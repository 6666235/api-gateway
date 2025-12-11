from fastapi import FastAPI, HTTPException, Header, Depends, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
import httpx
import os
import json
import sqlite3
import hashlib
import secrets
import time
import base64
import logging
import pyotp
import qrcode
import io
from datetime import datetime, timedelta
from typing import Optional, AsyncGenerator, List, Dict, Any
from contextlib import contextmanager
from collections import defaultdict
from cryptography.fernet import Fernet
from functools import lru_cache
import asyncio
import threading
from queue import Queue

load_dotenv()

# OAuth 配置
GITHUB_CLIENT_ID = os.getenv("GITHUB_CLIENT_ID", "")
GITHUB_CLIENT_SECRET = os.getenv("GITHUB_CLIENT_SECRET", "")

# ========== 日志配置 ==========
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(levelname)s - [%(name)s] %(message)s',
    handlers=[
        logging.FileHandler('app.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ========== 请求缓存 ==========
class ResponseCache:
    def __init__(self, max_size: int = 100, ttl: int = 3600):
        self.cache: Dict[str, tuple] = {}  # key -> (response, timestamp)
        self.max_size = max_size
        self.ttl = ttl
        self.lock = threading.Lock()
    
    def _make_key(self, messages: list, model: str) -> str:
        content = json.dumps(messages, sort_keys=True) + model
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, messages: list, model: str) -> Optional[dict]:
        key = self._make_key(messages, model)
        with self.lock:
            if key in self.cache:
                response, timestamp = self.cache[key]
                if time.time() - timestamp < self.ttl:
                    logger.debug(f"Cache hit for {key[:8]}")
                    return response
                del self.cache[key]
        return None
    
    def set(self, messages: list, model: str, response: dict):
        key = self._make_key(messages, model)
        with self.lock:
            if len(self.cache) >= self.max_size:
                # 删除最旧的条目
                oldest = min(self.cache.items(), key=lambda x: x[1][1])
                del self.cache[oldest[0]]
            self.cache[key] = (response, time.time())
            logger.debug(f"Cached response for {key[:8]}")
    
    def clear(self):
        with self.lock:
            self.cache.clear()

response_cache = ResponseCache(max_size=200, ttl=1800)  # 30分钟缓存

# ========== 数据库连接池 ==========
class DatabasePool:
    def __init__(self, db_path: str, pool_size: int = 5):
        self.db_path = db_path
        self.pool_size = pool_size
        self.pool: Queue = Queue(maxsize=pool_size)
        self.lock = threading.Lock()
        self._init_pool()
    
    def _init_pool(self):
        for _ in range(self.pool_size):
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            self.pool.put(conn)
        logger.info(f"Database pool initialized with {self.pool_size} connections")
    
    @contextmanager
    def get_connection(self):
        conn = self.pool.get()
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            self.pool.put(conn)
    
    def close_all(self):
        while not self.pool.empty():
            conn = self.pool.get()
            conn.close()

logger.info("AI Hub starting...")

# ========== API Key 加密 ==========
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY")
if not ENCRYPTION_KEY:
    # 首次运行自动生成密钥
    ENCRYPTION_KEY = Fernet.generate_key().decode()
    with open(".env", "a") as f:
        f.write(f"\nENCRYPTION_KEY={ENCRYPTION_KEY}")
    logger.info("Generated new encryption key")

cipher = Fernet(ENCRYPTION_KEY.encode() if isinstance(ENCRYPTION_KEY, str) else ENCRYPTION_KEY)

def encrypt_api_key(key: str) -> str:
    """加密 API Key"""
    if not key:
        return ""
    return cipher.encrypt(key.encode()).decode()

def decrypt_api_key(encrypted: str) -> str:
    """解密 API Key"""
    if not encrypted:
        return ""
    try:
        return cipher.decrypt(encrypted.encode()).decode()
    except:
        return encrypted  # 兼容未加密的旧数据

# ========== 速率限制 ==========
class RateLimiter:
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests = defaultdict(list)
        self.lock = asyncio.Lock()
    
    async def is_allowed(self, key: str) -> bool:
        async with self.lock:
            now = time.time()
            minute_ago = now - 60
            # 清理过期记录
            self.requests[key] = [t for t in self.requests[key] if t > minute_ago]
            if len(self.requests[key]) >= self.requests_per_minute:
                return False
            self.requests[key].append(now)
            return True
    
    def get_remaining(self, key: str) -> int:
        now = time.time()
        minute_ago = now - 60
        recent = [t for t in self.requests[key] if t > minute_ago]
        return max(0, self.requests_per_minute - len(recent))

rate_limiter = RateLimiter(requests_per_minute=60)

app = FastAPI(title="AI Hub", description="统一 AI 平台")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 数据库初始化
DB_PATH = "data.db"

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                email TEXT,
                plan TEXT DEFAULT 'free',
                tokens_used INTEGER DEFAULT 0,
                tokens_limit INTEGER DEFAULT 10000,
                totp_secret TEXT,
                totp_enabled INTEGER DEFAULT 0,
                github_id TEXT,
                avatar_url TEXT,
                locale TEXT DEFAULT 'zh-CN',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                user_id INTEGER,
                expires_at TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                user_id INTEGER,
                title TEXT,
                provider TEXT,
                model TEXT,
                tags TEXT DEFAULT '',
                pinned INTEGER DEFAULT 0,
                archived INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT,
                role TEXT,
                content TEXT,
                tokens INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            );
            CREATE TABLE IF NOT EXISTS notes (
                id TEXT PRIMARY KEY,
                user_id INTEGER,
                title TEXT,
                content TEXT,
                tags TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                content TEXT,
                category TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );
            CREATE TABLE IF NOT EXISTS shortcuts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                name TEXT,
                content TEXT,
                hotkey TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );
            CREATE TABLE IF NOT EXISTS settings (
                user_id INTEGER PRIMARY KEY,
                data TEXT DEFAULT '{}',
                FOREIGN KEY (user_id) REFERENCES users(id)
            );
            CREATE TABLE IF NOT EXISTS payments (
                id TEXT PRIMARY KEY,
                user_id INTEGER,
                plan TEXT,
                amount REAL,
                status TEXT DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );
            CREATE TABLE IF NOT EXISTS api_keys (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                provider TEXT,
                api_key TEXT,
                base_url TEXT,
                is_active INTEGER DEFAULT 1,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );
            CREATE TABLE IF NOT EXISTS api_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                provider TEXT,
                model TEXT,
                tokens_input INTEGER DEFAULT 0,
                tokens_output INTEGER DEFAULT 0,
                latency_ms INTEGER DEFAULT 0,
                status TEXT,
                error TEXT,
                ip_address TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );
            CREATE INDEX IF NOT EXISTS idx_api_logs_user ON api_logs(user_id);
            CREATE INDEX IF NOT EXISTS idx_api_logs_created ON api_logs(created_at);
            CREATE TABLE IF NOT EXISTS prompt_templates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                name TEXT NOT NULL,
                content TEXT NOT NULL,
                category TEXT DEFAULT 'general',
                is_public INTEGER DEFAULT 0,
                use_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );
            CREATE TABLE IF NOT EXISTS shared_conversations (
                id TEXT PRIMARY KEY,
                conversation_id TEXT,
                user_id INTEGER,
                expires_at TIMESTAMP,
                view_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id),
                FOREIGN KEY (user_id) REFERENCES users(id)
            );
            CREATE TABLE IF NOT EXISTS message_ratings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                conversation_id TEXT,
                message_index INTEGER,
                rating INTEGER,
                model TEXT,
                feedback TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );
            CREATE TABLE IF NOT EXISTS teams (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                owner_id INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (owner_id) REFERENCES users(id)
            );
            CREATE TABLE IF NOT EXISTS team_members (
                team_id INTEGER,
                user_id INTEGER,
                role TEXT DEFAULT 'member',
                joined_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (team_id, user_id),
                FOREIGN KEY (team_id) REFERENCES teams(id),
                FOREIGN KEY (user_id) REFERENCES users(id)
            );
            CREATE TABLE IF NOT EXISTS api_tokens (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                name TEXT,
                token TEXT UNIQUE,
                permissions TEXT DEFAULT '[]',
                last_used TIMESTAMP,
                expires_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );
            CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                action TEXT,
                resource TEXT,
                details TEXT,
                ip_address TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );
            CREATE TABLE IF NOT EXISTS knowledge_bases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                name TEXT,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );
            CREATE TABLE IF NOT EXISTS knowledge_documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                kb_id INTEGER,
                filename TEXT,
                content TEXT,
                embedding TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (kb_id) REFERENCES knowledge_bases(id)
            );
        ''')
    
    # 添加预设 Prompt 模板
    with sqlite3.connect(DB_PATH) as conn:
        count = conn.execute("SELECT COUNT(*) FROM prompt_templates WHERE user_id = 0").fetchone()[0]
        if count == 0:
            default_prompts = [
                ("翻译助手", "请将以下内容翻译成{目标语言}，保持原文的语气和风格：\n\n{内容}", "translate", 1),
                ("代码审查", "请审查以下代码，指出潜在的问题、性能优化建议和最佳实践改进：\n\n```\n{代码}\n```", "coding", 1),
                ("文章摘要", "请用3-5个要点总结以下文章的核心内容：\n\n{文章内容}", "analysis", 1),
                ("创意写作", "请以{主题}为题，写一篇{字数}字左右的{文体}，要求{要求}", "creative", 1),
                ("SQL生成", "根据以下需求生成SQL查询语句：\n\n表结构：{表结构}\n需求：{需求}", "coding", 1),
                ("邮件润色", "请帮我润色以下邮件，使其更加专业和礼貌：\n\n{邮件内容}", "writing", 1),
                ("解释概念", "请用简单易懂的语言解释{概念}，并举一个生活中的例子", "general", 1),
                ("Bug修复", "以下代码有bug，请找出问题并修复：\n\n```\n{代码}\n```\n\n错误信息：{错误}", "coding", 1),
            ]
            for name, content, category, is_public in default_prompts:
                conn.execute(
                    "INSERT INTO prompt_templates (user_id, name, content, category, is_public) VALUES (0, ?, ?, ?, ?)",
                    (name, content, category, is_public)
                )
            logger.info("Added default prompt templates")

init_db()

# ========== 扩展数据库表 ==========
def init_extended_db():
    """初始化扩展功能的数据库表"""
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript('''
            -- RBAC 权限系统
            CREATE TABLE IF NOT EXISTS roles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                description TEXT,
                permissions TEXT DEFAULT '[]',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS user_roles (
                user_id INTEGER,
                role_id INTEGER,
                PRIMARY KEY (user_id, role_id),
                FOREIGN KEY (user_id) REFERENCES users(id),
                FOREIGN KEY (role_id) REFERENCES roles(id)
            );
            
            -- 向量存储（RAG）
            CREATE TABLE IF NOT EXISTS vector_documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                kb_id INTEGER,
                chunk_id TEXT,
                content TEXT,
                embedding BLOB,
                metadata TEXT DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (kb_id) REFERENCES knowledge_bases(id)
            );
            
            CREATE INDEX IF NOT EXISTS idx_vector_kb ON vector_documents(kb_id);
            
            -- 计费系统
            CREATE TABLE IF NOT EXISTS billing_plans (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                price REAL,
                tokens_limit INTEGER,
                features TEXT DEFAULT '[]',
                is_active INTEGER DEFAULT 1
            );
            
            CREATE TABLE IF NOT EXISTS subscriptions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                plan_id TEXT,
                status TEXT DEFAULT 'active',
                start_date TIMESTAMP,
                end_date TIMESTAMP,
                auto_renew INTEGER DEFAULT 1,
                FOREIGN KEY (user_id) REFERENCES users(id),
                FOREIGN KEY (plan_id) REFERENCES billing_plans(id)
            );
            
            CREATE TABLE IF NOT EXISTS invoices (
                id TEXT PRIMARY KEY,
                user_id INTEGER,
                subscription_id INTEGER,
                amount REAL,
                status TEXT DEFAULT 'pending',
                paid_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );
            
            CREATE TABLE IF NOT EXISTS usage_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                type TEXT,
                amount INTEGER,
                unit_price REAL,
                total_cost REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );
            
            -- 实时协作
            CREATE TABLE IF NOT EXISTS collaboration_sessions (
                id TEXT PRIMARY KEY,
                conversation_id TEXT,
                created_by INTEGER,
                is_active INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            );
            
            CREATE TABLE IF NOT EXISTS collaboration_participants (
                session_id TEXT,
                user_id INTEGER,
                role TEXT DEFAULT 'viewer',
                joined_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (session_id, user_id),
                FOREIGN KEY (session_id) REFERENCES collaboration_sessions(id)
            );
            
            -- 插件系统
            CREATE TABLE IF NOT EXISTS plugins (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                version TEXT,
                author TEXT,
                config TEXT DEFAULT '{}',
                is_enabled INTEGER DEFAULT 0,
                installed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS user_plugins (
                user_id INTEGER,
                plugin_id TEXT,
                config TEXT DEFAULT '{}',
                enabled INTEGER DEFAULT 1,
                PRIMARY KEY (user_id, plugin_id),
                FOREIGN KEY (user_id) REFERENCES users(id),
                FOREIGN KEY (plugin_id) REFERENCES plugins(id)
            );
        ''')
        
        # 初始化默认角色
        default_roles = [
            ('admin', '管理员', '["*"]'),
            ('user', '普通用户', '["chat", "notes", "memory"]'),
            ('vip', 'VIP用户', '["chat", "notes", "memory", "image", "code", "rag"]'),
            ('guest', '访客', '["chat"]'),
        ]
        for name, desc, perms in default_roles:
            conn.execute(
                "INSERT OR IGNORE INTO roles (name, description, permissions) VALUES (?, ?, ?)",
                (name, desc, perms)
            )
        
        # 初始化计费套餐
        default_plans = [
            ('free', '免费版', 0, 10000, '["chat", "notes"]'),
            ('basic', '基础版', 19.9, 100000, '["chat", "notes", "memory", "shortcuts"]'),
            ('pro', '专业版', 49.9, 500000, '["chat", "notes", "memory", "shortcuts", "image", "code", "rag"]'),
            ('enterprise', '企业版', 199.9, -1, '["*"]'),
        ]
        for pid, name, price, tokens, features in default_plans:
            conn.execute(
                "INSERT OR IGNORE INTO billing_plans (id, name, price, tokens_limit, features) VALUES (?, ?, ?, ?, ?)",
                (pid, name, price, tokens, features)
            )
        
        logger.info("Extended database tables initialized")

init_extended_db()

# 密码哈希
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

# 获取当前用户（已禁用登录验证，返回默认用户）
async def get_current_user(authorization: Optional[str] = Header(None)):
    # 返回默认用户，无需登录
    default_user = {
        "id": 1,
        "username": "default",
        "email": None,
        "plan": "unlimited",
        "tokens_used": 0,
        "tokens_limit": -1,  # 无限制
        "totp_enabled": 0,
        "locale": "zh-CN"
    }
    
    # 确保默认用户存在于数据库中
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM users WHERE id = 1").fetchone()
        if row:
            return dict(row)
        else:
            # 创建默认用户
            conn.execute("""
                INSERT OR IGNORE INTO users (id, username, password_hash, plan, tokens_limit)
                VALUES (1, 'default', '', 'unlimited', -1)
            """)
            return default_user

# 平台配置
PROVIDERS = {
    "openai": {"base_url": "https://api.openai.com/v1", "env_key": "OPENAI_API_KEY", "type": "openai"},
    "claude": {"base_url": "https://api.anthropic.com/v1", "env_key": "ANTHROPIC_API_KEY", "type": "claude"},
    "gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta", "env_key": "GOOGLE_API_KEY", "type": "gemini"},
    "deepseek": {"base_url": "https://api.deepseek.com/v1", "env_key": "DEEPSEEK_API_KEY", "type": "openai"},
    "zhipu": {"base_url": "https://open.bigmodel.cn/api/paas/v4", "env_key": "ZHIPU_API_KEY", "type": "openai"},
    "moonshot": {"base_url": "https://api.moonshot.cn/v1", "env_key": "MOONSHOT_API_KEY", "type": "openai"},
    "qwen": {"base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "env_key": "QWEN_API_KEY", "type": "openai"},
    "doubao": {"base_url": "https://ark.cn-beijing.volces.com/api/v3", "env_key": "DOUBAO_API_KEY", "type": "openai"},
    "siliconflow": {"base_url": "https://api.siliconflow.cn/v1", "env_key": "SILICONFLOW_API_KEY", "type": "openai"},
    "groq": {"base_url": "https://api.groq.com/openai/v1", "env_key": "GROQ_API_KEY", "type": "openai"},
    "ollama": {"base_url": "http://localhost:11434/v1", "env_key": "OLLAMA_API_KEY", "type": "openai"},
}

PLANS = {
    "free": {"name": "免费版", "price": 0, "tokens": 10000, "features": ["基础对话", "3个对话历史"]},
    "basic": {"name": "基础版", "price": 19.9, "tokens": 100000, "features": ["无限对话", "笔记功能", "快捷短语"]},
    "pro": {"name": "专业版", "price": 49.9, "tokens": 500000, "features": ["所有基础功能", "网络搜索", "文档处理", "全局记忆"]},
    "unlimited": {"name": "无限版", "price": 99.9, "tokens": -1, "features": ["无限额度", "所有功能", "优先支持"]},
}

# ========== 数据模型 ==========
class UserRegister(BaseModel):
    username: str
    password: str
    email: Optional[str] = None

class UserLogin(BaseModel):
    username: str
    password: str

from typing import Union, Any

class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[Any]]  # 支持字符串或数组（图片消息）

class ChatRequest(BaseModel):
    provider: str
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 2048
    stream: Optional[bool] = False
    custom_url: Optional[str] = None
    api_key: Optional[str] = None
    conversation_id: Optional[str] = None
    web_search: Optional[bool] = False

class NoteCreate(BaseModel):
    title: str
    content: str
    tags: Optional[str] = ""

class MemoryCreate(BaseModel):
    content: str
    category: Optional[str] = "general"

class ShortcutCreate(BaseModel):
    name: str
    content: str
    hotkey: Optional[str] = ""

class SettingsUpdate(BaseModel):
    data: dict

class PaymentCreate(BaseModel):
    plan: str

# ========== 用户认证 ==========
@app.post("/api/auth/register")
async def register(user: UserRegister):
    with sqlite3.connect(DB_PATH) as conn:
        try:
            conn.execute(
                "INSERT INTO users (username, password_hash, email) VALUES (?, ?, ?)",
                (user.username, hash_password(user.password), user.email)
            )
            return {"success": True, "message": "注册成功"}
        except sqlite3.IntegrityError:
            raise HTTPException(status_code=400, detail="用户名已存在")

@app.post("/api/auth/login")
async def login(user: UserLogin):
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM users WHERE username = ? AND password_hash = ?",
            (user.username, hash_password(user.password))
        ).fetchone()
        if not row:
            raise HTTPException(status_code=401, detail="用户名或密码错误")
        token = secrets.token_hex(32)
        expires = datetime.now() + timedelta(days=30)
        conn.execute("INSERT INTO sessions (token, user_id, expires_at) VALUES (?, ?, ?)", (token, row["id"], expires))
        return {"token": token, "user": dict(row)}

@app.get("/api/auth/me")
async def get_me(user=Depends(get_current_user)):
    if not user:
        raise HTTPException(status_code=401, detail="未登录")
    return user

@app.post("/api/auth/logout")
async def logout(authorization: Optional[str] = Header(None)):
    if authorization and authorization.startswith("Bearer "):
        token = authorization[7:]
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
    return {"success": True}

# ========== 健康检查 ==========
@app.get("/health")
async def health_check():
    """健康检查接口"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("SELECT 1")
        db_status = "healthy"
    except:
        db_status = "unhealthy"
    
    return {
        "status": "ok" if db_status == "healthy" else "degraded",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "components": {
            "database": db_status,
            "api": "healthy"
        }
    }

@app.get("/api/stats")
async def get_stats(user=Depends(get_current_user)):
    """获取用户统计信息"""
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        
        # 今日调用次数
        today = datetime.now().strftime("%Y-%m-%d")
        today_calls = conn.execute(
            "SELECT COUNT(*) as cnt FROM api_logs WHERE user_id = ? AND date(created_at) = ?",
            (user["id"], today)
        ).fetchone()["cnt"]
        
        # 总调用次数
        total_calls = conn.execute(
            "SELECT COUNT(*) as cnt FROM api_logs WHERE user_id = ?",
            (user["id"],)
        ).fetchone()["cnt"]
        
        # 总 token 使用量
        token_stats = conn.execute(
            "SELECT COALESCE(SUM(tokens_input), 0) as input, COALESCE(SUM(tokens_output), 0) as output FROM api_logs WHERE user_id = ?",
            (user["id"],)
        ).fetchone()
        
        # 按模型统计
        model_stats = conn.execute(
            "SELECT provider, model, COUNT(*) as calls, SUM(tokens_input + tokens_output) as tokens FROM api_logs WHERE user_id = ? GROUP BY provider, model ORDER BY calls DESC LIMIT 10",
            (user["id"],)
        ).fetchall()
        
        # 最近7天趋势
        daily_stats = conn.execute(
            "SELECT date(created_at) as day, COUNT(*) as calls FROM api_logs WHERE user_id = ? AND created_at >= date('now', '-7 days') GROUP BY date(created_at) ORDER BY day",
            (user["id"],)
        ).fetchall()
        
        return {
            "today_calls": today_calls,
            "total_calls": total_calls,
            "tokens_input": token_stats["input"],
            "tokens_output": token_stats["output"],
            "model_stats": [dict(r) for r in model_stats],
            "daily_stats": [dict(r) for r in daily_stats]
        }

@app.get("/api/logs")
async def get_logs(limit: int = 50, user=Depends(get_current_user)):
    """获取 API 调用日志"""
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM api_logs WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
            (user["id"], limit)
        ).fetchall()
        return [dict(r) for r in rows]

# ========== 对话管理 ==========
@app.get("/api/conversations")
async def list_conversations(user=Depends(get_current_user)):
    user_id = user["id"] if user else 0
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM conversations WHERE user_id = ? ORDER BY updated_at DESC LIMIT 100",
            (user_id,)
        ).fetchall()
        return [dict(r) for r in rows]

@app.post("/api/conversations")
async def create_conversation(user=Depends(get_current_user)):
    user_id = user["id"] if user else 0
    conv_id = f"conv_{secrets.token_hex(8)}"
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO conversations (id, user_id, title) VALUES (?, ?, ?)",
            (conv_id, user_id, "新对话")
        )
    return {"id": conv_id, "title": "新对话"}

@app.get("/api/conversations/{conv_id}/messages")
async def get_messages(conv_id: str):
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at",
            (conv_id,)
        ).fetchall()
        return [dict(r) for r in rows]

@app.delete("/api/conversations/{conv_id}")
async def delete_conversation(conv_id: str):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conv_id,))
        conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
    return {"success": True}

@app.put("/api/conversations/{conv_id}")
async def update_conversation(conv_id: str, data: dict, user=Depends(get_current_user)):
    """更新对话信息（标题、标签等）"""
    with sqlite3.connect(DB_PATH) as conn:
        updates = []
        params = []
        if "title" in data:
            updates.append("title = ?")
            params.append(data["title"])
        if "tags" in data:
            updates.append("tags = ?")
            params.append(data["tags"])
        if "pinned" in data:
            updates.append("pinned = ?")
            params.append(1 if data["pinned"] else 0)
        
        if updates:
            params.append(conv_id)
            conn.execute(f"UPDATE conversations SET {', '.join(updates)}, updated_at = CURRENT_TIMESTAMP WHERE id = ?", params)
    
    return {"success": True}

@app.post("/api/conversations/{conv_id}/pin")
async def pin_conversation(conv_id: str, user=Depends(get_current_user)):
    """置顶对话"""
    with sqlite3.connect(DB_PATH) as conn:
        # 切换置顶状态
        current = conn.execute("SELECT pinned FROM conversations WHERE id = ?", (conv_id,)).fetchone()
        new_pinned = 0 if current and current[0] else 1
        conn.execute("UPDATE conversations SET pinned = ? WHERE id = ?", (new_pinned, conv_id))
    return {"success": True, "pinned": new_pinned == 1}

@app.post("/api/conversations/{conv_id}/archive")
async def archive_conversation(conv_id: str, user=Depends(get_current_user)):
    """归档对话"""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("UPDATE conversations SET archived = 1 WHERE id = ?", (conv_id,))
    return {"success": True}

@app.get("/api/conversations/archived")
async def list_archived_conversations(user=Depends(get_current_user)):
    """获取归档的对话"""
    if not user:
        return []
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM conversations WHERE user_id = ? AND archived = 1 ORDER BY updated_at DESC",
            (user["id"],)
        ).fetchall()
        return [dict(r) for r in rows]

@app.get("/api/search")
async def search_messages(q: str, user=Depends(get_current_user)):
    """搜索对话内容"""
    if not q or len(q) < 2:
        return {"results": []}
    user_id = user["id"] if user else 0
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT m.*, c.title as conv_title 
            FROM messages m 
            JOIN conversations c ON m.conversation_id = c.id 
            WHERE c.user_id = ? AND m.content LIKE ? 
            ORDER BY m.created_at DESC LIMIT 50
        """, (user_id, f"%{q}%")).fetchall()
        return {"results": [dict(r) for r in rows]}

@app.post("/api/tokens/estimate")
async def estimate_tokens(data: dict):
    """估算 Token 数量"""
    text = data.get("text", "")
    model = data.get("model", "gpt-4")
    try:
        import tiktoken
        try:
            enc = tiktoken.encoding_for_model(model)
        except:
            enc = tiktoken.get_encoding("cl100k_base")
        tokens = len(enc.encode(text))
        return {"tokens": tokens, "model": model}
    except Exception as e:
        # 简单估算：中文约1.5字符/token，英文约4字符/token
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        other_chars = len(text) - chinese_chars
        estimated = int(chinese_chars / 1.5 + other_chars / 4)
        return {"tokens": estimated, "model": model, "estimated": True}

@app.get("/api/conversations/{conv_id}/export")
async def export_conversation(conv_id: str, format: str = "markdown"):
    """导出对话为 Markdown 或 JSON"""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        conv = conn.execute("SELECT * FROM conversations WHERE id = ?", (conv_id,)).fetchone()
        if not conv:
            raise HTTPException(status_code=404, detail="对话不存在")
        
        messages = conn.execute(
            "SELECT role, content, created_at FROM messages WHERE conversation_id = ? ORDER BY created_at",
            (conv_id,)
        ).fetchall()
    
    if format == "json":
        return {
            "id": conv["id"],
            "title": conv["title"],
            "provider": conv["provider"],
            "model": conv["model"],
            "created_at": conv["created_at"],
            "messages": [dict(m) for m in messages]
        }
    else:
        # Markdown 格式
        lines = [f"# {conv['title'] or '对话记录'}", ""]
        lines.append(f"- 模型: {conv['provider']}/{conv['model']}")
        lines.append(f"- 时间: {conv['created_at']}")
        lines.append("")
        lines.append("---")
        lines.append("")
        
        for msg in messages:
            role_name = "👤 用户" if msg["role"] == "user" else "🤖 助手"
            content = msg["content"]
            # 处理图片消息
            if isinstance(content, str) and content.startswith("["):
                try:
                    content = json.loads(content)
                    content = "[图片消息]"
                except:
                    pass
            lines.append(f"### {role_name}")
            lines.append("")
            lines.append(str(content))
            lines.append("")
        
        return {"content": "\n".join(lines), "filename": f"{conv['title'] or 'chat'}_{conv_id}.md"}

# ========== Prompt 模板 ==========
@app.get("/api/prompts")
async def list_prompts(category: str = None, user=Depends(get_current_user)):
    """获取 Prompt 模板列表"""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        user_id = user["id"] if user else 0
        if category:
            rows = conn.execute(
                "SELECT * FROM prompt_templates WHERE (user_id = ? OR is_public = 1) AND category = ? ORDER BY use_count DESC",
                (user_id, category)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM prompt_templates WHERE user_id = ? OR is_public = 1 ORDER BY use_count DESC",
                (user_id,)
            ).fetchall()
        return [dict(r) for r in rows]

@app.post("/api/prompts")
async def create_prompt(data: dict, user=Depends(get_current_user)):
    """创建 Prompt 模板"""
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO prompt_templates (user_id, name, content, category, is_public) VALUES (?, ?, ?, ?, ?)",
            (user["id"], data.get("name"), data.get("content"), data.get("category", "general"), data.get("is_public", 0))
        )
    return {"success": True}

@app.put("/api/prompts/{prompt_id}")
async def update_prompt(prompt_id: int, data: dict, user=Depends(get_current_user)):
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "UPDATE prompt_templates SET name=?, content=?, category=? WHERE id=? AND user_id=?",
            (data.get("name"), data.get("content"), data.get("category"), prompt_id, user["id"])
        )
    return {"success": True}

@app.delete("/api/prompts/{prompt_id}")
async def delete_prompt(prompt_id: int, user=Depends(get_current_user)):
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM prompt_templates WHERE id=? AND user_id=?", (prompt_id, user["id"]))
    return {"success": True}

@app.post("/api/prompts/{prompt_id}/use")
async def use_prompt(prompt_id: int):
    """记录 Prompt 使用次数"""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("UPDATE prompt_templates SET use_count = use_count + 1 WHERE id = ?", (prompt_id,))
    return {"success": True}

# ========== 对话分享 ==========
@app.post("/api/conversations/{conv_id}/share")
async def share_conversation(conv_id: str, user=Depends(get_current_user)):
    """创建对话分享链接"""
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    share_id = secrets.token_urlsafe(12)
    expires = datetime.now() + timedelta(days=7)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO shared_conversations (id, conversation_id, user_id, expires_at) VALUES (?, ?, ?, ?)",
            (share_id, conv_id, user["id"], expires)
        )
    return {"share_id": share_id, "url": f"/share/{share_id}", "expires_at": expires.isoformat()}

@app.get("/api/share/{share_id}")
async def get_shared_conversation(share_id: str):
    """获取分享的对话"""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        share = conn.execute(
            "SELECT * FROM shared_conversations WHERE id = ? AND expires_at > ?",
            (share_id, datetime.now())
        ).fetchone()
        if not share:
            raise HTTPException(status_code=404, detail="分享链接不存在或已过期")
        
        # 更新查看次数
        conn.execute("UPDATE shared_conversations SET view_count = view_count + 1 WHERE id = ?", (share_id,))
        
        conv = conn.execute("SELECT * FROM conversations WHERE id = ?", (share["conversation_id"],)).fetchone()
        messages = conn.execute(
            "SELECT role, content, created_at FROM messages WHERE conversation_id = ? ORDER BY created_at",
            (share["conversation_id"],)
        ).fetchall()
        
        return {
            "conversation": dict(conv) if conv else None,
            "messages": [dict(m) for m in messages],
            "view_count": share["view_count"] + 1
        }

# ========== 消息评分 ==========
@app.post("/api/ratings")
async def create_rating(data: dict, user=Depends(get_current_user)):
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO message_ratings (user_id, conversation_id, message_index, rating, model, feedback) VALUES (?, ?, ?, ?, ?, ?)",
            (user["id"], data.get("conversation_id"), data.get("message_index"), data.get("rating"), data.get("model"), data.get("feedback"))
        )
    logger.info(f"User {user['id']} rated message: {data.get('rating')} stars")
    return {"success": True}

@app.get("/api/ratings/stats")
async def rating_stats(user=Depends(get_current_user)):
    """获取评分统计"""
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        stats = conn.execute("""
            SELECT model, AVG(rating) as avg_rating, COUNT(*) as count 
            FROM message_ratings WHERE user_id = ? 
            GROUP BY model ORDER BY avg_rating DESC
        """, (user["id"],)).fetchall()
        return [dict(r) for r in stats]

# ========== 团队空间 ==========
@app.post("/api/teams")
async def create_team(data: dict, user=Depends(get_current_user)):
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute("INSERT INTO teams (name, owner_id) VALUES (?, ?)", (data.get("name"), user["id"]))
        team_id = cursor.lastrowid
        conn.execute("INSERT INTO team_members (team_id, user_id, role) VALUES (?, ?, 'owner')", (team_id, user["id"]))
    return {"id": team_id, "success": True}

@app.get("/api/teams")
async def list_teams(user=Depends(get_current_user)):
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT t.*, tm.role FROM teams t 
            JOIN team_members tm ON t.id = tm.team_id 
            WHERE tm.user_id = ?
        """, (user["id"],)).fetchall()
        return [dict(r) for r in rows]

@app.post("/api/teams/{team_id}/members")
async def add_team_member(team_id: int, data: dict, user=Depends(get_current_user)):
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    with sqlite3.connect(DB_PATH) as conn:
        # 检查权限
        role = conn.execute("SELECT role FROM team_members WHERE team_id = ? AND user_id = ?", (team_id, user["id"])).fetchone()
        if not role or role[0] not in ['owner', 'admin']:
            raise HTTPException(status_code=403, detail="无权限")
        conn.execute("INSERT INTO team_members (team_id, user_id, role) VALUES (?, ?, ?)", 
                     (team_id, data.get("user_id"), data.get("role", "member")))
    return {"success": True}

# ========== API Token ==========
@app.post("/api/tokens")
async def create_api_token(data: dict, user=Depends(get_current_user)):
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    token = f"sk-{secrets.token_hex(24)}"
    expires = datetime.now() + timedelta(days=data.get("expires_days", 30))
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO api_tokens (user_id, name, token, permissions, expires_at) VALUES (?, ?, ?, ?, ?)",
            (user["id"], data.get("name", "API Token"), token, json.dumps(data.get("permissions", [])), expires)
        )
    return {"token": token, "expires_at": expires.isoformat()}

@app.get("/api/tokens")
async def list_api_tokens(user=Depends(get_current_user)):
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, name, token, permissions, last_used, expires_at, created_at FROM api_tokens WHERE user_id = ?",
            (user["id"],)
        ).fetchall()
        # 隐藏 token 中间部分
        result = []
        for r in rows:
            d = dict(r)
            d["token"] = d["token"][:10] + "..." + d["token"][-4:]
            result.append(d)
        return result

@app.delete("/api/tokens/{token_id}")
async def delete_api_token(token_id: int, user=Depends(get_current_user)):
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM api_tokens WHERE id = ? AND user_id = ?", (token_id, user["id"]))
    return {"success": True}

# ========== 审计日志 ==========
def log_audit(user_id: int, action: str, resource: str, details: str = "", ip: str = ""):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO audit_logs (user_id, action, resource, details, ip_address) VALUES (?, ?, ?, ?, ?)",
            (user_id, action, resource, details, ip)
        )

@app.get("/api/audit-logs")
async def get_audit_logs(limit: int = 100, user=Depends(get_current_user)):
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM audit_logs WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
            (user["id"], limit)
        ).fetchall()
        return [dict(r) for r in rows]

# ========== 知识库 ==========
@app.post("/api/knowledge-bases")
async def create_knowledge_base(data: dict, user=Depends(get_current_user)):
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(
            "INSERT INTO knowledge_bases (user_id, name, description) VALUES (?, ?, ?)",
            (user["id"], data.get("name"), data.get("description", ""))
        )
    return {"id": cursor.lastrowid, "success": True}

@app.get("/api/knowledge-bases")
async def list_knowledge_bases(user=Depends(get_current_user)):
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM knowledge_bases WHERE user_id = ?", (user["id"],)).fetchall()
        return [dict(r) for r in rows]

@app.post("/api/knowledge-bases/{kb_id}/documents")
async def upload_kb_document(kb_id: int, request: Request, user=Depends(get_current_user)):
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    form = await request.form()
    file = form.get("file")
    if not file:
        raise HTTPException(status_code=400, detail="请上传文件")
    content = await file.read()
    text = content.decode('utf-8', errors='ignore')
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO knowledge_documents (kb_id, filename, content) VALUES (?, ?, ?)",
            (kb_id, file.filename, text[:50000])  # 限制大小
        )
    return {"success": True, "filename": file.filename}

@app.get("/api/knowledge-bases/{kb_id}/search")
async def search_knowledge_base(kb_id: int, q: str, user=Depends(get_current_user)):
    """简单的知识库搜索"""
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM knowledge_documents WHERE kb_id = ? AND content LIKE ? LIMIT 10",
            (kb_id, f"%{q}%")
        ).fetchall()
        return [dict(r) for r in rows]

# ========== Agent 工作流 ==========
class WorkflowStep(BaseModel):
    name: str
    prompt: str
    model: Optional[str] = None
    depends_on: Optional[List[str]] = None

class WorkflowRequest(BaseModel):
    name: str
    steps: List[WorkflowStep]
    input_data: dict

@app.post("/api/workflows/run")
async def run_workflow(request: WorkflowRequest, user=Depends(get_current_user)):
    """执行 Agent 工作流"""
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    
    results = {}
    
    for step in request.steps:
        # 检查依赖
        if step.depends_on:
            for dep in step.depends_on:
                if dep not in results:
                    raise HTTPException(status_code=400, detail=f"步骤 {step.name} 依赖的 {dep} 未完成")
        
        # 构建 prompt，替换变量
        prompt = step.prompt
        for key, value in request.input_data.items():
            prompt = prompt.replace(f"{{{key}}}", str(value))
        for key, value in results.items():
            prompt = prompt.replace(f"{{result.{key}}}", str(value))
        
        # 调用 AI
        model = step.model or "gpt-4o"
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                # 使用用户配置的第一个 provider
                with sqlite3.connect(DB_PATH) as conn:
                    row = conn.execute(
                        "SELECT api_key, base_url FROM api_keys WHERE user_id = ? AND is_active = 1 LIMIT 1",
                        (user["id"],)
                    ).fetchone()
                
                if row:
                    api_key = decrypt_api_key(row[0]) if row[0] else ""
                    base_url = row[1] or "https://api.openai.com/v1"
                else:
                    api_key = os.getenv("OPENAI_API_KEY", "")
                    base_url = "https://api.openai.com/v1"
                
                response = await client.post(
                    f"{base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "stream": False
                    }
                )
                response.raise_for_status()
                data = response.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                results[step.name] = content
        except Exception as e:
            results[step.name] = f"Error: {str(e)}"
    
    return {"workflow": request.name, "results": results}

# 预设工作流模板
WORKFLOW_TEMPLATES = [
    {
        "id": "translate_review",
        "name": "翻译+审校",
        "description": "先翻译，再审校润色",
        "steps": [
            {"name": "translate", "prompt": "将以下内容翻译成{target_lang}：\n\n{content}"},
            {"name": "review", "prompt": "请审校以下翻译，修正错误并润色：\n\n{result.translate}", "depends_on": ["translate"]}
        ]
    },
    {
        "id": "code_review",
        "name": "代码审查+优化",
        "description": "审查代码问题，然后给出优化建议",
        "steps": [
            {"name": "review", "prompt": "审查以下代码的问题：\n\n{code}"},
            {"name": "optimize", "prompt": "基于以下审查结果，给出优化后的代码：\n\n审查结果：{result.review}\n\n原代码：{code}", "depends_on": ["review"]}
        ]
    },
    {
        "id": "content_pipeline",
        "name": "内容创作流水线",
        "description": "大纲→初稿→润色",
        "steps": [
            {"name": "outline", "prompt": "为主题「{topic}」创建一个详细大纲"},
            {"name": "draft", "prompt": "根据以下大纲写一篇文章：\n\n{result.outline}", "depends_on": ["outline"]},
            {"name": "polish", "prompt": "润色以下文章，使其更加流畅专业：\n\n{result.draft}", "depends_on": ["draft"]}
        ]
    }
]

@app.get("/api/workflows/templates")
async def get_workflow_templates():
    """获取预设工作流模板"""
    return WORKFLOW_TEMPLATES

# ========== 多 Agent 协作 ==========
class Agent:
    """AI Agent 基类"""
    def __init__(self, name: str, role: str, model: str = "gpt-4o-mini"):
        self.name = name
        self.role = role
        self.model = model
        self.memory = []
    
    async def think(self, task: str, context: dict = None) -> str:
        """Agent 思考并执行任务"""
        system_prompt = f"你是 {self.name}，角色是 {self.role}。请根据你的角色完成任务。"
        
        messages = [{"role": "system", "content": system_prompt}]
        
        if context:
            messages.append({"role": "user", "content": f"上下文信息：{json.dumps(context, ensure_ascii=False)}"})
        
        messages.append({"role": "user", "content": task})
        
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            return f"[{self.name}] 未配置 API Key"
        
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"model": self.model, "messages": messages, "max_tokens": 1000}
            )
            if response.status_code == 200:
                result = response.json()["choices"][0]["message"]["content"]
                self.memory.append({"task": task, "result": result})
                return result
        
        return f"[{self.name}] 执行失败"

class MultiAgentSystem:
    """多 Agent 协作系统"""
    def __init__(self):
        self.agents = {}
        self.conversations = {}
    
    def add_agent(self, agent: Agent):
        self.agents[agent.name] = agent
    
    async def collaborate(self, task: str, agent_names: list, mode: str = "sequential") -> dict:
        """多 Agent 协作执行任务"""
        results = {}
        context = {"task": task}
        
        if mode == "sequential":
            # 顺序执行
            for name in agent_names:
                if name in self.agents:
                    result = await self.agents[name].think(task, context)
                    results[name] = result
                    context[f"{name}_result"] = result
        
        elif mode == "parallel":
            # 并行执行
            import asyncio
            tasks = []
            for name in agent_names:
                if name in self.agents:
                    tasks.append(self.agents[name].think(task, context))
            
            if tasks:
                parallel_results = await asyncio.gather(*tasks)
                for i, name in enumerate(agent_names):
                    if i < len(parallel_results):
                        results[name] = parallel_results[i]
        
        elif mode == "debate":
            # 辩论模式
            for round_num in range(3):  # 3轮辩论
                for name in agent_names:
                    if name in self.agents:
                        debate_task = f"第{round_num + 1}轮辩论。任务：{task}\n\n其他观点：{json.dumps(results, ensure_ascii=False)}"
                        result = await self.agents[name].think(debate_task, context)
                        results[f"{name}_round{round_num + 1}"] = result
        
        return results

# 预设 Agent
multi_agent_system = MultiAgentSystem()
multi_agent_system.add_agent(Agent("分析师", "数据分析和问题诊断专家"))
multi_agent_system.add_agent(Agent("程序员", "代码编写和技术实现专家"))
multi_agent_system.add_agent(Agent("评审员", "代码审查和质量把控专家"))
multi_agent_system.add_agent(Agent("产品经理", "需求分析和产品设计专家"))
multi_agent_system.add_agent(Agent("测试员", "测试用例设计和质量验证专家"))

@app.get("/api/agents")
async def list_agents():
    """获取所有 Agent"""
    return [{"name": a.name, "role": a.role, "model": a.model} for a in multi_agent_system.agents.values()]

@app.post("/api/agents/collaborate")
async def agent_collaborate(data: dict, user=Depends(get_current_user)):
    """多 Agent 协作"""
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    
    task = data.get("task", "")
    agents = data.get("agents", ["分析师", "程序员"])
    mode = data.get("mode", "sequential")  # sequential, parallel, debate
    
    results = await multi_agent_system.collaborate(task, agents, mode)
    return {"task": task, "mode": mode, "results": results}

@app.post("/api/agents/create")
async def create_agent(data: dict, user=Depends(get_current_user)):
    """创建自定义 Agent"""
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    
    name = data.get("name")
    role = data.get("role")
    model = data.get("model", "gpt-4o-mini")
    
    if not name or not role:
        raise HTTPException(status_code=400, detail="名称和角色不能为空")
    
    multi_agent_system.add_agent(Agent(name, role, model))
    return {"success": True, "agent": {"name": name, "role": role, "model": model}}

# ========== 本地模型支持 (Ollama) ==========
OLLAMA_BASE_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

@app.get("/api/ollama/models")
async def list_ollama_models():
    """获取 Ollama 本地模型列表"""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            if response.status_code == 200:
                data = response.json()
                return {"models": data.get("models", [])}
    except:
        pass
    return {"models": [], "error": "Ollama 未运行或无法连接"}

@app.post("/api/ollama/pull")
async def pull_ollama_model(data: dict, user=Depends(get_current_user)):
    """拉取 Ollama 模型"""
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    
    model_name = data.get("model", "llama2")
    
    try:
        async with httpx.AsyncClient(timeout=300) as client:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/pull",
                json={"name": model_name}
            )
            return {"success": response.status_code == 200}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ollama/chat")
async def chat_with_ollama(data: dict, user=Depends(get_current_user)):
    """与 Ollama 本地模型对话"""
    model = data.get("model", "llama2")
    messages = data.get("messages", [])
    
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json={"model": model, "messages": messages, "stream": False}
            )
            if response.status_code == 200:
                return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ollama 错误: {str(e)}")
    
    raise HTTPException(status_code=500, detail="Ollama 请求失败")

# ========== vLLM 支持 ==========
VLLM_BASE_URL = os.getenv("VLLM_URL", "http://localhost:8080")

@app.get("/api/vllm/models")
async def list_vllm_models():
    """获取 vLLM 模型列表"""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(f"{VLLM_BASE_URL}/v1/models")
            if response.status_code == 200:
                return response.json()
    except:
        pass
    return {"data": [], "error": "vLLM 未运行或无法连接"}

@app.post("/api/vllm/chat")
async def chat_with_vllm(data: dict, user=Depends(get_current_user)):
    """与 vLLM 模型对话"""
    model = data.get("model")
    messages = data.get("messages", [])
    
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                f"{VLLM_BASE_URL}/v1/chat/completions",
                json={"model": model, "messages": messages}
            )
            if response.status_code == 200:
                return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"vLLM 错误: {str(e)}")
    
    raise HTTPException(status_code=500, detail="vLLM 请求失败")

# ========== Function Calling ==========
import re
import math

# 内置函数定义
BUILTIN_FUNCTIONS = {
    "get_weather": {
        "name": "get_weather",
        "description": "获取指定城市的天气信息",
        "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "城市名称"}}, "required": ["city"]}
    },
    "calculate": {
        "name": "calculate",
        "description": "计算数学表达式",
        "parameters": {"type": "object", "properties": {"expression": {"type": "string", "description": "数学表达式"}}, "required": ["expression"]}
    },
    "search_web": {
        "name": "search_web",
        "description": "搜索网络获取信息",
        "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "搜索关键词"}}, "required": ["query"]}
    },
    "get_time": {
        "name": "get_time",
        "description": "获取当前时间",
        "parameters": {"type": "object", "properties": {"timezone": {"type": "string", "description": "时区，如 Asia/Shanghai"}}}
    },
    "translate": {
        "name": "translate",
        "description": "翻译文本",
        "parameters": {"type": "object", "properties": {"text": {"type": "string"}, "target_lang": {"type": "string"}}, "required": ["text", "target_lang"]}
    }
}

async def execute_function(name: str, args: dict) -> str:
    """执行内置函数"""
    if name == "get_weather":
        city = args.get("city", "北京")
        # 模拟天气数据
        import random
        temp = random.randint(15, 30)
        conditions = ["晴", "多云", "阴", "小雨"]
        return f"{city}天气：{random.choice(conditions)}，温度 {temp}°C，湿度 {random.randint(40, 80)}%"
    
    elif name == "calculate":
        expr = args.get("expression", "0")
        try:
            # 安全计算
            allowed = set("0123456789+-*/().^ ")
            if all(c in allowed for c in expr):
                expr = expr.replace("^", "**")
                result = eval(expr)
                return f"计算结果：{expr} = {result}"
            return "表达式包含不允许的字符"
        except Exception as e:
            return f"计算错误：{str(e)}"
    
    elif name == "search_web":
        query = args.get("query", "")
        # 模拟搜索结果
        return f"搜索「{query}」的结果：\n1. {query}相关信息...\n2. {query}详细介绍...\n3. {query}最新动态..."
    
    elif name == "get_time":
        from datetime import datetime
        tz = args.get("timezone", "Asia/Shanghai")
        now = datetime.now()
        return f"当前时间：{now.strftime('%Y-%m-%d %H:%M:%S')} ({tz})"
    
    elif name == "translate":
        text = args.get("text", "")
        target = args.get("target_lang", "en")
        return f"[翻译结果] {text} -> ({target}) ..."
    
    return f"未知函数：{name}"

@app.post("/api/functions/call")
async def call_function(data: dict, user=Depends(get_current_user)):
    """调用内置函数"""
    name = data.get("name")
    args = data.get("arguments", {})
    if name not in BUILTIN_FUNCTIONS:
        raise HTTPException(status_code=400, detail=f"未知函数：{name}")
    result = await execute_function(name, args)
    return {"result": result}

@app.get("/api/functions")
async def list_functions():
    """获取可用函数列表"""
    return list(BUILTIN_FUNCTIONS.values())

# ========== 多模态支持 ==========
@app.post("/api/images/generate")
async def generate_image(data: dict, user=Depends(get_current_user)):
    """生成图片（DALL-E）"""
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    
    prompt = data.get("prompt", "")
    size = data.get("size", "1024x1024")
    
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=500, detail="未配置 OpenAI API Key")
    
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            "https://api.openai.com/v1/images/generations",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": "dall-e-3", "prompt": prompt, "n": 1, "size": size}
        )
        if response.status_code == 200:
            data = response.json()
            return {"url": data["data"][0]["url"], "revised_prompt": data["data"][0].get("revised_prompt")}
        else:
            raise HTTPException(status_code=response.status_code, detail=response.text)

@app.post("/api/audio/tts")
async def text_to_speech(data: dict, user=Depends(get_current_user)):
    """文字转语音（TTS）"""
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    
    text = data.get("text", "")
    voice = data.get("voice", "alloy")  # alloy, echo, fable, onyx, nova, shimmer
    
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=500, detail="未配置 OpenAI API Key")
    
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            "https://api.openai.com/v1/audio/speech",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": "tts-1", "input": text, "voice": voice}
        )
        if response.status_code == 200:
            audio_base64 = base64.b64encode(response.content).decode()
            return {"audio": f"data:audio/mp3;base64,{audio_base64}"}
        else:
            raise HTTPException(status_code=response.status_code, detail=response.text)

# ========== 长文本处理 ==========
def chunk_text(text: str, max_tokens: int = 4000) -> List[str]:
    """将长文本分块"""
    # 简单按字符数分块（约4字符=1token）
    chunk_size = max_tokens * 4
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])
    return chunks

@app.post("/api/text/chunk")
async def chunk_long_text(data: dict):
    """分块处理长文本"""
    text = data.get("text", "")
    max_tokens = data.get("max_tokens", 4000)
    chunks = chunk_text(text, max_tokens)
    return {"chunks": chunks, "count": len(chunks)}

@app.post("/api/text/summarize-long")
async def summarize_long_text(data: dict, user=Depends(get_current_user)):
    """摘要长文本（分块处理）"""
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    
    text = data.get("text", "")
    chunks = chunk_text(text, 3000)
    
    summaries = []
    api_key = os.getenv("OPENAI_API_KEY", "")
    
    async with httpx.AsyncClient(timeout=120) as client:
        for i, chunk in enumerate(chunks):
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": f"请用3-5句话总结以下内容：\n\n{chunk}"}],
                    "max_tokens": 500
                }
            )
            if response.status_code == 200:
                result = response.json()
                summaries.append(result["choices"][0]["message"]["content"])
    
    # 合并摘要
    if len(summaries) > 1:
        combined = "\n\n".join(summaries)
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": f"请将以下多段摘要合并为一个完整的摘要：\n\n{combined}"}],
                "max_tokens": 1000
            }
        )
        if response.status_code == 200:
            result = response.json()
            return {"summary": result["choices"][0]["message"]["content"], "chunks_processed": len(chunks)}
    
    return {"summary": summaries[0] if summaries else "", "chunks_processed": len(chunks)}

# ========== 上下文压缩 ==========
@app.post("/api/context/compress")
async def compress_context(data: dict, user=Depends(get_current_user)):
    """压缩对话上下文"""
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    
    messages = data.get("messages", [])
    max_tokens = data.get("max_tokens", 2000)
    
    if len(messages) <= 4:
        return {"messages": messages, "compressed": False}
    
    # 保留最近的消息，压缩较早的消息
    recent = messages[-4:]
    older = messages[:-4]
    
    # 将较早的消息压缩为摘要
    older_text = "\n".join([f"{m['role']}: {m['content']}" for m in older])
    
    api_key = os.getenv("OPENAI_API_KEY", "")
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": f"请用2-3句话总结以下对话的要点：\n\n{older_text}"}],
                "max_tokens": 200
            }
        )
        if response.status_code == 200:
            result = response.json()
            summary = result["choices"][0]["message"]["content"]
            compressed = [{"role": "system", "content": f"之前的对话摘要：{summary}"}] + recent
            return {"messages": compressed, "compressed": True, "original_count": len(messages), "new_count": len(compressed)}
    
    return {"messages": messages, "compressed": False}

# ========== 敏感词过滤 ==========
SENSITIVE_WORDS = ["暴力", "色情", "赌博", "毒品"]  # 示例敏感词

def filter_sensitive(text: str) -> tuple:
    """过滤敏感词"""
    found = []
    filtered = text
    for word in SENSITIVE_WORDS:
        if word in text:
            found.append(word)
            filtered = filtered.replace(word, "*" * len(word))
    return filtered, found

@app.post("/api/content/filter")
async def filter_content(data: dict):
    """内容安全过滤"""
    text = data.get("text", "")
    filtered, found = filter_sensitive(text)
    return {"filtered": filtered, "sensitive_words": found, "is_safe": len(found) == 0}

# ========== 数据脱敏 ==========
def mask_sensitive_data(text: str) -> str:
    """自动脱敏敏感信息"""
    # 手机号
    text = re.sub(r'1[3-9]\d{9}', lambda m: m.group()[:3] + '****' + m.group()[-4:], text)
    # 邮箱
    text = re.sub(r'[\w.-]+@[\w.-]+\.\w+', lambda m: m.group()[:3] + '***@***', text)
    # 身份证
    text = re.sub(r'\d{17}[\dXx]', lambda m: m.group()[:6] + '********' + m.group()[-4:], text)
    # 银行卡
    text = re.sub(r'\d{16,19}', lambda m: m.group()[:4] + ' **** **** ' + m.group()[-4:], text)
    return text

@app.post("/api/data/mask")
async def mask_data(data: dict):
    """数据脱敏"""
    text = data.get("text", "")
    masked = mask_sensitive_data(text)
    return {"masked": masked}

# ========== Prometheus 指标 ==========
METRICS = {
    "requests_total": 0,
    "requests_success": 0,
    "requests_error": 0,
    "tokens_used": 0,
    "active_users": 0,
    "response_time_sum": 0,
    "response_time_count": 0
}

@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus 指标端点"""
    lines = [
        f"# HELP ai_hub_requests_total Total requests",
        f"# TYPE ai_hub_requests_total counter",
        f"ai_hub_requests_total {METRICS['requests_total']}",
        f"# HELP ai_hub_requests_success Successful requests",
        f"ai_hub_requests_success {METRICS['requests_success']}",
        f"# HELP ai_hub_requests_error Error requests",
        f"ai_hub_requests_error {METRICS['requests_error']}",
        f"# HELP ai_hub_tokens_used Total tokens used",
        f"ai_hub_tokens_used {METRICS['tokens_used']}",
        f"# HELP ai_hub_active_users Active users",
        f"ai_hub_active_users {METRICS['active_users']}",
    ]
    if METRICS['response_time_count'] > 0:
        avg_time = METRICS['response_time_sum'] / METRICS['response_time_count']
        lines.append(f"# HELP ai_hub_response_time_avg Average response time")
        lines.append(f"ai_hub_response_time_avg {avg_time:.3f}")
    return "\n".join(lines)

# ========== 自动备份 ==========
import shutil

@app.post("/api/backup")
async def create_backup(user=Depends(get_current_user)):
    """创建数据库备份"""
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"backups/data_{timestamp}.db"
    os.makedirs("backups", exist_ok=True)
    shutil.copy(DB_PATH, backup_path)
    logger.info(f"Backup created: {backup_path}")
    return {"success": True, "path": backup_path, "timestamp": timestamp}

@app.get("/api/backups")
async def list_backups(user=Depends(get_current_user)):
    """列出备份"""
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    
    if not os.path.exists("backups"):
        return []
    
    backups = []
    for f in os.listdir("backups"):
        if f.endswith(".db"):
            path = os.path.join("backups", f)
            backups.append({
                "filename": f,
                "size": os.path.getsize(path),
                "created": datetime.fromtimestamp(os.path.getctime(path)).isoformat()
            })
    return sorted(backups, key=lambda x: x["created"], reverse=True)

# ========== Webhook ==========
WEBHOOKS = []

@app.post("/api/webhooks")
async def create_webhook(data: dict, user=Depends(get_current_user)):
    """创建 Webhook"""
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    
    webhook = {
        "id": secrets.token_hex(8),
        "user_id": user["id"],
        "url": data.get("url"),
        "events": data.get("events", ["message.created"]),
        "secret": secrets.token_hex(16),
        "created_at": datetime.now().isoformat()
    }
    WEBHOOKS.append(webhook)
    return webhook

@app.get("/api/webhooks")
async def list_webhooks(user=Depends(get_current_user)):
    """列出 Webhook"""
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    return [w for w in WEBHOOKS if w["user_id"] == user["id"]]

async def trigger_webhook(event: str, data: dict, user_id: int):
    """触发 Webhook"""
    for webhook in WEBHOOKS:
        if webhook["user_id"] == user_id and event in webhook["events"]:
            try:
                async with httpx.AsyncClient(timeout=10) as client:
                    await client.post(webhook["url"], json={"event": event, "data": data})
            except:
                pass

# ========== 代码运行器 ==========
import subprocess
import tempfile

@app.post("/api/code/run")
async def run_code(data: dict, user=Depends(get_current_user)):
    """在线运行代码（沙箱环境）"""
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    
    code = data.get("code", "")
    language = data.get("language", "python")
    timeout = min(data.get("timeout", 5), 10)  # 最大10秒
    
    if language == "python":
        # 安全检查
        forbidden = ["import os", "import subprocess", "import sys", "exec(", "eval(", "__import__", "open("]
        for f in forbidden:
            if f in code:
                return {"success": False, "error": f"禁止使用: {f}", "output": ""}
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
                f.write(code)
                f.flush()
                result = subprocess.run(
                    ["python", f.name],
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )
                os.unlink(f.name)
                return {
                    "success": result.returncode == 0,
                    "output": result.stdout,
                    "error": result.stderr
                }
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "执行超时", "output": ""}
        except Exception as e:
            return {"success": False, "error": str(e), "output": ""}
    
    elif language == "javascript":
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False, encoding='utf-8') as f:
                f.write(code)
                f.flush()
                result = subprocess.run(
                    ["node", f.name],
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )
                os.unlink(f.name)
                return {
                    "success": result.returncode == 0,
                    "output": result.stdout,
                    "error": result.stderr
                }
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "执行超时", "output": ""}
        except Exception as e:
            return {"success": False, "error": str(e), "output": ""}
    
    return {"success": False, "error": f"不支持的语言: {language}", "output": ""}

# ========== 思维导图 ==========
@app.post("/api/mindmap/generate")
async def generate_mindmap(data: dict, user=Depends(get_current_user)):
    """将对话转换为思维导图数据"""
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    
    messages = data.get("messages", [])
    title = data.get("title", "对话思维导图")
    
    # 提取关键信息生成思维导图结构
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        # 简单提取
        nodes = [{"id": "root", "label": title, "children": []}]
        for i, msg in enumerate(messages):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")[:100]
                nodes[0]["children"].append({"id": f"node_{i}", "label": content})
        return {"nodes": nodes}
    
    # 使用 AI 生成结构化思维导图
    conversation = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
    
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": "gpt-4o-mini",
                "messages": [{
                    "role": "user",
                    "content": f"请将以下对话内容转换为思维导图的JSON结构，格式为：{{\"nodes\": [{{\"id\": \"root\", \"label\": \"主题\", \"children\": [{{\"id\": \"1\", \"label\": \"子节点\"}}]}}]}}。对话内容：\n\n{conversation[:3000]}"
                }],
                "max_tokens": 1000
            }
        )
        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            try:
                # 提取 JSON
                import re
                json_match = re.search(r'\{[\s\S]*\}', content)
                if json_match:
                    return json.loads(json_match.group())
            except:
                pass
    
    return {"nodes": [{"id": "root", "label": title, "children": []}]}

# ========== 对话模板 ==========
@app.post("/api/chat-templates")
async def save_chat_template(data: dict, user=Depends(get_current_user)):
    """保存对话为模板"""
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO prompt_templates (user_id, name, content, category, is_public)
            VALUES (?, ?, ?, 'chat_template', 0)
        """, (user["id"], data.get("name"), json.dumps(data.get("messages", []))))
    return {"success": True}

@app.get("/api/chat-templates")
async def list_chat_templates(user=Depends(get_current_user)):
    """获取对话模板列表"""
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM prompt_templates WHERE user_id = ? AND category = 'chat_template'",
            (user["id"],)
        ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            try:
                d["messages"] = json.loads(d["content"])
            except:
                d["messages"] = []
            result.append(d)
        return result

# ========== 用量配额管理 ==========
@app.get("/api/quota")
async def get_quota(user=Depends(get_current_user)):
    """获取用户配额"""
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    
    return {
        "tokens_used": user.get("tokens_used", 0),
        "tokens_limit": user.get("tokens_limit", 10000),
        "plan": user.get("plan", "free"),
        "remaining": max(0, user.get("tokens_limit", 10000) - user.get("tokens_used", 0)),
        "usage_percent": min(100, (user.get("tokens_used", 0) / max(1, user.get("tokens_limit", 10000))) * 100)
    }

@app.post("/api/quota/reset")
async def reset_quota(data: dict, user=Depends(get_current_user)):
    """重置用户配额（管理员）"""
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    
    target_user_id = data.get("user_id", user["id"])
    new_limit = data.get("limit", 10000)
    
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "UPDATE users SET tokens_used = 0, tokens_limit = ? WHERE id = ?",
            (new_limit, target_user_id)
        )
    return {"success": True}

# ========== 链路追踪 ==========
import uuid

class TraceContext:
    def __init__(self):
        self.traces = {}
    
    def start_trace(self, name: str) -> str:
        trace_id = str(uuid.uuid4())[:8]
        self.traces[trace_id] = {
            "id": trace_id,
            "name": name,
            "start_time": time.time(),
            "spans": []
        }
        return trace_id
    
    def add_span(self, trace_id: str, name: str, duration: float, metadata: dict = None):
        if trace_id in self.traces:
            self.traces[trace_id]["spans"].append({
                "name": name,
                "duration": duration,
                "metadata": metadata or {}
            })
    
    def end_trace(self, trace_id: str) -> dict:
        if trace_id in self.traces:
            trace = self.traces[trace_id]
            trace["end_time"] = time.time()
            trace["total_duration"] = trace["end_time"] - trace["start_time"]
            return trace
        return None

tracer = TraceContext()

@app.get("/api/traces")
async def list_traces(limit: int = 20, user=Depends(get_current_user)):
    """获取最近的追踪记录"""
    traces = list(tracer.traces.values())[-limit:]
    return traces

@app.get("/api/traces/{trace_id}")
async def get_trace(trace_id: str):
    """获取单个追踪详情"""
    if trace_id in tracer.traces:
        return tracer.traces[trace_id]
    raise HTTPException(status_code=404, detail="追踪记录不存在")

@app.post("/api/traces/export")
async def export_traces(data: dict, user=Depends(get_current_user)):
    """导出追踪数据（Jaeger 格式）"""
    trace_ids = data.get("trace_ids", [])
    format_type = data.get("format", "jaeger")
    
    traces_data = []
    for tid in trace_ids:
        if tid in tracer.traces:
            traces_data.append(tracer.traces[tid])
    
    if format_type == "jaeger":
        # 转换为 Jaeger 格式
        jaeger_traces = []
        for t in traces_data:
            jaeger_traces.append({
                "traceID": t["id"],
                "spans": [{
                    "traceID": t["id"],
                    "spanID": s.get("name", ""),
                    "operationName": s.get("name", ""),
                    "duration": int(s.get("duration", 0) * 1000000),  # 微秒
                    "tags": [{"key": k, "value": v} for k, v in s.get("metadata", {}).items()]
                } for s in t.get("spans", [])],
                "processes": {"p1": {"serviceName": "ai-hub"}}
            })
        return {"data": jaeger_traces}
    
    return {"data": traces_data}

# ========== Slack 集成 ==========
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN", "")
SLACK_SIGNING_SECRET = os.getenv("SLACK_SIGNING_SECRET", "")

@app.post("/api/integrations/slack/webhook")
async def slack_webhook(request: Request):
    """Slack 事件 Webhook"""
    body = await request.json()
    
    # URL 验证
    if body.get("type") == "url_verification":
        return {"challenge": body.get("challenge")}
    
    # 处理事件
    event = body.get("event", {})
    event_type = event.get("type")
    
    if event_type == "app_mention":
        # 被 @ 提及时回复
        channel = event.get("channel")
        text = event.get("text", "")
        user = event.get("user")
        
        # 移除 @ 提及
        import re
        text = re.sub(r'<@[A-Z0-9]+>', '', text).strip()
        
        if text and SLACK_BOT_TOKEN:
            # 调用 AI 生成回复
            api_key = os.getenv("OPENAI_API_KEY", "")
            if api_key:
                async with httpx.AsyncClient(timeout=30) as client:
                    ai_response = await client.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers={"Authorization": f"Bearer {api_key}"},
                        json={
                            "model": "gpt-4o-mini",
                            "messages": [{"role": "user", "content": text}],
                            "max_tokens": 500
                        }
                    )
                    if ai_response.status_code == 200:
                        reply = ai_response.json()["choices"][0]["message"]["content"]
                        
                        # 发送到 Slack
                        await client.post(
                            "https://slack.com/api/chat.postMessage",
                            headers={"Authorization": f"Bearer {SLACK_BOT_TOKEN}"},
                            json={"channel": channel, "text": reply}
                        )
    
    return {"ok": True}

@app.post("/api/integrations/slack/send")
async def send_slack_message(data: dict, user=Depends(get_current_user)):
    """发送消息到 Slack"""
    if not SLACK_BOT_TOKEN:
        raise HTTPException(status_code=500, detail="Slack 未配置")
    
    channel = data.get("channel")
    text = data.get("text")
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://slack.com/api/chat.postMessage",
            headers={"Authorization": f"Bearer {SLACK_BOT_TOKEN}"},
            json={"channel": channel, "text": text}
        )
        return response.json()

# ========== 飞书集成 ==========
FEISHU_APP_ID = os.getenv("FEISHU_APP_ID", "")
FEISHU_APP_SECRET = os.getenv("FEISHU_APP_SECRET", "")

async def get_feishu_token():
    """获取飞书 tenant_access_token"""
    if not FEISHU_APP_ID or not FEISHU_APP_SECRET:
        return None
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal",
            json={"app_id": FEISHU_APP_ID, "app_secret": FEISHU_APP_SECRET}
        )
        if response.status_code == 200:
            return response.json().get("tenant_access_token")
    return None

@app.post("/api/integrations/feishu/webhook")
async def feishu_webhook(request: Request):
    """飞书事件 Webhook"""
    body = await request.json()
    
    # URL 验证
    if body.get("type") == "url_verification":
        return {"challenge": body.get("challenge")}
    
    # 处理消息事件
    event = body.get("event", {})
    message = event.get("message", {})
    
    if message.get("message_type") == "text":
        content = json.loads(message.get("content", "{}"))
        text = content.get("text", "")
        chat_id = message.get("chat_id")
        
        if text:
            token = await get_feishu_token()
            if token:
                # 调用 AI 生成回复
                api_key = os.getenv("OPENAI_API_KEY", "")
                if api_key:
                    async with httpx.AsyncClient(timeout=30) as client:
                        ai_response = await client.post(
                            "https://api.openai.com/v1/chat/completions",
                            headers={"Authorization": f"Bearer {api_key}"},
                            json={
                                "model": "gpt-4o-mini",
                                "messages": [{"role": "user", "content": text}],
                                "max_tokens": 500
                            }
                        )
                        if ai_response.status_code == 200:
                            reply = ai_response.json()["choices"][0]["message"]["content"]
                            
                            # 发送到飞书
                            await client.post(
                                "https://open.feishu.cn/open-apis/im/v1/messages",
                                headers={"Authorization": f"Bearer {token}"},
                                params={"receive_id_type": "chat_id"},
                                json={
                                    "receive_id": chat_id,
                                    "msg_type": "text",
                                    "content": json.dumps({"text": reply})
                                }
                            )
    
    return {"ok": True}

@app.post("/api/integrations/feishu/send")
async def send_feishu_message(data: dict, user=Depends(get_current_user)):
    """发送消息到飞书"""
    token = await get_feishu_token()
    if not token:
        raise HTTPException(status_code=500, detail="飞书未配置")
    
    chat_id = data.get("chat_id")
    text = data.get("text")
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://open.feishu.cn/open-apis/im/v1/messages",
            headers={"Authorization": f"Bearer {token}"},
            params={"receive_id_type": "chat_id"},
            json={
                "receive_id": chat_id,
                "msg_type": "text",
                "content": json.dumps({"text": text})
            }
        )
        return response.json()

# ========== 灰度发布增强 ==========
class FeatureFlags:
    """功能开关管理"""
    def __init__(self):
        self.flags = {
            "new_ui": {"enabled": False, "percentage": 10, "users": [], "groups": []},
            "advanced_search": {"enabled": True, "percentage": 50, "users": [], "groups": []},
            "beta_features": {"enabled": False, "percentage": 5, "users": [], "groups": []},
            "ai_v2": {"enabled": False, "percentage": 0, "users": [], "groups": []},
        }
        self._load_from_db()
    
    def _load_from_db(self):
        """从数据库加载配置"""
        try:
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS feature_flags (
                        name TEXT PRIMARY KEY,
                        config TEXT
                    )
                """)
                rows = conn.execute("SELECT name, config FROM feature_flags").fetchall()
                for name, config in rows:
                    self.flags[name] = json.loads(config)
        except:
            pass
    
    def _save_to_db(self, name: str):
        """保存配置到数据库"""
        try:
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO feature_flags (name, config) VALUES (?, ?)",
                    (name, json.dumps(self.flags.get(name, {})))
                )
        except:
            pass
    
    def is_enabled(self, feature: str, user_id: int = None, group: str = None) -> bool:
        """检查功能是否对用户启用"""
        flag = self.flags.get(feature)
        if not flag or not flag.get("enabled"):
            return False
        
        # 白名单用户
        if user_id and user_id in flag.get("users", []):
            return True
        
        # 白名单组
        if group and group in flag.get("groups", []):
            return True
        
        # 百分比灰度
        if user_id:
            return (user_id % 100) < flag.get("percentage", 0)
        
        return False
    
    def update(self, feature: str, config: dict):
        """更新功能配置"""
        if feature not in self.flags:
            self.flags[feature] = {}
        self.flags[feature].update(config)
        self._save_to_db(feature)

feature_flags = FeatureFlags()

@app.get("/api/features")
async def get_features(user=Depends(get_current_user)):
    """获取用户可用的功能"""
    user_id = user["id"] if user else None
    enabled_features = {}
    for feature in feature_flags.flags:
        enabled_features[feature] = feature_flags.is_enabled(feature, user_id)
    return enabled_features

@app.get("/api/features/all")
async def get_all_features(user=Depends(get_current_user)):
    """获取所有功能配置（管理员）"""
    return feature_flags.flags

@app.post("/api/features/{feature}")
async def update_feature(feature: str, data: dict, user=Depends(get_current_user)):
    """更新功能配置（管理员）"""
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    
    feature_flags.update(feature, data)
    return {"success": True, "feature": feature, "config": feature_flags.flags.get(feature)}

@app.post("/api/features/{feature}/users")
async def add_feature_user(feature: str, data: dict, user=Depends(get_current_user)):
    """添加功能白名单用户"""
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    
    user_id = data.get("user_id")
    if feature in feature_flags.flags and user_id:
        if "users" not in feature_flags.flags[feature]:
            feature_flags.flags[feature]["users"] = []
        if user_id not in feature_flags.flags[feature]["users"]:
            feature_flags.flags[feature]["users"].append(user_id)
            feature_flags._save_to_db(feature)
    
    return {"success": True}

# ========== Docker 沙箱代码执行 ==========
DOCKER_ENABLED = os.getenv("DOCKER_SANDBOX", "false").lower() == "true"

@app.post("/api/code/run-sandbox")
async def run_code_sandbox(data: dict, user=Depends(get_current_user)):
    """在 Docker 沙箱中运行代码"""
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    
    code = data.get("code", "")
    language = data.get("language", "python")
    timeout = min(data.get("timeout", 5), 30)
    
    if not DOCKER_ENABLED:
        # 降级到普通执行
        return await run_code(data, user)
    
    # Docker 镜像映射
    images = {
        "python": "python:3.11-slim",
        "javascript": "node:20-slim",
        "typescript": "node:20-slim",
        "go": "golang:1.21-alpine",
        "rust": "rust:1.74-slim",
        "java": "openjdk:17-slim",
    }
    
    image = images.get(language)
    if not image:
        return {"success": False, "error": f"不支持的语言: {language}", "output": ""}
    
    try:
        # 创建临时文件
        ext_map = {"python": ".py", "javascript": ".js", "typescript": ".ts", "go": ".go", "rust": ".rs", "java": ".java"}
        ext = ext_map.get(language, ".txt")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix=ext, delete=False, encoding='utf-8') as f:
            f.write(code)
            f.flush()
            
            # 运行命令映射
            cmd_map = {
                "python": f"python /code/main{ext}",
                "javascript": f"node /code/main{ext}",
                "typescript": f"npx ts-node /code/main{ext}",
                "go": f"go run /code/main{ext}",
                "rust": f"rustc /code/main{ext} -o /tmp/main && /tmp/main",
                "java": f"cd /code && javac Main.java && java Main",
            }
            
            cmd = cmd_map.get(language, f"cat /code/main{ext}")
            
            # Docker 运行
            docker_cmd = [
                "docker", "run", "--rm",
                "--network", "none",  # 禁用网络
                "--memory", "128m",   # 内存限制
                "--cpus", "0.5",      # CPU 限制
                "-v", f"{f.name}:/code/main{ext}:ro",
                image,
                "sh", "-c", cmd
            ]
            
            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            os.unlink(f.name)
            
            return {
                "success": result.returncode == 0,
                "output": result.stdout,
                "error": result.stderr,
                "sandbox": True
            }
    
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "执行超时", "output": "", "sandbox": True}
    except Exception as e:
        return {"success": False, "error": str(e), "output": "", "sandbox": True}

# ========== 笔记功能 ==========
@app.get("/api/notes")
async def list_notes(user=Depends(get_current_user)):
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM notes WHERE user_id = ? ORDER BY updated_at DESC", (user["id"],)).fetchall()
        return [dict(r) for r in rows]

@app.post("/api/notes")
async def create_note(note: NoteCreate, user=Depends(get_current_user)):
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    note_id = f"note_{secrets.token_hex(8)}"
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO notes (id, user_id, title, content, tags) VALUES (?, ?, ?, ?, ?)",
            (note_id, user["id"], note.title, note.content, note.tags)
        )
    return {"id": note_id, "success": True}

@app.put("/api/notes/{note_id}")
async def update_note(note_id: str, note: NoteCreate, user=Depends(get_current_user)):
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "UPDATE notes SET title=?, content=?, tags=?, updated_at=CURRENT_TIMESTAMP WHERE id=? AND user_id=?",
            (note.title, note.content, note.tags, note_id, user["id"])
        )
    return {"success": True}

@app.delete("/api/notes/{note_id}")
async def delete_note(note_id: str, user=Depends(get_current_user)):
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM notes WHERE id=? AND user_id=?", (note_id, user["id"]))
    return {"success": True}

# ========== 全局记忆 ==========
@app.get("/api/memories")
async def list_memories(user=Depends(get_current_user)):
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM memories WHERE user_id = ? ORDER BY created_at DESC", (user["id"],)).fetchall()
        return [dict(r) for r in rows]

@app.post("/api/memories")
async def create_memory(memory: MemoryCreate, user=Depends(get_current_user)):
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("INSERT INTO memories (user_id, content, category) VALUES (?, ?, ?)", (user["id"], memory.content, memory.category))
    return {"success": True}

@app.put("/api/memories/{memory_id}")
async def update_memory(memory_id: int, memory: MemoryCreate, user=Depends(get_current_user)):
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "UPDATE memories SET content=?, category=? WHERE id=? AND user_id=?",
            (memory.content, memory.category, memory_id, user["id"])
        )
    return {"success": True}

@app.delete("/api/memories/{memory_id}")
async def delete_memory(memory_id: int, user=Depends(get_current_user)):
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM memories WHERE id=? AND user_id=?", (memory_id, user["id"]))
    return {"success": True}

# ========== 快捷短语 ==========
@app.get("/api/shortcuts")
async def list_shortcuts(user=Depends(get_current_user)):
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM shortcuts WHERE user_id = ?", (user["id"],)).fetchall()
        return [dict(r) for r in rows]

@app.post("/api/shortcuts")
async def create_shortcut(shortcut: ShortcutCreate, user=Depends(get_current_user)):
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("INSERT INTO shortcuts (user_id, name, content, hotkey) VALUES (?, ?, ?, ?)", 
                     (user["id"], shortcut.name, shortcut.content, shortcut.hotkey))
    return {"success": True}

@app.delete("/api/shortcuts/{shortcut_id}")
async def delete_shortcut(shortcut_id: int, user=Depends(get_current_user)):
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM shortcuts WHERE id=? AND user_id=?", (shortcut_id, user["id"]))
    return {"success": True}

# ========== 设置 ==========
@app.get("/api/settings")
async def get_settings(user=Depends(get_current_user)):
    if not user:
        return {"data": {}}
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute("SELECT data FROM settings WHERE user_id = ?", (user["id"],)).fetchone()
        return {"data": json.loads(row[0]) if row else {}}

@app.put("/api/settings")
async def update_settings(settings: SettingsUpdate, user=Depends(get_current_user)):
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("INSERT OR REPLACE INTO settings (user_id, data) VALUES (?, ?)", (user["id"], json.dumps(settings.data)))
    return {"success": True}

# ========== 付费系统 ==========
@app.get("/api/plans")
async def get_plans():
    return PLANS

@app.post("/api/payments/create")
async def create_payment(payment: PaymentCreate, user=Depends(get_current_user)):
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    if payment.plan not in PLANS:
        raise HTTPException(status_code=400, detail="无效的套餐")
    plan = PLANS[payment.plan]
    payment_id = f"pay_{secrets.token_hex(8)}"
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO payments (id, user_id, plan, amount, status) VALUES (?, ?, ?, ?, ?)",
            (payment_id, user["id"], payment.plan, plan["price"], "pending")
        )
    # 返回支付信息（实际项目中这里会对接支付宝/微信支付）
    return {
        "payment_id": payment_id,
        "amount": plan["price"],
        "plan": payment.plan,
        "qrcode": f"https://example.com/pay/{payment_id}",  # 模拟支付二维码
    }

@app.post("/api/payments/{payment_id}/complete")
async def complete_payment(payment_id: str, user=Depends(get_current_user)):
    """模拟支付完成（实际项目中由支付回调触发）"""
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute("SELECT * FROM payments WHERE id = ? AND user_id = ?", (payment_id, user["id"])).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="订单不存在")
        plan = row[2]
        plan_info = PLANS.get(plan, {})
        expires = datetime.now() + timedelta(days=30)
        conn.execute("UPDATE payments SET status = 'completed' WHERE id = ?", (payment_id,))
        conn.execute(
            "UPDATE users SET plan = ?, tokens_limit = ?, expires_at = ? WHERE id = ?",
            (plan, plan_info.get("tokens", 10000), expires, user["id"])
        )
    return {"success": True, "message": "支付成功，套餐已激活"}

@app.get("/api/payments/history")
async def payment_history(user=Depends(get_current_user)):
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM payments WHERE user_id = ? ORDER BY created_at DESC", (user["id"],)).fetchall()
        return [dict(r) for r in rows]

# ========== 网络搜索 ==========
@app.post("/api/search")
async def web_search(query: dict, user=Depends(get_current_user)):
    """简单的网络搜索（实际项目中可对接搜索API）"""
    q = query.get("q", "")
    # 模拟搜索结果
    return {
        "results": [
            {"title": f"搜索结果: {q}", "url": "https://example.com", "snippet": f"关于 {q} 的相关信息..."},
        ]
    }

# ========== 代理 API ==========
class ProxyRequest(BaseModel):
    url: str
    key: str

def clean_api_key(key: str) -> str:
    """清理 API Key，移除或转换非 ASCII 字符，避免 HTTP 头编码报错"""
    import unicodedata

    if not key:
        return ""

    # 先做 NFKC 标准化，处理全角/半角与组合字符
    normalized = unicodedata.normalize("NFKC", key.strip())

    # 映射常见的全角标点为半角，避免被过滤掉
    punctuation_map = {
        "，": ",",
        "。": ".",
        "：": ":",
        "；": ";",
        "！": "!",
        "？": "?",
        "（": "(",
        "）": ")",
        "【": "[",
        "】": "]",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "￥": "$",
        "％": "%",
        "＾": "^",
        "　": " ",
    }

    cleaned_chars = []
    for ch in normalized:
        # 先做标点映射
        if ch in punctuation_map:
            ch = punctuation_map[ch]
        # 仅保留可打印的 ASCII 字符，避免 httpx/HTTP 头部编码报错
        if 32 <= ord(ch) < 127:
            cleaned_chars.append(ch)

    return "".join(cleaned_chars).strip()

@app.post("/api/proxy/test")
async def proxy_test(req: ProxyRequest):
    """测试 API 连接"""
    try:
        clean_key = clean_api_key(req.key)
        
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(
                f"{req.url.rstrip('/')}/models",
                headers={"Authorization": f"Bearer {clean_key}"}
            )
            if response.status_code == 200:
                return {"success": True}
            else:
                return {"success": False, "error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/api/proxy/models")
async def proxy_models(req: ProxyRequest):
    """获取模型列表"""
    try:
        clean_key = clean_api_key(req.key)
        
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(
                f"{req.url.rstrip('/')}/models",
                headers={"Authorization": f"Bearer {clean_key}"}
            )
            if response.status_code == 200:
                data = response.json()
                return data
            else:
                return {"error": f"HTTP {response.status_code}: {response.text[:200]}"}
    except Exception as e:
        return {"error": str(e)}

# ========== 文档处理 ==========
@app.post("/api/documents/parse")
async def parse_document(request: Request, user=Depends(get_current_user)):
    """解析上传的文档"""
    form = await request.form()
    file = form.get("file")
    if not file:
        raise HTTPException(status_code=400, detail="请上传文件")
    content = await file.read()
    filename = file.filename
    # 简单处理文本文件
    if filename.endswith(('.txt', '.md')):
        text = content.decode('utf-8', errors='ignore')
    else:
        text = f"[文件: {filename}, 大小: {len(content)} 字节]"
    return {"filename": filename, "content": text[:10000]}

# ========== API Key 管理 ==========
@app.get("/api/apikeys")
async def list_apikeys(user=Depends(get_current_user)):
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT id, provider, base_url, is_active FROM api_keys WHERE user_id = ?", (user["id"],)).fetchall()
        return [dict(r) for r in rows]

@app.post("/api/apikeys")
async def create_apikey(data: dict, user=Depends(get_current_user)):
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO api_keys (user_id, provider, api_key, base_url) VALUES (?, ?, ?, ?)",
            (user["id"], data.get("provider"), data.get("api_key"), data.get("base_url", ""))
        )
    return {"success": True}

@app.delete("/api/apikeys/{key_id}")
async def delete_apikey(key_id: int, user=Depends(get_current_user)):
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM api_keys WHERE id=? AND user_id=?", (key_id, user["id"]))
    return {"success": True}

# ========== 聊天完成 ==========
async def stream_openai(url: str, headers: dict, payload: dict) -> AsyncGenerator:
    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream("POST", url, headers=headers, json=payload) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        yield "data: [DONE]\n\n"
                        break
                    yield f"data: {data}\n\n"

async def stream_claude(url: str, headers: dict, payload: dict) -> AsyncGenerator:
    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream("POST", url, headers=headers, json=payload) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    try:
                        parsed = json.loads(data)
                        if parsed.get("type") == "content_block_delta":
                            text = parsed.get("delta", {}).get("text", "")
                            chunk = {"choices": [{"delta": {"content": text}, "index": 0}]}
                            yield f"data: {json.dumps(chunk)}\n\n"
                        elif parsed.get("type") == "message_stop":
                            yield "data: [DONE]\n\n"
                    except:
                        pass

@app.get("/api/providers")
async def list_providers():
    return {"providers": list(PROVIDERS.keys()), "details": PROVIDERS}

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest, req: Request, user=Depends(get_current_user)):
    provider = request.provider.lower()
    start_time = time.time()
    
    # 速率限制
    rate_key = f"user_{user['id']}" if user else f"ip_{req.client.host}"
    if not await rate_limiter.is_allowed(rate_key):
        raise HTTPException(status_code=429, detail="请求过于频繁，请稍后再试")
    
    # 检查用户额度
    if user:
        if user["tokens_limit"] > 0 and user["tokens_used"] >= user["tokens_limit"]:
            raise HTTPException(status_code=402, detail="额度已用完，请升级套餐")
    
    # 获取用户自定义API Key
    api_key = request.api_key
    base_url = request.custom_url
    
    if user and not api_key:
        with sqlite3.connect(DB_PATH) as conn:
            row = conn.execute(
                "SELECT api_key, base_url FROM api_keys WHERE user_id = ? AND provider = ? AND is_active = 1",
                (user["id"], provider)
            ).fetchone()
            if row:
                api_key = row[0]
                base_url = row[1] or base_url
    
    # 自定义平台
    if provider == "custom":
        if not base_url:
            raise HTTPException(status_code=400, detail="自定义平台需要填写 API 地址")
        if not api_key:
            raise HTTPException(status_code=400, detail="自定义平台需要填写 API Key")
        # 确保 URL 以 /v1 结尾
        base_url = base_url.rstrip('/')
        if not base_url.endswith('/v1'):
            base_url = base_url + '/v1'
        provider_type = "openai"
    elif provider not in PROVIDERS:
        raise HTTPException(status_code=400, detail=f"不支持的平台: {provider}")
    else:
        config = PROVIDERS[provider]
        api_key = api_key or os.getenv(config["env_key"])
        base_url = base_url or config["base_url"]
        provider_type = config.get("type", "openai")
        
        if not api_key:
            raise HTTPException(status_code=500, detail=f"未配置 {provider} 的 API Key")
    
    # 清理 API Key 中的中文标点
    api_key = clean_api_key(api_key)
    
    # 添加全局记忆到系统提示
    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    if user:
        with sqlite3.connect(DB_PATH) as conn:
            memories = conn.execute("SELECT content FROM memories WHERE user_id = ?", (user["id"],)).fetchall()
            if memories:
                memory_text = "\n".join([m[0] for m in memories])
                system_msg = f"用户记忆信息：\n{memory_text}\n\n请在回答时参考以上信息。"
                messages.insert(0, {"role": "system", "content": system_msg})
    
    async with httpx.AsyncClient(timeout=120) as client:
        try:
            if provider_type == "openai":
                url = f"{base_url}/chat/completions"
                headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                payload = {
                    "model": request.model,
                    "messages": messages,
                    "temperature": request.temperature,
                    "max_tokens": request.max_tokens,
                    "stream": request.stream,
                }
                if request.stream:
                    return StreamingResponse(stream_openai(url, headers, payload), media_type="text/event-stream")
                response = await client.post(url, headers=headers, json=payload)
                
            elif provider_type == "claude":
                url = f"{base_url}/messages"
                headers = {
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                }
                # Claude 不支持 system 在 messages 中，需要单独处理
                system_content = ""
                claude_messages = []
                for m in messages:
                    if m["role"] == "system":
                        system_content += m["content"] + "\n"
                    else:
                        claude_messages.append(m)
                payload = {
                    "model": request.model,
                    "messages": claude_messages,
                    "max_tokens": request.max_tokens,
                    "stream": request.stream,
                }
                if system_content:
                    payload["system"] = system_content
                if request.stream:
                    return StreamingResponse(stream_claude(url, headers, payload), media_type="text/event-stream")
                response = await client.post(url, headers=headers, json=payload)
                
            elif provider_type == "gemini":
                url = f"{base_url}/models/{request.model}:generateContent?key={api_key}"
                payload = {
                    "contents": [{"role": "user" if m["role"] == "user" else "model", "parts": [{"text": m["content"]}]} for m in messages if m["role"] != "system"],
                }
                response = await client.post(url, json=payload)
            
            response.raise_for_status()
            result = response.json()
            latency_ms = int((time.time() - start_time) * 1000)
            
            # 获取 token 使用量
            usage = result.get("usage", {})
            tokens_input = usage.get("prompt_tokens", 0)
            tokens_output = usage.get("completion_tokens", 0)
            
            # 保存消息到数据库
            if request.conversation_id and user:
                with sqlite3.connect(DB_PATH) as conn:
                    # 保存用户消息
                    user_msg = request.messages[-1]
                    user_content = user_msg.content if isinstance(user_msg.content, str) else json.dumps(user_msg.content)
                    conn.execute(
                        "INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)",
                        (request.conversation_id, user_msg.role, user_content)
                    )
                    # 保存助手回复
                    if provider_type == "claude":
                        assistant_content = result.get("content", [{}])[0].get("text", "")
                    else:
                        assistant_content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                    conn.execute(
                        "INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)",
                        (request.conversation_id, "assistant", assistant_content)
                    )
                    # 自动生成对话标题（使用首条消息的前30字符，或让AI生成）
                    if len(request.messages) == 1:
                        first_content = request.messages[0].content
                        if isinstance(first_content, str):
                            title = first_content[:30].replace('\n', ' ')
                        else:
                            title = "图片对话"
                        if len(first_content) > 30:
                            title += "..."
                        conn.execute("UPDATE conversations SET title = ? WHERE id = ?", (title, request.conversation_id))
                    conn.execute("UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = ?", (request.conversation_id,))
                    # 更新用户token使用量
                    total_tokens = tokens_input + tokens_output or 100
                    conn.execute("UPDATE users SET tokens_used = tokens_used + ? WHERE id = ?", (total_tokens, user["id"]))
            
            # 记录 API 调用日志
            if user:
                with sqlite3.connect(DB_PATH) as conn:
                    conn.execute(
                        "INSERT INTO api_logs (user_id, provider, model, tokens_input, tokens_output, latency_ms, status, ip_address) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (user["id"], provider, request.model, tokens_input, tokens_output, latency_ms, "success", req.client.host if req.client else "")
                    )
            
            return result
            
        except httpx.HTTPStatusError as e:
            latency_ms = int((time.time() - start_time) * 1000)
            error_detail = f"API 请求失败 ({e.response.status_code})"
            try:
                err_json = e.response.json()
                error_detail = err_json.get("error", {}).get("message", str(err_json))
            except:
                error_detail = e.response.text[:200] if e.response.text else str(e)
            # 记录错误日志
            if user:
                with sqlite3.connect(DB_PATH) as conn:
                    conn.execute(
                        "INSERT INTO api_logs (user_id, provider, model, latency_ms, status, error, ip_address) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (user["id"], provider, request.model, latency_ms, "error", error_detail[:500], req.client.host if req.client else "")
                    )
            raise HTTPException(status_code=e.response.status_code, detail=error_detail)
        except httpx.TimeoutException:
            latency_ms = int((time.time() - start_time) * 1000)
            if user:
                with sqlite3.connect(DB_PATH) as conn:
                    conn.execute(
                        "INSERT INTO api_logs (user_id, provider, model, latency_ms, status, error, ip_address) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (user["id"], provider, request.model, latency_ms, "timeout", "请求超时", req.client.host if req.client else "")
                    )
            raise HTTPException(status_code=504, detail="请求超时")
        except httpx.ConnectError:
            if user:
                with sqlite3.connect(DB_PATH) as conn:
                    conn.execute(
                        "INSERT INTO api_logs (user_id, provider, model, latency_ms, status, error, ip_address) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (user["id"], provider, request.model, 0, "error", "无法连接到 API 服务器", req.client.host if req.client else "")
                    )
            raise HTTPException(status_code=502, detail="无法连接到 API 服务器")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"内部错误: {str(e)}")

# ========== 模型对比 ==========
class CompareRequest(BaseModel):
    message: str
    models: List[dict]  # [{provider, model, api_key, custom_url}, ...]

@app.post("/api/compare")
async def compare_models(request: CompareRequest, user=Depends(get_current_user)):
    """同时向多个模型发送请求并对比结果"""
    if len(request.models) > 5:
        raise HTTPException(status_code=400, detail="最多同时对比5个模型")
    
    async def call_model(model_config: dict) -> dict:
        provider = model_config.get("provider", "custom")
        model = model_config.get("model")
        api_key = clean_api_key(model_config.get("api_key", ""))
        base_url = model_config.get("custom_url", "")
        
        if provider in PROVIDERS:
            config = PROVIDERS[provider]
            api_key = api_key or os.getenv(config["env_key"], "")
            base_url = base_url or config["base_url"]
        
        if not base_url.endswith('/v1'):
            base_url = base_url.rstrip('/') + '/v1'
        
        start_time = time.time()
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(
                    f"{base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json={"model": model, "messages": [{"role": "user", "content": request.message}], "stream": False}
                )
                response.raise_for_status()
                result = response.json()
                content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                return {
                    "provider": provider,
                    "model": model,
                    "content": content,
                    "latency": round(time.time() - start_time, 2),
                    "tokens": result.get("usage", {}).get("total_tokens", 0),
                    "success": True
                }
        except Exception as e:
            return {
                "provider": provider,
                "model": model,
                "content": "",
                "error": str(e),
                "latency": round(time.time() - start_time, 2),
                "success": False
            }
    
    # 并发调用所有模型
    tasks = [call_model(m) for m in request.models]
    results = await asyncio.gather(*tasks)
    
    return {"results": results, "message": request.message}

# ========== 缓存管理 ==========
@app.get("/api/cache/stats")
async def cache_stats(user=Depends(get_current_user)):
    """获取缓存统计"""
    return {
        "size": len(response_cache.cache),
        "max_size": response_cache.max_size,
        "ttl": response_cache.ttl
    }

@app.post("/api/cache/clear")
async def clear_cache(user=Depends(get_current_user)):
    """清空缓存"""
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    response_cache.clear()
    logger.info(f"Cache cleared by user {user['id']}")
    return {"success": True, "message": "缓存已清空"}

# ========== GitHub 推送 ==========
class GitPushRequest(BaseModel):
    repo_url: str
    token: str
    file_path: str
    content: str
    commit_message: str = "Add code from AI Hub"
    branch: str = "main"

@app.post("/api/git/push")
async def git_push(req: GitPushRequest, user=Depends(get_current_user)):
    """推送代码到 GitHub"""
    import base64
    import re
    
    # 解析仓库信息
    match = re.match(r'https://github\.com/([^/]+)/([^/]+?)(?:\.git)?$', req.repo_url.strip())
    if not match:
        raise HTTPException(status_code=400, detail="无效的 GitHub 仓库地址")
    
    owner, repo = match.groups()
    
    async with httpx.AsyncClient(timeout=30) as client:
        headers = {
            "Authorization": f"token {req.token}",
            "Accept": "application/vnd.github.v3+json"
        }
        
        # 获取文件的 SHA（如果存在）
        sha = None
        try:
            resp = await client.get(
                f"https://api.github.com/repos/{owner}/{repo}/contents/{req.file_path}",
                headers=headers,
                params={"ref": req.branch}
            )
            if resp.status_code == 200:
                sha = resp.json().get("sha")
        except:
            pass
        
        # 创建或更新文件
        payload = {
            "message": req.commit_message,
            "content": base64.b64encode(req.content.encode()).decode(),
            "branch": req.branch
        }
        if sha:
            payload["sha"] = sha
        
        resp = await client.put(
            f"https://api.github.com/repos/{owner}/{repo}/contents/{req.file_path}",
            headers=headers,
            json=payload
        )
        
        if resp.status_code in [200, 201]:
            data = resp.json()
            return {
                "success": True,
                "message": "推送成功",
                "url": data.get("content", {}).get("html_url", ""),
                "sha": data.get("content", {}).get("sha", "")
            }
        else:
            error = resp.json().get("message", resp.text)
            raise HTTPException(status_code=resp.status_code, detail=f"GitHub API 错误: {error}")

@app.post("/api/git/test")
async def git_test(data: dict):
    """测试 GitHub 连接"""
    repo_url = data.get("repo_url", "")
    token = data.get("token", "")
    
    import re
    match = re.match(r'https://github\.com/([^/]+)/([^/]+?)(?:\.git)?$', repo_url.strip())
    if not match:
        return {"success": False, "error": "无效的 GitHub 仓库地址"}
    
    owner, repo = match.groups()
    
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(
            f"https://api.github.com/repos/{owner}/{repo}",
            headers={
                "Authorization": f"token {token}",
                "Accept": "application/vnd.github.v3+json"
            }
        )
        if resp.status_code == 200:
            return {"success": True, "repo": resp.json().get("full_name")}
        else:
            return {"success": False, "error": f"HTTP {resp.status_code}"}

# ========== 本地 Git 推送 ==========
class LocalGitPushRequest(BaseModel):
    repo_path: str
    file_path: str
    content: str
    commit_message: str = "Add code from AI Hub"

@app.post("/api/git/local/push")
async def local_git_push(req: LocalGitPushRequest):
    """推送代码到本地 Git 仓库"""
    import subprocess
    import os
    
    repo_path = req.repo_path.strip()
    if not os.path.isdir(repo_path):
        raise HTTPException(status_code=400, detail="仓库路径不存在")
    
    git_dir = os.path.join(repo_path, ".git")
    if not os.path.isdir(git_dir):
        raise HTTPException(status_code=400, detail="该目录不是 Git 仓库")
    
    # 写入文件
    file_full_path = os.path.join(repo_path, req.file_path)
    os.makedirs(os.path.dirname(file_full_path), exist_ok=True)
    with open(file_full_path, "w", encoding="utf-8") as f:
        f.write(req.content)
    
    try:
        # Git add
        subprocess.run(["git", "add", req.file_path], cwd=repo_path, check=True, capture_output=True)
        # Git commit
        subprocess.run(["git", "commit", "-m", req.commit_message], cwd=repo_path, check=True, capture_output=True)
        # Git push
        result = subprocess.run(["git", "push"], cwd=repo_path, capture_output=True, text=True)
        
        if result.returncode == 0:
            return {"success": True, "message": "推送成功", "output": result.stdout}
        else:
            return {"success": True, "message": "已提交，但推送失败（可能没有配置远程仓库）", "error": result.stderr}
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Git 操作失败: {e.stderr.decode() if e.stderr else str(e)}")

@app.post("/api/git/local/test")
async def local_git_test(data: dict):
    """测试本地 Git 仓库"""
    import os
    import subprocess
    
    repo_path = data.get("repo_path", "").strip()
    if not os.path.isdir(repo_path):
        return {"success": False, "error": "路径不存在"}
    
    git_dir = os.path.join(repo_path, ".git")
    if not os.path.isdir(git_dir):
        return {"success": False, "error": "不是 Git 仓库"}
    
    try:
        result = subprocess.run(["git", "remote", "-v"], cwd=repo_path, capture_output=True, text=True)
        remotes = result.stdout.strip()
        return {"success": True, "remotes": remotes if remotes else "无远程仓库"}
    except Exception as e:
        return {"success": False, "error": str(e)}

# ========== WebSocket 实时通信 ==========
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, user_id: str):
        await websocket.accept()
        self.active_connections[user_id] = websocket
        logger.info(f"WebSocket connected: {user_id}")
    
    def disconnect(self, user_id: str):
        if user_id in self.active_connections:
            del self.active_connections[user_id]
            logger.info(f"WebSocket disconnected: {user_id}")
    
    async def send_message(self, user_id: str, message: dict):
        if user_id in self.active_connections:
            await self.active_connections[user_id].send_json(message)
    
    async def broadcast(self, message: dict):
        for connection in self.active_connections.values():
            await connection.send_json(message)

ws_manager = ConnectionManager()

@app.websocket("/ws/{token}")
async def websocket_endpoint(websocket: WebSocket, token: str):
    # 验证 token
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT u.id FROM users u JOIN sessions s ON u.id = s.user_id WHERE s.token = ? AND s.expires_at > ?",
            (token, datetime.now())
        ).fetchone()
        if not row:
            await websocket.close(code=4001)
            return
        user_id = str(row["id"])
    
    await ws_manager.connect(websocket, user_id)
    try:
        while True:
            data = await websocket.receive_json()
            # 处理不同类型的消息
            msg_type = data.get("type")
            if msg_type == "ping":
                await websocket.send_json({"type": "pong"})
            elif msg_type == "chat":
                # 可以在这里处理实时聊天
                pass
    except WebSocketDisconnect:
        ws_manager.disconnect(user_id)

# ========== GitHub OAuth ==========
@app.get("/api/auth/github")
async def github_auth():
    """重定向到 GitHub 授权页面"""
    if not GITHUB_CLIENT_ID:
        raise HTTPException(status_code=500, detail="GitHub OAuth 未配置")
    redirect_uri = os.getenv("GITHUB_REDIRECT_URI", "http://localhost:8000/api/auth/github/callback")
    return RedirectResponse(
        f"https://github.com/login/oauth/authorize?client_id={GITHUB_CLIENT_ID}&redirect_uri={redirect_uri}&scope=user:email"
    )

@app.get("/api/auth/github/callback")
async def github_callback(code: str):
    """GitHub OAuth 回调"""
    if not GITHUB_CLIENT_ID or not GITHUB_CLIENT_SECRET:
        raise HTTPException(status_code=500, detail="GitHub OAuth 未配置")
    
    async with httpx.AsyncClient() as client:
        # 获取 access token
        token_resp = await client.post(
            "https://github.com/login/oauth/access_token",
            data={
                "client_id": GITHUB_CLIENT_ID,
                "client_secret": GITHUB_CLIENT_SECRET,
                "code": code
            },
            headers={"Accept": "application/json"}
        )
        token_data = token_resp.json()
        access_token = token_data.get("access_token")
        
        if not access_token:
            raise HTTPException(status_code=400, detail="获取 token 失败")
        
        # 获取用户信息
        user_resp = await client.get(
            "https://api.github.com/user",
            headers={"Authorization": f"Bearer {access_token}"}
        )
        github_user = user_resp.json()
    
    github_id = str(github_user.get("id"))
    username = github_user.get("login")
    email = github_user.get("email")
    avatar = github_user.get("avatar_url")
    
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        # 查找或创建用户
        user = conn.execute("SELECT * FROM users WHERE github_id = ?", (github_id,)).fetchone()
        if not user:
            # 创建新用户
            conn.execute(
                "INSERT INTO users (username, password_hash, email, github_id, avatar_url) VALUES (?, ?, ?, ?, ?)",
                (f"gh_{username}", "", email, github_id, avatar)
            )
            user = conn.execute("SELECT * FROM users WHERE github_id = ?", (github_id,)).fetchone()
        
        # 创建 session
        token = secrets.token_hex(32)
        expires = datetime.now() + timedelta(days=30)
        conn.execute("INSERT INTO sessions (token, user_id, expires_at) VALUES (?, ?, ?)", (token, user["id"], expires))
    
    # 重定向回前端，带上 token
    return RedirectResponse(f"/?token={token}")

# ========== 2FA 双因素认证 ==========
@app.post("/api/auth/2fa/setup")
async def setup_2fa(user=Depends(get_current_user)):
    """设置 2FA"""
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    
    # 生成 TOTP 密钥
    secret = pyotp.random_base32()
    totp = pyotp.TOTP(secret)
    
    # 生成二维码
    provisioning_uri = totp.provisioning_uri(user["username"], issuer_name="AI Hub")
    
    # 保存密钥（未启用状态）
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("UPDATE users SET totp_secret = ? WHERE id = ?", (secret, user["id"]))
    
    # 生成二维码图片
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(provisioning_uri)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    qr_base64 = base64.b64encode(buffer.getvalue()).decode()
    
    return {
        "secret": secret,
        "qr_code": f"data:image/png;base64,{qr_base64}",
        "uri": provisioning_uri
    }

@app.post("/api/auth/2fa/verify")
async def verify_2fa(data: dict, user=Depends(get_current_user)):
    """验证并启用 2FA"""
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    
    code = data.get("code")
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute("SELECT totp_secret FROM users WHERE id = ?", (user["id"],)).fetchone()
        if not row or not row[0]:
            raise HTTPException(status_code=400, detail="请先设置 2FA")
        
        totp = pyotp.TOTP(row[0])
        if totp.verify(code):
            conn.execute("UPDATE users SET totp_enabled = 1 WHERE id = ?", (user["id"],))
            return {"success": True, "message": "2FA 已启用"}
        else:
            raise HTTPException(status_code=400, detail="验证码错误")

@app.post("/api/auth/2fa/disable")
async def disable_2fa(data: dict, user=Depends(get_current_user)):
    """禁用 2FA"""
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    
    code = data.get("code")
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute("SELECT totp_secret FROM users WHERE id = ?", (user["id"],)).fetchone()
        if row and row[0]:
            totp = pyotp.TOTP(row[0])
            if totp.verify(code):
                conn.execute("UPDATE users SET totp_enabled = 0, totp_secret = NULL WHERE id = ?", (user["id"],))
                return {"success": True, "message": "2FA 已禁用"}
        raise HTTPException(status_code=400, detail="验证码错误")

# ========== 国际化 ==========
I18N = {
    "zh-CN": {
        "welcome": "欢迎使用 AI Hub",
        "login": "登录",
        "register": "注册",
        "logout": "退出",
        "settings": "设置",
        "chat": "对话",
        "new_chat": "新建对话",
        "send": "发送",
        "copy": "复制",
        "delete": "删除",
        "export": "导出",
        "share": "分享",
    },
    "en": {
        "welcome": "Welcome to AI Hub",
        "login": "Login",
        "register": "Register",
        "logout": "Logout",
        "settings": "Settings",
        "chat": "Chat",
        "new_chat": "New Chat",
        "send": "Send",
        "copy": "Copy",
        "delete": "Delete",
        "export": "Export",
        "share": "Share",
    },
    "ja": {
        "welcome": "AI Hub へようこそ",
        "login": "ログイン",
        "register": "登録",
        "logout": "ログアウト",
        "settings": "設定",
        "chat": "チャット",
        "new_chat": "新規チャット",
        "send": "送信",
        "copy": "コピー",
        "delete": "削除",
        "export": "エクスポート",
        "share": "共有",
    }
}

@app.get("/api/i18n/{locale}")
async def get_i18n(locale: str):
    """获取国际化文本"""
    return I18N.get(locale, I18N["zh-CN"])

@app.put("/api/user/locale")
async def update_locale(data: dict, user=Depends(get_current_user)):
    """更新用户语言设置"""
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    locale = data.get("locale", "zh-CN")
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("UPDATE users SET locale = ? WHERE id = ?", (locale, user["id"]))
    return {"success": True}

# ========== 智能推荐 ==========
@app.get("/api/recommendations")
async def get_recommendations(user=Depends(get_current_user)):
    """基于使用习惯的智能推荐"""
    if not user:
        return {"prompts": [], "models": [], "features": []}
    
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        
        # 最常用的模型
        top_models = conn.execute("""
            SELECT provider, model, COUNT(*) as cnt 
            FROM api_logs WHERE user_id = ? 
            GROUP BY provider, model 
            ORDER BY cnt DESC LIMIT 3
        """, (user["id"],)).fetchall()
        
        # 最常用的 Prompt 模板
        top_prompts = conn.execute("""
            SELECT * FROM prompt_templates 
            WHERE user_id = ? OR is_public = 1 
            ORDER BY use_count DESC LIMIT 5
        """, (user["id"],)).fetchall()
        
        # 推荐功能
        features = []
        
        # 检查是否使用过知识库
        kb_count = conn.execute(
            "SELECT COUNT(*) FROM knowledge_bases WHERE user_id = ?", (user["id"],)
        ).fetchone()[0]
        if kb_count == 0:
            features.append({"name": "知识库", "desc": "上传文档，让 AI 基于您的资料回答", "icon": "📚"})
        
        # 检查是否使用过团队功能
        team_count = conn.execute(
            "SELECT COUNT(*) FROM team_members WHERE user_id = ?", (user["id"],)
        ).fetchone()[0]
        if team_count == 0:
            features.append({"name": "团队协作", "desc": "邀请团队成员一起使用", "icon": "👥"})
        
        return {
            "models": [dict(m) for m in top_models],
            "prompts": [dict(p) for p in top_prompts],
            "features": features
        }

# ========== 使用分析 ==========
@app.get("/api/analytics/usage")
async def get_usage_analytics(days: int = 30, user=Depends(get_current_user)):
    """获取详细使用分析"""
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        
        # 每日使用趋势
        daily_usage = conn.execute("""
            SELECT date(created_at) as day, 
                   COUNT(*) as calls,
                   SUM(tokens_input) as input_tokens,
                   SUM(tokens_output) as output_tokens,
                   AVG(latency_ms) as avg_latency
            FROM api_logs 
            WHERE user_id = ? AND created_at >= date('now', ?)
            GROUP BY date(created_at)
            ORDER BY day
        """, (user["id"], f'-{days} days')).fetchall()
        
        # 按小时分布
        hourly_dist = conn.execute("""
            SELECT strftime('%H', created_at) as hour, COUNT(*) as cnt
            FROM api_logs 
            WHERE user_id = ? AND created_at >= date('now', '-7 days')
            GROUP BY hour
            ORDER BY hour
        """, (user["id"],)).fetchall()
        
        # 错误率
        error_stats = conn.execute("""
            SELECT status, COUNT(*) as cnt
            FROM api_logs 
            WHERE user_id = ? AND created_at >= date('now', ?)
            GROUP BY status
        """, (user["id"], f'-{days} days')).fetchall()
        
        total = sum(e["cnt"] for e in error_stats)
        errors = sum(e["cnt"] for e in error_stats if e["status"] != "success")
        error_rate = (errors / total * 100) if total > 0 else 0
        
        return {
            "daily_usage": [dict(d) for d in daily_usage],
            "hourly_distribution": [dict(h) for h in hourly_dist],
            "error_rate": round(error_rate, 2),
            "total_calls": total
        }

# ========== 导入导出 ==========
@app.post("/api/data/export-all")
async def export_all_data(user=Depends(get_current_user)):
    """导出用户所有数据"""
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        
        data = {
            "user": {k: v for k, v in dict(user).items() if k not in ["password_hash", "totp_secret"]},
            "conversations": [],
            "notes": [],
            "memories": [],
            "shortcuts": [],
            "settings": {}
        }
        
        # 对话
        convs = conn.execute(
            "SELECT * FROM conversations WHERE user_id = ?", (user["id"],)
        ).fetchall()
        for conv in convs:
            msgs = conn.execute(
                "SELECT role, content, created_at FROM messages WHERE conversation_id = ?",
                (conv["id"],)
            ).fetchall()
            data["conversations"].append({
                **dict(conv),
                "messages": [dict(m) for m in msgs]
            })
        
        # 笔记
        notes = conn.execute(
            "SELECT * FROM notes WHERE user_id = ?", (user["id"],)
        ).fetchall()
        data["notes"] = [dict(n) for n in notes]
        
        # 记忆
        memories = conn.execute(
            "SELECT * FROM memories WHERE user_id = ?", (user["id"],)
        ).fetchall()
        data["memories"] = [dict(m) for m in memories]
        
        # 快捷短语
        shortcuts = conn.execute(
            "SELECT * FROM shortcuts WHERE user_id = ?", (user["id"],)
        ).fetchall()
        data["shortcuts"] = [dict(s) for s in shortcuts]
        
        # 设置
        settings = conn.execute(
            "SELECT data FROM settings WHERE user_id = ?", (user["id"],)
        ).fetchone()
        if settings:
            data["settings"] = json.loads(settings["data"])
        
        return data

@app.post("/api/data/import")
async def import_data(data: dict, user=Depends(get_current_user)):
    """导入用户数据"""
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    
    imported = {"conversations": 0, "notes": 0, "memories": 0, "shortcuts": 0}
    
    with sqlite3.connect(DB_PATH) as conn:
        # 导入对话
        for conv in data.get("conversations", []):
            conv_id = f"conv_{secrets.token_hex(8)}"
            conn.execute(
                "INSERT INTO conversations (id, user_id, title, provider, model) VALUES (?, ?, ?, ?, ?)",
                (conv_id, user["id"], conv.get("title", "导入的对话"), conv.get("provider"), conv.get("model"))
            )
            for msg in conv.get("messages", []):
                conn.execute(
                    "INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)",
                    (conv_id, msg.get("role"), msg.get("content"))
                )
            imported["conversations"] += 1
        
        # 导入笔记
        for note in data.get("notes", []):
            note_id = f"note_{secrets.token_hex(8)}"
            conn.execute(
                "INSERT INTO notes (id, user_id, title, content, tags) VALUES (?, ?, ?, ?, ?)",
                (note_id, user["id"], note.get("title"), note.get("content"), note.get("tags", ""))
            )
            imported["notes"] += 1
        
        # 导入记忆
        for mem in data.get("memories", []):
            conn.execute(
                "INSERT INTO memories (user_id, content, category) VALUES (?, ?, ?)",
                (user["id"], mem.get("content"), mem.get("category", "general"))
            )
            imported["memories"] += 1
        
        # 导入快捷短语
        for sc in data.get("shortcuts", []):
            conn.execute(
                "INSERT INTO shortcuts (user_id, name, content, hotkey) VALUES (?, ?, ?, ?)",
                (user["id"], sc.get("name"), sc.get("content"), sc.get("hotkey", ""))
            )
            imported["shortcuts"] += 1
    
    return {"success": True, "imported": imported}

# ========== 收藏功能 ==========
@app.post("/api/messages/favorite")
async def favorite_message(data: dict, user=Depends(get_current_user)):
    """收藏消息"""
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    
    conv_id = data.get("conversation_id")
    message_index = data.get("message_index")
    content = data.get("content")
    
    # 保存为记忆
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO memories (user_id, content, category) VALUES (?, ?, ?)",
            (user["id"], f"[收藏] {content[:500]}", "favorite")
        )
    
    return {"success": True}

# ========== 快速回复 ==========
QUICK_REPLIES = [
    {"id": "continue", "text": "继续", "prompt": "请继续"},
    {"id": "explain", "text": "解释", "prompt": "请详细解释一下"},
    {"id": "example", "text": "举例", "prompt": "请举个例子"},
    {"id": "simplify", "text": "简化", "prompt": "请用更简单的语言解释"},
    {"id": "code", "text": "代码", "prompt": "请给出代码示例"},
    {"id": "summary", "text": "总结", "prompt": "请总结一下要点"},
]

@app.get("/api/quick-replies")
async def get_quick_replies():
    """获取快速回复选项"""
    return QUICK_REPLIES

# ========== RAG 向量检索 ==========
import numpy as np
from typing import Tuple

class SimpleVectorStore:
    """简单的向量存储（生产环境建议使用 Pinecone/Milvus）"""
    
    def __init__(self):
        self.vectors = {}  # kb_id -> [(chunk_id, embedding, content)]
    
    def add(self, kb_id: int, chunk_id: str, embedding: list, content: str):
        if kb_id not in self.vectors:
            self.vectors[kb_id] = []
        self.vectors[kb_id].append((chunk_id, np.array(embedding), content))
    
    def search(self, kb_id: int, query_embedding: list, top_k: int = 5) -> list:
        if kb_id not in self.vectors:
            return []
        
        query_vec = np.array(query_embedding)
        results = []
        
        for chunk_id, vec, content in self.vectors[kb_id]:
            # 余弦相似度
            similarity = np.dot(query_vec, vec) / (np.linalg.norm(query_vec) * np.linalg.norm(vec) + 1e-8)
            results.append((chunk_id, float(similarity), content))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def delete(self, kb_id: int, chunk_id: str = None):
        if chunk_id:
            if kb_id in self.vectors:
                self.vectors[kb_id] = [(c, e, t) for c, e, t in self.vectors[kb_id] if c != chunk_id]
        else:
            if kb_id in self.vectors:
                del self.vectors[kb_id]

vector_store = SimpleVectorStore()

async def get_embedding(text: str) -> list:
    """获取文本的向量表示"""
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        # 简单的哈希向量（仅用于演示）
        import hashlib
        hash_val = hashlib.md5(text.encode()).hexdigest()
        return [int(hash_val[i:i+2], 16) / 255.0 for i in range(0, 32, 2)]
    
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            "https://api.openai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": "text-embedding-3-small", "input": text}
        )
        if response.status_code == 200:
            return response.json()["data"][0]["embedding"]
    
    return [0.0] * 1536

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    """将文本分块"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start = end - overlap
    return chunks

@app.post("/api/rag/index")
async def index_document(data: dict, user=Depends(get_current_user)):
    """索引文档到向量库"""
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    
    kb_id = data.get("kb_id")
    content = data.get("content", "")
    filename = data.get("filename", "document")
    
    if not content:
        raise HTTPException(status_code=400, detail="内容不能为空")
    
    # 分块
    chunks = chunk_text(content)
    indexed = 0
    
    for i, chunk in enumerate(chunks):
        chunk_id = f"{filename}_{i}"
        embedding = await get_embedding(chunk)
        
        # 存储到向量库
        vector_store.add(kb_id, chunk_id, embedding, chunk)
        
        # 存储到数据库
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                "INSERT INTO vector_documents (kb_id, chunk_id, content, embedding, metadata) VALUES (?, ?, ?, ?, ?)",
                (kb_id, chunk_id, chunk, json.dumps(embedding[:10]), json.dumps({"filename": filename, "index": i}))
            )
        indexed += 1
    
    return {"success": True, "indexed_chunks": indexed}

@app.post("/api/rag/search")
async def search_rag(data: dict, user=Depends(get_current_user)):
    """RAG 语义搜索"""
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    
    kb_id = data.get("kb_id")
    query = data.get("query", "")
    top_k = data.get("top_k", 5)
    
    if not query:
        raise HTTPException(status_code=400, detail="查询不能为空")
    
    # 获取查询向量
    query_embedding = await get_embedding(query)
    
    # 搜索
    results = vector_store.search(kb_id, query_embedding, top_k)
    
    return {
        "results": [
            {"chunk_id": r[0], "score": r[1], "content": r[2]}
            for r in results
        ]
    }

@app.post("/api/rag/chat")
async def rag_chat(data: dict, user=Depends(get_current_user)):
    """基于 RAG 的对话"""
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    
    kb_id = data.get("kb_id")
    question = data.get("question", "")
    
    # 搜索相关文档
    query_embedding = await get_embedding(question)
    results = vector_store.search(kb_id, query_embedding, top_k=3)
    
    # 构建上下文
    context = "\n\n".join([r[2] for r in results])
    
    # 调用 AI
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return {"answer": "未配置 API Key", "sources": []}
    
    prompt = f"""基于以下参考资料回答问题。如果资料中没有相关信息，请说明。

参考资料：
{context}

问题：{question}

请用中文回答："""
    
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1000
            }
        )
        if response.status_code == 200:
            answer = response.json()["choices"][0]["message"]["content"]
            return {
                "answer": answer,
                "sources": [{"chunk_id": r[0], "score": r[1]} for r in results]
            }
    
    return {"answer": "生成回答失败", "sources": []}

# ========== RBAC 权限系统 ==========
class RBACManager:
    """角色权限管理"""
    
    def __init__(self):
        self.permission_cache = {}
    
    def get_user_permissions(self, user_id: int) -> list:
        """获取用户所有权限"""
        if user_id in self.permission_cache:
            return self.permission_cache[user_id]
        
        with sqlite3.connect(DB_PATH) as conn:
            rows = conn.execute("""
                SELECT r.permissions FROM roles r
                JOIN user_roles ur ON r.id = ur.role_id
                WHERE ur.user_id = ?
            """, (user_id,)).fetchall()
            
            permissions = set()
            for row in rows:
                perms = json.loads(row[0])
                permissions.update(perms)
            
            self.permission_cache[user_id] = list(permissions)
            return list(permissions)
    
    def has_permission(self, user_id: int, permission: str) -> bool:
        """检查用户是否有某权限"""
        perms = self.get_user_permissions(user_id)
        return "*" in perms or permission in perms
    
    def assign_role(self, user_id: int, role_name: str):
        """分配角色给用户"""
        with sqlite3.connect(DB_PATH) as conn:
            role = conn.execute("SELECT id FROM roles WHERE name = ?", (role_name,)).fetchone()
            if role:
                conn.execute(
                    "INSERT OR IGNORE INTO user_roles (user_id, role_id) VALUES (?, ?)",
                    (user_id, role[0])
                )
                if user_id in self.permission_cache:
                    del self.permission_cache[user_id]
    
    def remove_role(self, user_id: int, role_name: str):
        """移除用户角色"""
        with sqlite3.connect(DB_PATH) as conn:
            role = conn.execute("SELECT id FROM roles WHERE name = ?", (role_name,)).fetchone()
            if role:
                conn.execute(
                    "DELETE FROM user_roles WHERE user_id = ? AND role_id = ?",
                    (user_id, role[0])
                )
                if user_id in self.permission_cache:
                    del self.permission_cache[user_id]
    
    def clear_cache(self, user_id: int = None):
        if user_id:
            self.permission_cache.pop(user_id, None)
        else:
            self.permission_cache.clear()

rbac = RBACManager()

def require_permission(permission: str):
    """权限检查装饰器"""
    async def check(user=Depends(get_current_user)):
        if not user:
            raise HTTPException(status_code=401, detail="请先登录")
        if not rbac.has_permission(user["id"], permission):
            raise HTTPException(status_code=403, detail=f"无权限: {permission}")
        return user
    return check

@app.get("/api/rbac/roles")
async def list_roles(user=Depends(get_current_user)):
    """获取所有角色"""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM roles").fetchall()
        return [dict(r) for r in rows]

@app.post("/api/rbac/roles")
async def create_role(data: dict, user=Depends(get_current_user)):
    """创建角色"""
    if not user or not rbac.has_permission(user["id"], "admin"):
        raise HTTPException(status_code=403, detail="需要管理员权限")
    
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO roles (name, description, permissions) VALUES (?, ?, ?)",
            (data.get("name"), data.get("description"), json.dumps(data.get("permissions", [])))
        )
    return {"success": True}

@app.post("/api/rbac/assign")
async def assign_role_to_user(data: dict, user=Depends(get_current_user)):
    """分配角色给用户"""
    if not user or not rbac.has_permission(user["id"], "admin"):
        raise HTTPException(status_code=403, detail="需要管理员权限")
    
    rbac.assign_role(data.get("user_id"), data.get("role"))
    return {"success": True}

@app.get("/api/rbac/my-permissions")
async def get_my_permissions(user=Depends(get_current_user)):
    """获取当前用户权限"""
    if not user:
        return {"permissions": []}
    return {"permissions": rbac.get_user_permissions(user["id"])}

# ========== 计费系统 ==========
class BillingManager:
    """计费管理"""
    
    def get_user_subscription(self, user_id: int) -> dict:
        """获取用户订阅"""
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            sub = conn.execute("""
                SELECT s.*, p.name as plan_name, p.price, p.tokens_limit, p.features
                FROM subscriptions s
                JOIN billing_plans p ON s.plan_id = p.id
                WHERE s.user_id = ? AND s.status = 'active'
                ORDER BY s.end_date DESC LIMIT 1
            """, (user_id,)).fetchone()
            return dict(sub) if sub else None
    
    def get_usage(self, user_id: int, period: str = "month") -> dict:
        """获取用量统计"""
        with sqlite3.connect(DB_PATH) as conn:
            if period == "month":
                date_filter = "date('now', 'start of month')"
            elif period == "day":
                date_filter = "date('now')"
            else:
                date_filter = "date('now', '-30 days')"
            
            usage = conn.execute(f"""
                SELECT type, SUM(amount) as total, SUM(total_cost) as cost
                FROM usage_records
                WHERE user_id = ? AND created_at >= {date_filter}
                GROUP BY type
            """, (user_id,)).fetchall()
            
            return {row[0]: {"amount": row[1], "cost": row[2]} for row in usage}
    
    def record_usage(self, user_id: int, usage_type: str, amount: int, unit_price: float = 0):
        """记录用量"""
        total_cost = amount * unit_price
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                "INSERT INTO usage_records (user_id, type, amount, unit_price, total_cost) VALUES (?, ?, ?, ?, ?)",
                (user_id, usage_type, amount, unit_price, total_cost)
            )
    
    def check_quota(self, user_id: int) -> Tuple[bool, int]:
        """检查配额"""
        sub = self.get_user_subscription(user_id)
        if not sub:
            # 免费用户
            limit = 10000
        else:
            limit = sub["tokens_limit"]
        
        if limit == -1:  # 无限
            return True, -1
        
        with sqlite3.connect(DB_PATH) as conn:
            used = conn.execute("""
                SELECT COALESCE(SUM(amount), 0) FROM usage_records
                WHERE user_id = ? AND type = 'tokens' AND created_at >= date('now', 'start of month')
            """, (user_id,)).fetchone()[0]
        
        return used < limit, limit - used
    
    def create_subscription(self, user_id: int, plan_id: str, months: int = 1) -> str:
        """创建订阅"""
        with sqlite3.connect(DB_PATH) as conn:
            # 获取套餐信息
            plan = conn.execute("SELECT * FROM billing_plans WHERE id = ?", (plan_id,)).fetchone()
            if not plan:
                raise ValueError("套餐不存在")
            
            start_date = datetime.now()
            end_date = start_date + timedelta(days=30 * months)
            
            conn.execute(
                "INSERT INTO subscriptions (user_id, plan_id, start_date, end_date) VALUES (?, ?, ?, ?)",
                (user_id, plan_id, start_date, end_date)
            )
            
            # 创建发票
            invoice_id = f"INV_{secrets.token_hex(8)}"
            amount = plan[2] * months  # price * months
            conn.execute(
                "INSERT INTO invoices (id, user_id, amount, status) VALUES (?, ?, ?, 'pending')",
                (invoice_id, user_id, amount)
            )
            
            return invoice_id

billing = BillingManager()

@app.get("/api/billing/plans")
async def get_billing_plans():
    """获取所有套餐"""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM billing_plans WHERE is_active = 1").fetchall()
        return [dict(r) for r in rows]

@app.get("/api/billing/subscription")
async def get_subscription(user=Depends(get_current_user)):
    """获取当前订阅"""
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    
    sub = billing.get_user_subscription(user["id"])
    return sub or {"plan": "free", "tokens_limit": 10000}

@app.post("/api/billing/subscribe")
async def subscribe(data: dict, user=Depends(get_current_user)):
    """订阅套餐"""
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    
    plan_id = data.get("plan_id")
    months = data.get("months", 1)
    
    try:
        invoice_id = billing.create_subscription(user["id"], plan_id, months)
        return {"success": True, "invoice_id": invoice_id}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/billing/usage")
async def get_usage(period: str = "month", user=Depends(get_current_user)):
    """获取用量统计"""
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    
    usage = billing.get_usage(user["id"], period)
    allowed, remaining = billing.check_quota(user["id"])
    
    return {
        "usage": usage,
        "quota": {"allowed": allowed, "remaining": remaining}
    }

@app.get("/api/billing/invoices")
async def get_invoices(user=Depends(get_current_user)):
    """获取发票列表"""
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM invoices WHERE user_id = ? ORDER BY created_at DESC",
            (user["id"],)
        ).fetchall()
        return [dict(r) for r in rows]

@app.post("/api/billing/pay/{invoice_id}")
async def pay_invoice(invoice_id: str, data: dict, user=Depends(get_current_user)):
    """支付发票（模拟）"""
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    
    payment_method = data.get("method", "alipay")  # alipay, wechat, stripe
    
    with sqlite3.connect(DB_PATH) as conn:
        invoice = conn.execute(
            "SELECT * FROM invoices WHERE id = ? AND user_id = ?",
            (invoice_id, user["id"])
        ).fetchone()
        
        if not invoice:
            raise HTTPException(status_code=404, detail="发票不存在")
        
        # 模拟支付成功
        conn.execute(
            "UPDATE invoices SET status = 'paid', paid_at = ? WHERE id = ?",
            (datetime.now(), invoice_id)
        )
        
        # 激活订阅
        conn.execute(
            "UPDATE subscriptions SET status = 'active' WHERE user_id = ? AND status = 'pending'",
            (user["id"],)
        )
    
    return {"success": True, "message": "支付成功"}

# ========== 实时协作 ==========
collaboration_sessions = {}  # session_id -> {participants: set, messages: list}

@app.post("/api/collab/create")
async def create_collab_session(data: dict, user=Depends(get_current_user)):
    """创建协作会话"""
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    
    conversation_id = data.get("conversation_id")
    session_id = f"collab_{secrets.token_hex(8)}"
    
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO collaboration_sessions (id, conversation_id, created_by) VALUES (?, ?, ?)",
            (session_id, conversation_id, user["id"])
        )
        conn.execute(
            "INSERT INTO collaboration_participants (session_id, user_id, role) VALUES (?, ?, 'owner')",
            (session_id, user["id"])
        )
    
    collaboration_sessions[session_id] = {"participants": {user["id"]}, "messages": []}
    
    return {"session_id": session_id, "share_link": f"/collab/{session_id}"}

@app.post("/api/collab/{session_id}/join")
async def join_collab_session(session_id: str, user=Depends(get_current_user)):
    """加入协作会话"""
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    
    with sqlite3.connect(DB_PATH) as conn:
        session = conn.execute(
            "SELECT * FROM collaboration_sessions WHERE id = ? AND is_active = 1",
            (session_id,)
        ).fetchone()
        
        if not session:
            raise HTTPException(status_code=404, detail="会话不存在")
        
        conn.execute(
            "INSERT OR IGNORE INTO collaboration_participants (session_id, user_id) VALUES (?, ?)",
            (session_id, user["id"])
        )
    
    if session_id in collaboration_sessions:
        collaboration_sessions[session_id]["participants"].add(user["id"])
    
    return {"success": True}

@app.websocket("/ws/collab/{session_id}")
async def collab_websocket(websocket: WebSocket, session_id: str):
    """协作 WebSocket"""
    await websocket.accept()
    
    if session_id not in collaboration_sessions:
        collaboration_sessions[session_id] = {"participants": set(), "messages": []}
    
    try:
        while True:
            data = await websocket.receive_json()
            
            # 广播消息给所有参与者
            message = {
                "type": data.get("type", "message"),
                "content": data.get("content"),
                "user_id": data.get("user_id"),
                "timestamp": datetime.now().isoformat()
            }
            
            collaboration_sessions[session_id]["messages"].append(message)
            
            # 这里应该广播给所有连接的 WebSocket
            await websocket.send_json(message)
            
    except WebSocketDisconnect:
        pass

# ========== 插件系统 ==========
@app.get("/api/plugins")
async def list_plugins():
    """获取插件列表"""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM plugins").fetchall()
        return [dict(r) for r in rows]

@app.post("/api/plugins/install")
async def install_plugin(data: dict, user=Depends(get_current_user)):
    """安装插件"""
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    
    plugin_id = data.get("plugin_id")
    
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT OR IGNORE INTO user_plugins (user_id, plugin_id) VALUES (?, ?)",
            (user["id"], plugin_id)
        )
    
    return {"success": True}

@app.post("/api/plugins/{plugin_id}/toggle")
async def toggle_plugin(plugin_id: str, user=Depends(get_current_user)):
    """启用/禁用插件"""
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    
    with sqlite3.connect(DB_PATH) as conn:
        current = conn.execute(
            "SELECT enabled FROM user_plugins WHERE user_id = ? AND plugin_id = ?",
            (user["id"], plugin_id)
        ).fetchone()
        
        if current:
            new_state = 0 if current[0] else 1
            conn.execute(
                "UPDATE user_plugins SET enabled = ? WHERE user_id = ? AND plugin_id = ?",
                (new_state, user["id"], plugin_id)
            )
            return {"success": True, "enabled": new_state == 1}
    
    raise HTTPException(status_code=404, detail="插件未安装")

# ========== 多租户系统 ==========
class TenantManager:
    """多租户管理"""
    
    def __init__(self):
        self._init_tables()
    
    def _init_tables(self):
        with sqlite3.connect(DB_PATH) as conn:
            conn.executescript('''
                CREATE TABLE IF NOT EXISTS tenants (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    domain TEXT UNIQUE,
                    config TEXT DEFAULT '{}',
                    status TEXT DEFAULT 'active',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS tenant_users (
                    tenant_id TEXT,
                    user_id INTEGER,
                    role TEXT DEFAULT 'member',
                    PRIMARY KEY (tenant_id, user_id),
                    FOREIGN KEY (tenant_id) REFERENCES tenants(id),
                    FOREIGN KEY (user_id) REFERENCES users(id)
                );
                
                CREATE TABLE IF NOT EXISTS tenant_quotas (
                    tenant_id TEXT PRIMARY KEY,
                    tokens_limit INTEGER DEFAULT 1000000,
                    tokens_used INTEGER DEFAULT 0,
                    users_limit INTEGER DEFAULT 100,
                    storage_limit INTEGER DEFAULT 10737418240,
                    FOREIGN KEY (tenant_id) REFERENCES tenants(id)
                );
            ''')
    
    def create_tenant(self, name: str, domain: str = None) -> str:
        tenant_id = f"tenant_{secrets.token_hex(8)}"
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                "INSERT INTO tenants (id, name, domain) VALUES (?, ?, ?)",
                (tenant_id, name, domain)
            )
            conn.execute(
                "INSERT INTO tenant_quotas (tenant_id) VALUES (?)",
                (tenant_id,)
            )
        return tenant_id
    
    def get_tenant(self, tenant_id: str) -> dict:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM tenants WHERE id = ?", (tenant_id,)).fetchone()
            return dict(row) if row else None
    
    def get_tenant_by_domain(self, domain: str) -> dict:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM tenants WHERE domain = ?", (domain,)).fetchone()
            return dict(row) if row else None
    
    def add_user_to_tenant(self, tenant_id: str, user_id: int, role: str = "member"):
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO tenant_users (tenant_id, user_id, role) VALUES (?, ?, ?)",
                (tenant_id, user_id, role)
            )
    
    def get_user_tenant(self, user_id: int) -> dict:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("""
                SELECT t.*, tu.role as user_role FROM tenants t
                JOIN tenant_users tu ON t.id = tu.tenant_id
                WHERE tu.user_id = ?
            """, (user_id,)).fetchone()
            return dict(row) if row else None
    
    def check_quota(self, tenant_id: str, quota_type: str) -> tuple:
        with sqlite3.connect(DB_PATH) as conn:
            row = conn.execute("SELECT * FROM tenant_quotas WHERE tenant_id = ?", (tenant_id,)).fetchone()
            if not row:
                return True, -1
            
            if quota_type == "tokens":
                return row[2] < row[1], row[1] - row[2]  # used < limit
            elif quota_type == "users":
                count = conn.execute(
                    "SELECT COUNT(*) FROM tenant_users WHERE tenant_id = ?", (tenant_id,)
                ).fetchone()[0]
                return count < row[3], row[3] - count
            
            return True, -1

tenant_manager = TenantManager()

@app.post("/api/tenants")
async def create_tenant(data: dict, user=Depends(get_current_user)):
    """创建租户"""
    if not user or not rbac.has_permission(user["id"], "admin"):
        raise HTTPException(status_code=403, detail="需要管理员权限")
    
    name = data.get("name")
    domain = data.get("domain")
    
    tenant_id = tenant_manager.create_tenant(name, domain)
    return {"success": True, "tenant_id": tenant_id}

@app.get("/api/tenants")
async def list_tenants(user=Depends(get_current_user)):
    """获取租户列表"""
    if not user or not rbac.has_permission(user["id"], "admin"):
        raise HTTPException(status_code=403, detail="需要管理员权限")
    
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM tenants").fetchall()
        return [dict(r) for r in rows]

@app.get("/api/tenants/my")
async def get_my_tenant(user=Depends(get_current_user)):
    """获取当前用户的租户"""
    if not user:
        return None
    return tenant_manager.get_user_tenant(user["id"])

# ========== 审计合规 ==========
class AuditLogger:
    """审计日志记录器"""
    
    AUDIT_EVENTS = [
        "user.login", "user.logout", "user.register",
        "data.export", "data.delete", "data.access",
        "settings.change", "permission.change",
        "payment.create", "payment.complete",
        "api.call", "security.alert"
    ]
    
    def log(self, user_id: int, event: str, resource: str, details: dict, ip: str = None):
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                INSERT INTO audit_logs (user_id, action, resource, details, ip_address)
                VALUES (?, ?, ?, ?, ?)
            """, (user_id, event, resource, json.dumps(details, ensure_ascii=False), ip))
    
    def get_logs(self, user_id: int = None, event: str = None, days: int = 30, limit: int = 100) -> list:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            
            query = "SELECT * FROM audit_logs WHERE created_at >= date('now', ?)"
            params = [f'-{days} days']
            
            if user_id:
                query += " AND user_id = ?"
                params.append(user_id)
            
            if event:
                query += " AND action = ?"
                params.append(event)
            
            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)
            
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]
    
    def generate_compliance_report(self, tenant_id: str = None, period: str = "month") -> dict:
        """生成合规报告"""
        with sqlite3.connect(DB_PATH) as conn:
            if period == "month":
                date_filter = "date('now', 'start of month')"
            elif period == "quarter":
                date_filter = "date('now', '-3 months')"
            else:
                date_filter = "date('now', '-1 year')"
            
            # 统计各类事件
            events = conn.execute(f"""
                SELECT action, COUNT(*) as count FROM audit_logs
                WHERE created_at >= {date_filter}
                GROUP BY action
            """).fetchall()
            
            # 安全事件
            security_events = conn.execute(f"""
                SELECT * FROM audit_logs
                WHERE action = 'security.alert' AND created_at >= {date_filter}
            """).fetchall()
            
            # 数据访问统计
            data_access = conn.execute(f"""
                SELECT user_id, COUNT(*) as count FROM audit_logs
                WHERE action LIKE 'data.%' AND created_at >= {date_filter}
                GROUP BY user_id
            """).fetchall()
            
            return {
                "period": period,
                "generated_at": datetime.now().isoformat(),
                "event_summary": {e[0]: e[1] for e in events},
                "security_alerts": len(security_events),
                "data_access_users": len(data_access),
                "compliance_status": "compliant" if len(security_events) == 0 else "review_required"
            }

audit_logger = AuditLogger()

@app.get("/api/audit/logs")
async def get_audit_logs(days: int = 30, event: str = None, user=Depends(get_current_user)):
    """获取审计日志"""
    if not user or not rbac.has_permission(user["id"], "admin"):
        raise HTTPException(status_code=403, detail="需要管理员权限")
    
    logs = audit_logger.get_logs(event=event, days=days)
    return logs

@app.get("/api/audit/report")
async def get_compliance_report(period: str = "month", user=Depends(get_current_user)):
    """获取合规报告"""
    if not user or not rbac.has_permission(user["id"], "admin"):
        raise HTTPException(status_code=403, detail="需要管理员权限")
    
    return audit_logger.generate_compliance_report(period=period)

@app.post("/api/audit/export")
async def export_audit_logs(data: dict, user=Depends(get_current_user)):
    """导出审计日志"""
    if not user or not rbac.has_permission(user["id"], "admin"):
        raise HTTPException(status_code=403, detail="需要管理员权限")
    
    days = data.get("days", 30)
    format_type = data.get("format", "json")
    
    logs = audit_logger.get_logs(days=days, limit=10000)
    
    # 记录导出操作
    audit_logger.log(user["id"], "data.export", "audit_logs", {"days": days, "count": len(logs)})
    
    if format_type == "csv":
        import csv
        import io
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=["id", "user_id", "action", "resource", "details", "ip_address", "created_at"])
        writer.writeheader()
        writer.writerows(logs)
        return {"content": output.getvalue(), "filename": f"audit_logs_{datetime.now().strftime('%Y%m%d')}.csv"}
    
    return {"logs": logs, "count": len(logs)}

# ========== 支付集成 ==========
class PaymentGateway:
    """支付网关"""
    
    def __init__(self):
        self.alipay_app_id = os.getenv("ALIPAY_APP_ID", "")
        self.wechat_app_id = os.getenv("WECHAT_APP_ID", "")
        self.stripe_key = os.getenv("STRIPE_SECRET_KEY", "")
    
    async def create_alipay_order(self, order_id: str, amount: float, subject: str) -> dict:
        """创建支付宝订单"""
        if not self.alipay_app_id:
            return {"success": False, "error": "支付宝未配置"}
        
        # 实际实现需要使用 alipay-sdk
        # 这里返回模拟数据
        return {
            "success": True,
            "order_id": order_id,
            "pay_url": f"https://openapi.alipay.com/gateway.do?order={order_id}",
            "qr_code": f"https://qr.alipay.com/{order_id}"
        }
    
    async def create_wechat_order(self, order_id: str, amount: float, subject: str) -> dict:
        """创建微信支付订单"""
        if not self.wechat_app_id:
            return {"success": False, "error": "微信支付未配置"}
        
        # 实际实现需要使用 wechatpay-python
        return {
            "success": True,
            "order_id": order_id,
            "prepay_id": f"wx_{secrets.token_hex(16)}",
            "qr_code": f"weixin://wxpay/bizpayurl?pr={order_id}"
        }
    
    async def create_stripe_order(self, order_id: str, amount: float, currency: str = "usd") -> dict:
        """创建 Stripe 订单"""
        if not self.stripe_key:
            return {"success": False, "error": "Stripe 未配置"}
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.stripe.com/v1/checkout/sessions",
                    auth=(self.stripe_key, ""),
                    data={
                        "payment_method_types[]": "card",
                        "line_items[0][price_data][currency]": currency,
                        "line_items[0][price_data][unit_amount]": int(amount * 100),
                        "line_items[0][price_data][product_data][name]": "AI Hub Subscription",
                        "line_items[0][quantity]": 1,
                        "mode": "payment",
                        "success_url": f"http://localhost:8000/payment/success?order={order_id}",
                        "cancel_url": f"http://localhost:8000/payment/cancel?order={order_id}"
                    }
                )
                if response.status_code == 200:
                    data = response.json()
                    return {"success": True, "order_id": order_id, "checkout_url": data.get("url")}
        except Exception as e:
            return {"success": False, "error": str(e)}
        
        return {"success": False, "error": "创建订单失败"}
    
    async def verify_payment(self, order_id: str, method: str) -> bool:
        """验证支付状态"""
        # 实际实现需要查询各支付平台
        # 这里模拟验证
        return True

payment_gateway = PaymentGateway()

@app.post("/api/payment/create")
async def create_payment_order(data: dict, user=Depends(get_current_user)):
    """创建支付订单"""
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    
    method = data.get("method", "alipay")  # alipay, wechat, stripe
    plan_id = data.get("plan_id")
    months = data.get("months", 1)
    
    # 获取套餐价格
    with sqlite3.connect(DB_PATH) as conn:
        plan = conn.execute("SELECT * FROM billing_plans WHERE id = ?", (plan_id,)).fetchone()
        if not plan:
            raise HTTPException(status_code=400, detail="套餐不存在")
        
        amount = plan[2] * months  # price * months
    
    order_id = f"order_{secrets.token_hex(12)}"
    
    # 创建订单记录
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO invoices (id, user_id, amount, status)
            VALUES (?, ?, ?, 'pending')
        """, (order_id, user["id"], amount))
    
    # 调用支付网关
    if method == "alipay":
        result = await payment_gateway.create_alipay_order(order_id, amount, f"AI Hub {plan[1]}")
    elif method == "wechat":
        result = await payment_gateway.create_wechat_order(order_id, amount, f"AI Hub {plan[1]}")
    elif method == "stripe":
        result = await payment_gateway.create_stripe_order(order_id, amount)
    else:
        raise HTTPException(status_code=400, detail="不支持的支付方式")
    
    # 记录审计日志
    audit_logger.log(user["id"], "payment.create", order_id, {"method": method, "amount": amount})
    
    return result

@app.post("/api/payment/callback/{method}")
async def payment_callback(method: str, request: Request):
    """支付回调"""
    body = await request.body()
    
    # 验证签名（实际实现需要各平台的签名验证）
    # ...
    
    # 解析订单信息
    if method == "alipay":
        form = await request.form()
        order_id = form.get("out_trade_no")
        status = form.get("trade_status")
    elif method == "wechat":
        # 解析 XML
        order_id = "..."
        status = "SUCCESS"
    elif method == "stripe":
        data = await request.json()
        order_id = data.get("data", {}).get("object", {}).get("metadata", {}).get("order_id")
        status = "SUCCESS"
    else:
        return {"success": False}
    
    if status in ["TRADE_SUCCESS", "SUCCESS"]:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("UPDATE invoices SET status = 'paid', paid_at = ? WHERE id = ?", (datetime.now(), order_id))
            
            # 获取用户并激活订阅
            invoice = conn.execute("SELECT user_id FROM invoices WHERE id = ?", (order_id,)).fetchone()
            if invoice:
                audit_logger.log(invoice[0], "payment.complete", order_id, {"method": method})
    
    return {"success": True}

# ========== 安全加固 ==========
class SecurityManager:
    """安全管理器"""
    
    def __init__(self):
        self.blocked_ips = set()
        self.failed_attempts = defaultdict(list)
        self.max_failed_attempts = 5
        self.block_duration = 3600  # 1小时
    
    def check_ip(self, ip: str) -> bool:
        """检查 IP 是否被封禁"""
        if ip in self.blocked_ips:
            return False
        
        # 清理过期的失败记录
        now = time.time()
        self.failed_attempts[ip] = [t for t in self.failed_attempts[ip] if now - t < self.block_duration]
        
        return len(self.failed_attempts[ip]) < self.max_failed_attempts
    
    def record_failed_attempt(self, ip: str):
        """记录失败尝试"""
        self.failed_attempts[ip].append(time.time())
        
        if len(self.failed_attempts[ip]) >= self.max_failed_attempts:
            self.blocked_ips.add(ip)
            logger.warning(f"IP {ip} blocked due to too many failed attempts")
    
    def unblock_ip(self, ip: str):
        """解封 IP"""
        self.blocked_ips.discard(ip)
        self.failed_attempts.pop(ip, None)
    
    def validate_password_strength(self, password: str) -> tuple:
        """验证密码强度"""
        errors = []
        
        if len(password) < 8:
            errors.append("密码长度至少8位")
        if not any(c.isupper() for c in password):
            errors.append("需要包含大写字母")
        if not any(c.islower() for c in password):
            errors.append("需要包含小写字母")
        if not any(c.isdigit() for c in password):
            errors.append("需要包含数字")
        if not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            errors.append("需要包含特殊字符")
        
        return len(errors) == 0, errors
    
    def generate_secure_token(self, length: int = 32) -> str:
        """生成安全令牌"""
        return secrets.token_urlsafe(length)
    
    def hash_sensitive_data(self, data: str) -> str:
        """哈希敏感数据"""
        return hashlib.sha256(data.encode()).hexdigest()

security_manager = SecurityManager()

@app.middleware("http")
async def security_middleware(request: Request, call_next):
    """安全中间件"""
    client_ip = request.client.host if request.client else "unknown"
    
    # 检查 IP 封禁
    if not security_manager.check_ip(client_ip):
        return JSONResponse(
            status_code=403,
            content={"error": "IP 已被封禁，请稍后再试"}
        )
    
    # 检查请求头安全
    user_agent = request.headers.get("user-agent", "")
    if not user_agent or len(user_agent) < 10:
        # 可疑请求
        logger.warning(f"Suspicious request from {client_ip}: missing/short user-agent")
    
    response = await call_next(request)
    
    # 添加安全响应头
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    
    return response

@app.get("/api/security/status")
async def get_security_status(user=Depends(get_current_user)):
    """获取安全状态"""
    if not user or not rbac.has_permission(user["id"], "admin"):
        raise HTTPException(status_code=403, detail="需要管理员权限")
    
    return {
        "blocked_ips": len(security_manager.blocked_ips),
        "failed_attempts_tracked": len(security_manager.failed_attempts),
        "2fa_enabled_users": 0,  # 需要查询数据库
        "last_security_scan": datetime.now().isoformat()
    }

@app.post("/api/security/unblock-ip")
async def unblock_ip(data: dict, user=Depends(get_current_user)):
    """解封 IP"""
    if not user or not rbac.has_permission(user["id"], "admin"):
        raise HTTPException(status_code=403, detail="需要管理员权限")
    
    ip = data.get("ip")
    security_manager.unblock_ip(ip)
    audit_logger.log(user["id"], "security.alert", "unblock_ip", {"ip": ip})
    
    return {"success": True}

# ========== 静态文件 ==========
import pathlib
BASE_DIR = pathlib.Path(__file__).parent.resolve()
STATIC_DIR = BASE_DIR / "static"

@app.get("/")
async def root():
    return FileResponse(STATIC_DIR / "index.html")

@app.get("/share/{share_id}")
async def share_page(share_id: str):
    """分享页面"""
    return FileResponse(STATIC_DIR / "share.html")

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ========== Redis 缓存支持（可选） ==========
class RedisCache:
    """Redis 缓存封装，支持降级到内存缓存"""
    def __init__(self):
        self.redis = None
        self.memory_cache = {}
        self._init_redis()
    
    def _init_redis(self):
        try:
            import redis
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
            self.redis = redis.from_url(redis_url, decode_responses=True)
            self.redis.ping()
            logger.info("✅ Redis 缓存已连接")
        except ImportError:
            logger.info("ℹ️ Redis 未安装，使用内存缓存")
            self.redis = None
        except Exception:
            # 尝试使用 fakeredis（内存模拟）
            try:
                import fakeredis
                self.redis = fakeredis.FakeRedis(decode_responses=True)
                logger.info("✅ 使用 FakeRedis 内存缓存（功能完整）")
            except ImportError:
                logger.info("ℹ️ Redis 服务未运行，使用简单内存缓存")
                self.redis = None
    
    async def get(self, key: str) -> Optional[str]:
        try:
            if self.redis:
                return self.redis.get(key)
            return self.memory_cache.get(key)
        except:
            return self.memory_cache.get(key)
    
    async def set(self, key: str, value: str, ttl: int = 3600):
        try:
            if self.redis:
                self.redis.setex(key, ttl, value)
            else:
                self.memory_cache[key] = value
        except:
            self.memory_cache[key] = value
    
    async def delete(self, key: str):
        try:
            if self.redis:
                self.redis.delete(key)
            elif key in self.memory_cache:
                del self.memory_cache[key]
        except:
            pass

redis_cache = RedisCache()

# ========== 异步任务队列 ==========
class TaskQueue:
    """简单的异步任务队列"""
    def __init__(self):
        self.tasks = asyncio.Queue()
        self.results = {}
        self.running = False
    
    async def start(self):
        self.running = True
        asyncio.create_task(self._worker())
        logger.info("Task queue started")
    
    async def _worker(self):
        while self.running:
            try:
                task_id, func, args, kwargs = await asyncio.wait_for(self.tasks.get(), timeout=1.0)
                try:
                    result = await func(*args, **kwargs)
                    self.results[task_id] = {"status": "completed", "result": result}
                except Exception as e:
                    self.results[task_id] = {"status": "failed", "error": str(e)}
            except asyncio.TimeoutError:
                continue
    
    async def submit(self, func, *args, **kwargs) -> str:
        task_id = secrets.token_hex(8)
        self.results[task_id] = {"status": "pending"}
        await self.tasks.put((task_id, func, args, kwargs))
        return task_id
    
    def get_result(self, task_id: str) -> dict:
        return self.results.get(task_id, {"status": "not_found"})
    
    async def stop(self):
        self.running = False
        logger.info("Task queue stopped")

task_queue = TaskQueue()

@app.on_event("startup")
async def startup_event():
    import time
    app.state.start_time = time.time()
    
    # 启动任务队列
    try:
        await task_queue.start()
    except Exception as e:
        logger.warning(f"Task queue start failed: {e}")
    
    # 启动定时任务调度器
    try:
        from modules.queue import scheduler
        await scheduler.start()
    except Exception as e:
        logger.warning(f"Scheduler start failed: {e}")
    
    logger.info("AI Hub started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    # 停止任务队列
    try:
        await task_queue.stop()
    except:
        pass
    
    # 停止调度器
    try:
        from modules.queue import scheduler
        await scheduler.stop()
    except:
        pass
    
    logger.info("AI Hub shutdown complete")

@app.get("/api/tasks/{task_id}")
async def get_task_status(task_id: str):
    """获取异步任务状态"""
    return task_queue.get_result(task_id)

# ========== 增强的健康检查 ==========
@app.get("/api/health/detailed")
async def detailed_health_check():
    """详细健康检查"""
    import psutil
    
    # 系统资源
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    # 数据库检查
    db_status = "healthy"
    db_latency = 0
    try:
        start = time.time()
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("SELECT 1")
        db_latency = (time.time() - start) * 1000
    except:
        db_status = "unhealthy"
    
    # Redis 检查
    redis_status = "healthy" if redis_cache.redis else "unavailable"
    
    # 缓存统计
    cache_stats = {
        "size": len(response_cache.cache),
        "max_size": response_cache.max_size
    }
    
    return {
        "status": "healthy" if db_status == "healthy" else "degraded",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "uptime_seconds": time.time() - app.state.start_time if hasattr(app.state, 'start_time') else 0,
        "system": {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_used_gb": round(memory.used / (1024**3), 2),
            "disk_percent": disk.percent
        },
        "components": {
            "database": {"status": db_status, "latency_ms": round(db_latency, 2)},
            "redis": {"status": redis_status},
            "cache": cache_stats,
            "task_queue": {"pending_tasks": task_queue.tasks.qsize()}
        }
    }

# ========== API 限流增强 ==========
class AdvancedRateLimiter:
    """高级限流器，支持多种限流策略"""
    def __init__(self):
        self.limits = {
            "default": {"requests": 60, "window": 60},
            "chat": {"requests": 30, "window": 60},
            "image": {"requests": 10, "window": 60},
            "code": {"requests": 20, "window": 60}
        }
        self.requests = defaultdict(list)
        self.lock = asyncio.Lock()
    
    async def check(self, key: str, limit_type: str = "default") -> tuple[bool, dict]:
        async with self.lock:
            now = time.time()
            limit = self.limits.get(limit_type, self.limits["default"])
            window = limit["window"]
            max_requests = limit["requests"]
            
            # 清理过期记录
            self.requests[key] = [t for t in self.requests[key] if t > now - window]
            
            remaining = max_requests - len(self.requests[key])
            reset_time = int(now + window)
            
            if remaining <= 0:
                return False, {
                    "limit": max_requests,
                    "remaining": 0,
                    "reset": reset_time,
                    "retry_after": int(self.requests[key][0] + window - now)
                }
            
            self.requests[key].append(now)
            return True, {
                "limit": max_requests,
                "remaining": remaining - 1,
                "reset": reset_time
            }

advanced_limiter = AdvancedRateLimiter()

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """全局限流中间件"""
    # 获取客户端标识
    client_ip = request.client.host if request.client else "unknown"
    auth = request.headers.get("authorization", "")
    key = auth[7:15] if auth.startswith("Bearer ") else client_ip
    
    # 确定限流类型
    path = request.url.path
    limit_type = "default"
    if "/chat" in path:
        limit_type = "chat"
    elif "/image" in path:
        limit_type = "image"
    elif "/code" in path:
        limit_type = "code"
    
    # 检查限流
    allowed, info = await advanced_limiter.check(key, limit_type)
    
    if not allowed:
        return JSONResponse(
            status_code=429,
            content={"error": "请求过于频繁，请稍后再试", "retry_after": info["retry_after"]},
            headers={
                "X-RateLimit-Limit": str(info["limit"]),
                "X-RateLimit-Remaining": str(info["remaining"]),
                "X-RateLimit-Reset": str(info["reset"]),
                "Retry-After": str(info["retry_after"])
            }
        )
    
    response = await call_next(request)
    response.headers["X-RateLimit-Limit"] = str(info["limit"])
    response.headers["X-RateLimit-Remaining"] = str(info["remaining"])
    response.headers["X-RateLimit-Reset"] = str(info["reset"])
    return response

# ========== 结构化日志 ==========
@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """请求日志中间件"""
    start_time = time.time()
    
    # 生成请求 ID
    request_id = secrets.token_hex(8)
    
    response = await call_next(request)
    
    # 计算耗时
    duration = (time.time() - start_time) * 1000
    
    # 记录日志
    log_data = {
        "request_id": request_id,
        "method": request.method,
        "path": request.url.path,
        "status": response.status_code,
        "duration_ms": round(duration, 2),
        "client_ip": request.client.host if request.client else "unknown"
    }
    
    if response.status_code >= 400:
        logger.warning(f"Request failed: {json.dumps(log_data)}")
    elif duration > 1000:
        logger.warning(f"Slow request: {json.dumps(log_data)}")
    else:
        logger.debug(f"Request: {json.dumps(log_data)}")
    
    response.headers["X-Request-ID"] = request_id
    return response

# ========== 数据库优化 ==========
@app.post("/api/admin/optimize-db")
async def optimize_database(user=Depends(get_current_user)):
    """优化数据库"""
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    
    try:
        with sqlite3.connect(DB_PATH) as conn:
            # 清理过期会话
            conn.execute("DELETE FROM sessions WHERE expires_at < ?", (datetime.now(),))
            
            # 清理旧日志（保留30天）
            conn.execute("DELETE FROM api_logs WHERE created_at < date('now', '-30 days')")
            
            # 清理旧审计日志（保留90天）
            conn.execute("DELETE FROM audit_logs WHERE created_at < date('now', '-90 days')")
            
            # 执行 VACUUM
            conn.execute("VACUUM")
            
            # 分析表
            conn.execute("ANALYZE")
        
        return {"success": True, "message": "数据库优化完成"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"优化失败: {str(e)}")

# ========== 批量操作 API ==========
@app.post("/api/conversations/batch-delete")
async def batch_delete_conversations(data: dict, user=Depends(get_current_user)):
    """批量删除对话"""
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    
    ids = data.get("ids", [])
    if not ids:
        return {"deleted": 0}
    
    with sqlite3.connect(DB_PATH) as conn:
        placeholders = ",".join("?" * len(ids))
        conn.execute(f"DELETE FROM messages WHERE conversation_id IN ({placeholders})", ids)
        result = conn.execute(f"DELETE FROM conversations WHERE id IN ({placeholders}) AND user_id = ?", ids + [user["id"]])
    
    return {"deleted": result.rowcount}

@app.post("/api/messages/batch-export")
async def batch_export_messages(data: dict, user=Depends(get_current_user)):
    """批量导出消息"""
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    
    conv_ids = data.get("conversation_ids", [])
    format_type = data.get("format", "json")
    
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        
        result = []
        for conv_id in conv_ids:
            conv = conn.execute(
                "SELECT * FROM conversations WHERE id = ? AND user_id = ?",
                (conv_id, user["id"])
            ).fetchone()
            
            if conv:
                messages = conn.execute(
                    "SELECT role, content, created_at FROM messages WHERE conversation_id = ? ORDER BY created_at",
                    (conv_id,)
                ).fetchall()
                
                result.append({
                    "id": conv["id"],
                    "title": conv["title"],
                    "messages": [dict(m) for m in messages]
                })
    
    return {"conversations": result, "count": len(result)}

# 记录启动时间
app.state.start_time = time.time()

# ========== 集成扩展模块 ==========
try:
    from modules.integration import setup_modules
    from modules.api_routes import router as extended_router
    from modules.rbac import rbac
    from modules.billing import billing
    from modules.security import waf, ai_detector, auditor
    from modules.enterprise import tenant_manager, compliance
    
    # 注册扩展路由
    app.include_router(extended_router)
    
    # 添加安全检查中间件
    @app.middleware("http")
    async def security_check_middleware(request: Request, call_next):
        # WAF 检查
        client_ip = request.client.host if request.client else "unknown"
        if waf.is_ip_blocked(client_ip):
            return JSONResponse(status_code=403, content={"detail": "IP 已被封禁"})
        
        # 检查请求体（仅对 POST/PUT）
        if request.method in ["POST", "PUT"]:
            try:
                body = await request.body()
                if body:
                    body_str = body.decode('utf-8', errors='ignore')
                    waf_result = waf.check(body_str, client_ip)
                    if waf_result["blocked"]:
                        auditor.log_event("waf_blocked", "high", ip=client_ip,
                                         details={"violations": waf_result["violations"]})
                        return JSONResponse(status_code=403, 
                                          content={"detail": "请求被安全策略拦截"})
            except:
                pass
        
        return await call_next(request)
    
    logger.info("✅ 扩展模块已加载: RBAC, 计费, RAG, 协作, 安全, 企业功能")
except ImportError as e:
    logger.warning(f"⚠️ 扩展模块未加载: {e}")
except Exception as e:
    logger.error(f"❌ 扩展模块加载失败: {e}")

# ========== 系统监控 API ==========
@app.get("/api/system/metrics")
async def get_system_metrics():
    """获取系统指标"""
    import psutil
    
    # 基础指标
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    # 计算运行时间
    import time
    uptime = time.time() - app.state.start_time if hasattr(app.state, 'start_time') else 0
    
    # 性能指标（从监控模块获取）
    performance = {}
    try:
        from modules.monitoring import profiler
        performance = profiler.get_stats()
    except:
        pass
    
    return {
        "cpu_percent": cpu_percent,
        "memory_percent": memory.percent,
        "memory_used_gb": round(memory.used / (1024**3), 2),
        "memory_total_gb": round(memory.total / (1024**3), 2),
        "disk_percent": disk.percent,
        "disk_used_gb": round(disk.used / (1024**3), 2),
        "disk_total_gb": round(disk.total / (1024**3), 2),
        "uptime": uptime,
        "performance": performance
    }

@app.get("/api/system/alerts")
async def get_system_alerts():
    """获取系统告警"""
    try:
        from modules.monitoring import alert_manager
        return alert_manager.get_alerts(limit=50)
    except:
        return []

@app.get("/api/system/traces")
async def get_system_traces(limit: int = 20):
    """获取链路追踪"""
    try:
        from modules.monitoring import tracer
        return tracer.get_traces(limit=limit)
    except:
        return []

# ========== 任务队列 API ==========
@app.get("/api/tasks/stats")
async def get_task_stats():
    """获取任务统计"""
    try:
        from modules.queue import task_queue
        return task_queue.get_stats()
    except:
        return {"total": 0, "by_status": {}, "workers": 0, "running": False}

@app.get("/api/tasks")
async def get_tasks(limit: int = 50):
    """获取任务列表"""
    try:
        from modules.queue import task_queue
        return task_queue.get_tasks(limit=limit)
    except:
        return []

@app.get("/api/tasks/scheduled")
async def get_scheduled_jobs():
    """获取定时任务"""
    try:
        from modules.queue import scheduler
        return scheduler.get_jobs()
    except:
        return []

# 挂载静态文件
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
