from fastapi import FastAPI, HTTPException, Header, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
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
from datetime import datetime, timedelta
from typing import Optional, AsyncGenerator, List
from contextlib import contextmanager

load_dotenv()

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
        ''')

init_db()

# 密码哈希
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

# 获取当前用户
async def get_current_user(authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        return None
    token = authorization[7:]
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT u.* FROM users u JOIN sessions s ON u.id = s.user_id WHERE s.token = ? AND s.expires_at > ?",
            (token, datetime.now())
        ).fetchone()
        return dict(row) if row else None

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

class ChatMessage(BaseModel):
    role: str
    content: str

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
async def chat_completions(request: ChatRequest, user=Depends(get_current_user)):
    provider = request.provider.lower()
    
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
            
            # 保存消息到数据库
            if request.conversation_id and user:
                with sqlite3.connect(DB_PATH) as conn:
                    # 保存用户消息
                    user_msg = request.messages[-1]
                    conn.execute(
                        "INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)",
                        (request.conversation_id, user_msg.role, user_msg.content)
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
                    # 更新对话标题
                    if len(request.messages) == 1:
                        title = request.messages[0].content[:30]
                        conn.execute("UPDATE conversations SET title = ? WHERE id = ?", (title, request.conversation_id))
                    conn.execute("UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = ?", (request.conversation_id,))
                    # 更新用户token使用量
                    tokens = result.get("usage", {}).get("total_tokens", 100)
                    conn.execute("UPDATE users SET tokens_used = tokens_used + ? WHERE id = ?", (tokens, user["id"]))
            
            return result
            
        except httpx.HTTPStatusError as e:
            error_detail = f"API 请求失败 ({e.response.status_code})"
            try:
                err_json = e.response.json()
                error_detail = err_json.get("error", {}).get("message", str(err_json))
            except:
                error_detail = e.response.text[:200] if e.response.text else str(e)
            raise HTTPException(status_code=e.response.status_code, detail=error_detail)
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="请求超时")
        except httpx.ConnectError:
            raise HTTPException(status_code=502, detail="无法连接到 API 服务器")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"内部错误: {str(e)}")

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

# ========== 静态文件 ==========
@app.get("/")
async def root():
    return FileResponse("static/index.html")

app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
