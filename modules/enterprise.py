"""
企业增强模块
支持：LDAP/AD集成、多租户、数据隔离、审计合规
"""
import sqlite3
import json
import secrets
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
import httpx
import os

DB_PATH = "data.db"

# ========== LDAP/AD 集成 ==========
@dataclass
class LDAPConfig:
    server: str
    port: int = 389
    use_ssl: bool = False
    bind_dn: str = ""
    bind_password: str = ""
    base_dn: str = ""
    user_filter: str = "(uid={username})"
    group_filter: str = "(member={user_dn})"


class LDAPAuthenticator:
    """LDAP 认证器"""
    
    def __init__(self, config: LDAPConfig = None):
        self.config = config
        self._ldap = None
    
    def _get_connection(self):
        """获取 LDAP 连接"""
        try:
            import ldap3
            server = ldap3.Server(
                self.config.server, 
                port=self.config.port,
                use_ssl=self.config.use_ssl
            )
            conn = ldap3.Connection(
                server,
                user=self.config.bind_dn,
                password=self.config.bind_password,
                auto_bind=True
            )
            return conn
        except ImportError:
            return None
        except Exception as e:
            print(f"LDAP connection error: {e}")
            return None
    
    def authenticate(self, username: str, password: str) -> Optional[Dict]:
        """LDAP 认证"""
        if not self.config:
            return None
        
        try:
            import ldap3
            
            # 搜索用户
            conn = self._get_connection()
            if not conn:
                return None
            
            user_filter = self.config.user_filter.format(username=username)
            conn.search(
                self.config.base_dn,
                user_filter,
                attributes=['cn', 'mail', 'memberOf', 'uid']
            )
            
            if not conn.entries:
                return None
            
            user_entry = conn.entries[0]
            user_dn = user_entry.entry_dn
            
            # 验证密码
            user_conn = ldap3.Connection(
                ldap3.Server(self.config.server, port=self.config.port),
                user=user_dn,
                password=password
            )
            
            if not user_conn.bind():
                return None
            
            # 获取用户组
            groups = []
            if hasattr(user_entry, 'memberOf'):
                groups = [str(g) for g in user_entry.memberOf]
            
            return {
                "username": username,
                "dn": user_dn,
                "email": str(user_entry.mail) if hasattr(user_entry, 'mail') else None,
                "display_name": str(user_entry.cn) if hasattr(user_entry, 'cn') else username,
                "groups": groups
            }
        
        except ImportError:
            print("ldap3 library not installed")
            return None
        except Exception as e:
            print(f"LDAP auth error: {e}")
            return None
    
    def sync_users(self) -> List[Dict]:
        """同步 LDAP 用户"""
        if not self.config:
            return []
        
        try:
            import ldap3
            
            conn = self._get_connection()
            if not conn:
                return []
            
            conn.search(
                self.config.base_dn,
                "(objectClass=person)",
                attributes=['uid', 'cn', 'mail']
            )
            
            users = []
            for entry in conn.entries:
                users.append({
                    "username": str(entry.uid) if hasattr(entry, 'uid') else None,
                    "display_name": str(entry.cn) if hasattr(entry, 'cn') else None,
                    "email": str(entry.mail) if hasattr(entry, 'mail') else None
                })
            
            return users
        
        except Exception as e:
            print(f"LDAP sync error: {e}")
            return []

# ========== 多租户支持 ==========
@dataclass
class Tenant:
    id: str
    name: str
    domain: str
    plan: str
    settings: Dict
    is_active: bool = True
    created_at: datetime = None

class MultiTenantManager:
    """多租户管理器"""
    
    def __init__(self):
        self._init_db()
        self._tenant_cache: Dict[str, Tenant] = {}
    
    def _init_db(self):
        """初始化多租户表"""
        with sqlite3.connect(DB_PATH) as conn:
            conn.executescript('''
                CREATE TABLE IF NOT EXISTS tenants (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    domain TEXT UNIQUE,
                    plan TEXT DEFAULT 'basic',
                    settings TEXT DEFAULT '{}',
                    is_active INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS tenant_users (
                    tenant_id TEXT,
                    user_id INTEGER,
                    role TEXT DEFAULT 'member',
                    joined_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (tenant_id, user_id)
                );
                
                CREATE TABLE IF NOT EXISTS tenant_quotas (
                    tenant_id TEXT PRIMARY KEY,
                    max_users INTEGER DEFAULT 10,
                    max_tokens INTEGER DEFAULT 1000000,
                    used_tokens INTEGER DEFAULT 0,
                    max_storage_mb INTEGER DEFAULT 1024,
                    used_storage_mb INTEGER DEFAULT 0
                );
                
                CREATE INDEX IF NOT EXISTS idx_tenant_domain ON tenants(domain);
                CREATE INDEX IF NOT EXISTS idx_tenant_users ON tenant_users(user_id);
            ''')
    
    def create_tenant(self, name: str, domain: str, plan: str = "basic",
                     admin_user_id: int = None) -> Tenant:
        """创建租户"""
        tenant_id = f"tenant_{secrets.token_hex(8)}"
        
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                INSERT INTO tenants (id, name, domain, plan)
                VALUES (?, ?, ?, ?)
            """, (tenant_id, name, domain, plan))
            
            # 初始化配额
            quotas = self._get_plan_quotas(plan)
            conn.execute("""
                INSERT INTO tenant_quotas (tenant_id, max_users, max_tokens, max_storage_mb)
                VALUES (?, ?, ?, ?)
            """, (tenant_id, quotas["max_users"], quotas["max_tokens"], quotas["max_storage_mb"]))
            
            # 添加管理员
            if admin_user_id:
                conn.execute("""
                    INSERT INTO tenant_users (tenant_id, user_id, role)
                    VALUES (?, ?, 'admin')
                """, (tenant_id, admin_user_id))
        
        return Tenant(
            id=tenant_id,
            name=name,
            domain=domain,
            plan=plan,
            settings={},
            created_at=datetime.now()
        )
    
    def _get_plan_quotas(self, plan: str) -> Dict:
        """获取套餐配额"""
        quotas = {
            "basic": {"max_users": 10, "max_tokens": 1000000, "max_storage_mb": 1024},
            "pro": {"max_users": 50, "max_tokens": 10000000, "max_storage_mb": 10240},
            "enterprise": {"max_users": -1, "max_tokens": -1, "max_storage_mb": -1},
        }
        return quotas.get(plan, quotas["basic"])
    
    def get_tenant_by_domain(self, domain: str) -> Optional[Tenant]:
        """通过域名获取租户"""
        if domain in self._tenant_cache:
            return self._tenant_cache[domain]
        
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM tenants WHERE domain = ? AND is_active = 1",
                (domain,)
            ).fetchone()
            
            if row:
                tenant = Tenant(
                    id=row["id"],
                    name=row["name"],
                    domain=row["domain"],
                    plan=row["plan"],
                    settings=json.loads(row["settings"]),
                    is_active=bool(row["is_active"])
                )
                self._tenant_cache[domain] = tenant
                return tenant
        
        return None
    
    def get_user_tenant(self, user_id: int) -> Optional[Tenant]:
        """获取用户所属租户"""
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("""
                SELECT t.* FROM tenants t
                JOIN tenant_users tu ON t.id = tu.tenant_id
                WHERE tu.user_id = ? AND t.is_active = 1
            """, (user_id,)).fetchone()
            
            if row:
                return Tenant(
                    id=row["id"],
                    name=row["name"],
                    domain=row["domain"],
                    plan=row["plan"],
                    settings=json.loads(row["settings"]),
                    is_active=bool(row["is_active"])
                )
        
        return None

    
    def add_user_to_tenant(self, tenant_id: str, user_id: int, 
                          role: str = "member") -> bool:
        """添加用户到租户"""
        with sqlite3.connect(DB_PATH) as conn:
            # 检查配额
            quota = conn.execute("""
                SELECT max_users, 
                       (SELECT COUNT(*) FROM tenant_users WHERE tenant_id = ?) as current_users
                FROM tenant_quotas WHERE tenant_id = ?
            """, (tenant_id, tenant_id)).fetchone()
            
            if quota and quota[0] > 0 and quota[1] >= quota[0]:
                return False  # 超出用户配额
            
            conn.execute("""
                INSERT OR REPLACE INTO tenant_users (tenant_id, user_id, role)
                VALUES (?, ?, ?)
            """, (tenant_id, user_id, role))
        
        return True
    
    def check_quota(self, tenant_id: str, resource: str, amount: int = 1) -> bool:
        """检查租户配额"""
        with sqlite3.connect(DB_PATH) as conn:
            row = conn.execute(
                "SELECT * FROM tenant_quotas WHERE tenant_id = ?",
                (tenant_id,)
            ).fetchone()
            
            if not row:
                return True
            
            if resource == "tokens":
                max_val, used_val = row[2], row[3]  # max_tokens, used_tokens
            elif resource == "storage":
                max_val, used_val = row[4], row[5]  # max_storage_mb, used_storage_mb
            else:
                return True
            
            if max_val < 0:  # 无限制
                return True
            
            return used_val + amount <= max_val
    
    def update_usage(self, tenant_id: str, resource: str, amount: int):
        """更新租户用量"""
        with sqlite3.connect(DB_PATH) as conn:
            if resource == "tokens":
                conn.execute("""
                    UPDATE tenant_quotas SET used_tokens = used_tokens + ?
                    WHERE tenant_id = ?
                """, (amount, tenant_id))
            elif resource == "storage":
                conn.execute("""
                    UPDATE tenant_quotas SET used_storage_mb = used_storage_mb + ?
                    WHERE tenant_id = ?
                """, (amount, tenant_id))

# ========== 数据隔离 ==========
class DataIsolation:
    """数据隔离管理"""
    
    @staticmethod
    def get_tenant_db_path(tenant_id: str) -> str:
        """获取租户专属数据库路径"""
        os.makedirs("tenant_data", exist_ok=True)
        return f"tenant_data/{tenant_id}.db"
    
    @staticmethod
    def init_tenant_db(tenant_id: str):
        """初始化租户数据库"""
        db_path = DataIsolation.get_tenant_db_path(tenant_id)
        
        with sqlite3.connect(db_path) as conn:
            conn.executescript('''
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    user_id INTEGER,
                    title TEXT,
                    provider TEXT,
                    model TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT,
                    role TEXT,
                    content TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS notes (
                    id TEXT PRIMARY KEY,
                    user_id INTEGER,
                    title TEXT,
                    content TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS knowledge_documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    kb_id INTEGER,
                    filename TEXT,
                    content TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            ''')
    
    @staticmethod
    def get_tenant_connection(tenant_id: str):
        """获取租户数据库连接"""
        db_path = DataIsolation.get_tenant_db_path(tenant_id)
        if not os.path.exists(db_path):
            DataIsolation.init_tenant_db(tenant_id)
        return sqlite3.connect(db_path)

# ========== 审计合规 ==========
class ComplianceManager:
    """合规管理器（SOX/GDPR）"""
    
    def __init__(self):
        self._init_db()
    
    def _init_db(self):
        """初始化合规表"""
        with sqlite3.connect(DB_PATH) as conn:
            conn.executescript('''
                CREATE TABLE IF NOT EXISTS compliance_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tenant_id TEXT,
                    user_id INTEGER,
                    action TEXT NOT NULL,
                    resource_type TEXT,
                    resource_id TEXT,
                    old_value TEXT,
                    new_value TEXT,
                    ip_address TEXT,
                    user_agent TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS data_retention_policies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tenant_id TEXT,
                    resource_type TEXT,
                    retention_days INTEGER DEFAULT 365,
                    delete_after_retention INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS gdpr_requests (
                    id TEXT PRIMARY KEY,
                    user_id INTEGER,
                    request_type TEXT,
                    status TEXT DEFAULT 'pending',
                    processed_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE INDEX IF NOT EXISTS idx_compliance_tenant ON compliance_logs(tenant_id);
                CREATE INDEX IF NOT EXISTS idx_compliance_user ON compliance_logs(user_id);
                CREATE INDEX IF NOT EXISTS idx_compliance_action ON compliance_logs(action);
            ''')
    
    def log_action(self, action: str, user_id: int = None, tenant_id: str = None,
                   resource_type: str = None, resource_id: str = None,
                   old_value: str = None, new_value: str = None,
                   ip: str = None, user_agent: str = None):
        """记录合规日志"""
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                INSERT INTO compliance_logs 
                (tenant_id, user_id, action, resource_type, resource_id, 
                 old_value, new_value, ip_address, user_agent)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (tenant_id, user_id, action, resource_type, resource_id,
                  old_value, new_value, ip, user_agent))
    
    def create_gdpr_request(self, user_id: int, request_type: str) -> str:
        """创建 GDPR 请求（数据导出/删除）"""
        request_id = f"gdpr_{secrets.token_hex(8)}"
        
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                INSERT INTO gdpr_requests (id, user_id, request_type)
                VALUES (?, ?, ?)
            """, (request_id, user_id, request_type))
        
        return request_id
    
    def process_gdpr_request(self, request_id: str) -> Dict:
        """处理 GDPR 请求"""
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            request = conn.execute(
                "SELECT * FROM gdpr_requests WHERE id = ?", (request_id,)
            ).fetchone()
            
            if not request:
                return {"error": "请求不存在"}
            
            user_id = request["user_id"]
            request_type = request["request_type"]
            
            if request_type == "export":
                # 导出用户数据
                data = self._export_user_data(user_id)
                result = {"type": "export", "data": data}
            
            elif request_type == "delete":
                # 删除用户数据
                self._delete_user_data(user_id)
                result = {"type": "delete", "success": True}
            
            elif request_type == "access":
                # 数据访问报告
                data = self._export_user_data(user_id)
                result = {"type": "access", "data": data}
            
            else:
                result = {"error": f"未知请求类型: {request_type}"}
            
            # 更新请求状态
            conn.execute("""
                UPDATE gdpr_requests SET status = 'completed', processed_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (request_id,))
            
            return result
    
    def _export_user_data(self, user_id: int) -> Dict:
        """导出用户数据"""
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            
            user = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
            conversations = conn.execute(
                "SELECT * FROM conversations WHERE user_id = ?", (user_id,)
            ).fetchall()
            notes = conn.execute(
                "SELECT * FROM notes WHERE user_id = ?", (user_id,)
            ).fetchall()
            
            return {
                "user": dict(user) if user else None,
                "conversations": [dict(c) for c in conversations],
                "notes": [dict(n) for n in notes],
                "exported_at": datetime.now().isoformat()
            }
    
    def _delete_user_data(self, user_id: int):
        """删除用户数据"""
        with sqlite3.connect(DB_PATH) as conn:
            # 删除消息
            conn.execute("""
                DELETE FROM messages WHERE conversation_id IN 
                (SELECT id FROM conversations WHERE user_id = ?)
            """, (user_id,))
            
            # 删除对话
            conn.execute("DELETE FROM conversations WHERE user_id = ?", (user_id,))
            
            # 删除笔记
            conn.execute("DELETE FROM notes WHERE user_id = ?", (user_id,))
            
            # 删除记忆
            conn.execute("DELETE FROM memories WHERE user_id = ?", (user_id,))
            
            # 匿名化用户（保留记录但删除个人信息）
            conn.execute("""
                UPDATE users SET 
                    username = ?, 
                    email = NULL, 
                    password_hash = ?
                WHERE id = ?
            """, (f"deleted_user_{user_id}", hashlib.sha256(secrets.token_bytes(32)).hexdigest(), user_id))
    
    def generate_compliance_report(self, tenant_id: str = None, days: int = 30) -> Dict:
        """生成合规报告"""
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            
            # 基础查询条件
            where = "WHERE created_at >= datetime('now', ?)"
            params = [f'-{days} days']
            
            if tenant_id:
                where += " AND tenant_id = ?"
                params.append(tenant_id)
            
            # 操作统计
            actions = conn.execute(f"""
                SELECT action, COUNT(*) as count
                FROM compliance_logs {where}
                GROUP BY action ORDER BY count DESC
            """, params).fetchall()
            
            # 用户活动
            user_activity = conn.execute(f"""
                SELECT user_id, COUNT(*) as actions
                FROM compliance_logs {where}
                GROUP BY user_id ORDER BY actions DESC LIMIT 20
            """, params).fetchall()
            
            # 敏感操作
            sensitive_actions = conn.execute(f"""
                SELECT * FROM compliance_logs 
                {where} AND action IN ('delete', 'export', 'permission_change', 'login_failed')
                ORDER BY created_at DESC LIMIT 100
            """, params).fetchall()
            
            return {
                "report_date": datetime.now().isoformat(),
                "period_days": days,
                "tenant_id": tenant_id,
                "action_summary": [dict(a) for a in actions],
                "top_users": [dict(u) for u in user_activity],
                "sensitive_operations": [dict(s) for s in sensitive_actions],
                "gdpr_compliance": self._check_gdpr_compliance(tenant_id)
            }
    
    def _check_gdpr_compliance(self, tenant_id: str = None) -> Dict:
        """检查 GDPR 合规状态"""
        with sqlite3.connect(DB_PATH) as conn:
            # 待处理的 GDPR 请求
            pending = conn.execute("""
                SELECT COUNT(*) FROM gdpr_requests WHERE status = 'pending'
            """).fetchone()[0]
            
            # 数据保留策略
            policies = conn.execute("""
                SELECT COUNT(*) FROM data_retention_policies
            """).fetchone()[0]
            
            return {
                "pending_requests": pending,
                "retention_policies_defined": policies > 0,
                "compliant": pending == 0 and policies > 0
            }

# 全局实例
ldap_auth = LDAPAuthenticator()
tenant_manager = MultiTenantManager()
compliance = ComplianceManager()
