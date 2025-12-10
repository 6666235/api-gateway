"""
RBAC 细粒度权限控制模块
支持：角色管理、权限检查、资源访问控制
"""
import sqlite3
import json
from typing import List, Dict, Optional, Set
from functools import wraps
from dataclasses import dataclass
from enum import Enum

DB_PATH = "data.db"

class Permission(str, Enum):
    # 基础权限
    CHAT = "chat"
    NOTES = "notes"
    MEMORY = "memory"
    SHORTCUTS = "shortcuts"
    
    # 高级权限
    IMAGE_GEN = "image"
    CODE_RUN = "code"
    RAG = "rag"
    AGENTS = "agents"
    WORKFLOWS = "workflows"
    
    # 管理权限
    ADMIN = "admin"
    USER_MANAGE = "user_manage"
    TEAM_MANAGE = "team_manage"
    BILLING = "billing"
    AUDIT = "audit"
    
    # 通配符
    ALL = "*"


@dataclass
class Role:
    id: int
    name: str
    description: str
    permissions: Set[str]

@dataclass
class Resource:
    type: str  # conversation, note, team, etc.
    id: str
    owner_id: int
    team_id: Optional[int] = None

class RBACManager:
    """RBAC 权限管理器"""
    
    # 默认角色权限映射
    DEFAULT_ROLES = {
        "admin": ["*"],
        "user": ["chat", "notes", "memory", "shortcuts"],
        "vip": ["chat", "notes", "memory", "shortcuts", "image", "code", "rag", "agents"],
        "guest": ["chat"],
        "team_admin": ["chat", "notes", "memory", "team_manage"],
        "team_member": ["chat", "notes"],
    }
    
    def __init__(self):
        self._init_db()
        self._cache = {}  # user_id -> permissions cache
    
    def _init_db(self):
        """初始化权限相关表"""
        with sqlite3.connect(DB_PATH) as conn:
            conn.executescript('''
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
                    granted_by INTEGER,
                    granted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (user_id, role_id)
                );
                
                CREATE TABLE IF NOT EXISTS resource_permissions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    resource_type TEXT,
                    resource_id TEXT,
                    user_id INTEGER,
                    permission TEXT,
                    granted_by INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(resource_type, resource_id, user_id, permission)
                );
                
                CREATE INDEX IF NOT EXISTS idx_user_roles ON user_roles(user_id);
                CREATE INDEX IF NOT EXISTS idx_resource_perms ON resource_permissions(resource_type, resource_id);
            ''')
            
            # 初始化默认角色
            for name, perms in self.DEFAULT_ROLES.items():
                conn.execute("""
                    INSERT OR IGNORE INTO roles (name, description, permissions)
                    VALUES (?, ?, ?)
                """, (name, f"默认{name}角色", json.dumps(perms)))
    
    def get_user_permissions(self, user_id: int) -> Set[str]:
        """获取用户所有权限"""
        if user_id in self._cache:
            return self._cache[user_id]
        
        permissions = set()
        
        with sqlite3.connect(DB_PATH) as conn:
            # 获取用户角色的权限
            rows = conn.execute("""
                SELECT r.permissions FROM roles r
                JOIN user_roles ur ON r.id = ur.role_id
                WHERE ur.user_id = ?
            """, (user_id,)).fetchall()
            
            for row in rows:
                perms = json.loads(row[0])
                permissions.update(perms)
            
            # 如果没有角色，给默认 user 角色
            if not permissions:
                permissions.update(self.DEFAULT_ROLES["user"])
        
        self._cache[user_id] = permissions
        return permissions
    
    def has_permission(self, user_id: int, permission: str) -> bool:
        """检查用户是否有某权限"""
        perms = self.get_user_permissions(user_id)
        return "*" in perms or permission in perms
    
    def check_resource_access(self, user_id: int, resource: Resource, 
                             action: str = "read") -> bool:
        """检查用户对资源的访问权限"""
        # 资源所有者有完全权限
        if resource.owner_id == user_id:
            return True
        
        # 检查是否有管理员权限
        if self.has_permission(user_id, "*"):
            return True
        
        # 检查资源级别权限
        with sqlite3.connect(DB_PATH) as conn:
            row = conn.execute("""
                SELECT 1 FROM resource_permissions
                WHERE resource_type = ? AND resource_id = ? 
                AND user_id = ? AND permission IN (?, '*')
            """, (resource.type, resource.id, user_id, action)).fetchone()
            
            if row:
                return True
            
            # 检查团队权限
            if resource.team_id:
                team_row = conn.execute("""
                    SELECT role FROM team_members
                    WHERE team_id = ? AND user_id = ?
                """, (resource.team_id, user_id)).fetchone()
                
                if team_row:
                    team_role = team_row[0]
                    if team_role in ["owner", "admin"]:
                        return True
                    if team_role == "member" and action == "read":
                        return True
        
        return False

    
    def assign_role(self, user_id: int, role_name: str, granted_by: int = None) -> bool:
        """给用户分配角色"""
        with sqlite3.connect(DB_PATH) as conn:
            role = conn.execute("SELECT id FROM roles WHERE name = ?", (role_name,)).fetchone()
            if not role:
                return False
            
            conn.execute("""
                INSERT OR REPLACE INTO user_roles (user_id, role_id, granted_by)
                VALUES (?, ?, ?)
            """, (user_id, role[0], granted_by))
        
        # 清除缓存
        self._cache.pop(user_id, None)
        return True
    
    def revoke_role(self, user_id: int, role_name: str) -> bool:
        """撤销用户角色"""
        with sqlite3.connect(DB_PATH) as conn:
            role = conn.execute("SELECT id FROM roles WHERE name = ?", (role_name,)).fetchone()
            if not role:
                return False
            
            conn.execute("DELETE FROM user_roles WHERE user_id = ? AND role_id = ?",
                        (user_id, role[0]))
        
        self._cache.pop(user_id, None)
        return True
    
    def grant_resource_permission(self, resource: Resource, user_id: int,
                                  permission: str, granted_by: int) -> bool:
        """授予资源权限"""
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO resource_permissions
                (resource_type, resource_id, user_id, permission, granted_by)
                VALUES (?, ?, ?, ?, ?)
            """, (resource.type, resource.id, user_id, permission, granted_by))
        return True
    
    def revoke_resource_permission(self, resource: Resource, user_id: int,
                                   permission: str) -> bool:
        """撤销资源权限"""
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                DELETE FROM resource_permissions
                WHERE resource_type = ? AND resource_id = ? 
                AND user_id = ? AND permission = ?
            """, (resource.type, resource.id, user_id, permission))
        return True
    
    def create_role(self, name: str, description: str, permissions: List[str]) -> int:
        """创建新角色"""
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.execute("""
                INSERT INTO roles (name, description, permissions)
                VALUES (?, ?, ?)
            """, (name, description, json.dumps(permissions)))
            return cursor.lastrowid
    
    def update_role(self, role_id: int, permissions: List[str]) -> bool:
        """更新角色权限"""
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("UPDATE roles SET permissions = ? WHERE id = ?",
                        (json.dumps(permissions), role_id))
        # 清除所有缓存
        self._cache.clear()
        return True
    
    def get_all_roles(self) -> List[Role]:
        """获取所有角色"""
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM roles").fetchall()
            return [Role(
                id=r["id"],
                name=r["name"],
                description=r["description"],
                permissions=set(json.loads(r["permissions"]))
            ) for r in rows]
    
    def get_user_roles(self, user_id: int) -> List[str]:
        """获取用户的角色列表"""
        with sqlite3.connect(DB_PATH) as conn:
            rows = conn.execute("""
                SELECT r.name FROM roles r
                JOIN user_roles ur ON r.id = ur.role_id
                WHERE ur.user_id = ?
            """, (user_id,)).fetchall()
            return [r[0] for r in rows]
    
    def clear_cache(self, user_id: int = None):
        """清除权限缓存"""
        if user_id:
            self._cache.pop(user_id, None)
        else:
            self._cache.clear()

# 全局实例
rbac = RBACManager()

# 装饰器：权限检查
def require_permission(permission: str):
    """权限检查装饰器"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            user = kwargs.get("user")
            if not user:
                from fastapi import HTTPException
                raise HTTPException(status_code=401, detail="请先登录")
            
            if not rbac.has_permission(user["id"], permission):
                from fastapi import HTTPException
                raise HTTPException(status_code=403, detail=f"无权限: {permission}")
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator
