"""
实时协作模块
支持：多人同时编辑对话、实时同步、协作会话管理
"""
import asyncio
import json
import secrets
import sqlite3
from datetime import datetime
from typing import Dict, List, Set, Optional
from dataclasses import dataclass, field
from enum import Enum
from fastapi import WebSocket

DB_PATH = "data.db"

class CollaborationRole(str, Enum):
    OWNER = "owner"
    EDITOR = "editor"
    VIEWER = "viewer"

class MessageType(str, Enum):
    JOIN = "join"
    LEAVE = "leave"
    MESSAGE = "message"
    TYPING = "typing"
    CURSOR = "cursor"
    SYNC = "sync"
    ERROR = "error"


@dataclass
class Participant:
    user_id: int
    username: str
    role: CollaborationRole
    websocket: Optional[WebSocket] = None
    cursor_position: int = 0
    is_typing: bool = False
    joined_at: datetime = field(default_factory=datetime.now)

@dataclass
class CollaborationSession:
    id: str
    conversation_id: str
    owner_id: int
    participants: Dict[int, Participant] = field(default_factory=dict)
    messages: List[Dict] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    is_active: bool = True

class CollaborationManager:
    """实时协作管理器"""
    
    def __init__(self):
        self.sessions: Dict[str, CollaborationSession] = {}
        self.user_sessions: Dict[int, Set[str]] = {}  # user_id -> session_ids
        self._init_db()
    
    def _init_db(self):
        """初始化协作相关表"""
        with sqlite3.connect(DB_PATH) as conn:
            conn.executescript('''
                CREATE TABLE IF NOT EXISTS collaboration_sessions (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT,
                    owner_id INTEGER,
                    is_active INTEGER DEFAULT 1,
                    settings TEXT DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS collaboration_participants (
                    session_id TEXT,
                    user_id INTEGER,
                    role TEXT DEFAULT 'viewer',
                    joined_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    left_at TIMESTAMP,
                    PRIMARY KEY (session_id, user_id)
                );
                
                CREATE TABLE IF NOT EXISTS collaboration_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    user_id INTEGER,
                    action TEXT,
                    data TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE INDEX IF NOT EXISTS idx_collab_conv ON collaboration_sessions(conversation_id);
            ''')
    
    def create_session(self, conversation_id: str, owner_id: int, 
                      owner_name: str) -> CollaborationSession:
        """创建协作会话"""
        session_id = f"collab_{secrets.token_hex(8)}"
        
        session = CollaborationSession(
            id=session_id,
            conversation_id=conversation_id,
            owner_id=owner_id
        )
        
        # 添加创建者为参与者
        session.participants[owner_id] = Participant(
            user_id=owner_id,
            username=owner_name,
            role=CollaborationRole.OWNER
        )
        
        self.sessions[session_id] = session
        
        if owner_id not in self.user_sessions:
            self.user_sessions[owner_id] = set()
        self.user_sessions[owner_id].add(session_id)
        
        # 持久化
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                INSERT INTO collaboration_sessions (id, conversation_id, owner_id)
                VALUES (?, ?, ?)
            """, (session_id, conversation_id, owner_id))
            
            conn.execute("""
                INSERT INTO collaboration_participants (session_id, user_id, role)
                VALUES (?, ?, 'owner')
            """, (session_id, owner_id))
        
        return session
    
    async def join_session(self, session_id: str, user_id: int, 
                          username: str, websocket: WebSocket,
                          role: CollaborationRole = CollaborationRole.VIEWER) -> bool:
        """加入协作会话"""
        if session_id not in self.sessions:
            # 尝试从数据库恢复
            session = self._load_session(session_id)
            if not session:
                return False
            self.sessions[session_id] = session
        
        session = self.sessions[session_id]
        
        if not session.is_active:
            return False
        
        # 添加参与者
        participant = Participant(
            user_id=user_id,
            username=username,
            role=role,
            websocket=websocket
        )
        session.participants[user_id] = participant
        
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = set()
        self.user_sessions[user_id].add(session_id)
        
        # 持久化
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO collaboration_participants 
                (session_id, user_id, role, joined_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            """, (session_id, user_id, role.value))
        
        # 广播加入消息
        await self.broadcast(session_id, {
            "type": MessageType.JOIN.value,
            "user_id": user_id,
            "username": username,
            "participants": self._get_participants_info(session_id)
        }, exclude_user=user_id)
        
        return True
    
    async def leave_session(self, session_id: str, user_id: int):
        """离开协作会话"""
        if session_id not in self.sessions:
            return
        
        session = self.sessions[session_id]
        
        if user_id in session.participants:
            participant = session.participants[user_id]
            del session.participants[user_id]
            
            # 更新数据库
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute("""
                    UPDATE collaboration_participants 
                    SET left_at = CURRENT_TIMESTAMP
                    WHERE session_id = ? AND user_id = ?
                """, (session_id, user_id))
            
            # 广播离开消息
            await self.broadcast(session_id, {
                "type": MessageType.LEAVE.value,
                "user_id": user_id,
                "username": participant.username,
                "participants": self._get_participants_info(session_id)
            })
        
        if user_id in self.user_sessions:
            self.user_sessions[user_id].discard(session_id)
        
        # 如果没有参与者了，关闭会话
        if not session.participants:
            session.is_active = False
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute(
                    "UPDATE collaboration_sessions SET is_active = 0 WHERE id = ?",
                    (session_id,)
                )

    
    async def broadcast(self, session_id: str, message: Dict, 
                       exclude_user: int = None):
        """广播消息给所有参与者"""
        if session_id not in self.sessions:
            return
        
        session = self.sessions[session_id]
        message_json = json.dumps(message, ensure_ascii=False)
        
        disconnected = []
        for user_id, participant in session.participants.items():
            if exclude_user and user_id == exclude_user:
                continue
            
            if participant.websocket:
                try:
                    await participant.websocket.send_text(message_json)
                except:
                    disconnected.append(user_id)
        
        # 清理断开的连接
        for user_id in disconnected:
            await self.leave_session(session_id, user_id)
    
    async def send_message(self, session_id: str, user_id: int, 
                          content: str, msg_type: str = "chat"):
        """发送消息"""
        if session_id not in self.sessions:
            return
        
        session = self.sessions[session_id]
        
        if user_id not in session.participants:
            return
        
        participant = session.participants[user_id]
        
        # 检查权限
        if participant.role == CollaborationRole.VIEWER and msg_type == "chat":
            return
        
        message = {
            "type": MessageType.MESSAGE.value,
            "msg_type": msg_type,
            "user_id": user_id,
            "username": participant.username,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        
        session.messages.append(message)
        
        # 记录历史
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                INSERT INTO collaboration_history (session_id, user_id, action, data)
                VALUES (?, ?, 'message', ?)
            """, (session_id, user_id, json.dumps(message)))
        
        await self.broadcast(session_id, message)
    
    async def update_typing(self, session_id: str, user_id: int, is_typing: bool):
        """更新输入状态"""
        if session_id not in self.sessions:
            return
        
        session = self.sessions[session_id]
        
        if user_id in session.participants:
            session.participants[user_id].is_typing = is_typing
            
            await self.broadcast(session_id, {
                "type": MessageType.TYPING.value,
                "user_id": user_id,
                "is_typing": is_typing
            }, exclude_user=user_id)
    
    async def update_cursor(self, session_id: str, user_id: int, position: int):
        """更新光标位置"""
        if session_id not in self.sessions:
            return
        
        session = self.sessions[session_id]
        
        if user_id in session.participants:
            session.participants[user_id].cursor_position = position
            
            await self.broadcast(session_id, {
                "type": MessageType.CURSOR.value,
                "user_id": user_id,
                "position": position
            }, exclude_user=user_id)
    
    async def sync_state(self, session_id: str, user_id: int):
        """同步会话状态"""
        if session_id not in self.sessions:
            return
        
        session = self.sessions[session_id]
        
        if user_id not in session.participants:
            return
        
        participant = session.participants[user_id]
        
        if participant.websocket:
            await participant.websocket.send_text(json.dumps({
                "type": MessageType.SYNC.value,
                "session_id": session_id,
                "conversation_id": session.conversation_id,
                "participants": self._get_participants_info(session_id),
                "messages": session.messages[-50:],  # 最近50条消息
                "your_role": participant.role.value
            }, ensure_ascii=False))
    
    def _get_participants_info(self, session_id: str) -> List[Dict]:
        """获取参与者信息"""
        if session_id not in self.sessions:
            return []
        
        session = self.sessions[session_id]
        return [{
            "user_id": p.user_id,
            "username": p.username,
            "role": p.role.value,
            "is_typing": p.is_typing,
            "cursor_position": p.cursor_position
        } for p in session.participants.values()]
    
    def _load_session(self, session_id: str) -> Optional[CollaborationSession]:
        """从数据库加载会话"""
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM collaboration_sessions WHERE id = ? AND is_active = 1",
                (session_id,)
            ).fetchone()
            
            if not row:
                return None
            
            return CollaborationSession(
                id=row["id"],
                conversation_id=row["conversation_id"],
                owner_id=row["owner_id"],
                is_active=bool(row["is_active"])
            )
    
    def get_user_sessions(self, user_id: int) -> List[Dict]:
        """获取用户的协作会话"""
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT cs.*, cp.role 
                FROM collaboration_sessions cs
                JOIN collaboration_participants cp ON cs.id = cp.session_id
                WHERE cp.user_id = ? AND cs.is_active = 1 AND cp.left_at IS NULL
            """, (user_id,)).fetchall()
            
            return [dict(r) for r in rows]
    
    def invite_user(self, session_id: str, inviter_id: int, 
                   invitee_id: int, role: CollaborationRole) -> str:
        """邀请用户加入协作"""
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        
        # 检查邀请者权限
        if inviter_id not in session.participants:
            return None
        
        inviter = session.participants[inviter_id]
        if inviter.role not in [CollaborationRole.OWNER, CollaborationRole.EDITOR]:
            return None
        
        # 生成邀请链接
        invite_token = secrets.token_urlsafe(16)
        
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS collaboration_invites (
                    token TEXT PRIMARY KEY,
                    session_id TEXT,
                    inviter_id INTEGER,
                    invitee_id INTEGER,
                    role TEXT,
                    expires_at TIMESTAMP,
                    used INTEGER DEFAULT 0
                )
            """)
            
            conn.execute("""
                INSERT INTO collaboration_invites 
                (token, session_id, inviter_id, invitee_id, role, expires_at)
                VALUES (?, ?, ?, ?, ?, datetime('now', '+24 hours'))
            """, (invite_token, session_id, inviter_id, invitee_id, role.value))
        
        return invite_token
    
    def close_session(self, session_id: str, user_id: int) -> bool:
        """关闭协作会话"""
        if session_id not in self.sessions:
            return False
        
        session = self.sessions[session_id]
        
        # 只有所有者可以关闭
        if session.owner_id != user_id:
            return False
        
        session.is_active = False
        
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                "UPDATE collaboration_sessions SET is_active = 0 WHERE id = ?",
                (session_id,)
            )
        
        # 通知所有参与者
        asyncio.create_task(self.broadcast(session_id, {
            "type": "session_closed",
            "message": "协作会话已关闭"
        }))
        
        return True

# 全局实例
collaboration = CollaborationManager()
