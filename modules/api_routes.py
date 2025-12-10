"""
扩展 API 路由
整合所有新模块的 API 端点
"""
from fastapi import APIRouter, HTTPException, Depends, Request, WebSocket, WebSocketDisconnect
from typing import Optional, List, Dict
import json

# 导入模块
from .rbac import rbac, require_permission, Resource
from .billing import billing, payment_gateway
from .rag import create_rag_engine, TextSplitter
from .collaboration import collaboration, CollaborationRole

router = APIRouter()

# ========== RBAC 权限 API ==========
@router.get("/api/rbac/roles")
async def list_roles(user=None):
    """获取所有角色"""
    roles = rbac.get_all_roles()
    return [{"id": r.id, "name": r.name, "description": r.description,
             "permissions": list(r.permissions)} for r in roles]


@router.post("/api/rbac/roles")
async def create_role(data: dict, user=None):
    """创建角色"""
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    if not rbac.has_permission(user["id"], "admin"):
        raise HTTPException(status_code=403, detail="需要管理员权限")
    
    role_id = rbac.create_role(
        data.get("name"),
        data.get("description", ""),
        data.get("permissions", [])
    )
    return {"id": role_id, "success": True}

@router.post("/api/rbac/users/{user_id}/roles")
async def assign_user_role(user_id: int, data: dict, user=None):
    """给用户分配角色"""
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    if not rbac.has_permission(user["id"], "user_manage"):
        raise HTTPException(status_code=403, detail="需要用户管理权限")
    
    success = rbac.assign_role(user_id, data.get("role"), user["id"])
    return {"success": success}

@router.delete("/api/rbac/users/{user_id}/roles/{role_name}")
async def revoke_user_role(user_id: int, role_name: str, user=None):
    """撤销用户角色"""
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    if not rbac.has_permission(user["id"], "user_manage"):
        raise HTTPException(status_code=403, detail="需要用户管理权限")
    
    success = rbac.revoke_role(user_id, role_name)
    return {"success": success}

@router.get("/api/rbac/users/{user_id}/permissions")
async def get_user_permissions(user_id: int, user=None):
    """获取用户权限"""
    permissions = rbac.get_user_permissions(user_id)
    roles = rbac.get_user_roles(user_id)
    return {"permissions": list(permissions), "roles": roles}

@router.get("/api/rbac/check")
async def check_permission(permission: str, user=None):
    """检查当前用户权限"""
    if not user:
        return {"has_permission": False}
    return {"has_permission": rbac.has_permission(user["id"], permission)}

# ========== 计费系统 API ==========
@router.get("/api/billing/plans")
async def list_plans():
    """获取所有套餐"""
    return [{"id": p.id, "name": p.name, "price": p.price,
             "tokens_limit": p.tokens_limit, "features": p.features}
            for p in billing.DEFAULT_PLANS.values()]

@router.get("/api/billing/subscription")
async def get_subscription(user=None):
    """获取当前订阅"""
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    return billing.get_subscription(user["id"])

@router.get("/api/billing/usage")
async def get_usage(days: int = 30, user=None):
    """获取用量统计"""
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    return billing.get_usage_summary(user["id"], days)

@router.post("/api/billing/subscribe")
async def subscribe(data: dict, user=None):
    """订阅套餐"""
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    
    plan_id = data.get("plan_id")
    if plan_id == "free":
        # 免费套餐直接激活
        return billing.create_subscription(user["id"], plan_id)
    
    # 付费套餐需要支付
    provider = data.get("payment_provider", "alipay")
    return await payment_gateway.create_payment(user["id"], plan_id, provider)

@router.post("/api/billing/payment/callback/{provider}")
async def payment_callback(provider: str, request: Request):
    """支付回调"""
    if provider == "stripe":
        body = await request.body()
        data = json.loads(body)
    else:
        data = await request.form()
        data = dict(data)
    
    success = await payment_gateway.handle_callback(provider, data)
    return {"success": success}

# ========== RAG 向量检索 API ==========
@router.post("/api/rag/index")
async def index_document(data: dict, user=None):
    """索引文档"""
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    if not rbac.has_permission(user["id"], "rag"):
        raise HTTPException(status_code=403, detail="需要 RAG 权限")
    
    kb_id = data.get("kb_id", 1)
    content = data.get("content", "")
    doc_id = data.get("doc_id")
    metadata = data.get("metadata", {})
    
    engine = create_rag_engine(kb_id)
    chunk_ids = await engine.index_document(content, doc_id, metadata)
    
    return {"success": True, "chunks": len(chunk_ids), "chunk_ids": chunk_ids}

@router.post("/api/rag/search")
async def rag_search(data: dict, user=None):
    """语义搜索"""
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    
    kb_id = data.get("kb_id", 1)
    query = data.get("query", "")
    top_k = data.get("top_k", 5)
    
    engine = create_rag_engine(kb_id)
    docs = await engine.search(query, top_k)
    
    return {"results": [{"id": d.id, "content": d.content, 
                        "score": d.score, "metadata": d.metadata} for d in docs]}

@router.post("/api/rag/query")
async def rag_query(data: dict, user=None):
    """RAG 查询（返回上下文）"""
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    
    kb_id = data.get("kb_id", 1)
    question = data.get("question", "")
    top_k = data.get("top_k", 3)
    
    engine = create_rag_engine(kb_id)
    context, docs = await engine.query(question, top_k)
    
    return {
        "context": context,
        "sources": [{"id": d.id, "content": d.content[:200], "score": d.score} for d in docs]
    }


# ========== 实时协作 API ==========
@router.post("/api/collaboration/sessions")
async def create_collab_session(data: dict, user=None):
    """创建协作会话"""
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    
    conversation_id = data.get("conversation_id")
    if not conversation_id:
        raise HTTPException(status_code=400, detail="需要对话ID")
    
    session = collaboration.create_session(
        conversation_id, 
        user["id"], 
        user.get("username", f"User{user['id']}")
    )
    
    return {"session_id": session.id, "conversation_id": session.conversation_id}

@router.get("/api/collaboration/sessions")
async def list_collab_sessions(user=None):
    """获取用户的协作会话"""
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    return collaboration.get_user_sessions(user["id"])

@router.post("/api/collaboration/sessions/{session_id}/invite")
async def invite_to_session(session_id: str, data: dict, user=None):
    """邀请用户加入协作"""
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    
    invitee_id = data.get("user_id")
    role = ColRole(data.get("role", "viewer"))
    
    token = collaboration.invite_user(session_id, user["id"], invitee_id, role)
    if not token:
        raise HTTPException(status_code=403, detail="无权邀请")
    
    return {"invite_token": token, "invite_url": f"/collaboration/join/{token}"}

@router.delete("/api/collaboration/sessions/{session_id}")
async def close_collab_session(session_id: str, user=None):
    """关闭协作会话"""
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    
    success = collaboration.close_session(session_id, user["id"])
    if not success:
        raise HTTPException(status_code=403, detail="无权关闭")
    
    return {"success": True}

# WebSocket 协作端点
@router.websocket("/ws/collaboration/{session_id}")
async def collaboration_websocket(websocket: WebSocket, session_id: str):
    """协作 WebSocket 连接"""
    await websocket.accept()
    
    # 从 query 参数获取用户信息
    token = websocket.query_params.get("token", "")
    user_id = int(websocket.query_params.get("user_id", 0))
    username = websocket.query_params.get("username", f"User{user_id}")
    
    if not user_id:
        await websocket.close(code=4001, reason="未授权")
        return
    
    # 加入会话
    joined = await collaboration.join_session(
        session_id, user_id, username, websocket, ColRole.EDITOR
    )
    
    if not joined:
        await websocket.close(code=4004, reason="会话不存在")
        return
    
    # 同步状态
    await collaboration.sync_state(session_id, user_id)
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            msg_type = message.get("type")
            
            if msg_type == "message":
                await collaboration.send_message(
                    session_id, user_id, 
                    message.get("content", ""),
                    message.get("msg_type", "chat")
                )
            elif msg_type == "typing":
                await collaboration.update_typing(
                    session_id, user_id, 
                    message.get("is_typing", False)
                )
            elif msg_type == "cursor":
                await collaboration.update_cursor(
                    session_id, user_id,
                    message.get("position", 0)
                )
            elif msg_type == "sync":
                await collaboration.sync_state(session_id, user_id)
    
    except WebSocketDisconnect:
        await collaboration.leave_session(session_id, user_id)
    except Exception as e:
        await collaboration.leave_session(session_id, user_id)
        raise

# ========== 插件系统 API ==========
@router.get("/api/plugins")
async def list_plugins(user=None):
    """获取插件列表"""
    import sqlite3
    with sqlite3.connect("data.db") as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM plugins WHERE is_enabled = 1").fetchall()
        return [dict(r) for r in rows]

@router.post("/api/plugins/{plugin_id}/enable")
async def enable_plugin(plugin_id: str, user=None):
    """启用插件"""
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    
    import sqlite3
    with sqlite3.connect("data.db") as conn:
        conn.execute("""
            INSERT OR REPLACE INTO user_plugins (user_id, plugin_id, enabled)
            VALUES (?, ?, 1)
        """, (user["id"], plugin_id))
    
    return {"success": True}

@router.post("/api/plugins/{plugin_id}/disable")
async def disable_plugin(plugin_id: str, user=None):
    """禁用插件"""
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    
    import sqlite3
    with sqlite3.connect("data.db") as conn:
        conn.execute("""
            UPDATE user_plugins SET enabled = 0
            WHERE user_id = ? AND plugin_id = ?
        """, (user["id"], plugin_id))
    
    return {"success": True}

# ========== 主题商店 API ==========
THEMES = [
    {"id": "default", "name": "默认", "colors": {"primary": "#667eea", "bg": "#ffffff"}},
    {"id": "dark", "name": "暗黑", "colors": {"primary": "#667eea", "bg": "#1a1a2e"}},
    {"id": "ocean", "name": "海洋", "colors": {"primary": "#0077b6", "bg": "#caf0f8"}},
    {"id": "forest", "name": "森林", "colors": {"primary": "#2d6a4f", "bg": "#d8f3dc"}},
    {"id": "sunset", "name": "日落", "colors": {"primary": "#e85d04", "bg": "#fff3e0"}},
    {"id": "purple", "name": "紫罗兰", "colors": {"primary": "#7b2cbf", "bg": "#f3e5f5"}},
]

@router.get("/api/themes")
async def list_themes():
    """获取主题列表"""
    return THEMES

@router.post("/api/themes/apply")
async def apply_theme(data: dict, user=None):
    """应用主题"""
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    
    theme_id = data.get("theme_id", "default")
    
    import sqlite3
    with sqlite3.connect("data.db") as conn:
        # 更新用户设置
        settings = conn.execute(
            "SELECT data FROM settings WHERE user_id = ?", (user["id"],)
        ).fetchone()
        
        if settings:
            current = json.loads(settings[0])
        else:
            current = {}
        
        current["theme"] = theme_id
        
        conn.execute("""
            INSERT OR REPLACE INTO settings (user_id, data)
            VALUES (?, ?)
        """, (user["id"], json.dumps(current)))
    
    return {"success": True, "theme": theme_id}
