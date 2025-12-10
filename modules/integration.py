"""
模块集成入口
将所有模块整合到 FastAPI 应用
"""
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import time

# 导入所有模块
from .rbac import rbac, require_permission
from .billing import billing, payment_gateway
from .rag import create_rag_engine
from .collaboration import collaboration
from .security import waf, ai_detector, key_manager, auditor
from .enterprise import tenant_manager, compliance, ldap_auth
from .api_routes import router as api_router

def setup_modules(app: FastAPI):
    """设置所有模块"""
    
    # 注册 API 路由
    app.include_router(api_router)
    
    # 添加安全中间件
    @app.middleware("http")
    async def security_middleware(request: Request, call_next):
        start_time = time.time()
        client_ip = request.client.host if request.client else "unknown"
        
        # WAF 检查
        if waf.is_ip_blocked(client_ip):
            auditor.log_event("blocked_request", "high", ip=client_ip)
            raise HTTPException(status_code=403, detail="IP 已被封禁")
        
        # 处理请求
        response = await call_next(request)
        
        # 记录请求
        duration = time.time() - start_time
        auditor.log_event(
            "http_request", "info",
            ip=client_ip,
            resource=str(request.url.path),
            action=request.method,
            details={"duration_ms": int(duration * 1000), "status": response.status_code}
        )
        
        return response
    
    # 多租户中间件
    @app.middleware("http")
    async def tenant_middleware(request: Request, call_next):
        # 从域名或 header 获取租户
        host = request.headers.get("host", "")
        tenant = tenant_manager.get_tenant_by_domain(host.split(":")[0])
        
        if tenant:
            request.state.tenant = tenant
            request.state.tenant_id = tenant.id
        else:
            request.state.tenant = None
            request.state.tenant_id = None
        
        return await call_next(request)
    
    print("✅ 所有模块已加载")
    return app
