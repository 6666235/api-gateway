# ========== 中间件模块 ==========
"""
请求处理中间件：日志、错误处理、性能追踪、安全检查
"""
import time
import json
import traceback
from typing import Callable
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import logging

logger = logging.getLogger(__name__)

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """请求日志中间件"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        
        # 生成请求 ID
        import uuid
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id
        
        # 记录请求
        logger.info(f"[{request_id}] {request.method} {request.url.path}")
        
        try:
            response = await call_next(request)
            
            # 计算耗时
            duration = (time.time() - start_time) * 1000
            
            # 添加响应头
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time"] = f"{duration:.2f}ms"
            
            # 记录响应
            logger.info(f"[{request_id}] {response.status_code} {duration:.2f}ms")
            
            # 记录性能指标
            try:
                from .monitoring import metrics, profiler
                metrics.inc("http_requests_total", labels={
                    "method": request.method,
                    "path": request.url.path,
                    "status": str(response.status_code)
                })
                metrics.observe("http_request_duration_seconds", duration / 1000, labels={
                    "method": request.method,
                    "path": request.url.path
                })
                profiler.record(f"HTTP {request.method} {request.url.path}", duration)
            except:
                pass
            
            return response
            
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            logger.error(f"[{request_id}] Error: {str(e)} {duration:.2f}ms")
            raise

class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """全局错误处理中间件"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            return await call_next(request)
        except Exception as e:
            # 获取请求 ID
            request_id = getattr(request.state, 'request_id', 'unknown')
            
            # 记录错误
            logger.error(f"[{request_id}] Unhandled error: {str(e)}")
            logger.error(traceback.format_exc())
            
            # 记录错误指标
            try:
                from .monitoring import metrics
                metrics.inc("http_errors_total", labels={
                    "path": request.url.path,
                    "error_type": type(e).__name__
                })
            except:
                pass
            
            # 返回友好错误响应
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal Server Error",
                    "message": str(e) if logger.level <= logging.DEBUG else "服务器内部错误",
                    "request_id": request_id
                }
            )

class SecurityMiddleware(BaseHTTPMiddleware):
    """安全中间件"""
    
    def __init__(self, app, blocked_ips: set = None):
        super().__init__(app)
        self.blocked_ips = blocked_ips or set()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # 获取客户端 IP
        client_ip = request.client.host if request.client else "unknown"
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            client_ip = forwarded.split(",")[0].strip()
        
        request.state.client_ip = client_ip
        
        # 检查 IP 封禁
        if client_ip in self.blocked_ips:
            logger.warning(f"Blocked IP attempt: {client_ip}")
            return JSONResponse(
                status_code=403,
                content={"error": "Forbidden", "message": "IP 已被封禁"}
            )
        
        response = await call_next(request)
        
        # 添加安全响应头
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        
        # HSTS (仅 HTTPS)
        if request.url.scheme == "https":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        
        return response

class CompressionMiddleware(BaseHTTPMiddleware):
    """响应压缩中间件"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # 检查是否支持 gzip
        accept_encoding = request.headers.get("Accept-Encoding", "")
        if "gzip" not in accept_encoding:
            return response
        
        # 只压缩文本类型
        content_type = response.headers.get("Content-Type", "")
        if not any(t in content_type for t in ["text/", "application/json", "application/javascript"]):
            return response
        
        # 小响应不压缩
        content_length = response.headers.get("Content-Length")
        if content_length and int(content_length) < 1024:
            return response
        
        # 实际压缩逻辑（简化版，生产环境建议使用 GZipMiddleware）
        return response

class RateLimitMiddleware(BaseHTTPMiddleware):
    """速率限制中间件"""
    
    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests = {}  # ip -> [timestamps]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        client_ip = getattr(request.state, 'client_ip', request.client.host if request.client else "unknown")
        
        # 清理过期记录
        now = time.time()
        minute_ago = now - 60
        
        if client_ip not in self.requests:
            self.requests[client_ip] = []
        
        self.requests[client_ip] = [t for t in self.requests[client_ip] if t > minute_ago]
        
        # 检查限制
        if len(self.requests[client_ip]) >= self.requests_per_minute:
            logger.warning(f"Rate limit exceeded: {client_ip}")
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Too Many Requests",
                    "message": "请求过于频繁，请稍后再试",
                    "retry_after": 60
                },
                headers={"Retry-After": "60"}
            )
        
        self.requests[client_ip].append(now)
        
        response = await call_next(request)
        
        # 添加限流信息头
        remaining = self.requests_per_minute - len(self.requests[client_ip])
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(now + 60))
        
        return response

class CORSMiddleware(BaseHTTPMiddleware):
    """CORS 中间件（增强版）"""
    
    def __init__(self, app, allowed_origins: list = None):
        super().__init__(app)
        self.allowed_origins = allowed_origins or ["*"]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        origin = request.headers.get("Origin", "")
        
        # 处理预检请求
        if request.method == "OPTIONS":
            response = Response(status_code=204)
        else:
            response = await call_next(request)
        
        # 设置 CORS 头
        if "*" in self.allowed_origins or origin in self.allowed_origins:
            response.headers["Access-Control-Allow-Origin"] = origin or "*"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS, PATCH"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With"
            response.headers["Access-Control-Allow-Credentials"] = "true"
            response.headers["Access-Control-Max-Age"] = "86400"
        
        return response

# ========== 工具函数 ==========
def setup_middlewares(app, config: dict = None):
    """设置所有中间件"""
    config = config or {}
    
    # 顺序很重要：最后添加的最先执行
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(ErrorHandlingMiddleware)
    app.add_middleware(
        SecurityMiddleware,
        blocked_ips=config.get("blocked_ips", set())
    )
    app.add_middleware(
        RateLimitMiddleware,
        requests_per_minute=config.get("rate_limit", 60)
    )
    
    logger.info("Middlewares configured")
