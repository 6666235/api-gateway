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

# ========== WAF 中间件 ==========
class WAFMiddleware(BaseHTTPMiddleware):
    """Web 应用防火墙中间件"""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        client_ip = getattr(request.state, "client_ip", "unknown")

        try:
            from .security import waf, auditor

            # 检查 IP 是否被封禁
            if waf.is_ip_blocked(client_ip):
                return JSONResponse(
                    status_code=403,
                    content={"error": "Forbidden", "message": "IP 已被临时封禁"},
                )

            # 检查请求内容
            if request.method in ["POST", "PUT", "PATCH"]:
                try:
                    body = await request.body()
                    if body:
                        content = body.decode("utf-8", errors="ignore")
                        result = waf.check(content, client_ip)

                        if result["blocked"]:
                            # 记录安全事件
                            auditor.log_event(
                                event_type="waf_blocked",
                                severity="warning",
                                ip=client_ip,
                                resource=request.url.path,
                                details={"violations": result["violations"]},
                            )

                            return JSONResponse(
                                status_code=403,
                                content={
                                    "error": "Forbidden",
                                    "message": "请求包含不安全内容",
                                },
                            )
                except:
                    pass

        except ImportError:
            pass

        return await call_next(request)


# ========== CSRF 中间件 ==========
class CSRFMiddleware(BaseHTTPMiddleware):
    """CSRF 防护中间件"""

    def __init__(self, app, exempt_paths: list = None):
        super().__init__(app)
        self.exempt_paths = exempt_paths or ["/api/auth/", "/health", "/docs", "/redoc"]

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # 跳过安全方法
        if request.method in ["GET", "HEAD", "OPTIONS"]:
            return await call_next(request)

        # 跳过豁免路径
        for path in self.exempt_paths:
            if request.url.path.startswith(path):
                return await call_next(request)

        # 验证 CSRF Token
        try:
            from .security import csrf_protection

            csrf_token = request.headers.get(
                "X-CSRF-Token"
            ) or request.headers.get("X-Csrf-Token")
            session_id = request.cookies.get("session_id", "")

            if not csrf_token or not csrf_protection.validate_token(
                csrf_token, session_id
            ):
                # 对于 API 请求，也检查 Authorization header
                if request.headers.get("Authorization"):
                    return await call_next(request)

                logger.warning(
                    f"CSRF validation failed: {request.url.path}"
                )
                return JSONResponse(
                    status_code=403,
                    content={"error": "Forbidden", "message": "CSRF 验证失败"},
                )
        except ImportError:
            pass

        return await call_next(request)


# ========== 请求签名验证中间件 ==========
class SignatureMiddleware(BaseHTTPMiddleware):
    """API 请求签名验证中间件"""

    def __init__(self, app, required_paths: list = None):
        super().__init__(app)
        self.required_paths = required_paths or []

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # 检查是否需要签名验证
        needs_signature = any(
            request.url.path.startswith(p) for p in self.required_paths
        )

        if not needs_signature:
            return await call_next(request)

        try:
            from .security import request_signer

            timestamp = request.headers.get("X-Timestamp")
            nonce = request.headers.get("X-Nonce")
            signature = request.headers.get("X-Signature")

            if not all([timestamp, nonce, signature]):
                return JSONResponse(
                    status_code=401,
                    content={"error": "Unauthorized", "message": "缺少签名信息"},
                )

            # 获取请求体
            body = None
            if request.method in ["POST", "PUT", "PATCH"]:
                body = (await request.body()).decode("utf-8", errors="ignore")

            # 验证签名
            is_valid, error = request_signer.verify_request(
                method=request.method,
                path=request.url.path,
                timestamp=timestamp,
                nonce=nonce,
                signature=signature,
                params=dict(request.query_params),
                body=body,
            )

            if not is_valid:
                return JSONResponse(
                    status_code=401,
                    content={"error": "Unauthorized", "message": error},
                )

        except ImportError:
            pass

        return await call_next(request)


# ========== 请求追踪中间件 ==========
class TracingMiddleware(BaseHTTPMiddleware):
    """分布式追踪中间件"""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            from .monitoring import tracer

            # 获取或生成 trace ID
            trace_id = request.headers.get("X-Trace-ID")
            parent_id = request.headers.get("X-Span-ID")

            span = tracer.start_span(
                f"{request.method} {request.url.path}",
                trace_id=trace_id,
                parent_id=parent_id,
            )

            span.set_tag("http.method", request.method)
            span.set_tag("http.url", str(request.url))
            span.set_tag(
                "http.client_ip",
                getattr(request.state, "client_ip", "unknown"),
            )

            try:
                response = await call_next(request)
                span.set_tag("http.status_code", str(response.status_code))

                # 添加追踪头到响应
                response.headers["X-Trace-ID"] = span.trace_id
                response.headers["X-Span-ID"] = span.span_id

                return response
            except Exception as e:
                span.set_error(e)
                raise
            finally:
                tracer.record_span(span)

        except ImportError:
            return await call_next(request)


# ========== 工具函数 ==========
def setup_middlewares(app, config: dict = None):
    """设置所有中间件"""
    config = config or {}

    # 顺序很重要：最后添加的最先执行
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(ErrorHandlingMiddleware)
    app.add_middleware(TracingMiddleware)
    app.add_middleware(SecurityMiddleware, blocked_ips=config.get("blocked_ips", set()))
    app.add_middleware(
        RateLimitMiddleware, requests_per_minute=config.get("rate_limit", 60)
    )

    # 可选中间件
    if config.get("enable_waf", True):
        app.add_middleware(WAFMiddleware)

    if config.get("enable_csrf", False):
        app.add_middleware(
            CSRFMiddleware, exempt_paths=config.get("csrf_exempt_paths", [])
        )

    if config.get("signature_required_paths"):
        app.add_middleware(
            SignatureMiddleware,
            required_paths=config.get("signature_required_paths", []),
        )

    logger.info("Middlewares configured")
