# ========== 通用装饰器模块 ==========
"""
API 装饰器：权限检查、日志记录、性能追踪、缓存
"""
from functools import wraps
from typing import Callable, List, Optional, Any
from fastapi import HTTPException, Request
import time
import logging
import asyncio

logger = logging.getLogger(__name__)


def require_auth(func: Callable) -> Callable:
    """要求用户登录的装饰器"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        user = kwargs.get("user")
        if not user:
            raise HTTPException(status_code=401, detail="请先登录")
        return await func(*args, **kwargs)
    return wrapper


def require_permissions(*permissions: str):
    """要求特定权限的装饰器"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            user = kwargs.get("user")
            if not user:
                raise HTTPException(status_code=401, detail="请先登录")
            
            # 检查权限
            from .rbac import rbac
            user_perms = rbac.get_user_permissions(user["id"])
            
            if "*" not in user_perms:
                for perm in permissions:
                    if perm not in user_perms:
                        raise HTTPException(
                            status_code=403, 
                            detail=f"需要权限: {perm}"
                        )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator


def require_plan(*plans: str):
    """要求特定套餐的装饰器"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            user = kwargs.get("user")
            if not user:
                raise HTTPException(status_code=401, detail="请先登录")
            
            user_plan = user.get("plan", "free")
            if user_plan not in plans and "unlimited" not in plans:
                raise HTTPException(
                    status_code=403,
                    detail=f"此功能需要 {', '.join(plans)} 套餐"
                )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator


def rate_limit(requests_per_minute: int = 60, key_func: Callable = None):
    """速率限制装饰器"""
    _requests = {}  # key -> [timestamps]
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # 获取限流键
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                user = kwargs.get("user")
                key = f"user_{user['id']}" if user else "anonymous"
            
            now = time.time()
            minute_ago = now - 60
            
            # 清理过期记录
            if key not in _requests:
                _requests[key] = []
            _requests[key] = [t for t in _requests[key] if t > minute_ago]
            
            # 检查限制
            if len(_requests[key]) >= requests_per_minute:
                raise HTTPException(
                    status_code=429,
                    detail="请求过于频繁，请稍后再试"
                )
            
            _requests[key].append(now)
            return await func(*args, **kwargs)
        return wrapper
    return decorator


def log_request(log_response: bool = False):
    """请求日志装饰器"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            user = kwargs.get("user")
            user_id = user["id"] if user else "anonymous"
            
            try:
                result = await func(*args, **kwargs)
                duration = (time.time() - start_time) * 1000
                
                log_msg = f"[{func.__name__}] user={user_id} duration={duration:.2f}ms"
                if log_response and result:
                    log_msg += f" response_size={len(str(result))}"
                
                logger.info(log_msg)
                return result
                
            except Exception as e:
                duration = (time.time() - start_time) * 1000
                logger.error(f"[{func.__name__}] user={user_id} duration={duration:.2f}ms error={str(e)}")
                raise
        return wrapper
    return decorator


def retry(max_retries: int = 3, delay: float = 1.0, exceptions: tuple = (Exception,)):
    """重试装饰器"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        wait_time = delay * (2 ** attempt)  # 指数退避
                        logger.warning(f"Retry {attempt + 1}/{max_retries} for {func.__name__}: {e}")
                        await asyncio.sleep(wait_time)
            
            raise last_exception
        return wrapper
    return decorator


def timeout(seconds: float = 30.0):
    """超时装饰器"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=seconds
                )
            except asyncio.TimeoutError:
                raise HTTPException(
                    status_code=504,
                    detail=f"请求超时 ({seconds}s)"
                )
        return wrapper
    return decorator


def validate_json_body(*required_fields: str):
    """验证 JSON 请求体装饰器"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            data = kwargs.get("data", {})
            
            missing = [f for f in required_fields if f not in data or data[f] is None]
            if missing:
                raise HTTPException(
                    status_code=400,
                    detail=f"缺少必填字段: {', '.join(missing)}"
                )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator


def deprecated(message: str = "此 API 已废弃"):
    """废弃 API 装饰器"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            logger.warning(f"Deprecated API called: {func.__name__}")
            result = await func(*args, **kwargs)
            
            # 如果返回的是字典，添加废弃警告
            if isinstance(result, dict):
                result["_deprecated"] = message
            
            return result
        return wrapper
    return decorator
