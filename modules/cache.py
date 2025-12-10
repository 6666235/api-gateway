# ========== Redis 缓存模块 ==========
"""
高性能缓存模块，支持 Redis 和内存缓存
"""
import os
import json
import hashlib
import asyncio
from typing import Optional, Any, Dict, Callable
from datetime import datetime, timedelta
from functools import wraps
import logging

logger = logging.getLogger(__name__)

# 尝试导入 Redis
try:
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis not available, using memory cache")

class CacheBackend:
    """缓存后端基类"""
    async def get(self, key: str) -> Optional[Any]:
        raise NotImplementedError
    
    async def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        raise NotImplementedError
    
    async def delete(self, key: str) -> bool:
        raise NotImplementedError
    
    async def exists(self, key: str) -> bool:
        raise NotImplementedError
    
    async def clear_pattern(self, pattern: str) -> int:
        raise NotImplementedError

class MemoryCache(CacheBackend):
    """内存缓存实现"""
    def __init__(self, max_size: int = 1000):
        self._cache: Dict[str, tuple] = {}  # key -> (value, expire_time)
        self._max_size = max_size
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        async with self._lock:
            if key in self._cache:
                value, expire_time = self._cache[key]
                if expire_time > datetime.now():
                    return value
                del self._cache[key]
            return None
    
    async def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        async with self._lock:
            if len(self._cache) >= self._max_size:
                # LRU: 删除最早过期的
                now = datetime.now()
                expired = [k for k, (_, exp) in self._cache.items() if exp <= now]
                for k in expired[:len(expired)//2 + 1]:
                    del self._cache[k]
            
            expire_time = datetime.now() + timedelta(seconds=ttl)
            self._cache[key] = (value, expire_time)
            return True
    
    async def delete(self, key: str) -> bool:
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    async def exists(self, key: str) -> bool:
        return await self.get(key) is not None
    
    async def clear_pattern(self, pattern: str) -> int:
        async with self._lock:
            import fnmatch
            keys_to_delete = [k for k in self._cache.keys() if fnmatch.fnmatch(k, pattern)]
            for k in keys_to_delete:
                del self._cache[k]
            return len(keys_to_delete)
    
    async def get_stats(self) -> dict:
        async with self._lock:
            now = datetime.now()
            valid = sum(1 for _, (_, exp) in self._cache.items() if exp > now)
            return {
                "type": "memory",
                "total_keys": len(self._cache),
                "valid_keys": valid,
                "max_size": self._max_size
            }

class RedisCache(CacheBackend):
    """Redis 缓存实现"""
    def __init__(self, url: str = None):
        self._url = url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self._client: Optional[aioredis.Redis] = None
        self._prefix = "aihub:"
    
    async def _get_client(self) -> aioredis.Redis:
        if self._client is None:
            self._client = await aioredis.from_url(
                self._url,
                encoding="utf-8",
                decode_responses=True
            )
        return self._client
    
    async def get(self, key: str) -> Optional[Any]:
        try:
            client = await self._get_client()
            value = await client.get(f"{self._prefix}{key}")
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        try:
            client = await self._get_client()
            await client.setex(
                f"{self._prefix}{key}",
                ttl,
                json.dumps(value, ensure_ascii=False, default=str)
            )
            return True
        except Exception as e:
            logger.error(f"Redis set error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        try:
            client = await self._get_client()
            await client.delete(f"{self._prefix}{key}")
            return True
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        try:
            client = await self._get_client()
            return await client.exists(f"{self._prefix}{key}") > 0
        except Exception as e:
            logger.error(f"Redis exists error: {e}")
            return False
    
    async def clear_pattern(self, pattern: str) -> int:
        try:
            client = await self._get_client()
            keys = []
            async for key in client.scan_iter(f"{self._prefix}{pattern}"):
                keys.append(key)
            if keys:
                await client.delete(*keys)
            return len(keys)
        except Exception as e:
            logger.error(f"Redis clear_pattern error: {e}")
            return 0
    
    async def get_stats(self) -> dict:
        try:
            client = await self._get_client()
            info = await client.info()
            return {
                "type": "redis",
                "connected_clients": info.get("connected_clients", 0),
                "used_memory": info.get("used_memory_human", "N/A"),
                "total_keys": info.get("db0", {}).get("keys", 0) if isinstance(info.get("db0"), dict) else 0,
                "uptime_days": info.get("uptime_in_days", 0)
            }
        except Exception as e:
            return {"type": "redis", "error": str(e)}
    
    async def close(self):
        if self._client:
            await self._client.close()
            self._client = None

# 全局缓存实例
_cache: Optional[CacheBackend] = None

async def get_cache() -> CacheBackend:
    """获取缓存实例"""
    global _cache
    if _cache is None:
        redis_url = os.getenv("REDIS_URL")
        if redis_url and REDIS_AVAILABLE:
            _cache = RedisCache(redis_url)
            logger.info(f"Using Redis cache: {redis_url}")
        else:
            _cache = MemoryCache(max_size=2000)
            logger.info("Using memory cache")
    return _cache

def cache_key(*args, **kwargs) -> str:
    """生成缓存键"""
    content = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True, default=str)
    return hashlib.md5(content.encode()).hexdigest()

def cached(ttl: int = 3600, prefix: str = ""):
    """缓存装饰器"""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            cache = await get_cache()
            key = f"{prefix}:{func.__name__}:{cache_key(*args[1:], **kwargs)}"  # 跳过 self
            
            # 尝试从缓存获取
            result = await cache.get(key)
            if result is not None:
                logger.debug(f"Cache hit: {key[:50]}")
                return result
            
            # 执行函数
            result = await func(*args, **kwargs)
            
            # 存入缓存
            if result is not None:
                await cache.set(key, result, ttl)
                logger.debug(f"Cache set: {key[:50]}")
            
            return result
        return wrapper
    return decorator

# ========== 分布式锁 ==========
class DistributedLock:
    """分布式锁（基于 Redis）"""
    def __init__(self, name: str, timeout: int = 10):
        self.name = f"lock:{name}"
        self.timeout = timeout
        self._token = None
    
    async def acquire(self) -> bool:
        cache = await get_cache()
        if isinstance(cache, RedisCache):
            import uuid
            self._token = str(uuid.uuid4())
            client = await cache._get_client()
            return await client.set(
                f"{cache._prefix}{self.name}",
                self._token,
                nx=True,
                ex=self.timeout
            )
        return True  # 内存模式直接返回 True
    
    async def release(self) -> bool:
        cache = await get_cache()
        if isinstance(cache, RedisCache) and self._token:
            client = await cache._get_client()
            # Lua 脚本确保原子性
            script = """
            if redis.call("get", KEYS[1]) == ARGV[1] then
                return redis.call("del", KEYS[1])
            else
                return 0
            end
            """
            return await client.eval(script, 1, f"{cache._prefix}{self.name}", self._token)
        return True
    
    async def __aenter__(self):
        if not await self.acquire():
            raise Exception(f"Failed to acquire lock: {self.name}")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.release()
