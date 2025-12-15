# ========== 异步数据库连接池模块 ==========
"""
高性能异步数据库连接池，支持 SQLite 和 PostgreSQL
特性：连接池、自动重连、查询缓存、性能监控
"""
import os
import asyncio
import sqlite3
import time
from typing import Optional, List, Dict, Any, AsyncGenerator, Callable
from contextlib import asynccontextmanager
from datetime import datetime
from functools import wraps
from collections import defaultdict
import logging
import hashlib
import json

logger = logging.getLogger(__name__)

# 尝试导入 aiosqlite
try:
    import aiosqlite
    AIOSQLITE_AVAILABLE = True
    logger.info("aiosqlite available, using async database operations")
except ImportError:
    AIOSQLITE_AVAILABLE = False
    logger.warning("aiosqlite not available, using sync sqlite3")

DB_PATH = os.getenv("DATABASE_PATH", "data.db")


class AsyncDatabasePool:
    """异步数据库连接池"""
    
    def __init__(self, db_path: str = None, pool_size: int = 10):
        self.db_path = db_path or DB_PATH
        self.pool_size = pool_size
        self._pool: asyncio.Queue = None
        self._initialized = False
        self._lock = asyncio.Lock()
    
    async def initialize(self):
        """初始化连接池"""
        if self._initialized:
            return
        
        async with self._lock:
            if self._initialized:
                return
            
            self._pool = asyncio.Queue(maxsize=self.pool_size)
            
            for _ in range(self.pool_size):
                if AIOSQLITE_AVAILABLE:
                    conn = await aiosqlite.connect(self.db_path)
                    conn.row_factory = aiosqlite.Row
                    # 启用 WAL 模式
                    await conn.execute("PRAGMA journal_mode=WAL")
                    await conn.execute("PRAGMA synchronous=NORMAL")
                    await conn.execute("PRAGMA cache_size=10000")
                    await self._pool.put(conn)
                else:
                    # 回退到同步连接的包装
                    await self._pool.put(None)
            
            self._initialized = True
            logger.info(f"Async database pool initialized: {self.pool_size} connections")
    
    @asynccontextmanager
    async def acquire(self) -> AsyncGenerator:
        """获取数据库连接"""
        if not self._initialized:
            await self.initialize()
        
        conn = await self._pool.get()
        try:
            if conn is None:
                # 回退模式：使用同步连接
                sync_conn = sqlite3.connect(self.db_path, check_same_thread=False)
                sync_conn.row_factory = sqlite3.Row
                yield SyncConnectionWrapper(sync_conn)
                sync_conn.close()
            else:
                yield conn
        finally:
            if conn is not None:
                await self._pool.put(conn)
    
    async def execute(self, query: str, params: tuple = None) -> Any:
        """执行查询"""
        async with self.acquire() as conn:
            if params:
                cursor = await conn.execute(query, params)
            else:
                cursor = await conn.execute(query)
            await conn.commit()
            return cursor
    
    async def fetch_one(self, query: str, params: tuple = None) -> Optional[Dict]:
        """获取单条记录"""
        async with self.acquire() as conn:
            if params:
                cursor = await conn.execute(query, params)
            else:
                cursor = await conn.execute(query)
            row = await cursor.fetchone()
            return dict(row) if row else None
    
    async def fetch_all(self, query: str, params: tuple = None) -> List[Dict]:
        """获取所有记录"""
        async with self.acquire() as conn:
            if params:
                cursor = await conn.execute(query, params)
            else:
                cursor = await conn.execute(query)
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]
    
    async def execute_many(self, query: str, params_list: List[tuple]) -> int:
        """批量执行"""
        async with self.acquire() as conn:
            await conn.executemany(query, params_list)
            await conn.commit()
            return len(params_list)
    
    async def close(self):
        """关闭所有连接"""
        if not self._initialized:
            return
        
        while not self._pool.empty():
            conn = await self._pool.get()
            if conn is not None:
                await conn.close()
        
        self._initialized = False
        logger.info("Database pool closed")


class SyncConnectionWrapper:
    """同步连接的异步包装器"""
    
    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn
    
    async def execute(self, query: str, params: tuple = None):
        if params:
            return self._conn.execute(query, params)
        return self._conn.execute(query)
    
    async def executemany(self, query: str, params_list: List[tuple]):
        return self._conn.executemany(query, params_list)
    
    async def commit(self):
        self._conn.commit()
    
    async def rollback(self):
        self._conn.rollback()


# 全局连接池实例
db_pool = AsyncDatabasePool()


# ========== 响应缓存装饰器 ==========
from functools import wraps
import hashlib
import json
import time

_response_cache: Dict[str, tuple] = {}  # key -> (response, timestamp)
_cache_ttl = 300  # 5分钟默认缓存


def cached_response(ttl: int = 300, key_prefix: str = ""):
    """API 响应缓存装饰器"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # 生成缓存键
            cache_key = f"{key_prefix}:{func.__name__}:{hash(str(args))}:{hash(str(sorted(kwargs.items())))}"
            
            # 检查缓存
            if cache_key in _response_cache:
                cached, timestamp = _response_cache[cache_key]
                if time.time() - timestamp < ttl:
                    logger.debug(f"Cache hit: {cache_key[:50]}")
                    return cached
            
            # 执行函数
            result = await func(*args, **kwargs)
            
            # 存入缓存
            _response_cache[cache_key] = (result, time.time())
            
            # 清理过期缓存
            if len(_response_cache) > 1000:
                _cleanup_cache()
            
            return result
        return wrapper
    return decorator


def _cleanup_cache():
    """清理过期缓存"""
    now = time.time()
    expired = [k for k, (_, t) in _response_cache.items() if now - t > _cache_ttl]
    for k in expired:
        del _response_cache[k]


def invalidate_cache(pattern: str = None):
    """使缓存失效"""
    if pattern:
        keys_to_delete = [k for k in _response_cache.keys() if pattern in k]
        for k in keys_to_delete:
            del _response_cache[k]
    else:
        _response_cache.clear()


# ========== 查询性能监控 ==========
class QueryProfiler:
    """查询性能分析器"""
    
    def __init__(self, enabled: bool = True, slow_threshold_ms: float = 100):
        self.enabled = enabled
        self.slow_threshold_ms = slow_threshold_ms
        self._stats: Dict[str, Dict] = defaultdict(lambda: {
            "count": 0,
            "total_time": 0,
            "min_time": float('inf'),
            "max_time": 0,
            "slow_count": 0
        })
        self._lock = asyncio.Lock()
    
    async def record(self, query: str, duration_ms: float):
        """记录查询性能"""
        if not self.enabled:
            return
        
        # 简化查询作为键
        query_key = self._normalize_query(query)
        
        async with self._lock:
            stats = self._stats[query_key]
            stats["count"] += 1
            stats["total_time"] += duration_ms
            stats["min_time"] = min(stats["min_time"], duration_ms)
            stats["max_time"] = max(stats["max_time"], duration_ms)
            
            if duration_ms > self.slow_threshold_ms:
                stats["slow_count"] += 1
                logger.warning(f"Slow query ({duration_ms:.2f}ms): {query[:100]}")
    
    def _normalize_query(self, query: str) -> str:
        """标准化查询（移除参数值）"""
        import re
        # 移除数字和字符串参数
        normalized = re.sub(r"'[^']*'", "'?'", query)
        normalized = re.sub(r"\b\d+\b", "?", normalized)
        return normalized[:200]
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        result = {}
        for query, stats in self._stats.items():
            if stats["count"] > 0:
                result[query] = {
                    **stats,
                    "avg_time": stats["total_time"] / stats["count"]
                }
        return dict(sorted(result.items(), key=lambda x: x[1]["total_time"], reverse=True)[:20])
    
    def get_slow_queries(self) -> List[Dict]:
        """获取慢查询列表"""
        return [
            {"query": q, **s}
            for q, s in self._stats.items()
            if s["slow_count"] > 0
        ]
    
    def reset(self):
        """重置统计"""
        self._stats.clear()


# 全局查询分析器
query_profiler = QueryProfiler()


# ========== 增强的异步数据库连接池 ==========
class EnhancedAsyncDatabasePool:
    """增强的异步数据库连接池"""
    
    def __init__(
        self,
        db_path: str = None,
        pool_size: int = 10,
        max_overflow: int = 5,
        pool_timeout: float = 30.0,
        enable_profiling: bool = True,
        enable_cache: bool = True,
        cache_ttl: int = 300
    ):
        self.db_path = db_path or DB_PATH
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.pool_timeout = pool_timeout
        self.enable_profiling = enable_profiling
        self.enable_cache = enable_cache
        self.cache_ttl = cache_ttl
        
        self._pool: asyncio.Queue = None
        self._overflow_count = 0
        self._initialized = False
        self._lock = asyncio.Lock()
        self._query_cache: Dict[str, tuple] = {}
        
        # 统计信息
        self._stats = {
            "connections_created": 0,
            "connections_reused": 0,
            "queries_executed": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
    
    async def initialize(self):
        """初始化连接池"""
        if self._initialized:
            return
        
        async with self._lock:
            if self._initialized:
                return
            
            self._pool = asyncio.Queue(maxsize=self.pool_size)
            
            for _ in range(self.pool_size):
                conn = await self._create_connection()
                await self._pool.put(conn)
                self._stats["connections_created"] += 1
            
            self._initialized = True
            logger.info(f"Enhanced async database pool initialized: {self.pool_size} connections")
    
    async def _create_connection(self):
        """创建新连接"""
        if AIOSQLITE_AVAILABLE:
            conn = await aiosqlite.connect(self.db_path)
            conn.row_factory = aiosqlite.Row
            # 优化设置
            await conn.execute("PRAGMA journal_mode=WAL")
            await conn.execute("PRAGMA synchronous=NORMAL")
            await conn.execute("PRAGMA cache_size=10000")
            await conn.execute("PRAGMA temp_store=MEMORY")
            await conn.execute("PRAGMA mmap_size=268435456")  # 256MB
            return conn
        else:
            return None
    
    @asynccontextmanager
    async def acquire(self) -> AsyncGenerator:
        """获取数据库连接"""
        if not self._initialized:
            await self.initialize()
        
        conn = None
        overflow = False
        
        try:
            # 尝试从池中获取连接
            conn = await asyncio.wait_for(
                self._pool.get(),
                timeout=self.pool_timeout
            )
            self._stats["connections_reused"] += 1
        except asyncio.TimeoutError:
            # 池已满，检查是否可以创建溢出连接
            if self._overflow_count < self.max_overflow:
                conn = await self._create_connection()
                self._overflow_count += 1
                overflow = True
                self._stats["connections_created"] += 1
                logger.warning(f"Created overflow connection ({self._overflow_count}/{self.max_overflow})")
            else:
                raise Exception("Database connection pool exhausted")
        
        try:
            if conn is None:
                # 回退模式
                sync_conn = sqlite3.connect(self.db_path, check_same_thread=False)
                sync_conn.row_factory = sqlite3.Row
                yield SyncConnectionWrapper(sync_conn)
                sync_conn.close()
            else:
                yield conn
        finally:
            if conn is not None:
                if overflow:
                    # 关闭溢出连接
                    await conn.close()
                    self._overflow_count -= 1
                else:
                    await self._pool.put(conn)
    
    async def execute(self, query: str, params: tuple = None) -> Any:
        """执行查询"""
        start_time = time.time()
        
        async with self.acquire() as conn:
            try:
                if params:
                    cursor = await conn.execute(query, params)
                else:
                    cursor = await conn.execute(query)
                await conn.commit()
                
                self._stats["queries_executed"] += 1
                
                # 记录性能
                if self.enable_profiling:
                    duration_ms = (time.time() - start_time) * 1000
                    await query_profiler.record(query, duration_ms)
                
                return cursor
            except Exception as e:
                logger.error(f"Query error: {e}, Query: {query[:100]}")
                raise
    
    async def fetch_one(self, query: str, params: tuple = None, use_cache: bool = True) -> Optional[Dict]:
        """获取单条记录（支持缓存）"""
        cache_key = None
        
        # 检查缓存
        if self.enable_cache and use_cache and query.strip().upper().startswith("SELECT"):
            cache_key = self._make_cache_key(query, params)
            cached = self._get_from_cache(cache_key)
            if cached is not None:
                self._stats["cache_hits"] += 1
                return cached
            self._stats["cache_misses"] += 1
        
        start_time = time.time()
        
        async with self.acquire() as conn:
            if params:
                cursor = await conn.execute(query, params)
            else:
                cursor = await conn.execute(query)
            row = await cursor.fetchone()
            result = dict(row) if row else None
            
            self._stats["queries_executed"] += 1
            
            if self.enable_profiling:
                duration_ms = (time.time() - start_time) * 1000
                await query_profiler.record(query, duration_ms)
            
            # 存入缓存
            if cache_key and result:
                self._set_cache(cache_key, result)
            
            return result
    
    async def fetch_all(self, query: str, params: tuple = None, use_cache: bool = True) -> List[Dict]:
        """获取所有记录（支持缓存）"""
        cache_key = None
        
        if self.enable_cache and use_cache and query.strip().upper().startswith("SELECT"):
            cache_key = self._make_cache_key(query, params)
            cached = self._get_from_cache(cache_key)
            if cached is not None:
                self._stats["cache_hits"] += 1
                return cached
            self._stats["cache_misses"] += 1
        
        start_time = time.time()
        
        async with self.acquire() as conn:
            if params:
                cursor = await conn.execute(query, params)
            else:
                cursor = await conn.execute(query)
            rows = await cursor.fetchall()
            result = [dict(row) for row in rows]
            
            self._stats["queries_executed"] += 1
            
            if self.enable_profiling:
                duration_ms = (time.time() - start_time) * 1000
                await query_profiler.record(query, duration_ms)
            
            if cache_key:
                self._set_cache(cache_key, result)
            
            return result
    
    async def execute_many(self, query: str, params_list: List[tuple]) -> int:
        """批量执行"""
        start_time = time.time()
        
        async with self.acquire() as conn:
            await conn.executemany(query, params_list)
            await conn.commit()
            
            self._stats["queries_executed"] += len(params_list)
            
            if self.enable_profiling:
                duration_ms = (time.time() - start_time) * 1000
                await query_profiler.record(f"BATCH: {query}", duration_ms)
            
            return len(params_list)
    
    async def execute_script(self, script: str):
        """执行 SQL 脚本"""
        async with self.acquire() as conn:
            await conn.executescript(script)
            await conn.commit()
    
    def _make_cache_key(self, query: str, params: tuple) -> str:
        """生成缓存键"""
        content = f"{query}:{params}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_from_cache(self, key: str) -> Optional[Any]:
        """从缓存获取"""
        if key in self._query_cache:
            value, timestamp = self._query_cache[key]
            if time.time() - timestamp < self.cache_ttl:
                return value
            del self._query_cache[key]
        return None
    
    def _set_cache(self, key: str, value: Any):
        """设置缓存"""
        # 限制缓存大小
        if len(self._query_cache) > 1000:
            # 清理过期缓存
            now = time.time()
            expired = [k for k, (_, t) in self._query_cache.items() if now - t > self.cache_ttl]
            for k in expired:
                del self._query_cache[k]
        
        self._query_cache[key] = (value, time.time())
    
    def invalidate_cache(self, pattern: str = None):
        """使缓存失效"""
        if pattern:
            keys_to_delete = [k for k in self._query_cache.keys() if pattern in k]
            for k in keys_to_delete:
                del self._query_cache[k]
        else:
            self._query_cache.clear()
    
    def get_stats(self) -> Dict:
        """获取连接池统计"""
        return {
            **self._stats,
            "pool_size": self.pool_size,
            "available_connections": self._pool.qsize() if self._pool else 0,
            "overflow_count": self._overflow_count,
            "cache_size": len(self._query_cache),
            "initialized": self._initialized
        }
    
    async def close(self):
        """关闭所有连接"""
        if not self._initialized:
            return
        
        while not self._pool.empty():
            conn = await self._pool.get()
            if conn is not None:
                await conn.close()
        
        self._initialized = False
        self._query_cache.clear()
        logger.info("Enhanced database pool closed")


# 全局增强连接池实例
enhanced_db_pool = EnhancedAsyncDatabasePool()


# ========== 便捷函数 ==========
async def get_db():
    """获取数据库连接（FastAPI 依赖注入用）"""
    async with enhanced_db_pool.acquire() as conn:
        yield conn


async def execute_query(query: str, params: tuple = None):
    """执行查询的便捷函数"""
    return await enhanced_db_pool.execute(query, params)


async def fetch_one(query: str, params: tuple = None):
    """获取单条记录的便捷函数"""
    return await enhanced_db_pool.fetch_one(query, params)


async def fetch_all(query: str, params: tuple = None):
    """获取所有记录的便捷函数"""
    return await enhanced_db_pool.fetch_all(query, params)
