# AI Hub 模块化架构
# 包含：RAG向量检索、RBAC权限、计费系统、实时协作、缓存、监控、任务队列等

__version__ = "2.2.0"

# 延迟导入，避免循环依赖
def __getattr__(name):
    """延迟导入模块"""
    if name == "get_cache":
        from .cache import get_cache
        return get_cache
    elif name == "cached":
        from .cache import cached
        return cached
    elif name == "DistributedLock":
        from .cache import DistributedLock
        return DistributedLock
    elif name == "metrics":
        from .monitoring import metrics
        return metrics
    elif name == "tracer":
        from .monitoring import tracer
        return tracer
    elif name == "profiler":
        from .monitoring import profiler
        return profiler
    elif name == "health_checker":
        from .monitoring import health_checker
        return health_checker
    elif name == "alert_manager":
        from .monitoring import alert_manager
        return alert_manager
    elif name == "shutdown_manager":
        from .monitoring import shutdown_manager
        return shutdown_manager
    elif name == "task_queue":
        from .queue import task_queue
        return task_queue
    elif name == "scheduler":
        from .queue import scheduler
        return scheduler
    elif name == "enhanced_db_pool":
        from .db_pool import enhanced_db_pool
        return enhanced_db_pool
    elif name == "query_profiler":
        from .db_pool import query_profiler
        return query_profiler
    elif name == "waf":
        from .security import waf
        return waf
    elif name == "csrf_protection":
        from .security import csrf_protection
        return csrf_protection
    elif name == "request_signer":
        from .security import request_signer
        return request_signer
    elif name == "auditor":
        from .security import auditor
        return auditor
    elif name == "config":
        from .config import config
        return config
    elif name == "get_config":
        from .config import get_config
        return get_config
    elif name == "exporter":
        from .data_export import exporter
        return exporter
    elif name == "importer":
        from .data_export import importer
        return importer
    raise AttributeError(f"module 'modules' has no attribute '{name}'")
