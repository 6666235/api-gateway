# AI Hub 模块化架构
# 包含：RAG向量检索、RBAC权限、计费系统、实时协作、缓存、监控、任务队列等

__version__ = "2.1.0"

# 核心模块
from .cache import get_cache, cached, DistributedLock
from .monitoring import metrics, tracer, profiler, health_checker, alert_manager
from .queue import task_queue, scheduler
from .ai_tools import tool_registry, function_handler, smart_assistant

__all__ = [
    # 缓存
    "get_cache",
    "cached", 
    "DistributedLock",
    # 监控
    "metrics",
    "tracer",
    "profiler",
    "health_checker",
    "alert_manager",
    # 任务队列
    "task_queue",
    "scheduler",
    # AI 工具
    "tool_registry",
    "function_handler",
    "smart_assistant",
]
