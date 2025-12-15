# ========== 监控与指标模块 ==========
"""
Prometheus 指标、链路追踪、性能监控
"""
import os
import time
import asyncio
import psutil
from typing import Dict, Optional, Callable
from datetime import datetime, timedelta
from functools import wraps
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

# ========== Prometheus 指标 ==========
class MetricsCollector:
    """指标收集器"""
    def __init__(self):
        self._counters: Dict[str, int] = defaultdict(int)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, list] = defaultdict(list)
        self._labels: Dict[str, Dict] = defaultdict(dict)
        self._start_time = time.time()
    
    def inc(self, name: str, value: int = 1, labels: dict = None):
        """增加计数器"""
        key = self._make_key(name, labels)
        self._counters[key] += value
        if labels:
            self._labels[key] = labels
    
    def set(self, name: str, value: float, labels: dict = None):
        """设置仪表盘值"""
        key = self._make_key(name, labels)
        self._gauges[key] = value
        if labels:
            self._labels[key] = labels
    
    def observe(self, name: str, value: float, labels: dict = None):
        """记录直方图观测值"""
        key = self._make_key(name, labels)
        self._histograms[key].append(value)
        # 保留最近1000个值
        if len(self._histograms[key]) > 1000:
            self._histograms[key] = self._histograms[key][-1000:]
        if labels:
            self._labels[key] = labels
    
    def _make_key(self, name: str, labels: dict = None) -> str:
        if labels:
            label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
            return f"{name}{{{label_str}}}"
        return name
    
    def get_prometheus_metrics(self) -> str:
        """生成 Prometheus 格式的指标"""
        lines = []
        
        # 系统指标
        lines.append("# HELP aihub_uptime_seconds Uptime in seconds")
        lines.append("# TYPE aihub_uptime_seconds gauge")
        lines.append(f"aihub_uptime_seconds {time.time() - self._start_time:.2f}")
        
        # CPU 和内存
        lines.append("# HELP aihub_cpu_percent CPU usage percent")
        lines.append("# TYPE aihub_cpu_percent gauge")
        lines.append(f"aihub_cpu_percent {psutil.cpu_percent()}")
        
        lines.append("# HELP aihub_memory_percent Memory usage percent")
        lines.append("# TYPE aihub_memory_percent gauge")
        lines.append(f"aihub_memory_percent {psutil.virtual_memory().percent}")
        
        # 计数器
        for key, value in self._counters.items():
            name = key.split("{")[0] if "{" in key else key
            lines.append(f"# TYPE {name} counter")
            lines.append(f"{key} {value}")
        
        # 仪表盘
        for key, value in self._gauges.items():
            name = key.split("{")[0] if "{" in key else key
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{key} {value}")
        
        # 直方图摘要
        for key, values in self._histograms.items():
            if values:
                name = key.split("{")[0] if "{" in key else key
                lines.append(f"# TYPE {name} summary")
                sorted_vals = sorted(values)
                count = len(sorted_vals)
                total = sum(sorted_vals)
                lines.append(f"{key}_count {count}")
                lines.append(f"{key}_sum {total:.4f}")
                # 分位数
                for q in [0.5, 0.9, 0.99]:
                    idx = int(count * q)
                    lines.append(f'{key}{{quantile="{q}"}} {sorted_vals[min(idx, count-1)]:.4f}')
        
        return "\n".join(lines)
    
    def get_stats(self) -> dict:
        """获取统计摘要"""
        return {
            "uptime": time.time() - self._start_time,
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
            "histogram_counts": {k: len(v) for k, v in self._histograms.items()}
        }

# 全局指标收集器
metrics = MetricsCollector()

def track_request(endpoint: str):
    """请求追踪装饰器"""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            status = "success"
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                raise
            finally:
                duration = time.time() - start_time
                metrics.inc("aihub_requests_total", labels={"endpoint": endpoint, "status": status})
                metrics.observe("aihub_request_duration_seconds", duration, labels={"endpoint": endpoint})
        return wrapper
    return decorator

# ========== 链路追踪 ==========
class TraceSpan:
    """追踪 Span"""
    def __init__(self, name: str, trace_id: str = None, parent_id: str = None):
        import uuid
        self.name = name
        self.trace_id = trace_id or str(uuid.uuid4())
        self.span_id = str(uuid.uuid4())[:16]
        self.parent_id = parent_id
        self.start_time = time.time()
        self.end_time: Optional[float] = None
        self.tags: Dict[str, str] = {}
        self.logs: list = []
        self.status = "OK"
    
    def set_tag(self, key: str, value: str):
        self.tags[key] = str(value)
        return self
    
    def log(self, message: str, **kwargs):
        self.logs.append({
            "timestamp": time.time(),
            "message": message,
            **kwargs
        })
        return self
    
    def set_error(self, error: Exception):
        self.status = "ERROR"
        self.set_tag("error", str(error))
        self.set_tag("error.type", type(error).__name__)
        return self
    
    def finish(self):
        self.end_time = time.time()
        return self
    
    @property
    def duration_ms(self) -> float:
        end = self.end_time or time.time()
        return (end - self.start_time) * 1000
    
    def to_dict(self) -> dict:
        return {
            "traceId": self.trace_id,
            "spanId": self.span_id,
            "parentId": self.parent_id,
            "operationName": self.name,
            "startTime": int(self.start_time * 1000000),
            "duration": int(self.duration_ms * 1000),
            "tags": self.tags,
            "logs": self.logs,
            "status": self.status
        }

class Tracer:
    """链路追踪器"""
    def __init__(self, service_name: str = "ai-hub"):
        self.service_name = service_name
        self._spans: list = []
        self._max_spans = 1000
    
    def start_span(self, name: str, trace_id: str = None, parent_id: str = None) -> TraceSpan:
        span = TraceSpan(name, trace_id, parent_id)
        span.set_tag("service", self.service_name)
        return span
    
    def record_span(self, span: TraceSpan):
        span.finish()
        self._spans.append(span.to_dict())
        if len(self._spans) > self._max_spans:
            self._spans = self._spans[-self._max_spans:]
    
    def get_traces(self, limit: int = 100) -> list:
        return self._spans[-limit:]
    
    def clear(self):
        self._spans.clear()

tracer = Tracer()

def trace(name: str = None):
    """追踪装饰器"""
    def decorator(func: Callable):
        span_name = name or func.__name__
        @wraps(func)
        async def wrapper(*args, **kwargs):
            span = tracer.start_span(span_name)
            try:
                result = await func(*args, **kwargs)
                span.set_tag("status", "success")
                return result
            except Exception as e:
                span.set_error(e)
                raise
            finally:
                tracer.record_span(span)
        return wrapper
    return decorator

# ========== 性能分析 ==========
class PerformanceProfiler:
    """性能分析器"""
    def __init__(self):
        self._profiles: Dict[str, list] = defaultdict(list)
        self._slow_threshold_ms = 1000  # 慢请求阈值
    
    def record(self, name: str, duration_ms: float, metadata: dict = None):
        self._profiles[name].append({
            "duration_ms": duration_ms,
            "timestamp": time.time(),
            "metadata": metadata or {}
        })
        # 保留最近500条
        if len(self._profiles[name]) > 500:
            self._profiles[name] = self._profiles[name][-500:]
        
        # 记录慢请求
        if duration_ms > self._slow_threshold_ms:
            logger.warning(f"Slow operation: {name} took {duration_ms:.2f}ms")
    
    def get_stats(self, name: str = None) -> dict:
        if name:
            records = self._profiles.get(name, [])
            if not records:
                return {}
            durations = [r["duration_ms"] for r in records]
            return {
                "name": name,
                "count": len(durations),
                "avg_ms": sum(durations) / len(durations),
                "min_ms": min(durations),
                "max_ms": max(durations),
                "p50_ms": sorted(durations)[len(durations)//2],
                "p99_ms": sorted(durations)[int(len(durations)*0.99)] if len(durations) > 100 else max(durations)
            }
        
        return {k: self.get_stats(k) for k in self._profiles.keys()}
    
    def get_slow_operations(self, threshold_ms: float = None) -> list:
        threshold = threshold_ms or self._slow_threshold_ms
        slow = []
        for name, records in self._profiles.items():
            for r in records:
                if r["duration_ms"] > threshold:
                    slow.append({"name": name, **r})
        return sorted(slow, key=lambda x: x["duration_ms"], reverse=True)[:50]

profiler = PerformanceProfiler()

# ========== 健康检查 ==========
class HealthChecker:
    """增强的健康检查器"""
    def __init__(self):
        self._checks: Dict[str, Callable] = {}
        self._last_results: Dict[str, dict] = {}
        self._check_history: list = []
        self._max_history = 100
    
    def register(self, name: str, check_func: Callable, critical: bool = False, timeout: float = 5.0):
        """注册健康检查"""
        self._checks[name] = {
            "func": check_func,
            "critical": critical,
            "timeout": timeout
        }
    
    async def check_all(self) -> dict:
        results = {}
        overall_healthy = True
        degraded = False
        
        for name, check_config in self._checks.items():
            check_func = check_config["func"]
            is_critical = check_config["critical"]
            timeout = check_config["timeout"]
            
            try:
                start = time.time()
                
                # 带超时的检查
                if asyncio.iscoroutinefunction(check_func):
                    result = await asyncio.wait_for(check_func(), timeout=timeout)
                else:
                    result = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(None, check_func),
                        timeout=timeout
                    )
                
                duration = (time.time() - start) * 1000
                
                if isinstance(result, dict):
                    results[name] = {
                        "status": "healthy" if result.get("healthy", True) else "unhealthy",
                        "latency_ms": round(duration, 2),
                        **result
                    }
                else:
                    results[name] = {
                        "status": "healthy" if result else "unhealthy",
                        "latency_ms": round(duration, 2)
                    }
                
                if not (result if not isinstance(result, dict) else result.get("healthy", True)):
                    if is_critical:
                        overall_healthy = False
                    else:
                        degraded = True
                        
            except asyncio.TimeoutError:
                results[name] = {
                    "status": "unhealthy",
                    "error": f"Check timed out after {timeout}s"
                }
                if is_critical:
                    overall_healthy = False
                else:
                    degraded = True
            except Exception as e:
                results[name] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                if is_critical:
                    overall_healthy = False
                else:
                    degraded = True
        
        # 确定整体状态
        if not overall_healthy:
            status = "unhealthy"
        elif degraded:
            status = "degraded"
        else:
            status = "healthy"
        
        health_result = {
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "version": os.getenv("APP_VERSION", "1.0.0"),
            "uptime_seconds": time.time() - metrics._start_time,
            "checks": results
        }
        
        # 保存历史
        self._last_results = results
        self._check_history.append({
            "timestamp": time.time(),
            "status": status
        })
        if len(self._check_history) > self._max_history:
            self._check_history = self._check_history[-self._max_history:]
        
        return health_result
    
    async def check_single(self, name: str) -> dict:
        """检查单个组件"""
        if name not in self._checks:
            return {"status": "unknown", "error": "Check not found"}
        
        check_config = self._checks[name]
        check_func = check_config["func"]
        
        try:
            start = time.time()
            if asyncio.iscoroutinefunction(check_func):
                result = await check_func()
            else:
                result = check_func()
            duration = (time.time() - start) * 1000
            
            return {
                "name": name,
                "status": "healthy" if result else "unhealthy",
                "latency_ms": round(duration, 2)
            }
        except Exception as e:
            return {
                "name": name,
                "status": "unhealthy",
                "error": str(e)
            }
    
    def get_history(self) -> list:
        """获取检查历史"""
        return self._check_history
    
    def get_uptime_percentage(self, hours: int = 24) -> float:
        """计算正常运行时间百分比"""
        cutoff = time.time() - (hours * 3600)
        recent = [h for h in self._check_history if h["timestamp"] > cutoff]
        if not recent:
            return 100.0
        healthy = sum(1 for h in recent if h["status"] == "healthy")
        return round(healthy / len(recent) * 100, 2)

health_checker = HealthChecker()


# ========== 默认健康检查 ==========
def check_database():
    """数据库健康检查"""
    import sqlite3
    try:
        with sqlite3.connect("data.db", timeout=5) as conn:
            conn.execute("SELECT 1")
        return {"healthy": True}
    except Exception as e:
        return {"healthy": False, "error": str(e)}

def check_disk():
    """磁盘健康检查"""
    try:
        disk = psutil.disk_usage('/')
        return {
            "healthy": disk.percent < 90,
            "used_percent": disk.percent,
            "free_gb": round(disk.free / (1024**3), 2)
        }
    except Exception as e:
        return {"healthy": False, "error": str(e)}

def check_memory():
    """内存健康检查"""
    try:
        mem = psutil.virtual_memory()
        return {
            "healthy": mem.percent < 90,
            "used_percent": mem.percent,
            "available_gb": round(mem.available / (1024**3), 2)
        }
    except Exception as e:
        return {"healthy": False, "error": str(e)}

def check_cpu():
    """CPU 健康检查"""
    try:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        return {
            "healthy": cpu_percent < 90,
            "used_percent": cpu_percent,
            "cores": psutil.cpu_count()
        }
    except Exception as e:
        return {"healthy": False, "error": str(e)}

async def check_external_api():
    """外部 API 健康检查"""
    import httpx
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("https://api.openai.com/v1/models", timeout=5)
            return {"healthy": response.status_code < 500}
    except:
        return {"healthy": True}  # 外部 API 不可用不影响整体健康

# 注册默认检查
health_checker.register("database", check_database, critical=True)
health_checker.register("disk", check_disk, critical=True)
health_checker.register("memory", check_memory, critical=False)
health_checker.register("cpu", check_cpu, critical=False)


# ========== 优雅关闭管理器 ==========
class GracefulShutdownManager:
    """优雅关闭管理器"""
    
    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self._shutdown_handlers: list = []
        self._is_shutting_down = False
        self._active_requests = 0
        self._lock = asyncio.Lock()
    
    def register_handler(self, handler: Callable, priority: int = 0):
        """注册关闭处理器（优先级越高越先执行）"""
        self._shutdown_handlers.append({
            "handler": handler,
            "priority": priority
        })
        self._shutdown_handlers.sort(key=lambda x: x["priority"], reverse=True)
    
    async def request_started(self):
        """标记请求开始"""
        async with self._lock:
            self._active_requests += 1
    
    async def request_finished(self):
        """标记请求结束"""
        async with self._lock:
            self._active_requests -= 1
    
    @property
    def is_shutting_down(self) -> bool:
        return self._is_shutting_down
    
    @property
    def active_requests(self) -> int:
        return self._active_requests
    
    async def shutdown(self):
        """执行优雅关闭"""
        if self._is_shutting_down:
            return
        
        self._is_shutting_down = True
        logger.info("Starting graceful shutdown...")
        
        # 等待活跃请求完成
        start_time = time.time()
        while self._active_requests > 0:
            if time.time() - start_time > self.timeout:
                logger.warning(f"Shutdown timeout, {self._active_requests} requests still active")
                break
            logger.info(f"Waiting for {self._active_requests} active requests...")
            await asyncio.sleep(1)
        
        # 执行关闭处理器
        for item in self._shutdown_handlers:
            handler = item["handler"]
            try:
                logger.info(f"Running shutdown handler: {handler.__name__}")
                if asyncio.iscoroutinefunction(handler):
                    await asyncio.wait_for(handler(), timeout=10)
                else:
                    handler()
            except asyncio.TimeoutError:
                logger.error(f"Shutdown handler timed out: {handler.__name__}")
            except Exception as e:
                logger.error(f"Shutdown handler failed: {handler.__name__}: {e}")
        
        logger.info("Graceful shutdown completed")
    
    def get_status(self) -> dict:
        """获取关闭状态"""
        return {
            "is_shutting_down": self._is_shutting_down,
            "active_requests": self._active_requests,
            "handlers_count": len(self._shutdown_handlers)
        }

shutdown_manager = GracefulShutdownManager()

# ========== 告警系统 ==========
class AlertManager:
    """告警管理器"""
    def __init__(self):
        self._rules: list = []
        self._alerts: list = []
        self._webhooks: list = []
    
    def add_rule(self, name: str, condition: Callable, severity: str = "warning", message: str = ""):
        self._rules.append({
            "name": name,
            "condition": condition,
            "severity": severity,
            "message": message
        })
    
    def add_webhook(self, url: str):
        self._webhooks.append(url)
    
    async def check_rules(self):
        for rule in self._rules:
            try:
                if rule["condition"]():
                    alert = {
                        "name": rule["name"],
                        "severity": rule["severity"],
                        "message": rule["message"],
                        "timestamp": datetime.now().isoformat()
                    }
                    self._alerts.append(alert)
                    await self._send_alert(alert)
            except Exception as e:
                logger.error(f"Alert rule check failed: {rule['name']}: {e}")
    
    async def _send_alert(self, alert: dict):
        import httpx
        for webhook in self._webhooks:
            try:
                async with httpx.AsyncClient() as client:
                    await client.post(webhook, json=alert, timeout=5)
            except Exception as e:
                logger.error(f"Failed to send alert to {webhook}: {e}")
    
    def get_alerts(self, limit: int = 100) -> list:
        return self._alerts[-limit:]
    
    def clear_alerts(self):
        self._alerts.clear()

alert_manager = AlertManager()

# 默认告警规则
alert_manager.add_rule(
    "high_cpu",
    lambda: psutil.cpu_percent() > 90,
    "critical",
    "CPU usage exceeds 90%"
)
alert_manager.add_rule(
    "high_memory",
    lambda: psutil.virtual_memory().percent > 90,
    "critical",
    "Memory usage exceeds 90%"
)
alert_manager.add_rule(
    "low_disk",
    lambda: psutil.disk_usage('/').percent > 90,
    "warning",
    "Disk usage exceeds 90%"
)
