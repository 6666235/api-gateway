# ========== 异步任务队列模块 ==========
"""
支持后台任务处理、定时任务、任务重试
"""
import os
import asyncio
import json
import time
import uuid
from typing import Callable, Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"

@dataclass
class Task:
    """任务定义"""
    id: str
    name: str
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: str = None
    retries: int = 0
    max_retries: int = 3
    retry_delay: int = 60  # 秒
    priority: int = 0  # 越大优先级越高
    created_at: float = field(default_factory=time.time)
    started_at: float = None
    completed_at: float = None
    scheduled_at: float = None  # 定时执行时间
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status.value,
            "result": str(self.result)[:200] if self.result else None,
            "error": self.error,
            "retries": self.retries,
            "created_at": datetime.fromtimestamp(self.created_at).isoformat(),
            "started_at": datetime.fromtimestamp(self.started_at).isoformat() if self.started_at else None,
            "completed_at": datetime.fromtimestamp(self.completed_at).isoformat() if self.completed_at else None,
            "duration_ms": int((self.completed_at - self.started_at) * 1000) if self.completed_at and self.started_at else None
        }

class TaskQueue:
    """异步任务队列（支持持久化）"""
    def __init__(self, max_workers: int = 5, persist: bool = True, db_path: str = "data.db"):
        self._handlers: Dict[str, Callable] = {}
        self._tasks: Dict[str, Task] = {}
        self._queue: asyncio.PriorityQueue = None
        self._workers: List[asyncio.Task] = []
        self._max_workers = max_workers
        self._running = False
        self._stats = defaultdict(int)
        self._persist = persist
        self._db_path = db_path
        
        if persist:
            self._init_persistence()
    
    def _init_persistence(self):
        """初始化持久化存储"""
        import sqlite3
        with sqlite3.connect(self._db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS task_queue (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    args TEXT,
                    kwargs TEXT,
                    status TEXT DEFAULT 'pending',
                    result TEXT,
                    error TEXT,
                    retries INTEGER DEFAULT 0,
                    max_retries INTEGER DEFAULT 3,
                    priority INTEGER DEFAULT 0,
                    created_at REAL,
                    started_at REAL,
                    completed_at REAL,
                    scheduled_at REAL
                )
            ''')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_task_status ON task_queue(status)')
    
    def _save_task(self, task: Task):
        """保存任务到数据库"""
        if not self._persist:
            return
        import sqlite3
        with sqlite3.connect(self._db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO task_queue 
                (id, name, args, kwargs, status, result, error, retries, max_retries, 
                 priority, created_at, started_at, completed_at, scheduled_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                task.id, task.name, json.dumps(task.args), json.dumps(task.kwargs),
                task.status.value, str(task.result)[:1000] if task.result else None,
                task.error, task.retries, task.max_retries, task.priority,
                task.created_at, task.started_at, task.completed_at, task.scheduled_at
            ))
    
    def _load_pending_tasks(self):
        """从数据库加载未完成的任务"""
        if not self._persist:
            return
        import sqlite3
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute('''
                SELECT * FROM task_queue 
                WHERE status IN ('pending', 'running', 'retrying')
                ORDER BY priority DESC, created_at ASC
            ''').fetchall()
            
            for row in rows:
                task = Task(
                    id=row['id'],
                    name=row['name'],
                    args=tuple(json.loads(row['args'] or '[]')),
                    kwargs=json.loads(row['kwargs'] or '{}'),
                    status=TaskStatus(row['status']),
                    retries=row['retries'],
                    max_retries=row['max_retries'],
                    priority=row['priority'],
                    created_at=row['created_at'],
                    scheduled_at=row['scheduled_at']
                )
                self._tasks[task.id] = task
                logger.info(f"Restored task from persistence: {task.id}")
    
    def register(self, name: str, handler: Callable, max_retries: int = 3):
        """注册任务处理器"""
        self._handlers[name] = {
            "handler": handler,
            "max_retries": max_retries
        }
        logger.info(f"Registered task handler: {name}")
    
    async def enqueue(
        self,
        name: str,
        *args,
        priority: int = 0,
        delay: int = 0,
        **kwargs
    ) -> str:
        """添加任务到队列"""
        if name not in self._handlers:
            raise ValueError(f"Unknown task: {name}")
        
        task_id = str(uuid.uuid4())
        handler_config = self._handlers[name]
        
        task = Task(
            id=task_id,
            name=name,
            args=args,
            kwargs=kwargs,
            priority=priority,
            max_retries=handler_config["max_retries"],
            scheduled_at=time.time() + delay if delay > 0 else None
        )
        
        self._tasks[task_id] = task
        self._save_task(task)  # 持久化
        
        if self._queue:
            # 优先级队列使用负数（越小越优先）
            await self._queue.put((-priority, task_id))
        
        self._stats["enqueued"] += 1
        logger.debug(f"Task enqueued: {name} ({task_id})")
        return task_id
    
    async def _worker(self, worker_id: int):
        """工作协程"""
        logger.info(f"Worker {worker_id} started")
        
        while self._running:
            try:
                # 等待任务
                priority, task_id = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=1.0
                )
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Worker {worker_id} queue error: {e}")
                continue
            
            task = self._tasks.get(task_id)
            if not task:
                continue
            
            # 检查定时任务
            if task.scheduled_at and time.time() < task.scheduled_at:
                # 重新入队
                await self._queue.put((priority, task_id))
                await asyncio.sleep(0.1)
                continue
            
            # 执行任务
            await self._execute_task(task)
            self._queue.task_done()
        
        logger.info(f"Worker {worker_id} stopped")
    
    async def _execute_task(self, task: Task):
        """执行任务"""
        task.status = TaskStatus.RUNNING
        task.started_at = time.time()
        
        handler_config = self._handlers.get(task.name)
        if not handler_config:
            task.status = TaskStatus.FAILED
            task.error = f"Handler not found: {task.name}"
            return
        
        handler = handler_config["handler"]
        
        try:
            if asyncio.iscoroutinefunction(handler):
                task.result = await handler(*task.args, **task.kwargs)
            else:
                task.result = handler(*task.args, **task.kwargs)
            
            task.status = TaskStatus.COMPLETED
            task.completed_at = time.time()
            self._stats["completed"] += 1
            logger.debug(f"Task completed: {task.name} ({task.id})")
            
        except Exception as e:
            task.error = str(e)
            task.retries += 1
            
            if task.retries < task.max_retries:
                task.status = TaskStatus.RETRYING
                # 指数退避重试
                delay = task.retry_delay * (2 ** (task.retries - 1))
                task.scheduled_at = time.time() + delay
                await self._queue.put((-task.priority, task.id))
                self._stats["retried"] += 1
                logger.warning(f"Task retry {task.retries}/{task.max_retries}: {task.name} ({task.id})")
            else:
                task.status = TaskStatus.FAILED
                task.completed_at = time.time()
                self._stats["failed"] += 1
                logger.error(f"Task failed: {task.name} ({task.id}): {e}")
        
        # 持久化任务状态
        self._save_task(task)
    
    async def start(self):
        """启动队列"""
        if self._running:
            return
        
        self._running = True
        self._queue = asyncio.PriorityQueue()
        
        # 从持久化存储恢复任务
        self._load_pending_tasks()
        for task_id, task in self._tasks.items():
            if task.status in (TaskStatus.PENDING, TaskStatus.RETRYING):
                await self._queue.put((-task.priority, task_id))
        
        # 启动工作协程
        for i in range(self._max_workers):
            worker = asyncio.create_task(self._worker(i))
            self._workers.append(worker)
        
        logger.info(f"Task queue started with {self._max_workers} workers, restored {len(self._tasks)} tasks")
    
    async def stop(self):
        """停止队列"""
        self._running = False
        
        # 等待所有工作协程结束
        for worker in self._workers:
            worker.cancel()
            try:
                await worker
            except asyncio.CancelledError:
                pass
        
        self._workers.clear()
        logger.info("Task queue stopped")
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """获取任务"""
        return self._tasks.get(task_id)
    
    def get_tasks(self, status: TaskStatus = None, limit: int = 100) -> List[dict]:
        """获取任务列表"""
        tasks = list(self._tasks.values())
        if status:
            tasks = [t for t in tasks if t.status == status]
        tasks.sort(key=lambda t: t.created_at, reverse=True)
        return [t.to_dict() for t in tasks[:limit]]
    
    def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        task = self._tasks.get(task_id)
        if task and task.status == TaskStatus.PENDING:
            task.status = TaskStatus.CANCELLED
            return True
        return False
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        status_counts = defaultdict(int)
        for task in self._tasks.values():
            status_counts[task.status.value] += 1
        
        return {
            "total": len(self._tasks),
            "by_status": dict(status_counts),
            "enqueued": self._stats["enqueued"],
            "completed": self._stats["completed"],
            "failed": self._stats["failed"],
            "retried": self._stats["retried"],
            "workers": self._max_workers,
            "running": self._running
        }
    
    def cleanup(self, max_age_hours: int = 24):
        """清理旧任务"""
        cutoff = time.time() - (max_age_hours * 3600)
        to_delete = [
            task_id for task_id, task in self._tasks.items()
            if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED)
            and task.created_at < cutoff
        ]
        for task_id in to_delete:
            del self._tasks[task_id]
        return len(to_delete)

# 全局任务队列
task_queue = TaskQueue(max_workers=5)

# ========== 定时任务调度器 ==========
class Scheduler:
    """定时任务调度器"""
    def __init__(self):
        self._jobs: Dict[str, dict] = {}
        self._running = False
        self._task: asyncio.Task = None
    
    def add_job(
        self,
        name: str,
        handler: Callable,
        interval: int = None,  # 秒
        cron: str = None,  # cron 表达式 (简化版)
        run_immediately: bool = False
    ):
        """添加定时任务"""
        self._jobs[name] = {
            "handler": handler,
            "interval": interval,
            "cron": cron,
            "last_run": None,
            "next_run": time.time() if run_immediately else None,
            "run_count": 0,
            "errors": 0
        }
        logger.info(f"Scheduled job added: {name}")
    
    def remove_job(self, name: str):
        """移除定时任务"""
        if name in self._jobs:
            del self._jobs[name]
    
    async def _run_loop(self):
        """调度循环"""
        while self._running:
            now = time.time()
            
            for name, job in self._jobs.items():
                should_run = False
                
                if job["interval"]:
                    if job["last_run"] is None or (now - job["last_run"]) >= job["interval"]:
                        should_run = True
                elif job["next_run"] and now >= job["next_run"]:
                    should_run = True
                
                if should_run:
                    try:
                        handler = job["handler"]
                        if asyncio.iscoroutinefunction(handler):
                            await handler()
                        else:
                            handler()
                        job["run_count"] += 1
                        logger.debug(f"Scheduled job executed: {name}")
                    except Exception as e:
                        job["errors"] += 1
                        logger.error(f"Scheduled job failed: {name}: {e}")
                    finally:
                        job["last_run"] = now
                        if job["interval"]:
                            job["next_run"] = now + job["interval"]
            
            await asyncio.sleep(1)
    
    async def start(self):
        """启动调度器"""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("Scheduler started")
    
    async def stop(self):
        """停止调度器"""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Scheduler stopped")
    
    def get_jobs(self) -> List[dict]:
        """获取所有任务"""
        return [
            {
                "name": name,
                "interval": job["interval"],
                "last_run": datetime.fromtimestamp(job["last_run"]).isoformat() if job["last_run"] else None,
                "run_count": job["run_count"],
                "errors": job["errors"]
            }
            for name, job in self._jobs.items()
        ]

scheduler = Scheduler()

# ========== 常用后台任务 ==========
async def cleanup_expired_sessions():
    """清理过期会话"""
    import sqlite3
    try:
        with sqlite3.connect("data.db") as conn:
            result = conn.execute(
                "DELETE FROM sessions WHERE expires_at < ?",
                (datetime.now(),)
            )
            if result.rowcount > 0:
                logger.info(f"Cleaned up {result.rowcount} expired sessions")
    except Exception as e:
        logger.error(f"Session cleanup failed: {e}")

async def cleanup_old_logs():
    """清理旧日志"""
    import sqlite3
    try:
        cutoff = datetime.now() - timedelta(days=30)
        with sqlite3.connect("data.db") as conn:
            result = conn.execute(
                "DELETE FROM api_logs WHERE created_at < ?",
                (cutoff,)
            )
            if result.rowcount > 0:
                logger.info(f"Cleaned up {result.rowcount} old API logs")
    except Exception as e:
        logger.error(f"Log cleanup failed: {e}")

async def update_usage_stats():
    """更新使用统计"""
    from .monitoring import metrics
    import sqlite3
    try:
        with sqlite3.connect("data.db") as conn:
            # 活跃用户数
            active_users = conn.execute(
                "SELECT COUNT(DISTINCT user_id) FROM api_logs WHERE created_at > datetime('now', '-1 day')"
            ).fetchone()[0]
            metrics.set("aihub_active_users", active_users)
            
            # 总对话数
            total_conversations = conn.execute(
                "SELECT COUNT(*) FROM conversations"
            ).fetchone()[0]
            metrics.set("aihub_total_conversations", total_conversations)
    except Exception as e:
        logger.error(f"Stats update failed: {e}")

# 注册默认定时任务
scheduler.add_job("cleanup_sessions", cleanup_expired_sessions, interval=3600)  # 每小时
scheduler.add_job("cleanup_logs", cleanup_old_logs, interval=86400)  # 每天
scheduler.add_job("update_stats", update_usage_stats, interval=300)  # 每5分钟
