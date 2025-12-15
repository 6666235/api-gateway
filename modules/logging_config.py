# ========== 日志配置模块 ==========
"""
支持：日志轮转、结构化日志、多输出目标、异步日志
"""
import os
import sys
import json
import logging
import asyncio
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from datetime import datetime
from typing import Optional, Dict, Any
from queue import Queue
from threading import Thread
import traceback
import gzip
import shutil


class JSONFormatter(logging.Formatter):
    """JSON 格式化器"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # 添加异常信息
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info) if record.exc_info[0] else None
            }
        
        # 添加额外字段
        if hasattr(record, 'extra_data'):
            log_data.update(record.extra_data)
        
        return json.dumps(log_data, ensure_ascii=False)


class ColoredFormatter(logging.Formatter):
    """彩色控制台格式化器"""
    
    COLORS = {
        'DEBUG': '\033[36m',     # 青色
        'INFO': '\033[32m',      # 绿色
        'WARNING': '\033[33m',   # 黄色
        'ERROR': '\033[31m',     # 红色
        'CRITICAL': '\033[35m',  # 紫色
    }
    RESET = '\033[0m'
    
    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logging(
    log_level: str = "INFO",
    log_file: str = "app.log",
    max_size_mb: int = 50,
    backup_count: int = 5,
    json_format: bool = False,
    colored_console: bool = True
) -> logging.Logger:
    """配置日志系统"""
    
    # 获取根日志器
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # 清除现有处理器
    root_logger.handlers.clear()
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    
    if colored_console and sys.stdout.isatty():
        console_formatter = ColoredFormatter(
            '%(asctime)s - %(levelname)s - [%(name)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    else:
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - [%(name)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # 文件处理器（带轮转）
    if log_file:
        # 确保日志目录存在
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # 按大小轮转
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_size_mb * 1024 * 1024,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        
        if json_format:
            file_formatter = JSONFormatter()
        else:
            file_formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - [%(name)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    # 错误日志单独文件
    error_log = log_file.replace('.log', '.error.log') if log_file else 'error.log'
    error_handler = RotatingFileHandler(
        error_log,
        maxBytes=max_size_mb * 1024 * 1024,
        backupCount=backup_count,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - [%(name)s] %(message)s\n%(pathname)s:%(lineno)d\n',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    root_logger.addHandler(error_handler)
    
    return root_logger


class RequestLogger:
    """请求日志记录器"""
    
    def __init__(self, logger_name: str = "request"):
        self.logger = logging.getLogger(logger_name)
    
    def log_request(
        self,
        method: str,
        path: str,
        status_code: int,
        duration_ms: float,
        client_ip: str = None,
        user_id: int = None,
        request_id: str = None,
        extra: dict = None
    ):
        """记录请求日志"""
        log_data = {
            "request_id": request_id,
            "method": method,
            "path": path,
            "status": status_code,
            "duration_ms": round(duration_ms, 2),
            "client_ip": client_ip,
            "user_id": user_id,
        }
        
        if extra:
            log_data.update(extra)
        
        # 根据状态码选择日志级别
        if status_code >= 500:
            self.logger.error(json.dumps(log_data, ensure_ascii=False))
        elif status_code >= 400:
            self.logger.warning(json.dumps(log_data, ensure_ascii=False))
        else:
            self.logger.info(json.dumps(log_data, ensure_ascii=False))


# 全局请求日志器
request_logger = RequestLogger()


# ========== 压缩日志轮转处理器 ==========
class CompressedRotatingFileHandler(RotatingFileHandler):
    """支持 gzip 压缩的日志轮转处理器"""
    
    def __init__(self, filename: str, mode: str = 'a', maxBytes: int = 0,
                 backupCount: int = 0, encoding: str = None, delay: bool = False,
                 compress: bool = True):
        super().__init__(filename, mode, maxBytes, backupCount, encoding, delay)
        self.compress = compress
    
    def doRollover(self):
        """执行轮转并压缩旧日志"""
        if self.stream:
            self.stream.close()
            self.stream = None
        
        if self.backupCount > 0:
            # 删除最旧的压缩文件
            for i in range(self.backupCount - 1, 0, -1):
                sfn = self.rotation_filename(f"{self.baseFilename}.{i}")
                dfn = self.rotation_filename(f"{self.baseFilename}.{i + 1}")
                
                # 检查压缩文件
                if self.compress:
                    sfn_gz = f"{sfn}.gz"
                    dfn_gz = f"{dfn}.gz"
                    if os.path.exists(sfn_gz):
                        if os.path.exists(dfn_gz):
                            os.remove(dfn_gz)
                        os.rename(sfn_gz, dfn_gz)
                else:
                    if os.path.exists(sfn):
                        if os.path.exists(dfn):
                            os.remove(dfn)
                        os.rename(sfn, dfn)
            
            # 轮转当前文件
            dfn = self.rotation_filename(f"{self.baseFilename}.1")
            if os.path.exists(dfn):
                os.remove(dfn)
            
            self.rotate(self.baseFilename, dfn)
            
            # 压缩轮转后的文件
            if self.compress and os.path.exists(dfn):
                self._compress_file(dfn)
        
        if not self.delay:
            self.stream = self._open()
    
    def _compress_file(self, filepath: str):
        """压缩文件"""
        try:
            with open(filepath, 'rb') as f_in:
                with gzip.open(f"{filepath}.gz", 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            os.remove(filepath)
        except Exception as e:
            logging.error(f"Failed to compress log file: {e}")


class TimedCompressedRotatingFileHandler(TimedRotatingFileHandler):
    """支持 gzip 压缩的时间轮转处理器"""
    
    def __init__(self, filename: str, when: str = 'midnight', interval: int = 1,
                 backupCount: int = 30, encoding: str = None, delay: bool = False,
                 utc: bool = False, compress: bool = True):
        super().__init__(filename, when, interval, backupCount, encoding, delay, utc)
        self.compress = compress
    
    def doRollover(self):
        """执行轮转并压缩"""
        super().doRollover()
        
        if self.compress:
            # 压缩旧日志文件
            for filename in self.getFilesToDelete():
                if os.path.exists(filename) and not filename.endswith('.gz'):
                    self._compress_file(filename)
    
    def _compress_file(self, filepath: str):
        """压缩文件"""
        try:
            with open(filepath, 'rb') as f_in:
                with gzip.open(f"{filepath}.gz", 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            os.remove(filepath)
        except Exception as e:
            logging.error(f"Failed to compress log file: {e}")


# ========== 异步日志处理器 ==========
class AsyncLogHandler(logging.Handler):
    """异步日志处理器，避免阻塞主线程"""
    
    def __init__(self, handler: logging.Handler, queue_size: int = 10000):
        super().__init__()
        self._handler = handler
        self._queue: Queue = Queue(maxsize=queue_size)
        self._thread = Thread(target=self._process_logs, daemon=True)
        self._running = True
        self._thread.start()
    
    def emit(self, record: logging.LogRecord):
        """将日志记录放入队列"""
        try:
            self._queue.put_nowait(record)
        except:
            # 队列满时丢弃日志
            pass
    
    def _process_logs(self):
        """后台线程处理日志"""
        while self._running:
            try:
                record = self._queue.get(timeout=1)
                self._handler.emit(record)
            except:
                continue
    
    def close(self):
        """关闭处理器"""
        self._running = False
        self._thread.join(timeout=5)
        self._handler.close()
        super().close()


# ========== 增强的日志配置函数 ==========
def setup_advanced_logging(
    log_level: str = "INFO",
    log_file: str = "app.log",
    max_size_mb: int = 50,
    backup_count: int = 10,
    json_format: bool = False,
    colored_console: bool = True,
    compress_old_logs: bool = True,
    async_logging: bool = True,
    time_rotation: bool = False,
    rotation_when: str = "midnight"
) -> logging.Logger:
    """
    高级日志配置
    
    Args:
        log_level: 日志级别
        log_file: 日志文件路径
        max_size_mb: 单个日志文件最大大小（MB）
        backup_count: 保留的备份文件数量
        json_format: 是否使用 JSON 格式
        colored_console: 是否使用彩色控制台输出
        compress_old_logs: 是否压缩旧日志
        async_logging: 是否使用异步日志
        time_rotation: 是否使用时间轮转（否则使用大小轮转）
        rotation_when: 时间轮转周期（midnight, H, D, W0-W6）
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    root_logger.handlers.clear()
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    
    if colored_console and sys.stdout.isatty():
        console_formatter = ColoredFormatter(
            '%(asctime)s - %(levelname)s - [%(name)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    else:
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - [%(name)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # 文件处理器
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        if time_rotation:
            file_handler = TimedCompressedRotatingFileHandler(
                log_file,
                when=rotation_when,
                backupCount=backup_count,
                encoding='utf-8',
                compress=compress_old_logs
            )
        else:
            file_handler = CompressedRotatingFileHandler(
                log_file,
                maxBytes=max_size_mb * 1024 * 1024,
                backupCount=backup_count,
                encoding='utf-8',
                compress=compress_old_logs
            )
        
        file_handler.setLevel(logging.DEBUG)
        
        if json_format:
            file_formatter = JSONFormatter()
        else:
            file_formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - [%(name)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        file_handler.setFormatter(file_formatter)
        
        # 使用异步处理器包装
        if async_logging:
            file_handler = AsyncLogHandler(file_handler)
        
        root_logger.addHandler(file_handler)
    
    # 错误日志单独文件
    error_log = log_file.replace('.log', '.error.log') if log_file else 'error.log'
    error_handler = CompressedRotatingFileHandler(
        error_log,
        maxBytes=max_size_mb * 1024 * 1024,
        backupCount=backup_count,
        encoding='utf-8',
        compress=compress_old_logs
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - [%(name)s] %(message)s\n%(pathname)s:%(lineno)d\n',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    
    if async_logging:
        error_handler = AsyncLogHandler(error_handler)
    
    root_logger.addHandler(error_handler)
    
    return root_logger


# ========== 日志清理工具 ==========
def cleanup_old_logs(log_dir: str = ".", max_age_days: int = 30, pattern: str = "*.log*"):
    """清理旧日志文件"""
    import glob
    from datetime import timedelta
    
    cutoff = datetime.now() - timedelta(days=max_age_days)
    deleted = 0
    
    for filepath in glob.glob(os.path.join(log_dir, pattern)):
        try:
            mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
            if mtime < cutoff:
                os.remove(filepath)
                deleted += 1
        except Exception as e:
            logging.error(f"Failed to delete old log {filepath}: {e}")
    
    return deleted
