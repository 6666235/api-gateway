"""
增强功能模块
包含：性能优化、Token 统计、对话摘要、智能建议等
"""
import asyncio
import hashlib
import json
import time
import re
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


# ========== Token 计数器 ==========
class TokenCounter:
    """简单的 Token 计数器（基于字符估算）"""
    
    @staticmethod
    def count_tokens(text: str, model: str = "gpt-4") -> int:
        """估算 token 数量"""
        if not text:
            return 0
        # 简单估算：中文约 1.5 字符/token，英文约 4 字符/token
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        other_chars = len(text) - chinese_chars
        return int(chinese_chars / 1.5 + other_chars / 4)
    
    @staticmethod
    def count_messages_tokens(messages: List[Dict]) -> int:
        """计算消息列表的 token 数"""
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total += TokenCounter.count_tokens(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        total += TokenCounter.count_tokens(item.get("text", ""))
        return total


# ========== 响应流优化器 ==========
class StreamOptimizer:
    """流式响应优化器"""
    
    def __init__(self, buffer_size: int = 5):
        self.buffer_size = buffer_size
        self.buffer = []
    
    def should_flush(self, chunk: str) -> bool:
        """判断是否应该刷新缓冲区"""
        self.buffer.append(chunk)
        # 遇到标点或缓冲区满时刷新
        if len(self.buffer) >= self.buffer_size:
            return True
        if chunk and chunk[-1] in '。！？.!?\n':
            return True
        return False
    
    def flush(self) -> str:
        """刷新缓冲区"""
        result = ''.join(self.buffer)
        self.buffer = []
        return result


# ========== 智能建议生成器 ==========
class SmartSuggestions:
    """基于上下文生成智能建议"""
    
    SUGGESTIONS = {
        "code": [
            "请解释这段代码的作用",
            "如何优化这段代码？",
            "有没有更好的实现方式？",
            "请添加注释",
            "请写单元测试"
        ],
        "error": [
            "如何修复这个错误？",
            "这个错误的原因是什么？",
            "如何避免类似错误？"
        ],
        "general": [
            "请继续",
            "请详细解释",
            "请举个例子",
            "请用更简单的语言解释",
            "请总结要点"
        ],
        "translation": [
            "请翻译成英文",
            "请翻译成中文",
            "请润色这段文字"
        ]
    }
    
    @classmethod
    def get_suggestions(cls, last_message: str, context: str = "general") -> List[str]:
        """根据上下文获取建议"""
        suggestions = cls.SUGGESTIONS.get(context, cls.SUGGESTIONS["general"])
        
        # 检测代码
        if "```" in last_message or "def " in last_message or "function " in last_message:
            suggestions = cls.SUGGESTIONS["code"]
        # 检测错误
        elif "error" in last_message.lower() or "错误" in last_message:
            suggestions = cls.SUGGESTIONS["error"]
        
        return suggestions[:5]


# ========== 对话摘要生成器 ==========
class ConversationSummarizer:
    """对话摘要生成器"""
    
    @staticmethod
    def generate_title(messages: List[Dict], max_length: int = 30) -> str:
        """从对话生成标题"""
        if not messages:
            return "新对话"
        
        # 取第一条用户消息
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    # 清理并截断
                    title = content.strip()[:max_length]
                    if len(content) > max_length:
                        title += "..."
                    return title
        return "新对话"
    
    @staticmethod
    def extract_keywords(text: str, top_n: int = 5) -> List[str]:
        """提取关键词"""
        # 简单实现：提取中文词和英文单词
        words = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+', text)
        # 过滤停用词
        stopwords = {'的', '是', '在', '了', '和', '与', 'the', 'a', 'an', 'is', 'are', 'to'}
        words = [w for w in words if w.lower() not in stopwords and len(w) > 1]
        # 统计词频
        freq = defaultdict(int)
        for w in words:
            freq[w.lower()] += 1
        # 返回高频词
        sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [w for w, _ in sorted_words[:top_n]]


# ========== 性能监控器 ==========
class PerformanceMonitor:
    """API 性能监控"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.lock = asyncio.Lock()
    
    async def record(self, provider: str, model: str, latency: float, tokens: int, success: bool):
        """记录一次 API 调用"""
        async with self.lock:
            key = f"{provider}/{model}"
            self.metrics[key].append({
                "timestamp": time.time(),
                "latency": latency,
                "tokens": tokens,
                "success": success
            })
            # 只保留最近 1000 条
            if len(self.metrics[key]) > 1000:
                self.metrics[key] = self.metrics[key][-1000:]
    
    def get_stats(self, provider: str = None, model: str = None) -> Dict:
        """获取统计信息"""
        stats = {}
        for key, records in self.metrics.items():
            if provider and not key.startswith(provider):
                continue
            if model and model not in key:
                continue
            
            recent = [r for r in records if time.time() - r["timestamp"] < 3600]
            if not recent:
                continue
            
            latencies = [r["latency"] for r in recent]
            success_count = sum(1 for r in recent if r["success"])
            
            stats[key] = {
                "calls": len(recent),
                "success_rate": success_count / len(recent) * 100,
                "avg_latency": sum(latencies) / len(latencies),
                "min_latency": min(latencies),
                "max_latency": max(latencies),
                "total_tokens": sum(r["tokens"] for r in recent)
            }
        return stats


# ========== 消息格式化器 ==========
class MessageFormatter:
    """消息格式化工具"""
    
    @staticmethod
    def format_code_blocks(text: str) -> str:
        """格式化代码块"""
        # 自动检测语言
        def detect_language(code: str) -> str:
            if "def " in code or "import " in code:
                return "python"
            elif "function " in code or "const " in code or "let " in code:
                return "javascript"
            elif "<html" in code or "<div" in code:
                return "html"
            elif "SELECT " in code.upper() or "INSERT " in code.upper():
                return "sql"
            return ""
        
        # 处理没有语言标记的代码块
        def replace_code_block(match):
            code = match.group(1)
            lang = detect_language(code)
            return f"```{lang}\n{code}\n```"
        
        text = re.sub(r'```\n([^`]+)\n```', replace_code_block, text)
        return text
    
    @staticmethod
    def escape_html(text: str) -> str:
        """转义 HTML 特殊字符"""
        return (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;"))


# ========== 快捷命令处理器 ==========
class CommandProcessor:
    """处理快捷命令"""
    
    COMMANDS = {
        "/help": "显示帮助信息",
        "/clear": "清空当前对话",
        "/export": "导出对话记录",
        "/model": "切换模型",
        "/system": "设置系统提示词",
        "/translate": "翻译模式",
        "/code": "代码模式",
        "/summary": "总结对话"
    }
    
    @classmethod
    def is_command(cls, text: str) -> bool:
        """检查是否是命令"""
        return text.strip().startswith("/")
    
    @classmethod
    def parse_command(cls, text: str) -> tuple:
        """解析命令"""
        parts = text.strip().split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        return cmd, args
    
    @classmethod
    def get_help(cls) -> str:
        """获取帮助信息"""
        lines = ["可用命令："]
        for cmd, desc in cls.COMMANDS.items():
            lines.append(f"  {cmd} - {desc}")
        return "\n".join(lines)


# 全局实例
token_counter = TokenCounter()
performance_monitor = PerformanceMonitor()
message_formatter = MessageFormatter()
command_processor = CommandProcessor()
