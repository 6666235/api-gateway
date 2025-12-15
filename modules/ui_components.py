"""
UI 组件增强模块
提供前端可用的 API 端点
"""
from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, List, Optional
import json

router = APIRouter(prefix="/api/ui", tags=["UI Components"])


# ========== 主题配置 ==========
THEMES = {
    "dark": {
        "name": "深色模式",
        "colors": {
            "bg": "#0f0f1a",
            "bg2": "#1a1a2e",
            "bg3": "#252542",
            "text": "#e8e8f0",
            "text2": "#8888a0",
            "accent": "#6366f1",
            "accent2": "#818cf8",
            "success": "#22c55e",
            "warning": "#f59e0b",
            "error": "#ef4444",
            "border": "#2d2d4a"
        }
    },
    "light": {
        "name": "浅色模式",
        "colors": {
            "bg": "#f8f9fa",
            "bg2": "#ffffff",
            "bg3": "#e9ecef",
            "text": "#212529",
            "text2": "#6c757d",
            "accent": "#6366f1",
            "accent2": "#818cf8",
            "success": "#22c55e",
            "warning": "#f59e0b",
            "error": "#ef4444",
            "border": "#dee2e6"
        }
    },
    "blue": {
        "name": "蓝色主题",
        "colors": {
            "bg": "#0a1628",
            "bg2": "#0f2744",
            "bg3": "#1a3a5c",
            "text": "#e8f4ff",
            "text2": "#88b4d8",
            "accent": "#3b82f6",
            "accent2": "#60a5fa",
            "success": "#22c55e",
            "warning": "#f59e0b",
            "error": "#ef4444",
            "border": "#1e4976"
        }
    },
    "green": {
        "name": "绿色主题",
        "colors": {
            "bg": "#0a1a14",
            "bg2": "#0f2e1f",
            "bg3": "#1a4a32",
            "text": "#e8fff0",
            "text2": "#88d8a8",
            "accent": "#22c55e",
            "accent2": "#4ade80",
            "success": "#22c55e",
            "warning": "#f59e0b",
            "error": "#ef4444",
            "border": "#1e5c3a"
        }
    },
    "purple": {
        "name": "紫色主题",
        "colors": {
            "bg": "#14081a",
            "bg2": "#1f0f2e",
            "bg3": "#321a4a",
            "text": "#f8e8ff",
            "text2": "#b888d8",
            "accent": "#a855f7",
            "accent2": "#c084fc",
            "success": "#22c55e",
            "warning": "#f59e0b",
            "error": "#ef4444",
            "border": "#4a1e76"
        }
    }
}


@router.get("/themes")
async def get_themes():
    """获取所有可用主题"""
    return {"themes": THEMES}


@router.get("/themes/{theme_id}")
async def get_theme(theme_id: str):
    """获取指定主题"""
    if theme_id not in THEMES:
        raise HTTPException(status_code=404, detail="主题不存在")
    return THEMES[theme_id]


# ========== 快捷键配置 ==========
SHORTCUTS = {
    "newChat": {"key": "Ctrl+N", "description": "新建对话"},
    "send": {"key": "Enter", "description": "发送消息"},
    "newLine": {"key": "Shift+Enter", "description": "换行"},
    "clearChat": {"key": "Ctrl+L", "description": "清空对话"},
    "search": {"key": "Ctrl+K", "description": "搜索"},
    "settings": {"key": "Ctrl+,", "description": "设置"},
    "export": {"key": "Ctrl+E", "description": "导出对话"},
    "toggleSidebar": {"key": "Ctrl+B", "description": "切换侧边栏"},
    "focusInput": {"key": "Ctrl+I", "description": "聚焦输入框"},
    "regenerate": {"key": "Ctrl+R", "description": "重新生成"},
    "copy": {"key": "Ctrl+C", "description": "复制选中内容"},
    "paste": {"key": "Ctrl+V", "description": "粘贴"},
    "undo": {"key": "Ctrl+Z", "description": "撤销"},
    "redo": {"key": "Ctrl+Y", "description": "重做"}
}


@router.get("/shortcuts")
async def get_shortcuts():
    """获取快捷键配置"""
    return {"shortcuts": SHORTCUTS}


# ========== 表情和图标 ==========
EMOJI_CATEGORIES = {
    "常用": ["👍", "👎", "❤️", "🔥", "✨", "🎉", "💡", "⚡", "🚀", "💪"],
    "表情": ["😀", "😂", "🤔", "😊", "😎", "🥳", "😅", "🙏", "👀", "💯"],
    "符号": ["✅", "❌", "⭐", "📌", "🔗", "📎", "📝", "💬", "🔔", "⚙️"],
    "代码": ["💻", "🖥️", "⌨️", "🐛", "🔧", "📦", "🗂️", "📊", "🔐", "🌐"]
}


@router.get("/emojis")
async def get_emojis():
    """获取表情列表"""
    return {"categories": EMOJI_CATEGORIES}


# ========== 提示词模板 ==========
PROMPT_TEMPLATES = [
    {
        "id": "translator",
        "name": "翻译助手",
        "icon": "🌐",
        "prompt": "你是一个专业的翻译助手。请将用户输入的内容翻译成目标语言，保持原文的语气和风格。"
    },
    {
        "id": "coder",
        "name": "代码助手",
        "icon": "💻",
        "prompt": "你是一个专业的编程助手。请帮助用户编写、调试和优化代码。提供清晰的解释和最佳实践建议。"
    },
    {
        "id": "writer",
        "name": "写作助手",
        "icon": "✍️",
        "prompt": "你是一个专业的写作助手。请帮助用户改进文章结构、润色语言、纠正语法错误。"
    },
    {
        "id": "analyst",
        "name": "数据分析师",
        "icon": "📊",
        "prompt": "你是一个数据分析专家。请帮助用户分析数据、生成报告、提供洞察和建议。"
    },
    {
        "id": "teacher",
        "name": "学习导师",
        "icon": "📚",
        "prompt": "你是一个耐心的学习导师。请用简单易懂的方式解释概念，提供例子，并回答学习相关的问题。"
    },
    {
        "id": "creative",
        "name": "创意助手",
        "icon": "🎨",
        "prompt": "你是一个富有创意的助手。请帮助用户进行头脑风暴、生成创意想法、设计方案。"
    }
]


@router.get("/templates")
async def get_templates():
    """获取提示词模板"""
    return {"templates": PROMPT_TEMPLATES}


# ========== 统计卡片数据 ==========
@router.get("/dashboard")
async def get_dashboard_data():
    """获取仪表盘数据"""
    return {
        "cards": [
            {"title": "今日对话", "value": "0", "icon": "💬", "trend": "+0%"},
            {"title": "总 Token", "value": "0", "icon": "🔢", "trend": "+0%"},
            {"title": "平均响应", "value": "0s", "icon": "⚡", "trend": "0%"},
            {"title": "成功率", "value": "100%", "icon": "✅", "trend": "0%"}
        ]
    }
