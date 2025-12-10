# ========== 数据验证模块 ==========
"""
输入验证、数据清洗、安全检查
"""
import re
import html
from typing import Optional, List, Any
from pydantic import BaseModel, validator, Field
import logging

logger = logging.getLogger(__name__)

# ========== 验证规则 ==========
class ValidationRules:
    """验证规则集合"""
    
    # 用户名规则
    USERNAME_MIN_LENGTH = 3
    USERNAME_MAX_LENGTH = 32
    USERNAME_PATTERN = r'^[a-zA-Z0-9_-]+$'
    
    # 密码规则
    PASSWORD_MIN_LENGTH = 8
    PASSWORD_MAX_LENGTH = 128
    PASSWORD_REQUIRE_UPPERCASE = True
    PASSWORD_REQUIRE_LOWERCASE = True
    PASSWORD_REQUIRE_DIGIT = True
    PASSWORD_REQUIRE_SPECIAL = False
    
    # 邮箱规则
    EMAIL_PATTERN = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    # 内容规则
    MAX_MESSAGE_LENGTH = 100000
    MAX_TITLE_LENGTH = 200
    
    # 敏感词列表（示例）
    SENSITIVE_WORDS = []

# ========== 验证函数 ==========
def validate_username(username: str) -> tuple[bool, str]:
    """验证用户名"""
    if not username:
        return False, "用户名不能为空"
    
    if len(username) < ValidationRules.USERNAME_MIN_LENGTH:
        return False, f"用户名至少 {ValidationRules.USERNAME_MIN_LENGTH} 个字符"
    
    if len(username) > ValidationRules.USERNAME_MAX_LENGTH:
        return False, f"用户名最多 {ValidationRules.USERNAME_MAX_LENGTH} 个字符"
    
    if not re.match(ValidationRules.USERNAME_PATTERN, username):
        return False, "用户名只能包含字母、数字、下划线和连字符"
    
    return True, ""

def validate_password(password: str) -> tuple[bool, str]:
    """验证密码强度"""
    if not password:
        return False, "密码不能为空"
    
    if len(password) < ValidationRules.PASSWORD_MIN_LENGTH:
        return False, f"密码至少 {ValidationRules.PASSWORD_MIN_LENGTH} 个字符"
    
    if len(password) > ValidationRules.PASSWORD_MAX_LENGTH:
        return False, f"密码最多 {ValidationRules.PASSWORD_MAX_LENGTH} 个字符"
    
    if ValidationRules.PASSWORD_REQUIRE_UPPERCASE and not re.search(r'[A-Z]', password):
        return False, "密码必须包含大写字母"
    
    if ValidationRules.PASSWORD_REQUIRE_LOWERCASE and not re.search(r'[a-z]', password):
        return False, "密码必须包含小写字母"
    
    if ValidationRules.PASSWORD_REQUIRE_DIGIT and not re.search(r'\d', password):
        return False, "密码必须包含数字"
    
    if ValidationRules.PASSWORD_REQUIRE_SPECIAL and not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        return False, "密码必须包含特殊字符"
    
    return True, ""

def validate_email(email: str) -> tuple[bool, str]:
    """验证邮箱格式"""
    if not email:
        return True, ""  # 邮箱可选
    
    if not re.match(ValidationRules.EMAIL_PATTERN, email):
        return False, "邮箱格式不正确"
    
    return True, ""

def validate_content(content: str, max_length: int = None) -> tuple[bool, str]:
    """验证内容"""
    max_len = max_length or ValidationRules.MAX_MESSAGE_LENGTH
    
    if not content:
        return False, "内容不能为空"
    
    if len(content) > max_len:
        return False, f"内容最多 {max_len} 个字符"
    
    return True, ""

# ========== 数据清洗 ==========
def sanitize_html(text: str) -> str:
    """清理 HTML 标签"""
    return html.escape(text)

def sanitize_sql(text: str) -> str:
    """清理 SQL 注入风险"""
    # 移除危险字符
    dangerous = ["'", '"', ";", "--", "/*", "*/", "xp_", "sp_"]
    result = text
    for d in dangerous:
        result = result.replace(d, "")
    return result

def sanitize_path(path: str) -> str:
    """清理路径遍历风险"""
    # 移除路径遍历字符
    path = path.replace("..", "")
    path = path.replace("//", "/")
    path = re.sub(r'[<>:"|?*]', '', path)
    return path

def sanitize_filename(filename: str) -> str:
    """清理文件名"""
    # 只保留安全字符
    safe = re.sub(r'[^a-zA-Z0-9._-]', '_', filename)
    # 移除开头的点
    safe = safe.lstrip('.')
    return safe[:255]  # 限制长度

def filter_sensitive_words(text: str, replacement: str = "***") -> str:
    """过滤敏感词"""
    result = text
    for word in ValidationRules.SENSITIVE_WORDS:
        result = re.sub(re.escape(word), replacement, result, flags=re.IGNORECASE)
    return result

# ========== XSS 防护 ==========
def prevent_xss(text: str) -> str:
    """防止 XSS 攻击"""
    # 转义 HTML 特殊字符
    text = html.escape(text)
    
    # 移除 JavaScript 协议
    text = re.sub(r'javascript:', '', text, flags=re.IGNORECASE)
    text = re.sub(r'vbscript:', '', text, flags=re.IGNORECASE)
    text = re.sub(r'data:', '', text, flags=re.IGNORECASE)
    
    # 移除事件处理器
    text = re.sub(r'on\w+\s*=', '', text, flags=re.IGNORECASE)
    
    return text

# ========== Pydantic 模型 ==========
class SafeUserRegister(BaseModel):
    """安全的用户注册模型"""
    username: str = Field(..., min_length=3, max_length=32)
    password: str = Field(..., min_length=8, max_length=128)
    email: Optional[str] = None
    
    @validator('username')
    def validate_username(cls, v):
        valid, msg = validate_username(v)
        if not valid:
            raise ValueError(msg)
        return v
    
    @validator('password')
    def validate_password(cls, v):
        valid, msg = validate_password(v)
        if not valid:
            raise ValueError(msg)
        return v
    
    @validator('email')
    def validate_email(cls, v):
        if v:
            valid, msg = validate_email(v)
            if not valid:
                raise ValueError(msg)
        return v

class SafeChatMessage(BaseModel):
    """安全的聊天消息模型"""
    role: str = Field(..., pattern=r'^(user|assistant|system)$')
    content: str = Field(..., max_length=100000)
    
    @validator('content')
    def sanitize_content(cls, v):
        # 基本清理，但保留 Markdown
        v = filter_sensitive_words(v)
        return v

class SafeNoteCreate(BaseModel):
    """安全的笔记创建模型"""
    title: str = Field(..., max_length=200)
    content: str = Field(..., max_length=100000)
    tags: Optional[str] = Field(default="", max_length=500)
    
    @validator('title')
    def sanitize_title(cls, v):
        return prevent_xss(v)
    
    @validator('tags')
    def sanitize_tags(cls, v):
        if v:
            return sanitize_html(v)
        return v

# ========== 验证装饰器 ==========
def validate_input(schema: type):
    """输入验证装饰器"""
    from functools import wraps
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # 查找请求体参数
            for key, value in kwargs.items():
                if isinstance(value, dict):
                    try:
                        validated = schema(**value)
                        kwargs[key] = validated.dict()
                    except Exception as e:
                        from fastapi import HTTPException
                        raise HTTPException(status_code=400, detail=str(e))
            return await func(*args, **kwargs)
        return wrapper
    return decorator

# ========== 批量验证 ==========
class BatchValidator:
    """批量验证器"""
    
    def __init__(self):
        self.errors: List[str] = []
    
    def validate(self, value: Any, validator_func, field_name: str) -> bool:
        """验证单个字段"""
        valid, msg = validator_func(value)
        if not valid:
            self.errors.append(f"{field_name}: {msg}")
        return valid
    
    def is_valid(self) -> bool:
        """检查是否全部通过"""
        return len(self.errors) == 0
    
    def get_errors(self) -> List[str]:
        """获取所有错误"""
        return self.errors
    
    def clear(self):
        """清除错误"""
        self.errors.clear()
