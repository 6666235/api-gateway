# ========== API 文档增强模块 ==========
"""
OpenAPI 文档增强、API 版本管理、变更日志
"""
from typing import Dict, List, Optional, Any
from datetime import datetime
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
import logging

logger = logging.getLogger(__name__)


# ========== API 版本管理 ==========
class APIVersion:
    """API 版本"""

    def __init__(
        self,
        version: str,
        release_date: str,
        status: str = "stable",
        deprecated: bool = False,
        sunset_date: str = None,
    ):
        self.version = version
        self.release_date = release_date
        self.status = status  # stable, beta, deprecated
        self.deprecated = deprecated
        self.sunset_date = sunset_date


class APIVersionManager:
    """API 版本管理器"""

    def __init__(self):
        self.versions: Dict[str, APIVersion] = {}
        self.current_version = "v1"

    def register_version(self, version: APIVersion):
        """注册 API 版本"""
        self.versions[version.version] = version

    def get_version(self, version: str) -> Optional[APIVersion]:
        """获取版本信息"""
        return self.versions.get(version)

    def get_all_versions(self) -> List[Dict]:
        """获取所有版本"""
        return [
            {
                "version": v.version,
                "release_date": v.release_date,
                "status": v.status,
                "deprecated": v.deprecated,
                "sunset_date": v.sunset_date,
            }
            for v in self.versions.values()
        ]

    def is_deprecated(self, version: str) -> bool:
        """检查版本是否已废弃"""
        v = self.versions.get(version)
        return v.deprecated if v else False


version_manager = APIVersionManager()

# 注册默认版本
version_manager.register_version(
    APIVersion("v1", "2024-01-01", "stable")
)


# ========== API 变更日志 ==========
class ChangelogEntry:
    """变更日志条目"""

    def __init__(
        self,
        version: str,
        date: str,
        changes: List[Dict[str, str]],
        breaking: bool = False,
    ):
        self.version = version
        self.date = date
        self.changes = changes  # [{"type": "added/changed/fixed/removed", "description": "..."}]
        self.breaking = breaking


class APIChangelog:
    """API 变更日志"""

    def __init__(self):
        self.entries: List[ChangelogEntry] = []

    def add_entry(self, entry: ChangelogEntry):
        """添加变更日志"""
        self.entries.append(entry)
        self.entries.sort(key=lambda x: x.date, reverse=True)

    def get_changelog(self, limit: int = 10) -> List[Dict]:
        """获取变更日志"""
        return [
            {
                "version": e.version,
                "date": e.date,
                "changes": e.changes,
                "breaking": e.breaking,
            }
            for e in self.entries[:limit]
        ]

    def get_breaking_changes(self) -> List[Dict]:
        """获取破坏性变更"""
        return [
            {"version": e.version, "date": e.date, "changes": e.changes}
            for e in self.entries
            if e.breaking
        ]


changelog = APIChangelog()

# 添加示例变更日志
changelog.add_entry(
    ChangelogEntry(
        "1.0.0",
        "2024-01-01",
        [
            {"type": "added", "description": "初始版本发布"},
            {"type": "added", "description": "支持多模型对话"},
            {"type": "added", "description": "用户认证系统"},
        ],
    )
)


# ========== OpenAPI 文档增强 ==========
def custom_openapi(app: FastAPI) -> Dict:
    """自定义 OpenAPI 文档"""
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="AI Hub API",
        version="1.0.0",
        description="""
## AI Hub - 统一 AI 平台 API

### 功能特性
- 🤖 多模型对话（OpenAI、Claude、Gemini 等）
- 📝 笔记管理
- 🧠 全局记忆
- 🔍 RAG 向量检索
- 👥 团队协作
- 📊 使用统计

### 认证方式
- Bearer Token: 在请求头中添加 `Authorization: Bearer <token>`
- API Key: 在请求头中添加 `X-API-Key: <key>`

### 速率限制
- 免费用户: 60 请求/分钟
- 付费用户: 300 请求/分钟
- 企业用户: 无限制

### 错误码
| 状态码 | 说明 |
|--------|------|
| 400 | 请求参数错误 |
| 401 | 未授权 |
| 403 | 禁止访问 |
| 404 | 资源不存在 |
| 429 | 请求过于频繁 |
| 500 | 服务器错误 |
        """,
        routes=app.routes,
        tags=[
            {"name": "认证", "description": "用户认证相关接口"},
            {"name": "对话", "description": "AI 对话相关接口"},
            {"name": "笔记", "description": "笔记管理接口"},
            {"name": "记忆", "description": "全局记忆接口"},
            {"name": "设置", "description": "用户设置接口"},
            {"name": "统计", "description": "使用统计接口"},
            {"name": "RBAC", "description": "权限管理接口"},
            {"name": "计费", "description": "订阅计费接口"},
            {"name": "RAG", "description": "向量检索接口"},
            {"name": "协作", "description": "实时协作接口"},
            {"name": "系统", "description": "系统管理接口"},
        ],
    )

    # 添加服务器信息
    openapi_schema["servers"] = [
        {"url": "/", "description": "当前服务器"},
        {"url": "http://localhost:8000", "description": "本地开发"},
    ]

    # 添加安全方案
    openapi_schema["components"]["securitySchemes"] = {
        "bearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
        },
        "apiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
        },
    }

    # 添加外部文档链接
    openapi_schema["externalDocs"] = {
        "description": "完整文档",
        "url": "https://github.com/your-repo/ai-hub",
    }

    app.openapi_schema = openapi_schema
    return app.openapi_schema


def setup_api_docs(app: FastAPI):
    """设置 API 文档"""
    app.openapi = lambda: custom_openapi(app)

    # 添加版本和变更日志端点
    @app.get("/api/versions", tags=["系统"])
    async def get_api_versions():
        """获取 API 版本列表"""
        return {
            "current": version_manager.current_version,
            "versions": version_manager.get_all_versions(),
        }

    @app.get("/api/changelog", tags=["系统"])
    async def get_api_changelog(limit: int = 10):
        """获取 API 变更日志"""
        return {
            "changelog": changelog.get_changelog(limit),
            "breaking_changes": changelog.get_breaking_changes(),
        }

    logger.info("API documentation configured")


# ========== API 响应模型 ==========
from pydantic import BaseModel, Field


class APIResponse(BaseModel):
    """标准 API 响应"""

    success: bool = True
    data: Any = None
    message: str = ""
    error: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class PaginatedResponse(BaseModel):
    """分页响应"""

    items: List[Any]
    total: int
    page: int
    page_size: int
    total_pages: int


class ErrorResponse(BaseModel):
    """错误响应"""

    error: str
    message: str
    details: Optional[Dict] = None
    request_id: Optional[str] = None


# ========== 响应工具函数 ==========
def success_response(data: Any = None, message: str = "") -> Dict:
    """成功响应"""
    return {
        "success": True,
        "data": data,
        "message": message,
        "timestamp": datetime.now().isoformat(),
    }


def error_response(
    error: str, message: str, details: Dict = None, request_id: str = None
) -> Dict:
    """错误响应"""
    return {
        "success": False,
        "error": error,
        "message": message,
        "details": details,
        "request_id": request_id,
        "timestamp": datetime.now().isoformat(),
    }


def paginate(
    items: List, total: int, page: int = 1, page_size: int = 20
) -> Dict:
    """分页响应"""
    total_pages = (total + page_size - 1) // page_size
    return {
        "items": items,
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages,
        "has_next": page < total_pages,
        "has_prev": page > 1,
    }
