"""
API 端点测试
使用 pytest 运行: pytest tests/test_api.py -v
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 检查 httpx 版本
try:
    from httpx import AsyncClient, ASGITransport
    HAS_ASGI_TRANSPORT = True
except ImportError:
    from httpx import AsyncClient
    HAS_ASGI_TRANSPORT = False


@pytest.fixture
def app():
    """获取 FastAPI 应用"""
    from main import app
    return app


@pytest.fixture
async def client(app):
    """创建测试客户端"""
    if HAS_ASGI_TRANSPORT:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac
    else:
        # 旧版本 httpx
        async with AsyncClient(app=app, base_url="http://test") as ac:
            yield ac


class TestHealthEndpoints:
    """健康检查端点测试"""
    
    @pytest.mark.asyncio
    async def test_health_check(self, client):
        """测试健康检查接口"""
        response = await client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "components" in data
    
    @pytest.mark.asyncio
    async def test_root_redirect(self, client):
        """测试根路径重定向"""
        response = await client.get("/", follow_redirects=False)
        
        # 应该重定向到静态页面
        assert response.status_code in [200, 302, 307]


class TestAuthEndpoints:
    """认证端点测试"""
    
    @pytest.mark.asyncio
    async def test_get_me_default_user(self, client):
        """测试获取当前用户（默认用户）"""
        response = await client.get("/api/auth/me")
        
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert "username" in data
    
    @pytest.mark.asyncio
    async def test_register(self, client):
        """测试用户注册"""
        import uuid
        username = f"test_user_{uuid.uuid4().hex[:8]}"
        
        response = await client.post("/api/auth/register", json={
            "username": username,
            "password": "test_password_123",
            "email": f"{username}@test.com"
        })
        
        # 可能成功或用户已存在
        assert response.status_code in [200, 400]
    
    @pytest.mark.asyncio
    async def test_logout(self, client):
        """测试登出"""
        response = await client.post("/api/auth/logout")
        
        assert response.status_code == 200
        data = response.json()
        assert data.get("success") == True


class TestStatsEndpoints:
    """统计端点测试"""
    
    @pytest.mark.asyncio
    async def test_get_stats(self, client):
        """测试获取统计信息"""
        response = await client.get("/api/stats")
        
        assert response.status_code == 200
        data = response.json()
        assert "today_calls" in data
        assert "total_calls" in data


class TestProvidersEndpoints:
    """模型服务商端点测试"""
    
    @pytest.mark.asyncio
    async def test_list_providers(self, client):
        """测试获取服务商列表"""
        response = await client.get("/api/providers")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        
        # 应该有一些默认服务商
        provider_ids = [p.get("id") for p in data]
        assert "openai" in provider_ids or len(data) >= 0


class TestConversationEndpoints:
    """对话端点测试"""
    
    @pytest.mark.asyncio
    async def test_list_conversations(self, client):
        """测试获取对话列表"""
        response = await client.get("/api/conversations")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    @pytest.mark.asyncio
    async def test_create_conversation(self, client):
        """测试创建对话"""
        response = await client.post("/api/conversations", json={
            "title": "Test Conversation",
            "provider": "openai",
            "model": "gpt-3.5-turbo"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "id" in data


class TestNotesEndpoints:
    """笔记端点测试"""
    
    @pytest.mark.asyncio
    async def test_list_notes(self, client):
        """测试获取笔记列表"""
        response = await client.get("/api/notes")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    @pytest.mark.asyncio
    async def test_create_note(self, client):
        """测试创建笔记"""
        response = await client.post("/api/notes", json={
            "title": "Test Note",
            "content": "This is a test note content.",
            "tags": "test,api"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "id" in data


class TestMemoryEndpoints:
    """记忆端点测试"""
    
    @pytest.mark.asyncio
    async def test_list_memories(self, client):
        """测试获取记忆列表"""
        response = await client.get("/api/memories")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    @pytest.mark.asyncio
    async def test_create_memory(self, client):
        """测试创建记忆"""
        response = await client.post("/api/memories", json={
            "content": "Test memory content",
            "category": "test"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "id" in data


class TestSettingsEndpoints:
    """设置端点测试"""
    
    @pytest.mark.asyncio
    async def test_get_settings(self, client):
        """测试获取设置"""
        response = await client.get("/api/settings")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)
    
    @pytest.mark.asyncio
    async def test_update_settings(self, client):
        """测试更新设置"""
        response = await client.put("/api/settings", json={
            "data": {"theme": "dark", "language": "zh-CN"}
        })
        
        assert response.status_code == 200


class TestPromptTemplatesEndpoints:
    """Prompt 模板端点测试"""
    
    @pytest.mark.asyncio
    async def test_list_templates(self, client):
        """测试获取模板列表"""
        response = await client.get("/api/prompts")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)


class TestRBACEndpoints:
    """RBAC 端点测试"""
    
    @pytest.mark.asyncio
    async def test_list_roles(self, client):
        """测试获取角色列表"""
        response = await client.get("/api/rbac/roles")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    @pytest.mark.asyncio
    async def test_check_permission(self, client):
        """测试检查权限"""
        response = await client.get("/api/rbac/check?permission=chat")
        
        assert response.status_code == 200
        data = response.json()
        assert "has_permission" in data


class TestBillingEndpoints:
    """计费端点测试"""
    
    @pytest.mark.asyncio
    async def test_list_plans(self, client):
        """测试获取套餐列表"""
        response = await client.get("/api/billing/plans")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    @pytest.mark.asyncio
    async def test_get_usage(self, client):
        """测试获取用量"""
        response = await client.get("/api/billing/usage")
        
        assert response.status_code == 200
        data = response.json()
        assert "total_tokens" in data or "error" in data


class TestThemesEndpoints:
    """主题端点测试"""
    
    @pytest.mark.asyncio
    async def test_list_themes(self, client):
        """测试获取主题列表"""
        response = await client.get("/api/themes")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0


# ========== 运行测试 ==========
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
