"""
Pytest 配置和共享 fixtures
"""
import pytest
import asyncio
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session")
def event_loop():
    """创建事件循环"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_db_path(tmp_path_factory):
    """创建测试数据库路径"""
    return str(tmp_path_factory.mktemp("data") / "test.db")


@pytest.fixture
def mock_user():
    """模拟用户"""
    return {
        "id": 1,
        "username": "test_user",
        "email": "test@example.com",
        "plan": "free",
        "tokens_used": 0,
        "tokens_limit": 10000
    }


@pytest.fixture
def mock_admin_user():
    """模拟管理员用户"""
    return {
        "id": 1,
        "username": "admin",
        "email": "admin@example.com",
        "plan": "unlimited",
        "tokens_used": 0,
        "tokens_limit": -1
    }


@pytest.fixture
def sample_messages():
    """示例消息列表"""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi! How can I help you today?"}
    ]


@pytest.fixture
def sample_conversation():
    """示例对话"""
    return {
        "id": "test-conv-123",
        "title": "Test Conversation",
        "provider": "openai",
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
    }


# 配置 pytest-asyncio
def pytest_configure(config):
    """配置 pytest"""
    config.addinivalue_line(
        "markers", "asyncio: mark test as async"
    )
