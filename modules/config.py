# ========== 配置管理模块 ==========
"""
支持：环境变量、配置文件、配置中心、热更新
"""
import os
import json
import yaml
from typing import Any, Dict, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
import logging
import asyncio
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class ConfigSource:
    """配置源"""
    name: str
    priority: int  # 优先级越高越优先
    data: Dict[str, Any] = field(default_factory=dict)
    last_updated: datetime = None


class ConfigManager:
    """配置管理器"""
    
    def __init__(self):
        self._sources: List[ConfigSource] = []
        self._cache: Dict[str, Any] = {}
        self._watchers: Dict[str, List[callable]] = {}
        self._last_hash: str = ""
    
    def add_source(self, name: str, priority: int, data: Dict[str, Any]):
        """添加配置源"""
        source = ConfigSource(
            name=name,
            priority=priority,
            data=data,
            last_updated=datetime.now()
        )
        self._sources.append(source)
        self._sources.sort(key=lambda x: x.priority, reverse=True)
        self._rebuild_cache()
        logger.info(f"Config source added: {name} (priority={priority})")
    
    def _rebuild_cache(self):
        """重建配置缓存"""
        old_hash = self._last_hash
        self._cache = {}
        
        # 按优先级从低到高合并（高优先级覆盖低优先级）
        for source in reversed(self._sources):
            self._deep_merge(self._cache, source.data)
        
        # 计算哈希检测变化
        self._last_hash = hashlib.md5(
            json.dumps(self._cache, sort_keys=True).encode()
        ).hexdigest()
        
        # 触发变更通知
        if old_hash and old_hash != self._last_hash:
            self._notify_watchers()
    
    def _deep_merge(self, base: dict, override: dict):
        """深度合并字典"""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值（支持点号分隔的路径）"""
        keys = key.split('.')
        value = self._cache
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any, source_name: str = "runtime"):
        """设置配置值"""
        # 查找或创建运行时配置源
        runtime_source = None
        for source in self._sources:
            if source.name == source_name:
                runtime_source = source
                break
        
        if not runtime_source:
            runtime_source = ConfigSource(
                name=source_name,
                priority=100,  # 运行时配置最高优先级
                data={},
                last_updated=datetime.now()
            )
            self._sources.append(runtime_source)
            self._sources.sort(key=lambda x: x.priority, reverse=True)
        
        # 设置值
        keys = key.split('.')
        data = runtime_source.data
        for k in keys[:-1]:
            if k not in data:
                data[k] = {}
            data = data[k]
        data[keys[-1]] = value
        
        runtime_source.last_updated = datetime.now()
        self._rebuild_cache()
    
    def watch(self, key: str, callback: callable):
        """监听配置变化"""
        if key not in self._watchers:
            self._watchers[key] = []
        self._watchers[key].append(callback)
    
    def _notify_watchers(self):
        """通知配置变化"""
        for key, callbacks in self._watchers.items():
            value = self.get(key)
            for callback in callbacks:
                try:
                    callback(key, value)
                except Exception as e:
                    logger.error(f"Config watcher error: {e}")
    
    def get_all(self) -> Dict[str, Any]:
        """获取所有配置"""
        return self._cache.copy()
    
    def get_sources(self) -> List[Dict]:
        """获取所有配置源信息"""
        return [
            {
                "name": s.name,
                "priority": s.priority,
                "keys_count": len(s.data),
                "last_updated": s.last_updated.isoformat() if s.last_updated else None
            }
            for s in self._sources
        ]


# 全局配置管理器
config = ConfigManager()


# ========== 配置加载器 ==========
def load_env_config() -> Dict[str, Any]:
    """从环境变量加载配置"""
    prefix = "AIHUB_"
    data = {}
    
    for key, value in os.environ.items():
        if key.startswith(prefix):
            config_key = key[len(prefix):].lower().replace('__', '.')
            
            # 尝试解析 JSON
            try:
                data[config_key] = json.loads(value)
            except:
                # 尝试解析布尔值
                if value.lower() in ('true', 'yes', '1'):
                    data[config_key] = True
                elif value.lower() in ('false', 'no', '0'):
                    data[config_key] = False
                else:
                    data[config_key] = value
    
    return data


def load_file_config(filepath: str) -> Dict[str, Any]:
    """从文件加载配置"""
    if not os.path.exists(filepath):
        return {}
    
    with open(filepath, 'r', encoding='utf-8') as f:
        if filepath.endswith('.json'):
            return json.load(f)
        elif filepath.endswith(('.yml', '.yaml')):
            return yaml.safe_load(f)
        else:
            # 尝试 JSON
            try:
                return json.load(f)
            except:
                return {}


def load_dotenv_config(filepath: str = ".env") -> Dict[str, Any]:
    """从 .env 文件加载配置"""
    if not os.path.exists(filepath):
        return {}
    
    data = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                data[key] = value
    
    return data


# ========== 配置中心客户端 ==========
class ConfigCenterClient:
    """配置中心客户端（支持 Consul、Nacos 等）"""
    
    def __init__(self, center_type: str = "consul", **kwargs):
        self.center_type = center_type
        self.config = kwargs
        self._client = None
    
    async def connect(self):
        """连接配置中心"""
        if self.center_type == "consul":
            await self._connect_consul()
        elif self.center_type == "nacos":
            await self._connect_nacos()
        elif self.center_type == "etcd":
            await self._connect_etcd()
    
    async def _connect_consul(self):
        """连接 Consul"""
        try:
            import consul.aio
            host = self.config.get("host", "localhost")
            port = self.config.get("port", 8500)
            self._client = consul.aio.Consul(host=host, port=port)
            logger.info(f"Connected to Consul at {host}:{port}")
        except ImportError:
            logger.warning("python-consul not installed")
    
    async def _connect_nacos(self):
        """连接 Nacos"""
        # Nacos 客户端实现
        pass
    
    async def _connect_etcd(self):
        """连接 etcd"""
        # etcd 客户端实现
        pass
    
    async def get_config(self, key: str) -> Optional[str]:
        """获取配置"""
        if not self._client:
            return None
        
        if self.center_type == "consul":
            index, data = await self._client.kv.get(key)
            if data:
                return data['Value'].decode('utf-8')
        
        return None
    
    async def set_config(self, key: str, value: str):
        """设置配置"""
        if not self._client:
            return
        
        if self.center_type == "consul":
            await self._client.kv.put(key, value)
    
    async def watch_config(self, key: str, callback: callable):
        """监听配置变化"""
        if self.center_type == "consul":
            index = None
            while True:
                try:
                    index, data = await self._client.kv.get(key, index=index)
                    if data:
                        value = data['Value'].decode('utf-8')
                        callback(key, value)
                except Exception as e:
                    logger.error(f"Config watch error: {e}")
                    await asyncio.sleep(5)


# ========== 初始化默认配置 ==========
def init_config():
    """初始化配置"""
    # 1. 加载 .env 文件（最低优先级）
    dotenv_config = load_dotenv_config()
    if dotenv_config:
        config.add_source("dotenv", 10, dotenv_config)
    
    # 2. 加载配置文件
    for filepath in ["config.json", "config.yml", "config.yaml"]:
        file_config = load_file_config(filepath)
        if file_config:
            config.add_source(f"file:{filepath}", 20, file_config)
    
    # 3. 加载环境变量（最高优先级）
    env_config = load_env_config()
    if env_config:
        config.add_source("env", 50, env_config)
    
    logger.info(f"Config initialized with {len(config._sources)} sources")


# 自动初始化
init_config()


# ========== 便捷函数 ==========
def get_config(key: str, default: Any = None) -> Any:
    """获取配置值"""
    return config.get(key, default)


def set_config(key: str, value: Any):
    """设置配置值"""
    config.set(key, value)
