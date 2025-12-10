# 🤖 AI Hub - 企业级统一 AI 平台

<p align="center">
  <img src="https://img.shields.io/badge/version-2.0.0-blue.svg" alt="Version">
  <img src="https://img.shields.io/badge/python-3.9+-green.svg" alt="Python">
  <img src="https://img.shields.io/badge/license-MIT-orange.svg" alt="License">
</p>

一站式企业级 AI 对话平台，支持 10+ 模型服务商，包含完整的用户系统、团队协作、付费功能和运维监控。

## ✨ 核心功能

### 🎯 AI 能力
| 功能 | 描述 |
|------|------|
| 💬 多模型对话 | OpenAI、Claude、Gemini、DeepSeek、通义千问、智谱、Moonshot 等 |
| 🎨 图片生成 | DALL-E 3 AI 绘图 |
| 🔊 语音合成 | OpenAI TTS 文字转语音 |
| 🎤 语音输入 | Web Speech API 语音识别 |
| 📝 长文本处理 | 自动分块摘要 |
| 🧠 上下文压缩 | 智能压缩历史消息 |
| 🔧 Function Calling | 天气、计算、搜索、翻译等工具 |

### 👥 用户体验
| 功能 | 描述 |
|------|------|
| 📌 对话置顶/归档 | 管理重要对话 |
| 🌿 对话分支 | 从任意消息创建分支 |
| ⭐ 消息评分 | 1-5星评分系统 |
| ❤️ 消息收藏 | 收藏到全局记忆 |
| 💾 对话模板 | 保存常用对话 |
| 🧠 思维导图 | 对话转思维导图 |
| ▶️ 代码运行器 | Python/JavaScript 在线执行 |
| 📝 Markdown 编辑器 | 实时预览 |

### 🏢 企业级功能
| 功能 | 描述 |
|------|------|
| 🔐 SSO 单点登录 | SAML 集成框架 |
| 🐙 GitHub OAuth | 第三方登录 |
| 🔑 2FA 双因素认证 | TOTP 验证 |
| 👥 团队空间 | 多人协作 |
| 📊 用量配额 | 按用户/团队限制 |
| 🔑 API Token | 第三方访问 |
| 🛡️ 敏感词过滤 | 内容安全审核 |
| 🔒 数据脱敏 | 自动隐藏敏感信息 |

### 🔗 第三方集成
| 功能 | 描述 |
|------|------|
| 💬 Slack 集成 | Webhook 自动回复 |
| 🐦 飞书集成 | 机器人消息 |
| 🔔 Webhook 通知 | 事件推送 |
| 🌐 浏览器插件 | 划词翻译/摘要 |

### 📊 运维监控
| 功能 | 描述 |
|------|------|
| 📈 Prometheus 指标 | /metrics 端点 |
| 🔍 链路追踪 | 请求追踪分析，支持 Jaeger 导出 |
| 🖥️ 系统状态 | CPU/内存/磁盘监控 |
| 🎛️ 灰度发布 | A/B 测试支持 |
| 💾 自动备份 | 数据库备份 |
| 🗄️ Redis 缓存 | 分布式缓存（可选） |

### 📱 移动端支持
| 功能 | 描述 |
|------|------|
| 📲 PWA | 可安装桌面应用 |
| 🔔 推送通知 | 消息推送 |
| 📴 离线模式 | Service Worker 缓存 |
| 📱 响应式设计 | 移动端适配 |

## 🚀 快速开始

### 方式一：直接运行

```bash
# 克隆项目
git clone https://github.com/6666235/api-gateway.git
cd api-gateway

# 安装依赖
pip install -r requirements.txt

# 启动服务
python main.py
```

### 方式二：Docker 部署

```bash
# 构建镜像
docker build -t ai-hub .

# 运行容器
docker run -d -p 8000:8000 --name ai-hub ai-hub
```

### 方式三：Windows 启动脚本

双击 `start.bat` 即可启动

### 访问

打开浏览器访问 http://localhost:8000

## ⚙️ 配置

### 环境变量

编辑 `.env` 文件：

```env
# AI 模型 API Key
OPENAI_API_KEY=sk-xxx
ANTHROPIC_API_KEY=sk-ant-xxx
DEEPSEEK_API_KEY=sk-xxx
GOOGLE_API_KEY=xxx

# OAuth 登录（可选）
GITHUB_CLIENT_ID=xxx
GITHUB_CLIENT_SECRET=xxx

# 第三方集成（可选）
SLACK_BOT_TOKEN=xoxb-xxx
FEISHU_APP_ID=cli_xxx
FEISHU_APP_SECRET=xxx

# Redis 缓存（可选）
REDIS_URL=redis://localhost:6379/0

# Docker 沙箱（可选）
DOCKER_SANDBOX=true
```

### 界面配置

也可以在网页界面的「模型服务」中直接配置 API Key。

## 🔌 浏览器插件

支持 Chrome/Edge 浏览器插件：

1. 打开 `chrome://extensions/`
2. 开启「开发者模式」
3. 点击「加载已解压的扩展程序」
4. 选择 `extension` 文件夹

**插件功能：**
- 🌐 划词翻译
- 💡 智能解释
- 📝 快速摘要
- ❓ 智能问答
- 📄 页面摘要

## 📖 API 文档

启动服务后访问：
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### OpenAI 兼容 API

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

## 🛠️ 技术栈

| 类别 | 技术 |
|------|------|
| 后端 | FastAPI + SQLite/PostgreSQL |
| 前端 | 原生 HTML/CSS/JS |
| 实时通信 | WebSocket + SSE |
| 缓存 | Redis（可选） |
| 向量检索 | 本地/Pinecone/Milvus |
| 监控 | Prometheus + 自定义链路追踪 |
| 安全 | WAF + TOTP 2FA + API Key 加密 |
| 企业 | LDAP/AD + 多租户 + RBAC |

## 🆕 v2.0 新增功能

### 🔍 RAG 向量检索
```python
# 索引文档
POST /api/rag/index
{"kb_id": 1, "content": "文档内容...", "metadata": {"source": "file.pdf"}}

# 语义搜索
POST /api/rag/search
{"kb_id": 1, "query": "如何使用?", "top_k": 5}

# RAG 查询（返回上下文）
POST /api/rag/query
{"kb_id": 1, "question": "产品有什么功能?"}
```

### 🔐 RBAC 权限控制
```python
# 获取用户权限
GET /api/rbac/users/{user_id}/permissions

# 分配角色
POST /api/rbac/users/{user_id}/roles
{"role": "vip"}

# 检查权限
GET /api/rbac/check?permission=rag
```

### 💰 计费系统
```python
# 获取套餐
GET /api/billing/plans

# 订阅套餐
POST /api/billing/subscribe
{"plan_id": "pro", "payment_provider": "alipay"}

# 获取用量
GET /api/billing/usage?days=30
```

### 👥 实时协作
```python
# 创建协作会话
POST /api/collaboration/sessions
{"conversation_id": "conv_xxx"}

# WebSocket 连接
ws://localhost:8000/ws/collaboration/{session_id}?user_id=1&username=test
```

### 🛡️ 安全加固
- WAF 防护（SQL注入、XSS、命令注入检测）
- AI 攻击检测（Prompt Injection 防护）
- 密钥自动轮换
- 安全审计日志

### 🏢 企业功能
- LDAP/AD 集成
- 多租户支持
- 数据隔离
- GDPR 合规

## 📁 项目结构

```
api-gateway/
├── main.py              # 主程序
├── requirements.txt     # Python 依赖
├── Dockerfile          # Docker 配置
├── start.bat           # Windows 启动脚本
├── .env                # 环境变量
├── static/
│   ├── index.html      # 主页面
│   ├── share.html      # 分享页面
│   ├── manifest.json   # PWA 配置
│   └── sw.js           # Service Worker
└── extension/          # 浏览器插件
    ├── manifest.json
    ├── background.js
    ├── content.js
    ├── popup.html
    └── popup.js
```

## 🔑 快捷键

| 快捷键 | 功能 |
|--------|------|
| `Ctrl + N` | 新建对话 |
| `Ctrl + K` | 全局搜索 |
| `Ctrl + D` | 切换主题 |
| `Ctrl + H` | 系统状态 |
| `Ctrl + B` | 批量选择 |
| `Ctrl + F` | 搜索当前对话 |
| `Ctrl + /` | 快捷键帮助 |
| `Enter` | 发送消息 |
| `Shift + Enter` | 换行 |

## 📄 许可证

MIT License

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

<p align="center">
  Made with ❤️ by AI Hub Team
</p>