# AI Hub - 统一 AI 对话平台

一站式 AI 对话平台，支持多种模型服务，包含完整的用户系统和付费功能。

## 功能特性

- 💬 **多模型对话** - 支持 OpenAI、Claude、DeepSeek、通义千问等多种模型
- 🔌 **模型服务管理** - 自定义添加和管理 API 服务
- 📝 **笔记功能** - 记录和管理你的笔记
- 🧠 **全局记忆** - AI 会记住你的偏好和信息
- ⚡ **快捷短语** - 创建常用提示词模板
- 📄 **文档处理** - 上传文档让 AI 分析
- 💳 **付费系统** - 完整的套餐和支付功能
- 🔐 **用户系统** - 注册登录，数据云端同步

## 快速开始

### 方式一：直接运行
```bash
# 安装依赖
pip install -r requirements.txt

# 启动服务
python main.py
```

### 方式二：使用启动脚本 (Windows)
双击 `start.bat`

### 访问
打开浏览器访问 http://localhost:8000

## 配置 API Key

编辑 `.env` 文件，填入你的 API Key：

```env
OPENAI_API_KEY=sk-xxx
ANTHROPIC_API_KEY=sk-ant-xxx
DEEPSEEK_API_KEY=sk-xxx
```

或者在网页界面的「模型服务」中直接配置。

## 技术栈

- 后端：FastAPI + SQLite
- 前端：原生 HTML/CSS/JS
- 支持流式输出 (SSE)
