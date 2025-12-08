# API Gateway

统一 API 网关，支持 55+ AI 平台。

## 支持平台

- **国际主流**: OpenAI, Claude, Gemini, Mistral, Groq, Together, Fireworks, Perplexity, OpenRouter, Grok 等
- **国内平台**: DeepSeek, 智谱GLM, 月之暗面, 百川, 零一万物, 阿里云百炼, 阶跃星辰, MiniMax, 豆包, 腾讯混元 等
- **API聚合**: 硅基流动, AiHubMix, 302.AI, TokenFlux 等
- **本地部署**: Ollama, LM Studio, GPUStack

## 功能

- 多平台统一接口
- 流式输出
- 多会话管理
- Markdown 渲染 + 代码高亮
- Token 统计
- 自定义 API 地址
- 导出对话 (JSON/Markdown)
- 暗色/亮色主题

## 部署

### Docker 部署

```bash
cp .env.example .env
# 编辑 .env 填入 API Key
docker-compose up -d
```

### 直接运行

```bash
pip install -r requirements.txt
cp .env.example .env
# 编辑 .env 填入 API Key
python main.py
```

访问 http://localhost:8000
