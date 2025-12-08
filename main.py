from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
import httpx
import os
import json
from typing import Optional, AsyncGenerator

load_dotenv()

app = FastAPI(title="API Gateway", description="统一 API 网关")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROVIDERS = {
    # 国际主流
    "openai": {"base_url": "https://api.openai.com/v1", "env_key": "OPENAI_API_KEY", "type": "openai"},
    "azure_openai": {"base_url": "", "env_key": "AZURE_OPENAI_API_KEY", "type": "openai"},
    "claude": {"base_url": "https://api.anthropic.com/v1", "env_key": "ANTHROPIC_API_KEY", "type": "claude"},
    "gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta", "env_key": "GOOGLE_API_KEY", "type": "gemini"},
    "vertex_ai": {"base_url": "", "env_key": "VERTEX_API_KEY", "type": "openai"},
    "mistral": {"base_url": "https://api.mistral.ai/v1", "env_key": "MISTRAL_API_KEY", "type": "openai"},
    "groq": {"base_url": "https://api.groq.com/openai/v1", "env_key": "GROQ_API_KEY", "type": "openai"},
    "together": {"base_url": "https://api.together.xyz/v1", "env_key": "TOGETHER_API_KEY", "type": "openai"},
    "fireworks": {"base_url": "https://api.fireworks.ai/inference/v1", "env_key": "FIREWORKS_API_KEY", "type": "openai"},
    "perplexity": {"base_url": "https://api.perplexity.ai", "env_key": "PERPLEXITY_API_KEY", "type": "openai"},
    "openrouter": {"base_url": "https://openrouter.ai/api/v1", "env_key": "OPENROUTER_API_KEY", "type": "openai"},
    "grok": {"base_url": "https://api.x.ai/v1", "env_key": "XAI_API_KEY", "type": "openai"},
    "hyperbolic": {"base_url": "https://api.hyperbolic.xyz/v1", "env_key": "HYPERBOLIC_API_KEY", "type": "openai"},
    "nvidia": {"base_url": "https://integrate.api.nvidia.com/v1", "env_key": "NVIDIA_API_KEY", "type": "openai"},
    "github_models": {"base_url": "https://models.inference.ai.azure.com", "env_key": "GITHUB_TOKEN", "type": "openai"},
    "github_copilot": {"base_url": "https://api.githubcopilot.com", "env_key": "GITHUB_COPILOT_TOKEN", "type": "openai"},
    "aws_bedrock": {"base_url": "", "env_key": "AWS_BEDROCK_API_KEY", "type": "openai"},
    "jina": {"base_url": "https://api.jina.ai/v1", "env_key": "JINA_API_KEY", "type": "openai"},
    "voyage": {"base_url": "https://api.voyageai.com/v1", "env_key": "VOYAGE_API_KEY", "type": "openai"},
    "poe": {"base_url": "https://api.poe.com/bot", "env_key": "POE_API_KEY", "type": "openai"},
    # 国内平台
    "deepseek": {"base_url": "https://api.deepseek.com/v1", "env_key": "DEEPSEEK_API_KEY", "type": "openai"},
    "zhipu": {"base_url": "https://open.bigmodel.cn/api/paas/v4", "env_key": "ZHIPU_API_KEY", "type": "openai"},
    "moonshot": {"base_url": "https://api.moonshot.cn/v1", "env_key": "MOONSHOT_API_KEY", "type": "openai"},
    "baichuan": {"base_url": "https://api.baichuan-ai.com/v1", "env_key": "BAICHUAN_API_KEY", "type": "openai"},
    "yi": {"base_url": "https://api.lingyiwanwu.com/v1", "env_key": "YI_API_KEY", "type": "openai"},
    "qwen": {"base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "env_key": "QWEN_API_KEY", "type": "openai"},
    "stepfun": {"base_url": "https://api.stepfun.com/v1", "env_key": "STEPFUN_API_KEY", "type": "openai"},
    "minimax": {"base_url": "https://api.minimax.chat/v1", "env_key": "MINIMAX_API_KEY", "type": "openai"},
    "doubao": {"base_url": "https://ark.cn-beijing.volces.com/api/v3", "env_key": "DOUBAO_API_KEY", "type": "openai"},
    "hunyuan": {"base_url": "https://api.hunyuan.cloud.tencent.com/v1", "env_key": "HUNYUAN_API_KEY", "type": "openai"},
    "siliconflow": {"base_url": "https://api.siliconflow.cn/v1", "env_key": "SILICONFLOW_API_KEY", "type": "openai"},
    "aihubmix": {"base_url": "https://aihubmix.com/v1", "env_key": "AIHUBMIX_API_KEY", "type": "openai"},
    "ocoolai": {"base_url": "https://api.ocoolai.com/v1", "env_key": "OCOOLAI_API_KEY", "type": "openai"},
    "alaya": {"base_url": "https://api.alayanew.com/v1", "env_key": "ALAYA_API_KEY", "type": "openai"},
    "dmxapi": {"base_url": "https://api.dmxapi.com/v1", "env_key": "DMXAPI_API_KEY", "type": "openai"},
    "aionly": {"base_url": "https://api.aionly.me/v1", "env_key": "AIONLY_API_KEY", "type": "openai"},
    "burncloud": {"base_url": "https://api.burncloud.com/v1", "env_key": "BURNCLOUD_API_KEY", "type": "openai"},
    "tokenflux": {"base_url": "https://api.tokenflux.ai/v1", "env_key": "TOKENFLUX_API_KEY", "type": "openai"},
    "ai302": {"base_url": "https://api.302.ai/v1", "env_key": "AI302_API_KEY", "type": "openai"},
    "cephalon": {"base_url": "https://api.cephalon.cloud/v1", "env_key": "CEPHALON_API_KEY", "type": "openai"},
    "lanfun": {"base_url": "https://api.lanfun.com/v1", "env_key": "LANFUN_API_KEY", "type": "openai"},
    "ph8": {"base_url": "https://api.ph8.ai/v1", "env_key": "PH8_API_KEY", "type": "openai"},
    "ppio": {"base_url": "https://api.ppio.cloud/v1", "env_key": "PPIO_API_KEY", "type": "openai"},
    "qiniu": {"base_url": "https://api.qiniu.com/v1", "env_key": "QINIU_API_KEY", "type": "openai"},
    "intel_ovms": {"base_url": "", "env_key": "INTEL_OVMS_API_KEY", "type": "openai"},
    "tianyi": {"base_url": "https://api.ctyun.cn/v1", "env_key": "TIANYI_API_KEY", "type": "openai"},
    "tencent_ti": {"base_url": "https://api.ti.tencent.com/v1", "env_key": "TENCENT_TI_API_KEY", "type": "openai"},
    "baidu_qianfan": {"base_url": "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop", "env_key": "BAIDU_API_KEY", "type": "openai"},
    "gpustack": {"base_url": "http://localhost:8080/v1", "env_key": "GPUSTACK_API_KEY", "type": "openai"},
    "wuxinqiong": {"base_url": "https://api.wuxinqiong.com/v1", "env_key": "WUXINQIONG_API_KEY", "type": "openai"},
    "longcat": {"base_url": "https://api.longcat.ai/v1", "env_key": "LONGCAT_API_KEY", "type": "openai"},
    # 本地/自托管
    "ollama": {"base_url": "http://localhost:11434/v1", "env_key": "OLLAMA_API_KEY", "type": "openai"},
    "lmstudio": {"base_url": "http://localhost:1234/v1", "env_key": "LMSTUDIO_API_KEY", "type": "openai"},
}

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    provider: str
    model: str
    messages: list[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 2048
    stream: Optional[bool] = False
    custom_url: Optional[str] = None
    api_key: Optional[str] = None

@app.get("/")
async def root():
    return FileResponse("static/index.html")

@app.get("/providers")
async def list_providers():
    return {"providers": list(PROVIDERS.keys())}


async def stream_openai(client: httpx.AsyncClient, url: str, headers: dict, payload: dict) -> AsyncGenerator:
    async with client.stream("POST", url, headers=headers, json=payload) as response:
        async for line in response.aiter_lines():
            if line.startswith("data: "):
                data = line[6:]
                if data == "[DONE]":
                    yield "data: [DONE]\n\n"
                    break
                yield f"data: {data}\n\n"

async def stream_claude(client: httpx.AsyncClient, url: str, headers: dict, payload: dict) -> AsyncGenerator:
    async with client.stream("POST", url, headers=headers, json=payload) as response:
        async for line in response.aiter_lines():
            if line.startswith("data: "):
                data = line[6:]
                try:
                    parsed = json.loads(data)
                    if parsed.get("type") == "content_block_delta":
                        text = parsed.get("delta", {}).get("text", "")
                        chunk = {"choices": [{"delta": {"content": text}, "index": 0}]}
                        yield f"data: {json.dumps(chunk)}\n\n"
                    elif parsed.get("type") == "message_stop":
                        yield "data: [DONE]\n\n"
                except:
                    pass

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest, authorization: Optional[str] = Header(None)):
    provider = request.provider.lower()
    
    if provider not in PROVIDERS:
        raise HTTPException(status_code=400, detail=f"不支持的平台: {provider}")
    
    config = PROVIDERS[provider]
    api_key = request.api_key or os.getenv(config["env_key"])
    base_url = request.custom_url or config["base_url"]
    
    if not api_key:
        raise HTTPException(status_code=500, detail=f"未配置 {provider} 的 API Key，请在网页上输入或在 .env 文件中设置")
    
    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    
    provider_type = config.get("type", "openai")
    
    async with httpx.AsyncClient(timeout=120) as client:
        try:
            if provider_type == "openai":
                url = f"{base_url}/chat/completions"
                headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                payload = {
                    "model": request.model,
                    "messages": messages,
                    "temperature": request.temperature,
                    "max_tokens": request.max_tokens,
                    "stream": request.stream,
                }
                if request.stream:
                    return StreamingResponse(stream_openai(client, url, headers, payload), media_type="text/event-stream")
                response = await client.post(url, headers=headers, json=payload)
                
            elif provider_type == "claude":
                url = f"{base_url}/messages"
                headers = {
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                }
                payload = {
                    "model": request.model,
                    "messages": messages,
                    "max_tokens": request.max_tokens,
                    "stream": request.stream,
                }
                if request.stream:
                    return StreamingResponse(stream_claude(client, url, headers, payload), media_type="text/event-stream")
                response = await client.post(url, headers=headers, json=payload)
                
            elif provider_type == "gemini":
                url = f"{base_url}/models/{request.model}:generateContent?key={api_key}"
                payload = {
                    "contents": [{"role": "user" if m.role == "user" else "model", "parts": [{"text": m.content}]} for m in request.messages],
                }
                response = await client.post(url, json=payload)
            
            response.raise_for_status()
            return response.json()
            
        except httpx.HTTPStatusError as e:
            error_detail = f"API 请求失败 ({e.response.status_code})"
            try:
                err_json = e.response.json()
                error_detail = err_json.get("error", {}).get("message", str(err_json))
            except:
                error_detail = e.response.text[:200] if e.response.text else str(e)
            raise HTTPException(status_code=e.response.status_code, detail=error_detail)
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="请求超时，请稍后重试")
        except httpx.ConnectError:
            raise HTTPException(status_code=502, detail="无法连接到 API 服务器，请检查网络或自定义地址")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"内部错误: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
