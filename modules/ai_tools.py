# ========== AI 工具增强模块 ==========
"""
Function Calling、Agent 工具、智能助手增强
"""
import os
import json
import httpx
import asyncio
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# ========== 工具注册表 ==========
class ToolRegistry:
    """工具注册表"""
    def __init__(self):
        self._tools: Dict[str, dict] = {}
    
    def register(
        self,
        name: str,
        description: str,
        parameters: dict,
        handler: Callable,
        category: str = "general"
    ):
        """注册工具"""
        self._tools[name] = {
            "name": name,
            "description": description,
            "parameters": parameters,
            "handler": handler,
            "category": category
        }
        logger.info(f"Tool registered: {name}")
    
    def get_tool(self, name: str) -> Optional[dict]:
        return self._tools.get(name)
    
    def get_all_tools(self) -> List[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["parameters"]
                }
            }
            for t in self._tools.values()
        ]
    
    def get_tools_by_category(self, category: str) -> List[dict]:
        return [
            t for t in self._tools.values()
            if t["category"] == category
        ]
    
    async def execute(self, name: str, arguments: dict) -> Any:
        """执行工具"""
        tool = self._tools.get(name)
        if not tool:
            raise ValueError(f"Unknown tool: {name}")
        
        handler = tool["handler"]
        if asyncio.iscoroutinefunction(handler):
            return await handler(**arguments)
        return handler(**arguments)

tool_registry = ToolRegistry()

# ========== 内置工具 ==========

async def get_weather(city: str) -> dict:
    """获取天气信息"""
    # 使用免费天气 API
    api_key = os.getenv("WEATHER_API_KEY", "")
    if not api_key:
        return {"error": "Weather API key not configured", "city": city, "weather": "未知"}
    
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"https://api.openweathermap.org/data/2.5/weather",
                params={"q": city, "appid": api_key, "units": "metric", "lang": "zh_cn"},
                timeout=10
            )
            data = resp.json()
            return {
                "city": city,
                "temperature": data["main"]["temp"],
                "feels_like": data["main"]["feels_like"],
                "humidity": data["main"]["humidity"],
                "weather": data["weather"][0]["description"],
                "wind_speed": data["wind"]["speed"]
            }
    except Exception as e:
        return {"error": str(e), "city": city}

async def web_search(query: str, num_results: int = 5) -> List[dict]:
    """网络搜索"""
    api_key = os.getenv("SERPER_API_KEY", "")
    if not api_key:
        return [{"error": "Search API key not configured"}]
    
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://google.serper.dev/search",
                headers={"X-API-KEY": api_key},
                json={"q": query, "num": num_results},
                timeout=10
            )
            data = resp.json()
            results = []
            for item in data.get("organic", [])[:num_results]:
                results.append({
                    "title": item.get("title"),
                    "link": item.get("link"),
                    "snippet": item.get("snippet")
                })
            return results
    except Exception as e:
        return [{"error": str(e)}]

def calculate(expression: str) -> dict:
    """安全计算器"""
    import ast
    import operator
    
    # 允许的操作符
    operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.Mod: operator.mod,
    }
    
    def eval_expr(node):
        if isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.BinOp):
            return operators[type(node.op)](eval_expr(node.left), eval_expr(node.right))
        elif isinstance(node, ast.UnaryOp):
            return operators[type(node.op)](eval_expr(node.operand))
        else:
            raise ValueError(f"Unsupported operation: {type(node)}")
    
    try:
        tree = ast.parse(expression, mode='eval')
        result = eval_expr(tree.body)
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"expression": expression, "error": str(e)}

async def translate_text(text: str, target_language: str = "en") -> dict:
    """文本翻译"""
    # 使用 DeepL 或其他翻译 API
    api_key = os.getenv("DEEPL_API_KEY", "")
    if not api_key:
        return {"error": "Translation API not configured", "text": text}
    
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://api-free.deepl.com/v2/translate",
                data={
                    "auth_key": api_key,
                    "text": text,
                    "target_lang": target_language.upper()
                },
                timeout=10
            )
            data = resp.json()
            return {
                "original": text,
                "translated": data["translations"][0]["text"],
                "target_language": target_language
            }
    except Exception as e:
        return {"error": str(e), "text": text}

def get_current_time(timezone: str = "Asia/Shanghai") -> dict:
    """获取当前时间"""
    from datetime import datetime
    try:
        import pytz
        tz = pytz.timezone(timezone)
        now = datetime.now(tz)
    except:
        now = datetime.now()
    
    return {
        "timezone": timezone,
        "datetime": now.isoformat(),
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "weekday": now.strftime("%A")
    }

async def fetch_url(url: str) -> dict:
    """获取网页内容"""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, timeout=10, follow_redirects=True)
            # 简单提取文本
            from html.parser import HTMLParser
            
            class TextExtractor(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self.text = []
                    self.skip = False
                
                def handle_starttag(self, tag, attrs):
                    if tag in ['script', 'style', 'nav', 'footer', 'header']:
                        self.skip = True
                
                def handle_endtag(self, tag):
                    if tag in ['script', 'style', 'nav', 'footer', 'header']:
                        self.skip = False
                
                def handle_data(self, data):
                    if not self.skip:
                        text = data.strip()
                        if text:
                            self.text.append(text)
            
            parser = TextExtractor()
            parser.feed(resp.text)
            content = " ".join(parser.text)[:5000]  # 限制长度
            
            return {
                "url": url,
                "status": resp.status_code,
                "content": content
            }
    except Exception as e:
        return {"url": url, "error": str(e)}

async def generate_image_prompt(description: str) -> dict:
    """生成图片提示词"""
    # 优化用户描述为更好的图片生成提示词
    prompt_template = f"""Based on this description: "{description}"
    
Generate an optimized image generation prompt that includes:
- Art style (photorealistic, digital art, oil painting, etc.)
- Lighting conditions
- Color palette
- Composition details
- Quality modifiers (4K, detailed, professional, etc.)

Return only the optimized prompt, nothing else."""
    
    return {
        "original": description,
        "optimized_prompt": f"A highly detailed, professional quality image of {description}, 4K resolution, dramatic lighting, vibrant colors"
    }

def code_execute(code: str, language: str = "python") -> dict:
    """执行代码（沙箱环境）"""
    if language != "python":
        return {"error": f"Language {language} not supported yet"}
    
    # 安全限制
    forbidden = ['import os', 'import sys', 'exec(', 'eval(', '__import__', 'open(', 'file(']
    for f in forbidden:
        if f in code:
            return {"error": f"Forbidden operation: {f}"}
    
    try:
        # 使用受限的全局变量
        safe_globals = {
            "__builtins__": {
                "print": print,
                "len": len,
                "range": range,
                "str": str,
                "int": int,
                "float": float,
                "list": list,
                "dict": dict,
                "sum": sum,
                "min": min,
                "max": max,
                "abs": abs,
                "round": round,
                "sorted": sorted,
                "enumerate": enumerate,
                "zip": zip,
                "map": map,
                "filter": filter,
            }
        }
        
        # 捕获输出
        import io
        import sys
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        
        try:
            exec(code, safe_globals)
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout
        
        return {"code": code, "output": output, "success": True}
    except Exception as e:
        return {"code": code, "error": str(e), "success": False}

# 注册内置工具
tool_registry.register(
    "get_weather",
    "获取指定城市的天气信息",
    {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "城市名称，如 Beijing, Shanghai"}
        },
        "required": ["city"]
    },
    get_weather,
    "utility"
)

tool_registry.register(
    "web_search",
    "搜索互联网获取最新信息",
    {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "搜索关键词"},
            "num_results": {"type": "integer", "description": "返回结果数量", "default": 5}
        },
        "required": ["query"]
    },
    web_search,
    "search"
)

tool_registry.register(
    "calculate",
    "执行数学计算",
    {
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "数学表达式，如 2+2, 3*4, 10/2"}
        },
        "required": ["expression"]
    },
    calculate,
    "utility"
)

tool_registry.register(
    "translate",
    "翻译文本到指定语言",
    {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "要翻译的文本"},
            "target_language": {"type": "string", "description": "目标语言代码，如 en, zh, ja, ko"}
        },
        "required": ["text", "target_language"]
    },
    translate_text,
    "language"
)

tool_registry.register(
    "get_time",
    "获取当前日期和时间",
    {
        "type": "object",
        "properties": {
            "timezone": {"type": "string", "description": "时区，如 Asia/Shanghai, America/New_York", "default": "Asia/Shanghai"}
        }
    },
    get_current_time,
    "utility"
)

tool_registry.register(
    "fetch_url",
    "获取网页内容",
    {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "要获取的网页URL"}
        },
        "required": ["url"]
    },
    fetch_url,
    "web"
)

tool_registry.register(
    "code_execute",
    "执行Python代码",
    {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "要执行的Python代码"},
            "language": {"type": "string", "description": "编程语言", "default": "python"}
        },
        "required": ["code"]
    },
    code_execute,
    "coding"
)

# ========== Function Calling 处理器 ==========
class FunctionCallingHandler:
    """Function Calling 处理器"""
    
    def __init__(self, registry: ToolRegistry):
        self.registry = registry
    
    def get_tools_for_model(self, model: str) -> List[dict]:
        """获取适合模型的工具定义"""
        return self.registry.get_all_tools()
    
    async def process_tool_calls(self, tool_calls: List[dict]) -> List[dict]:
        """处理工具调用"""
        results = []
        
        for call in tool_calls:
            tool_name = call.get("function", {}).get("name")
            arguments_str = call.get("function", {}).get("arguments", "{}")
            
            try:
                arguments = json.loads(arguments_str)
                result = await self.registry.execute(tool_name, arguments)
                results.append({
                    "tool_call_id": call.get("id"),
                    "role": "tool",
                    "content": json.dumps(result, ensure_ascii=False)
                })
            except Exception as e:
                results.append({
                    "tool_call_id": call.get("id"),
                    "role": "tool",
                    "content": json.dumps({"error": str(e)})
                })
        
        return results
    
    async def auto_execute(
        self,
        messages: List[dict],
        model_response: dict,
        max_iterations: int = 5
    ) -> dict:
        """自动执行工具调用并继续对话"""
        iterations = 0
        current_messages = messages.copy()
        
        while iterations < max_iterations:
            # 检查是否有工具调用
            tool_calls = model_response.get("choices", [{}])[0].get("message", {}).get("tool_calls")
            
            if not tool_calls:
                break
            
            # 添加助手消息
            current_messages.append(model_response["choices"][0]["message"])
            
            # 执行工具调用
            tool_results = await self.process_tool_calls(tool_calls)
            current_messages.extend(tool_results)
            
            # 继续对话（这里需要调用模型 API）
            # model_response = await call_model(current_messages)
            
            iterations += 1
        
        return {
            "messages": current_messages,
            "final_response": model_response,
            "iterations": iterations
        }

function_handler = FunctionCallingHandler(tool_registry)

# ========== 智能助手增强 ==========
class SmartAssistant:
    """智能助手增强功能"""
    
    def __init__(self):
        self._context_memory: Dict[str, List[dict]] = {}
        self._user_preferences: Dict[str, dict] = {}
    
    def add_context(self, user_id: str, context: dict):
        """添加上下文"""
        if user_id not in self._context_memory:
            self._context_memory[user_id] = []
        self._context_memory[user_id].append({
            "timestamp": datetime.now().isoformat(),
            **context
        })
        # 保留最近20条
        self._context_memory[user_id] = self._context_memory[user_id][-20:]
    
    def get_context(self, user_id: str) -> List[dict]:
        """获取上下文"""
        return self._context_memory.get(user_id, [])
    
    def set_preference(self, user_id: str, key: str, value: Any):
        """设置用户偏好"""
        if user_id not in self._user_preferences:
            self._user_preferences[user_id] = {}
        self._user_preferences[user_id][key] = value
    
    def get_preferences(self, user_id: str) -> dict:
        """获取用户偏好"""
        return self._user_preferences.get(user_id, {})
    
    def build_system_prompt(self, user_id: str, base_prompt: str = "") -> str:
        """构建增强的系统提示"""
        prefs = self.get_preferences(user_id)
        context = self.get_context(user_id)
        
        prompt_parts = [base_prompt] if base_prompt else []
        
        # 添加用户偏好
        if prefs:
            prompt_parts.append(f"\n用户偏好: {json.dumps(prefs, ensure_ascii=False)}")
        
        # 添加最近上下文摘要
        if context:
            recent = context[-5:]
            context_summary = "\n".join([
                f"- {c.get('type', 'info')}: {c.get('content', '')[:100]}"
                for c in recent
            ])
            prompt_parts.append(f"\n最近上下文:\n{context_summary}")
        
        return "\n".join(prompt_parts)

smart_assistant = SmartAssistant()
