"""
计费系统模块
支持：套餐管理、用量计费、支付集成（支付宝/微信/Stripe）
"""
import sqlite3
import json
import hashlib
import hmac
import time
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import httpx
import os

DB_PATH = "data.db"

class PaymentStatus(str, Enum):
    PENDING = "pending"
    PAID = "paid"
    FAILED = "failed"
    REFUNDED = "refunded"
    CANCELLED = "cancelled"

class SubscriptionStatus(str, Enum):
    ACTIVE = "active"
    EXPIRED = "expired"
    CANCELLED = "cancelled"
    SUSPENDED = "suspended"


@dataclass
class Plan:
    id: str
    name: str
    price: float
    tokens_limit: int
    features: List[str]
    billing_cycle: str = "monthly"  # monthly, yearly
    is_active: bool = True

@dataclass
class UsageRecord:
    user_id: int
    type: str  # tokens, api_calls, image_gen, etc.
    amount: int
    unit_price: float
    total_cost: float
    timestamp: datetime

class BillingManager:
    """计费管理器"""
    
    # 默认套餐
    DEFAULT_PLANS = {
        "free": Plan("free", "免费版", 0, 10000, ["chat", "notes"], "monthly"),
        "basic": Plan("basic", "基础版", 19.9, 100000, 
                     ["chat", "notes", "memory", "shortcuts"], "monthly"),
        "pro": Plan("pro", "专业版", 49.9, 500000,
                   ["chat", "notes", "memory", "shortcuts", "image", "code", "rag"], "monthly"),
        "enterprise": Plan("enterprise", "企业版", 199.9, -1, ["*"], "monthly"),
    }
    
    # Token 单价（每1000 tokens）
    TOKEN_PRICES = {
        "gpt-4o": 0.01,
        "gpt-4o-mini": 0.001,
        "gpt-4-turbo": 0.03,
        "claude-3-opus": 0.075,
        "claude-3-sonnet": 0.015,
        "deepseek-chat": 0.001,
        "default": 0.002,
    }
    
    def __init__(self):
        self._init_db()
    
    def _init_db(self):
        """初始化计费相关表"""
        with sqlite3.connect(DB_PATH) as conn:
            conn.executescript('''
                CREATE TABLE IF NOT EXISTS billing_plans (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    price REAL,
                    tokens_limit INTEGER,
                    features TEXT DEFAULT '[]',
                    billing_cycle TEXT DEFAULT 'monthly',
                    is_active INTEGER DEFAULT 1
                );
                
                CREATE TABLE IF NOT EXISTS subscriptions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER UNIQUE,
                    plan_id TEXT,
                    status TEXT DEFAULT 'active',
                    start_date TIMESTAMP,
                    end_date TIMESTAMP,
                    auto_renew INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS invoices (
                    id TEXT PRIMARY KEY,
                    user_id INTEGER,
                    subscription_id INTEGER,
                    amount REAL,
                    currency TEXT DEFAULT 'CNY',
                    status TEXT DEFAULT 'pending',
                    payment_method TEXT,
                    payment_id TEXT,
                    paid_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS usage_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    type TEXT,
                    model TEXT,
                    amount INTEGER,
                    unit_price REAL,
                    total_cost REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS payment_transactions (
                    id TEXT PRIMARY KEY,
                    user_id INTEGER,
                    invoice_id TEXT,
                    amount REAL,
                    currency TEXT DEFAULT 'CNY',
                    provider TEXT,
                    provider_tx_id TEXT,
                    status TEXT DEFAULT 'pending',
                    metadata TEXT DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE INDEX IF NOT EXISTS idx_usage_user ON usage_records(user_id);
                CREATE INDEX IF NOT EXISTS idx_usage_date ON usage_records(created_at);
                CREATE INDEX IF NOT EXISTS idx_invoices_user ON invoices(user_id);
            ''')
            
            # 初始化默认套餐
            for plan in self.DEFAULT_PLANS.values():
                conn.execute("""
                    INSERT OR IGNORE INTO billing_plans 
                    (id, name, price, tokens_limit, features, billing_cycle)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (plan.id, plan.name, plan.price, plan.tokens_limit,
                      json.dumps(plan.features), plan.billing_cycle))

    
    def record_usage(self, user_id: int, usage_type: str, amount: int,
                    model: str = None) -> float:
        """记录用量"""
        unit_price = self.TOKEN_PRICES.get(model, self.TOKEN_PRICES["default"])
        total_cost = (amount / 1000) * unit_price
        
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                INSERT INTO usage_records (user_id, type, model, amount, unit_price, total_cost)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (user_id, usage_type, model, amount, unit_price, total_cost))
            
            # 更新用户已用额度
            conn.execute("""
                UPDATE users SET tokens_used = tokens_used + ? WHERE id = ?
            """, (amount, user_id))
        
        return total_cost
    
    def get_usage_summary(self, user_id: int, days: int = 30) -> Dict:
        """获取用量汇总"""
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            
            # 总用量
            total = conn.execute("""
                SELECT SUM(amount) as tokens, SUM(total_cost) as cost
                FROM usage_records 
                WHERE user_id = ? AND created_at >= date('now', ?)
            """, (user_id, f'-{days} days')).fetchone()
            
            # 按模型统计
            by_model = conn.execute("""
                SELECT model, SUM(amount) as tokens, SUM(total_cost) as cost
                FROM usage_records
                WHERE user_id = ? AND created_at >= date('now', ?)
                GROUP BY model ORDER BY tokens DESC
            """, (user_id, f'-{days} days')).fetchall()
            
            # 按日期统计
            daily = conn.execute("""
                SELECT date(created_at) as day, SUM(amount) as tokens
                FROM usage_records
                WHERE user_id = ? AND created_at >= date('now', ?)
                GROUP BY date(created_at) ORDER BY day
            """, (user_id, f'-{days} days')).fetchall()
            
            return {
                "total_tokens": total["tokens"] or 0,
                "total_cost": round(total["cost"] or 0, 4),
                "by_model": [dict(r) for r in by_model],
                "daily": [dict(r) for r in daily]
            }
    
    def get_subscription(self, user_id: int) -> Optional[Dict]:
        """获取用户订阅"""
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("""
                SELECT s.*, p.name as plan_name, p.price, p.tokens_limit, p.features
                FROM subscriptions s
                JOIN billing_plans p ON s.plan_id = p.id
                WHERE s.user_id = ?
            """, (user_id,)).fetchone()
            
            if row:
                return dict(row)
        return None
    
    def create_subscription(self, user_id: int, plan_id: str) -> Dict:
        """创建订阅"""
        plan = self.DEFAULT_PLANS.get(plan_id)
        if not plan:
            raise ValueError(f"无效的套餐: {plan_id}")
        
        start_date = datetime.now()
        end_date = start_date + timedelta(days=30 if plan.billing_cycle == "monthly" else 365)
        
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO subscriptions 
                (user_id, plan_id, status, start_date, end_date)
                VALUES (?, ?, 'active', ?, ?)
            """, (user_id, plan_id, start_date, end_date))
            
            # 更新用户套餐
            conn.execute("""
                UPDATE users SET plan = ?, tokens_limit = ?, expires_at = ?
                WHERE id = ?
            """, (plan_id, plan.tokens_limit, end_date, user_id))
        
        return {
            "plan_id": plan_id,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat()
        }
    
    def create_invoice(self, user_id: int, plan_id: str) -> Dict:
        """创建发票/订单"""
        plan = self.DEFAULT_PLANS.get(plan_id)
        if not plan:
            raise ValueError(f"无效的套餐: {plan_id}")
        
        invoice_id = f"INV{int(time.time())}{secrets.token_hex(4).upper()}"
        
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                INSERT INTO invoices (id, user_id, amount, status)
                VALUES (?, ?, ?, 'pending')
            """, (invoice_id, user_id, plan.price))
        
        return {
            "invoice_id": invoice_id,
            "amount": plan.price,
            "plan": plan_id,
            "status": "pending"
        }


class AlipayProvider:
    """支付宝支付"""
    def __init__(self):
        self.app_id = os.getenv("ALIPAY_APP_ID", "")
        self.private_key = os.getenv("ALIPAY_PRIVATE_KEY", "")
        self.public_key = os.getenv("ALIPAY_PUBLIC_KEY", "")
        self.gateway = "https://openapi.alipay.com/gateway.do"
    
    async def create_payment(self, invoice_id: str, amount: float, 
                            subject: str) -> Dict:
        """创建支付宝支付"""
        if not self.app_id:
            return {"error": "支付宝未配置", "mock": True,
                    "qr_code": f"https://example.com/pay/alipay/{invoice_id}"}
        
        params = {
            "app_id": self.app_id,
            "method": "alipay.trade.precreate",
            "charset": "utf-8",
            "sign_type": "RSA2",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "version": "1.0",
            "biz_content": json.dumps({
                "out_trade_no": invoice_id,
                "total_amount": str(amount),
                "subject": subject
            })
        }
        
        # 实际项目中需要签名
        async with httpx.AsyncClient() as client:
            response = await client.post(self.gateway, data=params)
            return response.json()
    
    def verify_callback(self, params: Dict) -> bool:
        """验证支付宝回调"""
        # 实际项目中需要验签
        return True

class WechatPayProvider:
    """微信支付"""
    def __init__(self):
        self.app_id = os.getenv("WECHAT_APP_ID", "")
        self.mch_id = os.getenv("WECHAT_MCH_ID", "")
        self.api_key = os.getenv("WECHAT_API_KEY", "")
        self.api_v3_key = os.getenv("WECHAT_API_V3_KEY", "")
    
    async def create_payment(self, invoice_id: str, amount: float,
                            description: str) -> Dict:
        """创建微信支付（Native）"""
        if not self.app_id:
            return {"error": "微信支付未配置", "mock": True,
                    "qr_code": f"https://example.com/pay/wechat/{invoice_id}"}
        
        url = "https://api.mch.weixin.qq.com/v3/pay/transactions/native"
        
        data = {
            "appid": self.app_id,
            "mchid": self.mch_id,
            "description": description,
            "out_trade_no": invoice_id,
            "notify_url": os.getenv("WECHAT_NOTIFY_URL", ""),
            "amount": {"total": int(amount * 100), "currency": "CNY"}
        }
        
        # 实际项目中需要签名
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=data)
            return response.json()
    
    def verify_callback(self, headers: Dict, body: str) -> bool:
        """验证微信支付回调"""
        return True

class StripeProvider:
    """Stripe 支付"""
    def __init__(self):
        self.secret_key = os.getenv("STRIPE_SECRET_KEY", "")
        self.webhook_secret = os.getenv("STRIPE_WEBHOOK_SECRET", "")
    
    async def create_payment(self, invoice_id: str, amount: float,
                            currency: str = "usd") -> Dict:
        """创建 Stripe 支付"""
        if not self.secret_key:
            return {"error": "Stripe 未配置", "mock": True,
                    "url": f"https://example.com/pay/stripe/{invoice_id}"}
        
        url = "https://api.stripe.com/v1/checkout/sessions"
        
        data = {
            "payment_method_types[]": "card",
            "line_items[0][price_data][currency]": currency,
            "line_items[0][price_data][product_data][name]": "AI Hub 订阅",
            "line_items[0][price_data][unit_amount]": int(amount * 100),
            "line_items[0][quantity]": 1,
            "mode": "payment",
            "success_url": os.getenv("STRIPE_SUCCESS_URL", ""),
            "cancel_url": os.getenv("STRIPE_CANCEL_URL", ""),
            "client_reference_id": invoice_id
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                data=data,
                auth=(self.secret_key, "")
            )
            return response.json()
    
    def verify_webhook(self, payload: str, signature: str) -> bool:
        """验证 Stripe Webhook"""
        if not self.webhook_secret:
            return True
        
        try:
            timestamp, sig = signature.split(",")[0].split("=")[1], signature.split(",")[1].split("=")[1]
            expected = hmac.new(
                self.webhook_secret.encode(),
                f"{timestamp}.{payload}".encode(),
                hashlib.sha256
            ).hexdigest()
            return hmac.compare_digest(expected, sig)
        except:
            return False

class PaymentGateway:
    """统一支付网关"""
    def __init__(self):
        self.alipay = AlipayProvider()
        self.wechat = WechatPayProvider()
        self.stripe = StripeProvider()
        self.billing = BillingManager()
    
    async def create_payment(self, user_id: int, plan_id: str,
                            provider: str = "alipay") -> Dict:
        """创建支付"""
        invoice = self.billing.create_invoice(user_id, plan_id)
        invoice_id = invoice["invoice_id"]
        amount = invoice["amount"]
        
        if provider == "alipay":
            result = await self.alipay.create_payment(invoice_id, amount, f"AI Hub {plan_id}套餐")
        elif provider == "wechat":
            result = await self.wechat.create_payment(invoice_id, amount, f"AI Hub {plan_id}套餐")
        elif provider == "stripe":
            result = await self.stripe.create_payment(invoice_id, amount)
        else:
            return {"error": f"不支持的支付方式: {provider}"}
        
        # 记录交易
        tx_id = f"TX{int(time.time())}{secrets.token_hex(4)}"
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                INSERT INTO payment_transactions 
                (id, user_id, invoice_id, amount, provider, status, metadata)
                VALUES (?, ?, ?, ?, ?, 'pending', ?)
            """, (tx_id, user_id, invoice_id, amount, provider, json.dumps(result)))
        
        return {
            "transaction_id": tx_id,
            "invoice_id": invoice_id,
            "amount": amount,
            "provider": provider,
            **result
        }
    
    async def handle_callback(self, provider: str, data: Dict) -> bool:
        """处理支付回调"""
        if provider == "alipay":
            invoice_id = data.get("out_trade_no")
            status = "paid" if data.get("trade_status") == "TRADE_SUCCESS" else "failed"
        elif provider == "wechat":
            invoice_id = data.get("out_trade_no")
            status = "paid" if data.get("trade_state") == "SUCCESS" else "failed"
        elif provider == "stripe":
            invoice_id = data.get("client_reference_id")
            status = "paid" if data.get("payment_status") == "paid" else "failed"
        else:
            return False
        
        with sqlite3.connect(DB_PATH) as conn:
            # 更新发票状态
            conn.execute("""
                UPDATE invoices SET status = ?, paid_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (status, invoice_id))
            
            # 更新交易状态
            conn.execute("""
                UPDATE payment_transactions SET status = ?, updated_at = CURRENT_TIMESTAMP
                WHERE invoice_id = ?
            """, (status, invoice_id))
            
            if status == "paid":
                # 获取用户和套餐信息
                invoice = conn.execute(
                    "SELECT user_id FROM invoices WHERE id = ?", (invoice_id,)
                ).fetchone()
                
                if invoice:
                    # 激活订阅（这里简化处理）
                    self.billing.create_subscription(invoice[0], "pro")
        
        return status == "paid"

# 全局实例
billing = BillingManager()
payment_gateway = PaymentGateway()
