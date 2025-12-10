"""
安全加固模块
支持：WAF防护、密钥轮换、安全审计、AI攻击检测
"""
import re
import time
import hashlib
import secrets
import sqlite3
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from collections import defaultdict
from cryptography.fernet import Fernet
import asyncio

DB_PATH = "data.db"

# ========== WAF 防护 ==========
class WAFRule:
    """WAF 规则"""
    def __init__(self, name: str, pattern: str, action: str = "block"):
        self.name = name
        self.pattern = re.compile(pattern, re.IGNORECASE)
        self.action = action  # block, log, sanitize


class WebApplicationFirewall:
    """Web 应用防火墙"""
    
    DEFAULT_RULES = [
        # SQL 注入
        WAFRule("sql_injection", r"(\b(union|select|insert|update|delete|drop|truncate)\b.*\b(from|into|table|where)\b)", "block"),
        WAFRule("sql_injection_2", r"('|\")\s*(or|and)\s*('|\"|1|true)", "block"),
        
        # XSS 攻击
        WAFRule("xss_script", r"<script[^>]*>.*?</script>", "sanitize"),
        WAFRule("xss_event", r"\bon\w+\s*=", "sanitize"),
        WAFRule("xss_javascript", r"javascript:", "block"),
        
        # 路径遍历
        WAFRule("path_traversal", r"\.\./|\.\.\\", "block"),
        
        # 命令注入
        WAFRule("cmd_injection", r"[;&|`$]|\b(cat|ls|rm|wget|curl|bash|sh|nc)\b", "block"),
        
        # SSRF
        WAFRule("ssrf_localhost", r"(localhost|127\.0\.0\.1|0\.0\.0\.0|::1)", "log"),
        WAFRule("ssrf_internal", r"(10\.\d+\.\d+\.\d+|172\.(1[6-9]|2\d|3[01])\.\d+\.\d+|192\.168\.\d+\.\d+)", "log"),
    ]
    
    def __init__(self):
        self.rules = self.DEFAULT_RULES.copy()
        self.blocked_ips: Set[str] = set()
        self.ip_violations: Dict[str, List[float]] = defaultdict(list)
        self.violation_threshold = 10  # 10次违规后封禁
        self.violation_window = 300  # 5分钟窗口
    
    def check(self, content: str, ip: str = None) -> Dict:
        """检查内容是否违规"""
        violations = []
        sanitized = content
        
        for rule in self.rules:
            if rule.pattern.search(content):
                violations.append({
                    "rule": rule.name,
                    "action": rule.action
                })
                
                if rule.action == "sanitize":
                    sanitized = rule.pattern.sub("[FILTERED]", sanitized)
        
        # 记录 IP 违规
        if ip and violations:
            self._record_violation(ip)
            if self._should_block_ip(ip):
                self.blocked_ips.add(ip)
        
        should_block = any(v["action"] == "block" for v in violations)
        
        return {
            "blocked": should_block,
            "violations": violations,
            "sanitized": sanitized,
            "ip_blocked": ip in self.blocked_ips if ip else False
        }
    
    def _record_violation(self, ip: str):
        """记录违规"""
        now = time.time()
        self.ip_violations[ip].append(now)
        # 清理过期记录
        self.ip_violations[ip] = [
            t for t in self.ip_violations[ip] 
            if now - t < self.violation_window
        ]
    
    def _should_block_ip(self, ip: str) -> bool:
        """判断是否应该封禁 IP"""
        return len(self.ip_violations.get(ip, [])) >= self.violation_threshold
    
    def is_ip_blocked(self, ip: str) -> bool:
        """检查 IP 是否被封禁"""
        return ip in self.blocked_ips
    
    def unblock_ip(self, ip: str):
        """解封 IP"""
        self.blocked_ips.discard(ip)
        self.ip_violations.pop(ip, None)
    
    def add_rule(self, name: str, pattern: str, action: str = "block"):
        """添加规则"""
        self.rules.append(WAFRule(name, pattern, action))

# ========== AI 攻击检测 ==========
class AIAttackDetector:
    """AI 攻击检测器（Prompt Injection 等）"""
    
    INJECTION_PATTERNS = [
        # 角色扮演攻击
        r"ignore (previous|all|above) (instructions|prompts)",
        r"forget (everything|all|your) (you|instructions)",
        r"you are now",
        r"pretend (to be|you are)",
        r"act as",
        r"roleplay as",
        
        # 系统提示泄露
        r"(show|reveal|display|print|output) (your|the|system) (prompt|instructions)",
        r"what (are|is) your (system|initial) (prompt|instructions)",
        
        # 越狱尝试
        r"DAN mode",
        r"jailbreak",
        r"bypass (safety|content|filter)",
        r"disable (safety|content|filter)",
        
        # 数据提取
        r"(list|show|display) all (users|data|information)",
        r"(export|dump) (database|data)",
    ]
    
    def __init__(self):
        self.patterns = [re.compile(p, re.IGNORECASE) for p in self.INJECTION_PATTERNS]
        self.suspicious_users: Dict[int, int] = defaultdict(int)
    
    def detect(self, content: str, user_id: int = None) -> Dict:
        """检测 AI 攻击"""
        threats = []
        risk_score = 0
        
        for i, pattern in enumerate(self.patterns):
            if pattern.search(content):
                threats.append({
                    "type": "prompt_injection",
                    "pattern": self.INJECTION_PATTERNS[i],
                    "severity": "high" if i < 6 else "medium"
                })
                risk_score += 30 if i < 6 else 15
        
        # 检测异常长度
        if len(content) > 10000:
            threats.append({"type": "excessive_length", "severity": "low"})
            risk_score += 10
        
        # 检测重复模式
        if self._has_repetition(content):
            threats.append({"type": "repetition_attack", "severity": "medium"})
            risk_score += 20
        
        # 记录可疑用户
        if user_id and threats:
            self.suspicious_users[user_id] += len(threats)
        
        return {
            "is_attack": risk_score >= 30,
            "risk_score": min(100, risk_score),
            "threats": threats,
            "user_suspicious_count": self.suspicious_users.get(user_id, 0) if user_id else 0
        }
    
    def _has_repetition(self, content: str, threshold: int = 5) -> bool:
        """检测重复模式"""
        words = content.lower().split()
        if len(words) < 10:
            return False
        
        word_counts = defaultdict(int)
        for word in words:
            word_counts[word] += 1
        
        max_count = max(word_counts.values())
        return max_count > len(words) * 0.3  # 超过30%重复


# ========== 密钥轮换 ==========
class KeyRotationManager:
    """密钥轮换管理器"""
    
    def __init__(self):
        self._init_db()
        self.current_key = self._get_current_key()
        self.cipher = Fernet(self.current_key.encode()) if self.current_key else None
    
    def _init_db(self):
        """初始化密钥表"""
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS encryption_keys (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key_value TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    is_active INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP,
                    rotated_at TIMESTAMP
                )
            ''')
            
            # 确保有一个活跃密钥
            active = conn.execute(
                "SELECT key_value FROM encryption_keys WHERE is_active = 1"
            ).fetchone()
            
            if not active:
                new_key = Fernet.generate_key().decode()
                conn.execute("""
                    INSERT INTO encryption_keys (key_value, version, is_active, expires_at)
                    VALUES (?, 1, 1, datetime('now', '+90 days'))
                """, (new_key,))
    
    def _get_current_key(self) -> Optional[str]:
        """获取当前活跃密钥"""
        with sqlite3.connect(DB_PATH) as conn:
            row = conn.execute(
                "SELECT key_value FROM encryption_keys WHERE is_active = 1 ORDER BY version DESC LIMIT 1"
            ).fetchone()
            return row[0] if row else None
    
    def rotate_key(self) -> str:
        """轮换密钥"""
        new_key = Fernet.generate_key().decode()
        
        with sqlite3.connect(DB_PATH) as conn:
            # 获取当前版本
            current = conn.execute(
                "SELECT version FROM encryption_keys WHERE is_active = 1 ORDER BY version DESC LIMIT 1"
            ).fetchone()
            new_version = (current[0] + 1) if current else 1
            
            # 停用旧密钥
            conn.execute("""
                UPDATE encryption_keys 
                SET is_active = 0, rotated_at = CURRENT_TIMESTAMP
                WHERE is_active = 1
            """)
            
            # 创建新密钥
            conn.execute("""
                INSERT INTO encryption_keys (key_value, version, is_active, expires_at)
                VALUES (?, ?, 1, datetime('now', '+90 days'))
            """, (new_key, new_version))
        
        self.current_key = new_key
        self.cipher = Fernet(new_key.encode())
        
        return new_key
    
    def encrypt(self, data: str) -> str:
        """加密数据"""
        if not self.cipher:
            return data
        return self.cipher.encrypt(data.encode()).decode()
    
    def decrypt(self, encrypted: str) -> str:
        """解密数据（尝试所有密钥）"""
        if not encrypted:
            return encrypted
        
        with sqlite3.connect(DB_PATH) as conn:
            keys = conn.execute(
                "SELECT key_value FROM encryption_keys ORDER BY version DESC"
            ).fetchall()
        
        for (key,) in keys:
            try:
                cipher = Fernet(key.encode())
                return cipher.decrypt(encrypted.encode()).decode()
            except:
                continue
        
        return encrypted  # 返回原文（可能未加密）
    
    def check_expiration(self) -> Dict:
        """检查密钥是否即将过期"""
        with sqlite3.connect(DB_PATH) as conn:
            row = conn.execute("""
                SELECT expires_at, version FROM encryption_keys 
                WHERE is_active = 1 ORDER BY version DESC LIMIT 1
            """).fetchone()
            
            if not row:
                return {"needs_rotation": True, "reason": "no_active_key"}
            
            expires_at = datetime.fromisoformat(row[0])
            days_until_expiry = (expires_at - datetime.now()).days
            
            return {
                "needs_rotation": days_until_expiry < 7,
                "days_until_expiry": days_until_expiry,
                "current_version": row[1]
            }

# ========== 安全审计 ==========
class SecurityAuditor:
    """安全审计器"""
    
    def __init__(self):
        self._init_db()
    
    def _init_db(self):
        """初始化审计表"""
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS security_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    severity TEXT DEFAULT 'info',
                    user_id INTEGER,
                    ip_address TEXT,
                    user_agent TEXT,
                    resource TEXT,
                    action TEXT,
                    details TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_security_events_type ON security_events(event_type)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_security_events_user ON security_events(user_id)')
    
    def log_event(self, event_type: str, severity: str = "info", 
                  user_id: int = None, ip: str = None, 
                  user_agent: str = None, resource: str = None,
                  action: str = None, details: Dict = None):
        """记录安全事件"""
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                INSERT INTO security_events 
                (event_type, severity, user_id, ip_address, user_agent, resource, action, details)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (event_type, severity, user_id, ip, user_agent, resource, action,
                  json.dumps(details) if details else None))
    
    def get_events(self, event_type: str = None, severity: str = None,
                   user_id: int = None, days: int = 7, limit: int = 100) -> List[Dict]:
        """获取安全事件"""
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            
            query = "SELECT * FROM security_events WHERE created_at >= datetime('now', ?)"
            params = [f'-{days} days']
            
            if event_type:
                query += " AND event_type = ?"
                params.append(event_type)
            if severity:
                query += " AND severity = ?"
                params.append(severity)
            if user_id:
                query += " AND user_id = ?"
                params.append(user_id)
            
            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)
            
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]
    
    def get_summary(self, days: int = 7) -> Dict:
        """获取安全事件汇总"""
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            
            # 按类型统计
            by_type = conn.execute("""
                SELECT event_type, COUNT(*) as count
                FROM security_events
                WHERE created_at >= datetime('now', ?)
                GROUP BY event_type ORDER BY count DESC
            """, (f'-{days} days',)).fetchall()
            
            # 按严重程度统计
            by_severity = conn.execute("""
                SELECT severity, COUNT(*) as count
                FROM security_events
                WHERE created_at >= datetime('now', ?)
                GROUP BY severity
            """, (f'-{days} days',)).fetchall()
            
            # 高危事件
            critical = conn.execute("""
                SELECT COUNT(*) as count
                FROM security_events
                WHERE severity IN ('critical', 'high')
                AND created_at >= datetime('now', ?)
            """, (f'-{days} days',)).fetchone()
            
            return {
                "by_type": [dict(r) for r in by_type],
                "by_severity": [dict(r) for r in by_severity],
                "critical_count": critical["count"],
                "period_days": days
            }
    
    def generate_report(self, days: int = 30) -> Dict:
        """生成安全审计报告"""
        summary = self.get_summary(days)
        events = self.get_events(days=days, limit=500)
        
        # 分析趋势
        with sqlite3.connect(DB_PATH) as conn:
            daily = conn.execute("""
                SELECT date(created_at) as day, COUNT(*) as count
                FROM security_events
                WHERE created_at >= datetime('now', ?)
                GROUP BY date(created_at) ORDER BY day
            """, (f'-{days} days',)).fetchall()
        
        return {
            "generated_at": datetime.now().isoformat(),
            "period_days": days,
            "summary": summary,
            "daily_trend": [{"day": r[0], "count": r[1]} for r in daily],
            "recent_critical": [e for e in events if e.get("severity") in ("critical", "high")][:20],
            "recommendations": self._generate_recommendations(summary, events)
        }
    
    def _generate_recommendations(self, summary: Dict, events: List[Dict]) -> List[str]:
        """生成安全建议"""
        recommendations = []
        
        if summary["critical_count"] > 10:
            recommendations.append("检测到大量高危事件，建议立即审查安全策略")
        
        # 检查常见攻击
        attack_types = {e.get("event_type") for e in events}
        if "sql_injection" in attack_types:
            recommendations.append("检测到 SQL 注入尝试，建议加强输入验证")
        if "xss_attack" in attack_types:
            recommendations.append("检测到 XSS 攻击尝试，建议启用 CSP 策略")
        if "brute_force" in attack_types:
            recommendations.append("检测到暴力破解尝试，建议启用账户锁定策略")
        
        if not recommendations:
            recommendations.append("当前安全状况良好，建议继续保持监控")
        
        return recommendations

# 全局实例
waf = WebApplicationFirewall()
ai_detector = AIAttackDetector()
key_manager = KeyRotationManager()
auditor = SecurityAuditor()
