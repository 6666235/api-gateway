# ========== 数据导出导入模块 ==========
"""
支持：JSON、CSV、Excel 导出，数据备份恢复
"""
import os
import json
import csv
import io
import sqlite3
import zipfile
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

DB_PATH = os.getenv("DATABASE_PATH", "data.db")


@dataclass
class ExportConfig:
    """导出配置"""

    format: str = "json"  # json, csv, xlsx
    include_metadata: bool = True
    compress: bool = False
    encrypt: bool = False


class DataExporter:
    """数据导出器"""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or DB_PATH

    def export_conversations(
        self, user_id: int, config: ExportConfig = None
    ) -> bytes:
        """导出用户对话"""
        config = config or ExportConfig()

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            # 获取对话
            conversations = conn.execute(
                """
                SELECT * FROM conversations 
                WHERE user_id = ? 
                ORDER BY updated_at DESC
            """,
                (user_id,),
            ).fetchall()

            data = []
            for conv in conversations:
                conv_data = dict(conv)

                # 获取消息
                messages = conn.execute(
                    """
                    SELECT role, content, created_at 
                    FROM messages 
                    WHERE conversation_id = ?
                    ORDER BY created_at
                """,
                    (conv["id"],),
                ).fetchall()

                conv_data["messages"] = [dict(m) for m in messages]
                data.append(conv_data)

        return self._format_output(data, config, "conversations")

    def export_notes(self, user_id: int, config: ExportConfig = None) -> bytes:
        """导出用户笔记"""
        config = config or ExportConfig()

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            notes = conn.execute(
                """
                SELECT * FROM notes 
                WHERE user_id = ? 
                ORDER BY updated_at DESC
            """,
                (user_id,),
            ).fetchall()

            data = [dict(n) for n in notes]

        return self._format_output(data, config, "notes")

    def export_memories(self, user_id: int, config: ExportConfig = None) -> bytes:
        """导出用户记忆"""
        config = config or ExportConfig()

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            memories = conn.execute(
                """
                SELECT * FROM memories 
                WHERE user_id = ? 
                ORDER BY created_at DESC
            """,
                (user_id,),
            ).fetchall()

            data = [dict(m) for m in memories]

        return self._format_output(data, config, "memories")

    def export_all_user_data(
        self, user_id: int, config: ExportConfig = None
    ) -> bytes:
        """导出用户所有数据（GDPR 合规）"""
        config = config or ExportConfig()

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            # 用户信息
            user = conn.execute(
                "SELECT id, username, email, plan, created_at FROM users WHERE id = ?",
                (user_id,),
            ).fetchone()

            # 对话
            conversations = conn.execute(
                "SELECT * FROM conversations WHERE user_id = ?", (user_id,)
            ).fetchall()

            # 消息
            messages = []
            for conv in conversations:
                msgs = conn.execute(
                    "SELECT * FROM messages WHERE conversation_id = ?",
                    (conv["id"],),
                ).fetchall()
                messages.extend([dict(m) for m in msgs])

            # 笔记
            notes = conn.execute(
                "SELECT * FROM notes WHERE user_id = ?", (user_id,)
            ).fetchall()

            # 记忆
            memories = conn.execute(
                "SELECT * FROM memories WHERE user_id = ?", (user_id,)
            ).fetchall()

            # 设置
            settings = conn.execute(
                "SELECT * FROM settings WHERE user_id = ?", (user_id,)
            ).fetchone()

            # API 日志
            api_logs = conn.execute(
                """
                SELECT * FROM api_logs 
                WHERE user_id = ? 
                ORDER BY created_at DESC 
                LIMIT 1000
            """,
                (user_id,),
            ).fetchall()

        data = {
            "export_date": datetime.now().isoformat(),
            "user": dict(user) if user else None,
            "conversations": [dict(c) for c in conversations],
            "messages": messages,
            "notes": [dict(n) for n in notes],
            "memories": [dict(m) for m in memories],
            "settings": dict(settings) if settings else None,
            "api_logs": [dict(l) for l in api_logs],
        }

        if config.format == "json":
            content = json.dumps(data, ensure_ascii=False, indent=2, default=str)
            return content.encode("utf-8")
        else:
            # 创建 ZIP 包含多个文件
            return self._create_zip_export(data)

    def _format_output(
        self, data: List[Dict], config: ExportConfig, name: str
    ) -> bytes:
        """格式化输出"""
        if config.include_metadata:
            output = {
                "export_date": datetime.now().isoformat(),
                "count": len(data),
                "data": data,
            }
        else:
            output = data

        if config.format == "json":
            content = json.dumps(output, ensure_ascii=False, indent=2, default=str)
            result = content.encode("utf-8")

        elif config.format == "csv":
            result = self._to_csv(data)

        else:
            result = json.dumps(output, ensure_ascii=False, default=str).encode(
                "utf-8"
            )

        if config.compress:
            import gzip

            result = gzip.compress(result)

        return result

    def _to_csv(self, data: List[Dict]) -> bytes:
        """转换为 CSV"""
        if not data:
            return b""

        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=data[0].keys())
        writer.writeheader()

        for row in data:
            # 处理嵌套数据
            flat_row = {}
            for k, v in row.items():
                if isinstance(v, (dict, list)):
                    flat_row[k] = json.dumps(v, ensure_ascii=False)
                else:
                    flat_row[k] = v
            writer.writerow(flat_row)

        return output.getvalue().encode("utf-8")

    def _create_zip_export(self, data: Dict) -> bytes:
        """创建 ZIP 导出包"""
        buffer = io.BytesIO()

        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            # 用户信息
            if data.get("user"):
                zf.writestr(
                    "user.json",
                    json.dumps(data["user"], ensure_ascii=False, indent=2, default=str),
                )

            # 对话
            if data.get("conversations"):
                zf.writestr(
                    "conversations.json",
                    json.dumps(
                        data["conversations"], ensure_ascii=False, indent=2, default=str
                    ),
                )

            # 消息
            if data.get("messages"):
                zf.writestr(
                    "messages.json",
                    json.dumps(
                        data["messages"], ensure_ascii=False, indent=2, default=str
                    ),
                )

            # 笔记
            if data.get("notes"):
                zf.writestr(
                    "notes.json",
                    json.dumps(
                        data["notes"], ensure_ascii=False, indent=2, default=str
                    ),
                )

            # 记忆
            if data.get("memories"):
                zf.writestr(
                    "memories.json",
                    json.dumps(
                        data["memories"], ensure_ascii=False, indent=2, default=str
                    ),
                )

            # 设置
            if data.get("settings"):
                zf.writestr(
                    "settings.json",
                    json.dumps(
                        data["settings"], ensure_ascii=False, indent=2, default=str
                    ),
                )

            # 导出元数据
            zf.writestr(
                "metadata.json",
                json.dumps(
                    {
                        "export_date": data["export_date"],
                        "version": "1.0",
                        "files": [
                            "user.json",
                            "conversations.json",
                            "messages.json",
                            "notes.json",
                            "memories.json",
                            "settings.json",
                        ],
                    },
                    indent=2,
                ),
            )

        return buffer.getvalue()


class DataImporter:
    """数据导入器"""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or DB_PATH

    def import_conversations(
        self, user_id: int, data: bytes, format: str = "json"
    ) -> Dict:
        """导入对话"""
        if format == "json":
            content = json.loads(data.decode("utf-8"))
            if isinstance(content, dict) and "data" in content:
                conversations = content["data"]
            else:
                conversations = content
        else:
            return {"success": False, "error": "Unsupported format"}

        imported = 0
        errors = []

        with sqlite3.connect(self.db_path) as conn:
            for conv in conversations:
                try:
                    # 生成新 ID
                    import uuid

                    new_id = str(uuid.uuid4())

                    conn.execute(
                        """
                        INSERT INTO conversations 
                        (id, user_id, title, provider, model, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            new_id,
                            user_id,
                            conv.get("title", "导入的对话"),
                            conv.get("provider", "openai"),
                            conv.get("model", "gpt-3.5-turbo"),
                            conv.get("created_at", datetime.now().isoformat()),
                            datetime.now().isoformat(),
                        ),
                    )

                    # 导入消息
                    for msg in conv.get("messages", []):
                        conn.execute(
                            """
                            INSERT INTO messages 
                            (conversation_id, role, content, created_at)
                            VALUES (?, ?, ?, ?)
                        """,
                            (
                                new_id,
                                msg.get("role", "user"),
                                msg.get("content", ""),
                                msg.get("created_at", datetime.now().isoformat()),
                            ),
                        )

                    imported += 1
                except Exception as e:
                    errors.append(str(e))

        return {
            "success": True,
            "imported": imported,
            "errors": errors,
        }

    def import_notes(self, user_id: int, data: bytes, format: str = "json") -> Dict:
        """导入笔记"""
        if format == "json":
            content = json.loads(data.decode("utf-8"))
            if isinstance(content, dict) and "data" in content:
                notes = content["data"]
            else:
                notes = content
        else:
            return {"success": False, "error": "Unsupported format"}

        imported = 0
        errors = []

        with sqlite3.connect(self.db_path) as conn:
            for note in notes:
                try:
                    import uuid

                    conn.execute(
                        """
                        INSERT INTO notes 
                        (id, user_id, title, content, tags, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            str(uuid.uuid4()),
                            user_id,
                            note.get("title", "导入的笔记"),
                            note.get("content", ""),
                            note.get("tags", ""),
                            note.get("created_at", datetime.now().isoformat()),
                            datetime.now().isoformat(),
                        ),
                    )
                    imported += 1
                except Exception as e:
                    errors.append(str(e))

        return {
            "success": True,
            "imported": imported,
            "errors": errors,
        }

    def import_from_zip(self, user_id: int, data: bytes) -> Dict:
        """从 ZIP 导入"""
        buffer = io.BytesIO(data)
        results = {}

        with zipfile.ZipFile(buffer, "r") as zf:
            # 导入对话
            if "conversations.json" in zf.namelist():
                conv_data = zf.read("conversations.json")
                results["conversations"] = self.import_conversations(
                    user_id, conv_data
                )

            # 导入笔记
            if "notes.json" in zf.namelist():
                notes_data = zf.read("notes.json")
                results["notes"] = self.import_notes(user_id, notes_data)

        return results


# 全局实例
exporter = DataExporter()
importer = DataImporter()
