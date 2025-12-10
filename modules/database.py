# ========== 数据库管理模块 ==========
"""
数据库连接池、迁移、备份、优化
"""
import os
import sqlite3
import json
import shutil
from datetime import datetime
from typing import Optional, List, Dict, Any
from contextlib import contextmanager
from queue import Queue
import threading
import logging

logger = logging.getLogger(__name__)

# ========== 数据库连接池 ==========
class DatabasePool:
    """SQLite 连接池"""
    
    def __init__(self, db_path: str, pool_size: int = 5):
        self.db_path = db_path
        self.pool_size = pool_size
        self._pool: Queue = Queue(maxsize=pool_size)
        self._lock = threading.Lock()
        self._init_pool()
    
    def _init_pool(self):
        """初始化连接池"""
        for _ in range(self.pool_size):
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            # 启用 WAL 模式提高并发性能
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=10000")
            self._pool.put(conn)
        logger.info(f"Database pool initialized: {self.pool_size} connections")
    
    @contextmanager
    def get_connection(self):
        """获取数据库连接"""
        conn = self._pool.get()
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            self._pool.put(conn)
    
    def execute(self, query: str, params: tuple = None) -> sqlite3.Cursor:
        """执行查询"""
        with self.get_connection() as conn:
            if params:
                return conn.execute(query, params)
            return conn.execute(query)
    
    def fetch_one(self, query: str, params: tuple = None) -> Optional[Dict]:
        """获取单条记录"""
        with self.get_connection() as conn:
            if params:
                row = conn.execute(query, params).fetchone()
            else:
                row = conn.execute(query).fetchone()
            return dict(row) if row else None
    
    def fetch_all(self, query: str, params: tuple = None) -> List[Dict]:
        """获取所有记录"""
        with self.get_connection() as conn:
            if params:
                rows = conn.execute(query, params).fetchall()
            else:
                rows = conn.execute(query).fetchall()
            return [dict(row) for row in rows]
    
    def close_all(self):
        """关闭所有连接"""
        while not self._pool.empty():
            conn = self._pool.get()
            conn.close()
        logger.info("Database pool closed")

# ========== 数据库迁移 ==========
class Migration:
    """数据库迁移"""
    
    def __init__(self, version: int, name: str, up_sql: str, down_sql: str = ""):
        self.version = version
        self.name = name
        self.up_sql = up_sql
        self.down_sql = down_sql

class MigrationManager:
    """迁移管理器"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.migrations: List[Migration] = []
        self._ensure_migration_table()
    
    def _ensure_migration_table(self):
        """确保迁移表存在"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS _migrations (
                    version INTEGER PRIMARY KEY,
                    name TEXT,
                    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
    
    def add_migration(self, migration: Migration):
        """添加迁移"""
        self.migrations.append(migration)
        self.migrations.sort(key=lambda m: m.version)
    
    def get_current_version(self) -> int:
        """获取当前版本"""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT MAX(version) FROM _migrations"
            ).fetchone()
            return row[0] or 0
    
    def get_pending_migrations(self) -> List[Migration]:
        """获取待执行的迁移"""
        current = self.get_current_version()
        return [m for m in self.migrations if m.version > current]
    
    def migrate(self, target_version: int = None) -> List[str]:
        """执行迁移"""
        results = []
        current = self.get_current_version()
        
        if target_version is None:
            target_version = max(m.version for m in self.migrations) if self.migrations else 0
        
        with sqlite3.connect(self.db_path) as conn:
            if target_version > current:
                # 升级
                for migration in self.migrations:
                    if current < migration.version <= target_version:
                        try:
                            conn.executescript(migration.up_sql)
                            conn.execute(
                                "INSERT INTO _migrations (version, name) VALUES (?, ?)",
                                (migration.version, migration.name)
                            )
                            results.append(f"Applied: {migration.version} - {migration.name}")
                            logger.info(f"Migration applied: {migration.version} - {migration.name}")
                        except Exception as e:
                            logger.error(f"Migration failed: {migration.version}: {e}")
                            raise
            elif target_version < current:
                # 降级
                for migration in reversed(self.migrations):
                    if target_version < migration.version <= current:
                        if migration.down_sql:
                            try:
                                conn.executescript(migration.down_sql)
                                conn.execute(
                                    "DELETE FROM _migrations WHERE version = ?",
                                    (migration.version,)
                                )
                                results.append(f"Reverted: {migration.version} - {migration.name}")
                                logger.info(f"Migration reverted: {migration.version} - {migration.name}")
                            except Exception as e:
                                logger.error(f"Migration rollback failed: {migration.version}: {e}")
                                raise
                        else:
                            raise Exception(f"No down migration for version {migration.version}")
        
        return results
    
    def status(self) -> Dict:
        """获取迁移状态"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            applied = conn.execute(
                "SELECT * FROM _migrations ORDER BY version"
            ).fetchall()
        
        return {
            "current_version": self.get_current_version(),
            "latest_version": max(m.version for m in self.migrations) if self.migrations else 0,
            "pending": len(self.get_pending_migrations()),
            "applied": [dict(row) for row in applied]
        }

# ========== 数据库备份 ==========
class BackupManager:
    """备份管理器"""
    
    def __init__(self, db_path: str, backup_dir: str = "backups"):
        self.db_path = db_path
        self.backup_dir = backup_dir
        os.makedirs(backup_dir, exist_ok=True)
    
    def create_backup(self, name: str = None) -> str:
        """创建备份"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = name or f"backup_{timestamp}"
        backup_path = os.path.join(self.backup_dir, f"{name}.db")
        
        # 使用 SQLite 的备份 API
        with sqlite3.connect(self.db_path) as source:
            with sqlite3.connect(backup_path) as dest:
                source.backup(dest)
        
        logger.info(f"Backup created: {backup_path}")
        return backup_path
    
    def restore_backup(self, backup_name: str) -> bool:
        """恢复备份"""
        backup_path = os.path.join(self.backup_dir, f"{backup_name}.db")
        
        if not os.path.exists(backup_path):
            raise FileNotFoundError(f"Backup not found: {backup_path}")
        
        # 先备份当前数据库
        self.create_backup("pre_restore")
        
        # 恢复
        with sqlite3.connect(backup_path) as source:
            with sqlite3.connect(self.db_path) as dest:
                source.backup(dest)
        
        logger.info(f"Backup restored: {backup_path}")
        return True
    
    def list_backups(self) -> List[Dict]:
        """列出所有备份"""
        backups = []
        for filename in os.listdir(self.backup_dir):
            if filename.endswith(".db"):
                path = os.path.join(self.backup_dir, filename)
                stat = os.stat(path)
                backups.append({
                    "name": filename[:-3],
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "created_at": datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
        return sorted(backups, key=lambda x: x["created_at"], reverse=True)
    
    def delete_backup(self, backup_name: str) -> bool:
        """删除备份"""
        backup_path = os.path.join(self.backup_dir, f"{backup_name}.db")
        if os.path.exists(backup_path):
            os.remove(backup_path)
            logger.info(f"Backup deleted: {backup_path}")
            return True
        return False
    
    def cleanup_old_backups(self, keep_count: int = 10) -> int:
        """清理旧备份"""
        backups = self.list_backups()
        deleted = 0
        
        if len(backups) > keep_count:
            for backup in backups[keep_count:]:
                self.delete_backup(backup["name"])
                deleted += 1
        
        return deleted

# ========== 数据库优化 ==========
class DatabaseOptimizer:
    """数据库优化器"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    def analyze(self) -> Dict:
        """分析数据库"""
        with sqlite3.connect(self.db_path) as conn:
            # 表统计
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            
            table_stats = {}
            for (table_name,) in tables:
                if table_name.startswith("_"):
                    continue
                count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
                table_stats[table_name] = count
            
            # 数据库大小
            db_size = os.path.getsize(self.db_path)
            
            # 索引统计
            indexes = conn.execute(
                "SELECT name, tbl_name FROM sqlite_master WHERE type='index'"
            ).fetchall()
            
            return {
                "size_mb": round(db_size / (1024 * 1024), 2),
                "tables": table_stats,
                "total_rows": sum(table_stats.values()),
                "index_count": len(indexes)
            }
    
    def vacuum(self) -> Dict:
        """压缩数据库"""
        before_size = os.path.getsize(self.db_path)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("VACUUM")
        
        after_size = os.path.getsize(self.db_path)
        saved = before_size - after_size
        
        logger.info(f"Database vacuumed: saved {saved} bytes")
        
        return {
            "before_mb": round(before_size / (1024 * 1024), 2),
            "after_mb": round(after_size / (1024 * 1024), 2),
            "saved_mb": round(saved / (1024 * 1024), 2)
        }
    
    def reindex(self) -> int:
        """重建索引"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("REINDEX")
            
            indexes = conn.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='index'"
            ).fetchone()[0]
        
        logger.info(f"Database reindexed: {indexes} indexes")
        return indexes
    
    def integrity_check(self) -> Dict:
        """完整性检查"""
        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute("PRAGMA integrity_check").fetchone()[0]
        
        return {
            "status": "ok" if result == "ok" else "error",
            "message": result
        }
    
    def optimize(self) -> Dict:
        """执行全面优化"""
        results = {
            "integrity": self.integrity_check(),
            "vacuum": self.vacuum(),
            "reindex": self.reindex(),
            "analysis": self.analyze()
        }
        
        logger.info("Database optimization completed")
        return results

# ========== 数据导出导入 ==========
class DataExporter:
    """数据导出器"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    def export_table_json(self, table_name: str) -> str:
        """导出表为 JSON"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(f"SELECT * FROM {table_name}").fetchall()
            data = [dict(row) for row in rows]
        
        return json.dumps(data, ensure_ascii=False, indent=2, default=str)
    
    def export_table_csv(self, table_name: str) -> str:
        """导出表为 CSV"""
        import csv
        import io
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(f"SELECT * FROM {table_name}").fetchall()
            
            if not rows:
                return ""
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            # 写入表头
            writer.writerow(rows[0].keys())
            
            # 写入数据
            for row in rows:
                writer.writerow(dict(row).values())
            
            return output.getvalue()
    
    def import_table_json(self, table_name: str, json_data: str) -> int:
        """从 JSON 导入数据"""
        data = json.loads(json_data)
        
        if not data:
            return 0
        
        columns = list(data[0].keys())
        placeholders = ",".join(["?" for _ in columns])
        
        with sqlite3.connect(self.db_path) as conn:
            for row in data:
                values = [row.get(col) for col in columns]
                conn.execute(
                    f"INSERT OR REPLACE INTO {table_name} ({','.join(columns)}) VALUES ({placeholders})",
                    values
                )
        
        return len(data)
