#!/usr/bin/env python
"""
数据库迁移脚本

用法:
    python scripts/migrate.py status     # 查看迁移状态
    python scripts/migrate.py up         # 执行所有待迁移
    python scripts/migrate.py down       # 回滚最后一次迁移
    python scripts/migrate.py create NAME # 创建新迁移
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.database import MigrationManager, Migration

DB_PATH = os.getenv("DATABASE_PATH", "data.db")


def get_migrations():
    """获取所有迁移"""
    return [
        Migration(
            version=1,
            name="initial_schema",
            up_sql="""
                -- 基础表已在 main.py 中创建
                SELECT 1;
            """,
            down_sql="SELECT 1;",
        ),
        Migration(
            version=2,
            name="add_task_queue_table",
            up_sql="""
                CREATE TABLE IF NOT EXISTS task_queue (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    args TEXT,
                    kwargs TEXT,
                    status TEXT DEFAULT 'pending',
                    result TEXT,
                    error TEXT,
                    retries INTEGER DEFAULT 0,
                    max_retries INTEGER DEFAULT 3,
                    priority INTEGER DEFAULT 0,
                    created_at REAL,
                    started_at REAL,
                    completed_at REAL,
                    scheduled_at REAL
                );
                CREATE INDEX IF NOT EXISTS idx_task_status ON task_queue(status);
            """,
            down_sql="DROP TABLE IF EXISTS task_queue;",
        ),
        Migration(
            version=3,
            name="add_security_events_table",
            up_sql="""
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
                );
                CREATE INDEX IF NOT EXISTS idx_security_events_type ON security_events(event_type);
                CREATE INDEX IF NOT EXISTS idx_security_events_user ON security_events(user_id);
            """,
            down_sql="DROP TABLE IF EXISTS security_events;",
        ),
        Migration(
            version=4,
            name="add_encryption_keys_table",
            up_sql="""
                CREATE TABLE IF NOT EXISTS encryption_keys (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key_value TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    is_active INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP,
                    rotated_at TIMESTAMP
                );
            """,
            down_sql="DROP TABLE IF EXISTS encryption_keys;",
        ),
        Migration(
            version=5,
            name="add_user_indexes",
            up_sql="""
                CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
                CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(user_id);
                CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id);
                CREATE INDEX IF NOT EXISTS idx_notes_user ON notes(user_id);
            """,
            down_sql="""
                DROP INDEX IF EXISTS idx_users_username;
                DROP INDEX IF EXISTS idx_conversations_user;
                DROP INDEX IF EXISTS idx_messages_conversation;
                DROP INDEX IF EXISTS idx_notes_user;
            """,
        ),
    ]


def main():
    parser = argparse.ArgumentParser(description="数据库迁移工具")
    parser.add_argument(
        "command",
        choices=["status", "up", "down", "create"],
        help="迁移命令",
    )
    parser.add_argument("name", nargs="?", help="迁移名称（用于 create）")
    parser.add_argument(
        "--target", type=int, help="目标版本（用于 up/down）"
    )

    args = parser.parse_args()

    manager = MigrationManager(DB_PATH)

    # 注册迁移
    for migration in get_migrations():
        manager.add_migration(migration)

    if args.command == "status":
        status = manager.status()
        print(f"当前版本: {status['current_version']}")
        print(f"最新版本: {status['latest_version']}")
        print(f"待迁移数: {status['pending']}")
        print("\n已应用的迁移:")
        for m in status["applied"]:
            print(f"  - v{m['version']}: {m['name']} ({m['applied_at']})")

        pending = manager.get_pending_migrations()
        if pending:
            print("\n待执行的迁移:")
            for m in pending:
                print(f"  - v{m.version}: {m.name}")

    elif args.command == "up":
        target = args.target
        results = manager.migrate(target)
        if results:
            print("迁移完成:")
            for r in results:
                print(f"  ✅ {r}")
        else:
            print("没有待执行的迁移")

    elif args.command == "down":
        current = manager.get_current_version()
        if current == 0:
            print("已经是最初版本")
            return

        target = args.target if args.target is not None else current - 1
        results = manager.migrate(target)
        if results:
            print("回滚完成:")
            for r in results:
                print(f"  ⬇️ {r}")
        else:
            print("没有可回滚的迁移")

    elif args.command == "create":
        if not args.name:
            print("请提供迁移名称")
            sys.exit(1)

        # 获取下一个版本号
        current = manager.get_current_version()
        next_version = max(m.version for m in get_migrations()) + 1

        template = f'''
Migration(
    version={next_version},
    name="{args.name}",
    up_sql="""
        -- 在这里添加升级 SQL
    """,
    down_sql="""
        -- 在这里添加回滚 SQL
    """,
),
'''
        print(f"请将以下迁移添加到 get_migrations() 函数中:\n")
        print(template)


if __name__ == "__main__":
    main()
