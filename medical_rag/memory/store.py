"""SQLite 用户长期记忆存储。"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Iterable, List

from medical_rag.core.config import settings


ALLOWED_MEMORY_TYPES = {"medical_history", "allergy", "medication", "preference"}


@dataclass(frozen=True)
class MemoryRecord:
    id: int
    username: str
    memory_type: str
    content: str
    created_at: str
    updated_at: str

    def to_dict(self) -> dict:
        return asdict(self)


class SQLiteMemoryStore:
    """按用户名隔离的 SQLite 长期记忆。"""

    def __init__(self, db_path: str | None = None) -> None:
        self.db_path = db_path or settings.MEMORY_DB_PATH
        folder = os.path.dirname(self.db_path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(username, memory_type, content)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_user_memory_username "
                "ON user_memory(username)"
            )

    def add_memories(
        self,
        username: str,
        memories: Iterable[dict],
    ) -> List[MemoryRecord]:
        """写入合法记忆，重复内容只更新时间。"""
        username = username.strip()
        if not username:
            return []
        memory_items = list(memories)
        saved: List[MemoryRecord] = []
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            for item in memory_items:
                memory_type = str(item.get("memory_type", "")).strip()
                content = str(item.get("content", "")).strip()
                if memory_type not in ALLOWED_MEMORY_TYPES or not content:
                    continue
                conn.execute(
                    """
                    INSERT INTO user_memory
                        (username, memory_type, content, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(username, memory_type, content)
                    DO UPDATE SET updated_at = excluded.updated_at
                    """,
                    (username, memory_type, content, now, now),
                )
            conn.commit()
        for record in self.list_memories(username):
            if any(
                record.memory_type == str(item.get("memory_type", "")).strip()
                and record.content == str(item.get("content", "")).strip()
                for item in memory_items
            ):
                saved.append(record)
        return saved

    def list_memories(self, username: str) -> List[MemoryRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, username, memory_type, content, created_at, updated_at
                FROM user_memory
                WHERE username = ?
                ORDER BY updated_at DESC, id DESC
                """,
                (username,),
            ).fetchall()
        return [MemoryRecord(**dict(row)) for row in rows]

    def delete_memory(self, username: str, memory_id: int) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM user_memory WHERE id = ? AND username = ?",
                (memory_id, username),
            )
            conn.commit()
        return cursor.rowcount > 0

    def delete_all(self, username: str) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM user_memory WHERE username = ?",
                (username,),
            )
            conn.commit()
        return cursor.rowcount
