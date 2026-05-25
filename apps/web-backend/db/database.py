#!/usr/bin/env python
# @FileName  : database.py
# @Time      : 2026/5/25
# @Project   : ODPlatform
# @Function  : SQLite 数据库连接管理（单连接 + WAL 模式）

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Optional

# 数据库文件放在 web-backend 工程目录下
DB_DIR: Path = Path(__file__).resolve().parent.parent  # apps/web-backend/
DB_PATH: Path = DB_DIR / "odplatform.db"

# 线程本地存储每一个线程自己的连接，保证 FastAPI 异步/多线程安全
_local: threading.local = threading.local()


def get_db() -> sqlite3.Connection:
    """获取当前线程的 SQLite 连接（自动创建 if 不存在）。

    返回的连接配置为：
      - row_factory = sqlite3.Row（可通过列名访问字段）
      - WAL 模式（允许并发读）
      - 外键约束启用

    Returns:
        线程级复用的 sqlite3.Connection
    """
    conn = getattr(_local, "connection", None)
    if conn is None:
        conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")       # 写入不阻塞读取
        conn.execute("PRAGMA foreign_keys=ON")         # 启用外键约束
        _local.connection = conn
    return conn


def close_db() -> None:
    """关闭当前线程的数据库连接（应用退出时调用）。"""
    conn = getattr(_local, "connection", None)
    if conn is not None:
        conn.close()
        _local.connection = None


def init_db() -> None:
    """初始化数据库：执行建表语句（幂等 — 表已存在则不重复创建）。"""
    from .init_db import create_tables

    db = get_db()
    create_tables(db)
    db.commit()
