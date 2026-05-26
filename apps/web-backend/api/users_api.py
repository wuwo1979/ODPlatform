#!/usr/bin/env python
# @FileName  : users_api.py
# @Time      : 2026/5/26
# @Project   : ODPlatform
# @Function  : 用户信息 + 检测历史

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, Query

from api.auth import get_current_user
from db.database import get_db

logger = logging.getLogger("odp-backend.users")

router = APIRouter(tags=["users"])


# ============================================================
#  GET /api/v1/users/me  — 当前用户信息
# ============================================================
@router.get("/users/me")
def get_me(user: dict = Depends(get_current_user)):
    """返回当前登录用户的信息。"""
    return {"user": user}


# ============================================================
#  GET /api/v1/users/me/history  — 检测历史
# ============================================================
@router.get("/users/me/history")
def get_detection_history(
    limit: int = Query(default=20, ge=1, le=200),
    user: dict = Depends(get_current_user),
):
    """返回当前用户的检测任务历史（含每个任务的结果数量）。"""
    db = get_db()

    tasks = db.execute(
        """SELECT id, model_name, image_filename, status, config_json, created_at
           FROM detection_tasks
           WHERE user_id = ?
           ORDER BY created_at DESC
           LIMIT ?""",
        (user["id"], limit),
    ).fetchall()

    history: list[dict] = []
    for t in tasks:
        result_count = db.execute(
            "SELECT COUNT(*) AS cnt FROM detection_results WHERE task_id = ?",
            (t["id"],),
        ).fetchone()["cnt"]

        history.append({
            "id": t["id"],
            "model_name": t["model_name"],
            "image_filename": t["image_filename"],
            "status": t["status"],
            "config_json": t["config_json"],
            "result_count": result_count,
            "created_at": t["created_at"],
        })

    return {"count": len(history), "tasks": history}
