#!/usr/bin/env python
# @FileName  : models.py
# @Time      : 2026/5/25
# @Project   : ODPlatform
# @Function  : 模型注册与查询接口

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from db.database import get_db
from schemas import ModelCreate

logger = logging.getLogger("odp-backend.models")

router = APIRouter(tags=["models"])


# ============================================================
#  POST /api/models  — 注册模型
# ============================================================
@router.post("/models", status_code=201)
def register_model(data: ModelCreate):
    """注册一个训练产出的模型文件。"""
    db = get_db()

    # 验证引用的实验存在
    existing = db.execute(
        "SELECT id FROM experiments WHERE id = ?", (data.experiment_id,)
    ).fetchone()
    if existing is None:
        raise HTTPException(
            status_code=404,
            detail=f"实验 id={data.experiment_id} 不存在，请先创建实验",
        )

    try:
        cursor = db.execute(
            """INSERT INTO models
               (experiment_id, filename, format, map50, map50_95, file_size_mb)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                data.experiment_id, data.filename, data.format,
                data.map50, data.map50_95, data.file_size_mb,
            ),
        )
        db.commit()
        model_id = cursor.lastrowid
        logger.info(f"模型注册成功: id={model_id}, file={data.filename}")
        return {"id": model_id, "filename": data.filename}
    except Exception as exc:
        db.rollback()
        logger.error(f"注册模型失败: {exc}")
        raise HTTPException(status_code=500, detail=f"数据库错误: {exc}")


# ============================================================
#  GET /api/models  — 查询模型列表
# ============================================================
@router.get("/models")
def list_models(
    experiment_id: Optional[int] = Query(default=None, ge=1),
    limit: int = Query(default=50, ge=1, le=200),
):
    """查询模型列表，可按 experiment_id 过滤。"""
    db = get_db()
    conditions: list[str] = []
    params: list = []

    if experiment_id is not None:
        conditions.append("experiment_id = ?")
        params.append(experiment_id)

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    params.append(limit)

    rows = db.execute(
        f"SELECT * FROM models {where} ORDER BY created_at DESC LIMIT ?",
        params,
    ).fetchall()

    return {
        "count": len(rows),
        "models": [dict(r) for r in rows],
    }
