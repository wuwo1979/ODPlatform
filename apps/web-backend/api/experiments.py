#!/usr/bin/env python
# @FileName  : experiments.py
# @Time      : 2026/5/25
# @Project   : ODPlatform
# @Function  : 实验 CRUD + Epoch 数据接口

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from db.database import get_db
from schemas import EpochData, ExperimentCreate, ExperimentUpdate

logger = logging.getLogger("odp-backend.experiments")

router = APIRouter(tags=["experiments"])


# ============================================================
#  POST /api/experiments  — 创建实验
# ============================================================
@router.post("/experiments", status_code=201)
def create_experiment(data: ExperimentCreate):
    """注册一个新实验。name 必须唯一，重复则返回 409。"""
    db = get_db()
    try:
        cursor = db.execute(
            """INSERT INTO experiments (name, dataset, model, task, config_json, status)
               VALUES (?, ?, ?, ?, ?, 'running')""",
            (data.name, data.dataset, data.model, data.task, data.config_json),
        )
        db.commit()
        exp_id = cursor.lastrowid
        logger.info(f"实验创建成功: id={exp_id}, name={data.name}")
        return {"id": exp_id, "name": data.name}
    except Exception as exc:
        db.rollback()
        error_msg = str(exc).lower()
        if "unique" in error_msg:
            raise HTTPException(
                status_code=409,
                detail=f"实验 '{data.name}' 已存在",
            )
        logger.error(f"创建实验失败: {exc}")
        raise HTTPException(status_code=500, detail=f"数据库错误: {exc}")


# ============================================================
#  PATCH /api/experiments/{id}  — 更新状态 / 最终指标
# ============================================================
@router.patch("/experiments/{exp_id}")
def update_experiment_status(exp_id: int, data: ExperimentUpdate):
    """更新实验状态（running/completed/failed）及最终指标。

    只更新请求中提供的非 None 字段。
    """
    db = get_db()

    # 先验证实验存在
    existing = db.execute(
        "SELECT id FROM experiments WHERE id = ?", (exp_id,)
    ).fetchone()
    if existing is None:
        raise HTTPException(status_code=404, detail=f"实验 id={exp_id} 不存在")

    # 动态拼接 SET 子句
    updates: list[str] = []
    params: list = []
    model_data = data.model_dump(exclude_none=True, by_alias=True)

    for field, value in model_data.items():
        updates.append(f"{field} = ?")
        params.append(value)

    # 终态时自动填充 end_time（如果调用方未提供）
    status_value = model_data.get("status")
    if status_value in ("completed", "failed") and "end_time" not in model_data:
        updates.append("end_time = ?")
        params.append(datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"))

    if not updates:
        raise HTTPException(status_code=400, detail="没有可更新的字段")

    params.append(exp_id)
    sql = f"UPDATE experiments SET {', '.join(updates)} WHERE id = ?"

    try:
        db.execute(sql, params)
        db.commit()
        logger.info(f"实验状态更新: id={exp_id}, fields={list(model_data.keys())}")
        return {"id": exp_id, "updated": list(model_data.keys())}
    except Exception as exc:
        db.rollback()
        logger.error(f"更新实验状态失败: {exc}")
        raise HTTPException(status_code=500, detail=f"数据库错误: {exc}")


# ============================================================
#  POST /api/experiments/{id}/epochs  — 写入 epoch 数据
# ============================================================
@router.post("/experiments/{exp_id}/epochs", status_code=201)
def add_epoch_data(exp_id: int, data: EpochData):
    """为指定实验写入单个 epoch 的训练/验证指标。

    如果同一 (experiment_id, epoch) 已存在，则更新。
    """
    db = get_db()

    # 验证实验存在
    existing = db.execute(
        "SELECT id FROM experiments WHERE id = ?", (exp_id,)
    ).fetchone()
    if existing is None:
        raise HTTPException(status_code=404, detail=f"实验 id={exp_id} 不存在")

    try:
        db.execute(
            """INSERT INTO training_epochs
               (experiment_id, epoch, train_loss, val_loss,
                map50, map50_95, precision, recall, lr)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(experiment_id, epoch) DO UPDATE SET
                   train_loss = excluded.train_loss,
                   val_loss   = excluded.val_loss,
                   map50      = excluded.map50,
                   map50_95   = excluded.map50_95,
                   precision  = excluded.precision,
                   recall     = excluded.recall,
                   lr         = excluded.lr""",
            (
                exp_id, data.epoch,
                data.train_loss, data.val_loss,
                data.map50, data.map50_95,
                data.precision, data.recall, data.lr,
            ),
        )

        # ── 自动同步最佳指标到 experiments 表 ──
        exp = db.execute(
            "SELECT best_map50, best_map50_95, best_epoch, start_time FROM experiments WHERE id = ?",
            (exp_id,),
        ).fetchone()

        exp_updates: list[str] = []
        exp_params: list = []

        # 首次写 epoch → 自动设 start_time
        if exp["start_time"] is None:
            exp_updates.append("start_time = ?")
            exp_params.append(datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"))

        # 如果新 epoch 的 map50 更好 → 更新 best_map50 和 best_epoch
        if data.map50 is not None:
            current_best50 = exp["best_map50"] or 0
            if data.map50 > current_best50:
                exp_updates.append("best_map50 = ?")
                exp_params.append(data.map50)
                exp_updates.append("best_epoch = ?")
                exp_params.append(data.epoch)

        # 如果新 epoch 的 map50_95 更好 → 更新 best_map50_95
        if data.map50_95 is not None:
            current_best95 = exp["best_map50_95"] or 0
            if data.map50_95 > current_best95:
                exp_updates.append("best_map50_95 = ?")
                exp_params.append(data.map50_95)

        if exp_updates:
            exp_params.append(exp_id)
            db.execute(
                f"UPDATE experiments SET {', '.join(exp_updates)} WHERE id = ?",
                exp_params,
            )
            logger.info(
                f"epoch {data.epoch} 自动同步实验表: {exp_updates}"
            )

        db.commit()
        return {"experiment_id": exp_id, "epoch": data.epoch, "status": "ok"}
    except Exception as exc:
        db.rollback()
        logger.error(f"写入 epoch 数据失败 (exp_id={exp_id}, epoch={data.epoch}): {exc}")
        raise HTTPException(status_code=500, detail=f"数据库错误: {exc}")


# ============================================================
#  GET /api/experiments  — 查询实验列表
# ============================================================
@router.get("/experiments")
def list_experiments(
    dataset: Optional[str] = Query(default=None),
    status: Optional[str] = Query(default=None),
    limit: int = Query(default=20, ge=1, le=200),
):
    """查询实验列表，可按 dataset 和 status 过滤。"""
    db = get_db()
    conditions: list[str] = []
    params: list = []

    if dataset:
        conditions.append("dataset = ?")
        params.append(dataset)
    if status:
        conditions.append("status = ?")
        params.append(status)

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    params.append(limit)

    rows = db.execute(
        f"SELECT * FROM experiments {where} ORDER BY created_at DESC LIMIT ?",
        params,
    ).fetchall()

    return {
        "count": len(rows),
        "experiments": [dict(r) for r in rows],
    }


# ============================================================
#  GET /api/experiments/{id}/epochs  — 查询训练曲线
# ============================================================
@router.get("/experiments/{exp_id}/epochs")
def get_epoch_data(exp_id: int):
    """查询指定实验的所有 epoch 数据（按 epoch 升序），用于绘制训练曲线。"""
    db = get_db()

    existing = db.execute(
        "SELECT id FROM experiments WHERE id = ?", (exp_id,)
    ).fetchone()
    if existing is None:
        raise HTTPException(status_code=404, detail=f"实验 id={exp_id} 不存在")

    rows = db.execute(
        "SELECT * FROM training_epochs WHERE experiment_id = ? ORDER BY epoch ASC",
        (exp_id,),
    ).fetchall()

    return {
        "experiment_id": exp_id,
        "count": len(rows),
        "epochs": [dict(r) for r in rows],
    }
