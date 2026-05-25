#!/usr/bin/env python
# @FileName  : validation.py
# @Time      : 2026/5/25
# @Project   : ODPlatform
# @Function  : 数据质检报告存储与查询接口

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from db.database import get_db
from schemas import ValidationReportCreate

logger = logging.getLogger("odp-backend.validation")

router = APIRouter(tags=["validation"])


# ============================================================
#  POST /api/validation/reports  — 写入质检报告
# ============================================================
@router.post("/validation/reports", status_code=201)
def create_validation_report(data: ValidationReportCreate):
    """写入一份新的数据质检报告。"""
    db = get_db()

    try:
        cursor = db.execute(
            """INSERT INTO validation_reports
               (dataset, run_id, passed, warnings, errors, report_json)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                data.dataset, data.run_id,
                data.passed, data.warnings, data.errors,
                data.report_json,
            ),
        )
        db.commit()
        report_id = cursor.lastrowid
        logger.info(
            f"质检报告写入成功: id={report_id}, dataset={data.dataset}, "
            f"run_id={data.run_id}, passed={data.passed}, "
            f"warnings={data.warnings}, errors={data.errors}"
        )
        return {"id": report_id, "dataset": data.dataset, "run_id": data.run_id}
    except Exception as exc:
        db.rollback()
        logger.error(f"写入质检报告失败: {exc}")
        raise HTTPException(status_code=500, detail=f"数据库错误: {exc}")


# ============================================================
#  GET /api/validation/reports  — 查询质检历史
# ============================================================
@router.get("/validation/reports")
def list_validation_reports(
    dataset: Optional[str] = Query(default=None),
    limit: int = Query(default=20, ge=1, le=200),
):
    """查询质检报告历史，可按 dataset 过滤。"""
    db = get_db()
    conditions: list[str] = []
    params: list = []

    if dataset:
        conditions.append("dataset = ?")
        params.append(dataset)

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    params.append(limit)

    rows = db.execute(
        f"SELECT * FROM validation_reports {where} ORDER BY created_at DESC LIMIT ?",
        params,
    ).fetchall()

    return {
        "count": len(rows),
        "reports": [dict(r) for r in rows],
    }
