#!/usr/bin/env python
# @FileName  : models_api.py
# @Time      : 2026/5/26
# @Project   : ODPlatform
# @Function  : 用户面模型管理 —— 列表 / 上传 / 删除

from __future__ import annotations

import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile

from api.auth import get_current_user
from db.database import get_db

logger = logging.getLogger("odp-backend.models_api")

router = APIRouter(tags=["models"])

# 模型存储目录
MODELS_DIR = Path(__file__).parent.parent.parent.parent / "data" / "models" / "checkpoints"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
#  GET /api/v1/models  — 模型列表
# ============================================================
@router.get("/models")
def list_models(
    format: Optional[str] = Query(default=None, description="过滤格式，如 pt"),
    limit: int = Query(default=50, ge=1, le=200),
    user: dict = Depends(get_current_user),
):
    """返回所有可用的检测模型列表。"""
    # 扫描磁盘上的 .pt 文件
    models: list[dict] = []
    for pt_file in sorted(MODELS_DIR.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True):
        if format and pt_file.suffix.lstrip(".") != format:
            continue
        stat = pt_file.stat()
        models.append({
            "filename": pt_file.name,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "modified_at": Path(pt_file).stat().st_mtime,
        })
        if len(models) >= limit:
            break

    return {"count": len(models), "models": models}


# ============================================================
#  POST /api/v1/models/upload  — 上传 .pt 模型
# ============================================================
@router.post("/models/upload", status_code=201)
async def upload_model(
    file: UploadFile = File(..., description=".pt 模型文件"),
    user: dict = Depends(get_current_user),
):
    """上传一个 .pt 模型文件到 checkpoints 目录。"""
    filename = file.filename or "unknown.pt"
    ext = os.path.splitext(filename)[1].lower()

    if ext != ".pt":
        raise HTTPException(status_code=400, detail=f"仅支持 .pt 格式，收到: {ext}")

    # 防止文件名冲突
    safe_name = f"{Path(filename).stem}_{uuid.uuid4().hex[:8]}{ext}"
    dest = MODELS_DIR / safe_name

    try:
        content = await file.read()
        dest.write_bytes(content)
        size_mb = len(content) / (1024 * 1024)
        logger.info(f"模型上传成功: {safe_name} ({size_mb:.2f} MB), by user={user['id']}")
    except Exception as exc:
        logger.error(f"模型上传失败: {exc}")
        raise HTTPException(status_code=500, detail=f"文件保存失败: {exc}")

    return {
        "filename": safe_name,
        "original_name": filename,
        "size_mb": round(size_mb, 2),
    }


# ============================================================
#  DELETE /api/v1/models/{id}  — 删除模型
# ============================================================
@router.delete("/models/{filename:path}")
def delete_model(
    filename: str,
    user: dict = Depends(get_current_user),
):
    """删除指定的 .pt 模型文件。

    文件名需通过 path 参数传递，如 /api/v1/models/best_xxx.pt
    """
    target = MODELS_DIR / filename
    if not target.exists():
        raise HTTPException(status_code=404, detail=f"模型文件不存在: {filename}")
    if target.suffix != ".pt":
        raise HTTPException(status_code=400, detail=f"只能删除 .pt 文件")

    try:
        target.unlink()
        logger.info(f"模型已删除: {filename}, by user={user['id']}")
        return {"deleted": filename}
    except Exception as exc:
        logger.error(f"删除模型失败: {exc}")
        raise HTTPException(status_code=500, detail=f"删除失败: {exc}")
