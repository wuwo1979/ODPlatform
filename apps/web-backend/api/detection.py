#!/usr/bin/env python
# @FileName  : detection.py
# @Time      : 2026/5/26
# @Project   : ODPlatform
# @Function  : 检测任务提交 + 结果查询

from __future__ import annotations

import json
import logging
import os
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from api.auth import get_current_user
from db.database import get_db

logger = logging.getLogger("odp-backend.detection")

router = APIRouter(tags=["detection"])

# 上传图片存储目录
UPLOAD_DIR = Path(__file__).parent.parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


def _try_inference(model_name: str, image_path: Path) -> list[dict]:
    """尝试调用本地推理模块执行检测。

    如果推理模块可用则返回检测结果列表，否则返回空列表。
    """
    try:
        import sys
        from pathlib import Path as Pth

        # 将 platform src 加入路径
        platform_src = Pth(__file__).parent.parent.parent.parent / "platform" / "src"
        if str(platform_src) not in sys.path:
            sys.path.insert(0, str(platform_src))

        from odp_platform.inference.engine import Detector
        import cv2

        # 查找模型文件
        checkpoints = Pth(__file__).parent.parent.parent.parent / "data" / "models" / "checkpoints"
        model_path = None
        for candidate in checkpoints.glob("*.pt"):
            if model_name in candidate.stem:
                model_path = str(candidate)
                break

        if model_path is None:
            logger.warning(f"未找到模型 {model_name} 的 .pt 文件")
            return []

        detector = Detector(model_path)
        image = cv2.imread(str(image_path))
        if image is None:
            return []

        result = detector.detect(image)
        return [
            {
                "class_name": d.class_name,
                "confidence": round(d.confidence, 4),
                "bbox_x1": round(d.bbox[0], 4),
                "bbox_y1": round(d.bbox[1], 4),
                "bbox_x2": round(d.bbox[2], 4),
                "bbox_y2": round(d.bbox[3], 4),
            }
            for d in result.detections
        ]
    except Exception as exc:
        logger.warning(f"推理调用失败: {exc}")
        return []


# ============================================================
#  POST /api/v1/detection  — 提交检测
# ============================================================
@router.post("/detection", status_code=201)
async def submit_detection(
    model_name: str = Form(..., description="模型名称"),
    conf: float = Form(default=0.25, description="置信度阈值"),
    iou: float = Form(default=0.45, description="IoU 阈值"),
    image: UploadFile = File(..., description="待检测图片"),
    user: dict = Depends(get_current_user),
):
    """提交检测任务：上传图片，指定模型，返回任务 ID。

    支持格式: jpg / png / jpeg / bmp / webp
    """
    # 校验文件类型
    ext = os.path.splitext(image.filename or "unknown.jpg")[1].lower()
    allowed = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"不支持的图片格式: {ext}，仅支持 {allowed}")

    # 保存图片
    safe_name = f"{uuid.uuid4().hex}{ext}"
    save_path = UPLOAD_DIR / safe_name
    content = await image.read()
    save_path.write_bytes(content)

    db = get_db()
    config_json = json.dumps({"conf": conf, "iou": iou}, ensure_ascii=False)

    try:
        cursor = db.execute(
            """INSERT INTO detection_tasks (user_id, model_name, image_filename, config_json)
               VALUES (?, ?, ?, ?)""",
            (user["id"], model_name, image.filename, config_json),
        )
        task_id = cursor.lastrowid

        # 尝试执行推理
        detections = _try_inference(model_name, save_path)
        if detections:
            status = "completed"
            for det in detections:
                db.execute(
                    """INSERT INTO detection_results
                       (task_id, class_name, confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (task_id, det["class_name"], det["confidence"],
                     det["bbox_x1"], det["bbox_y1"], det["bbox_x2"], det["bbox_y2"]),
                )
            db.execute(
                "UPDATE detection_tasks SET status = ? WHERE id = ?",
                (status, task_id),
            )
        else:
            status = "pending"

        db.commit()
        logger.info(f"检测任务创建: id={task_id}, model={model_name}, image={image.filename}, status={status}")

        return {
            "id": task_id,
            "model_name": model_name,
            "image_filename": image.filename,
            "status": status,
            "detection_count": len(detections),
        }
    except Exception as exc:
        db.rollback()
        logger.error(f"提交检测失败: {exc}")
        raise HTTPException(status_code=500, detail=f"数据库错误: {exc}")


# ============================================================
#  GET /api/v1/detection/{id}  — 查询检测结果
# ============================================================
@router.get("/detection/{task_id}")
def get_detection_result(
    task_id: int,
    user: dict = Depends(get_current_user),
):
    """查询指定检测任务的结果。

    返回任务信息和所有检测到的目标列表。
    """
    db = get_db()

    task = db.execute(
        """SELECT id, user_id, model_name, image_filename, status, config_json, created_at
           FROM detection_tasks WHERE id = ?""",
        (task_id,),
    ).fetchone()

    if task is None:
        raise HTTPException(status_code=404, detail=f"检测任务 id={task_id} 不存在")
    if task["user_id"] != user["id"]:
        raise HTTPException(status_code=403, detail="无权查看他人的检测任务")

    results = db.execute(
        """SELECT class_name, confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2
           FROM detection_results WHERE task_id = ?""",
        (task_id,),
    ).fetchall()

    return {
        "task": {
            "id": task["id"],
            "model_name": task["model_name"],
            "image_filename": task["image_filename"],
            "status": task["status"],
            "config_json": task["config_json"],
            "created_at": task["created_at"],
        },
        "detections": [dict(r) for r in results],
    }
