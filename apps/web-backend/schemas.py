#!/usr/bin/env python
# @FileName  : schemas.py
# @Time      : 2026/5/25
# @Project   : ODPlatform
# @Function  : Pydantic schemas —— 后端 API 的请求/响应数据模型
#
# 对齐原则:
#   ExperimentCreate 字段名和语义必须与 ExperimentConfig 一致
#   EpochData 字段名必须与训练侧 metrics 字典一致

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


# ============================================================
#  实验相关
# ============================================================

class ExperimentCreate(BaseModel):
    """创建新实验（对齐 ExperimentConfig）。"""
    name: str = Field(..., description="唯一标识，如 rsod_yolo11n_640_001")
    dataset: str = Field(..., description="数据集名：rsod | visdrone")
    model: str = Field(..., description="模型文件，如 yolo11n.pt")
    task: str = Field(default="detect", description="任务类型")
    config_json: str = Field(..., description="ExperimentConfig.to_json() 快照")


class ExperimentUpdate(BaseModel):
    """更新实验状态 / 最终指标（对齐 ExperimentResult）。"""
    status: Optional[str] = Field(default=None, description="running | completed | failed")
    best_map50: Optional[float] = Field(default=None, alias="best_map50")
    best_map50_95: Optional[float] = Field(default=None, alias="best_map50_95")
    best_epoch: Optional[int] = Field(default=None)
    model_path: Optional[str] = Field(default=None)
    start_time: Optional[str] = Field(default=None)
    end_time: Optional[str] = Field(default=None)

    class Config:
        populate_by_name = True


class EpochData(BaseModel):
    """单 epoch 的训练/验证指标。"""
    epoch: int = Field(..., ge=1, description="epoch 编号（从 1 开始）")
    train_loss: Optional[float] = Field(default=None)
    val_loss: Optional[float] = Field(default=None)
    map50: Optional[float] = Field(default=None)
    map50_95: Optional[float] = Field(default=None)
    precision: Optional[float] = Field(default=None)
    recall: Optional[float] = Field(default=None)
    lr: Optional[float] = Field(default=None)


# ============================================================
#  模型注册
# ============================================================

class ModelCreate(BaseModel):
    """注册一个模型文件。"""
    experiment_id: int = Field(..., ge=1)
    filename: str = Field(..., description="如 best_rsod_yolo11n_640.pt")
    format: str = Field(default="pt")
    map50: Optional[float] = Field(default=None)
    map50_95: Optional[float] = Field(default=None)
    file_size_mb: Optional[float] = Field(default=None)


# ============================================================
#  质检报告
# ============================================================

class ValidationReportCreate(BaseModel):
    """写入一份新的质检报告。"""
    dataset: str = Field(..., description="数据集名")
    run_id: str = Field(..., description="质检运行唯一标识")
    passed: int = Field(default=0, ge=0)
    warnings: int = Field(default=0, ge=0)
    errors: int = Field(default=0, ge=0)
    report_json: Optional[str] = Field(default=None, description="完整质检报告 JSON 字符串")
