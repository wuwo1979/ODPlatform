#!/usr/bin/env python
# @FileName  : hooks.py
# @Time      : 2026/5/25
# @Project   : ODPlatform
# @Function  : 训练侧客户端 SDK —— 供 callbacks.py 引用
#
# 使用方式（训练工程师在 callbacks.py 中）:
#   from hooks import on_training_start, on_epoch_end, on_training_end
#
# 约定: 函数签名不变，后端负责 HTTP 细节，训练侧不关心 API 路径。

from __future__ import annotations

import logging
import time
from typing import Optional

import requests

logger = logging.getLogger("odp-backend.hooks")

# 后端服务地址（可通过环境变量覆盖）
BASE_URL: str = "http://127.0.0.1:8000"

# 重试配置
MAX_RETRIES: int = 3
RETRY_DELAY: float = 1.0  # 秒
REQUEST_TIMEOUT: float = 5.0  # 秒


def _post_with_retry(
    url: str,
    json_data: dict,
    label: str = "",
    method: str = "POST",
) -> Optional[dict]:
    """带重试的 HTTP 请求，后端暂不可用时记录告警并返回 None。

    Args:
        url: API 完整 URL
        json_data: 请求体
        label: 日志标签
        method: HTTP 方法 ("POST" | "PATCH")

    Returns:
        响应的 JSON dict，失败则返回 None
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if method == "PATCH":
                r = requests.patch(url, json=json_data, timeout=REQUEST_TIMEOUT)
            else:
                r = requests.post(url, json=json_data, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.RequestException as e:
            logger.warning(
                f"[{label}] 后端通信失败 (尝试 {attempt}/{MAX_RETRIES}): {e}"
            )
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
    logger.error(f"[{label}] 重试耗尽，放弃本次同步")
    return None


def on_training_start(
    name: str,
    config_json: str,
    dataset: str,
    model: str,
    base_url: str = BASE_URL,
) -> Optional[int]:
    """训练开始时调用，注册实验并返回 exp_id。

    Returns:
        成功返回实验 id，失败返回 None
    """
    result = _post_with_retry(
        f"{base_url}/api/experiments",
        {
            "name": name,
            "config_json": config_json,
            "dataset": dataset,
            "model": model,
        },
        label="train_start",
    )
    if result:
        logger.info(f"实验注册成功: id={result['id']}, name={result['name']}")
        return result["id"]
    return None


def on_epoch_end(
    experiment_id: int,
    epoch: int,
    metrics: dict,
    base_url: str = BASE_URL,
) -> None:
    """每个 epoch 结束时调用，写入训练指标。

    metrics 字典格式（o=overall）:
        {"train_loss": 2.3, "val_loss": 1.8, "map50": 0.872,
         "map50_95": 0.651, "precision": 0.91, "recall": 0.85, "lr": 0.008}
    """
    _post_with_retry(
        f"{base_url}/api/experiments/{experiment_id}/epochs",
        {"epoch": epoch, **metrics},
        label=f"epoch_{epoch}",
    )


def on_training_end(
    experiment_id: int,
    map50: float,
    model_path: str,
    base_url: str = BASE_URL,
) -> None:
    """训练完成时调用，更新最终状态和指标。"""
    result = _post_with_retry(
        f"{base_url}/api/experiments/{experiment_id}",
        {
            "status": "completed",
            "best_map50": map50,
            "model_path": model_path,
        },
        label="train_end",
        method="PATCH",
    )
    if result:
        logger.info(f"实验完成同步: id={experiment_id}")


def on_training_failed(
    experiment_id: int,
    reason: str = "",
    base_url: str = BASE_URL,
) -> None:
    """训练失败时调用，标记实验状态为 failed。"""
    _post_with_retry(
        f"{base_url}/api/experiments/{experiment_id}",
        {"status": "failed"},
        label="train_failed",
        method="PATCH",
    )
    if reason:
        logger.warning(f"实验失败: id={experiment_id}, reason={reason}")
