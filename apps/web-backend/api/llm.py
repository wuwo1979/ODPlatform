#!/usr/bin/env python
# @FileName  : llm.py
# @Time      : 2026/5/26
# @Project   : ODPlatform
# @Function  : LLM 透传代理 —— chat 转发 + 模型列表

from __future__ import annotations

import logging
import os
from typing import Optional

import requests
from fastapi import APIRouter, Depends, HTTPException, Query

from api.auth import get_current_user
from schemas import LLMChatRequest

logger = logging.getLogger("odp-backend.llm")

router = APIRouter(tags=["llm"])

# LLM 后端配置（可通过环境变量覆盖）
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "30"))


# ============================================================
#  POST /api/v1/llm/chat  — LLM 对话透传
# ============================================================
@router.post("/llm/chat")
def llm_chat(
    data: LLMChatRequest,
    user: dict = Depends(get_current_user),
):
    """将对话请求透传到 LLM 后端，返回模型回复。

    需要环境变量 LLM_API_KEY 已配置。
    """
    if not LLM_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="LLM 服务未配置（缺少 LLM_API_KEY 环境变量）",
        )

    payload = {
        "model": data.model,
        "messages": data.messages,
        "temperature": data.temperature,
        "max_tokens": data.max_tokens,
    }

    try:
        resp = requests.post(
            f"{LLM_BASE_URL}/chat/completions",
            json=payload,
            headers={
                "Authorization": f"Bearer {LLM_API_KEY}",
                "Content-Type": "application/json",
            },
            timeout=LLM_TIMEOUT,
        )
        resp.raise_for_status()
        result = resp.json()
        logger.info(f"LLM chat: model={data.model}, user={user['id']}, tokens_used")
        return result
    except requests.exceptions.Timeout:
        logger.error(f"LLM 请求超时 ({LLM_TIMEOUT}s)")
        raise HTTPException(status_code=504, detail=f"LLM 请求超时（{LLM_TIMEOUT}s）")
    except requests.exceptions.HTTPError as exc:
        logger.error(f"LLM 后端返回错误: {exc}")
        raise HTTPException(
            status_code=502,
            detail=f"LLM 后端错误: {exc.response.status_code if exc.response else '未知'}",
        )
    except requests.exceptions.RequestException as exc:
        logger.error(f"LLM 连接失败: {exc}")
        raise HTTPException(status_code=502, detail=f"无法连接 LLM 后端: {exc}")


# ============================================================
#  GET /api/v1/llm/models  — 可用模型列表
# ============================================================
@router.get("/llm/models")
def list_llm_models(
    user: dict = Depends(get_current_user),
):
    """获取 LLM 后端提供的可用模型列表。"""
    if not LLM_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="LLM 服务未配置（缺少 LLM_API_KEY 环境变量）",
        )

    try:
        resp = requests.get(
            f"{LLM_BASE_URL}/models",
            headers={"Authorization": f"Bearer {LLM_API_KEY}"},
            timeout=LLM_TIMEOUT,
        )
        resp.raise_for_status()
        result = resp.json()
        return result
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail=f"LLM 请求超时（{LLM_TIMEOUT}s）")
    except requests.exceptions.RequestException as exc:
        logger.error(f"获取 LLM 模型列表失败: {exc}")
        raise HTTPException(status_code=502, detail=f"无法连接 LLM 后端: {exc}")
