#!/usr/bin/env python
# @FileName  : main.py
# @Time      : 2026/5/25
# @Project   : ODPlatform
# @Function  : FastAPI 应用入口 —— 实验数据存储与查询服务
#
# 启动方式:
#   cd apps/web-backend
#   pip install -r requirements.txt
#   python main.py
#
# 验证:
#   curl -X POST http://127.0.0.1:8000/api/experiments \
#     -H "Content-Type: application/json" \
#     -d '{"name":"test","dataset":"rsod","model":"yolo11n.pt","config_json":"{}"}'

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api import experiments, models, validation
from db.database import init_db

# ── 日志 ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(name)-25s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("odp-backend")


# ── 生命周期 ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动时初始化数据库。"""
    logger.info("正在初始化数据库...")
    init_db()
    logger.info(f"数据库已就绪: {Path(__file__).parent / 'odplatform.db'}")
    yield
    logger.info("应用关闭")


# ── FastAPI 应用 ──────────────────────────────────────────
app = FastAPI(
    title="ODPlatform API",
    version="0.1.0",
    description="目标检测开发平台 — 实验数据存储与查询服务",
    lifespan=lifespan,
)

# CORS（开发阶段允许所有来源）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(experiments.router, prefix="/api")
app.include_router(models.router, prefix="/api")
app.include_router(validation.router, prefix="/api")


# ── 健康检查 ──────────────────────────────────────────────
@app.get("/")
def root():
    """服务可用性探针。"""
    return {
        "service": "ODPlatform API",
        "version": "0.1.0",
        "status": "running",
    }


@app.get("/health")
def health():
    """健康检查端点。"""
    return {"status": "healthy"}


# ── 入口 ──────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    logger.info("启动 ODPlatform API 服务...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
