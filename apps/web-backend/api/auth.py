#!/usr/bin/env python
# @FileName  : auth.py
# @Time      : 2026/5/26
# @Project   : ODPlatform
# @Function  : 用户认证 —— 注册 / 登录

from __future__ import annotations

import hashlib
import logging
import secrets
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Header

from db.database import get_db
from schemas import UserLogin, UserRegister

logger = logging.getLogger("odp-backend.auth")

router = APIRouter(tags=["auth"])


# ── 密码哈希 ──────────────────────────────────────────────
def _hash_password(password: str, salt: Optional[str] = None) -> tuple[str, str]:
    """PBKDF2-SHA256 哈希密码，返回 (hash_hex, salt_hex)。"""
    if salt is None:
        salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000)
    return dk.hex(), salt


def _verify_password(password: str, stored_hash: str) -> bool:
    """验证密码。stored_hash 格式: salt_hex:hash_hex"""
    salt, expected = stored_hash.split(":", 1)
    actual, _ = _hash_password(password, salt)
    return secrets.compare_digest(actual, expected)


def _generate_token() -> str:
    return secrets.token_hex(32)


# ── 认证依赖（其他 API 复用） ─────────────────────────────
def get_current_user(authorization: str = Header(default="")) -> dict:
    """从 Authorization: Bearer <token> 中解析当前用户。

    其他 API 通过 Depends(get_current_user) 注入用户信息。
    """
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="未提供认证令牌")
    token = authorization[7:]
    db = get_db()
    user = db.execute(
        "SELECT id, username, email, created_at FROM users WHERE token = ?",
        (token,),
    ).fetchone()
    if user is None:
        raise HTTPException(status_code=401, detail="令牌无效或已过期")
    return dict(user)


# ============================================================
#  POST /api/v1/auth/register  — 注册
# ============================================================
@router.post("/auth/register", status_code=201)
def register(data: UserRegister):
    """用户注册。username 唯一，重复返回 409。"""
    db = get_db()

    hashed, salt = _hash_password(data.password)
    stored = f"{salt}:{hashed}"
    token = _generate_token()

    try:
        cursor = db.execute(
            """INSERT INTO users (username, password_hash, email, token)
               VALUES (?, ?, ?, ?)""",
            (data.username, stored, data.email, token),
        )
        db.commit()
        user_id = cursor.lastrowid
        logger.info(f"用户注册成功: id={user_id}, username={data.username}")
        return {
            "id": user_id,
            "username": data.username,
            "token": token,
        }
    except Exception as exc:
        db.rollback()
        if "unique" in str(exc).lower():
            raise HTTPException(status_code=409, detail=f"用户名 '{data.username}' 已被占用")
        logger.error(f"注册失败: {exc}")
        raise HTTPException(status_code=500, detail=f"数据库错误: {exc}")


# ============================================================
#  POST /api/v1/auth/login  — 登录
# ============================================================
@router.post("/auth/login")
def login(data: UserLogin):
    """用户名 + 密码登录，返回新 token。"""
    db = get_db()

    user = db.execute(
        "SELECT id, username, password_hash, email FROM users WHERE username = ?",
        (data.username,),
    ).fetchone()

    if user is None:
        raise HTTPException(status_code=401, detail="用户名或密码错误")

    if not _verify_password(data.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="用户名或密码错误")

    token = _generate_token()
    db.execute(
        "UPDATE users SET token = ? WHERE id = ?",
        (token, user["id"]),
    )
    db.commit()
    logger.info(f"用户登录: id={user['id']}, username={data.username}")

    return {
        "id": user["id"],
        "username": user["username"],
        "token": token,
    }
