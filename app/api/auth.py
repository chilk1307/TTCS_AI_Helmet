import hashlib
import time
from typing import Optional
from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel

from app.core import config

router = APIRouter(prefix="/api/auth", tags=["auth"])

active_tokens: dict = {}


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    token: str
    message: str


def generate_token(username: str) -> str:
    raw = f"{username}:{time.time()}:{config.SECRET_KEY}"
    return hashlib.sha256(raw.encode()).hexdigest()


def verify_token(token: str) -> bool:
    if not token:
        return False
    return token in active_tokens


@router.post("/login", response_model=LoginResponse)
async def login(req: LoginRequest):
    if req.username == config.ADMIN_USERNAME and req.password == config.ADMIN_PASSWORD:
        token = generate_token(req.username)
        active_tokens[token] = {
            "username": req.username,
            "login_time": time.time(),
        }
        return LoginResponse(token=token, message="Đăng nhập thành công")
    raise HTTPException(status_code=401, detail="Sai tên đăng nhập hoặc mật khẩu")
