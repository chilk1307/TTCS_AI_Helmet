import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.responses import HTMLResponse

from app.core import config
from app.core.logger import get_logger
from app.api import auth, video, violations, webrtc

logger = get_logger(__name__)

app = FastAPI(
    title="Hệ thống Giám sát Giao thông",
    description="Demo phát hiện vi phạm giao thông realtime",
    version="1.0.0",
)

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend")
STATIC_DIR = os.path.join(FRONTEND_DIR, "static")
TEMPLATES_DIR = os.path.join(FRONTEND_DIR, "templates")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/evidence", StaticFiles(directory=config.EVIDENCE_DIR), name="evidence")

templates = Jinja2Templates(directory=TEMPLATES_DIR)

app.include_router(auth.router)
app.include_router(video.router)
app.include_router(violations.router)
app.include_router(webrtc.router)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse(request, "dashboard.html")


@app.on_event("startup")
async def startup():
    logger.info("=" * 50)
    logger.info("HE THONG GIAM SAT GIAO THONG - STARTING")
    logger.info(f"Server: http://{config.HOST}:{config.PORT}")
    logger.info("=" * 50)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.HOST, port=config.PORT)
