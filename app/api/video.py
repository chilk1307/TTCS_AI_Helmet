import os
import uuid
import aiofiles
from fastapi import APIRouter, UploadFile, File, HTTPException

from app.core import config
from app.core.logger import get_logger
from app.services.inference.pipeline import pipeline_manager

logger = get_logger(__name__)
router = APIRouter(prefix="/api/videos", tags=["videos"])


@router.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    video_id = str(uuid.uuid4())[:8]
    ext = os.path.splitext(file.filename)[1] or ".mp4"
    save_path = os.path.join(config.UPLOADS_DIR, f"{video_id}{ext}")

    async with aiofiles.open(save_path, "wb") as f:
        content = await file.read()
        await f.write(content)

    logger.info(f"Video uploaded: {video_id} -> {save_path}")
    return {
        "video_id": video_id,
        "filename": file.filename,
        "path": save_path,
        "message": "Upload thành công",
    }


@router.post("/{video_id}/start")
async def start_analysis(video_id: str):
    uploads = os.listdir(config.UPLOADS_DIR)
    video_file = None
    for f in uploads:
        if f.startswith(video_id):
            video_file = os.path.join(config.UPLOADS_DIR, f)
            break

    if not video_file:
        raise HTTPException(status_code=404, detail="Video không tìm thấy")

    existing = pipeline_manager.get_pipeline(video_id)
    if existing and existing.running:
        return {"message": "Pipeline đang chạy", "video_id": video_id}

    pipeline = pipeline_manager.create_pipeline(video_id, video_file)
    await pipeline.start()

    return {"message": "Bắt đầu phân tích", "video_id": video_id}


@router.post("/{video_id}/stop")
async def stop_analysis(video_id: str):
    pipeline_manager.stop_pipeline(video_id)
    return {"message": "Đã dừng phân tích", "video_id": video_id}
