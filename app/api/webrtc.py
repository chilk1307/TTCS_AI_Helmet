import asyncio
import cv2
import json
import time
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core import config
from app.core.logger import get_logger
from app.services.inference.pipeline import pipeline_manager

logger = get_logger(__name__)
router = APIRouter(tags=["webrtc"])

_encode_params = [cv2.IMWRITE_JPEG_QUALITY, config.STREAM_JPEG_QUALITY]


@router.websocket("/ws/stream/{video_id}")
async def websocket_stream(websocket: WebSocket, video_id: str):
    """Stream annotated frames via WebSocket (binary JPEG + JSON metadata)."""
    await websocket.accept()
    logger.info(f"WS connected: {video_id}")

    last_frame_id = -1
    try:
        while True:
            pipeline = pipeline_manager.get_pipeline(video_id)
            if pipeline is None:
                await asyncio.sleep(0.3)
                continue

            frame = pipeline.current_frame
            if frame is not None and id(frame) != last_frame_id:
                last_frame_id = id(frame)

                _, buf = cv2.imencode(".jpg", frame, _encode_params)

                meta = json.dumps({
                    "fps": round(pipeline.stream_fps.fps, 1),
                    "s1fps": round(pipeline.stage1_fps.fps, 1),
                    "status": pipeline.status,
                    "vc": pipeline.violations_count,
                    "fq": pipeline.frame_queue.qsize(),
                    "dq": pipeline.display_queue.qsize(),
                    "oq": pipeline.ocr_queue.qsize(),
                })

                # Send metadata as text, then frame as binary
                await websocket.send_text(meta)
                await websocket.send_bytes(buf.tobytes())

            if pipeline.status == "completed" and frame is not None:
                await websocket.send_text(json.dumps({"status": "completed", "vc": pipeline.violations_count}))
                break

            await asyncio.sleep(0.025)

    except WebSocketDisconnect:
        logger.info(f"WS disconnected: {video_id}")
    except Exception as e:
        logger.error(f"WS error: {e}")


@router.get("/api/status/{video_id}")
async def get_pipeline_status(video_id: str):
    pipeline = pipeline_manager.get_pipeline(video_id)
    if not pipeline:
        return {"status": "not_found", "video_id": video_id}
    return pipeline.get_status()
