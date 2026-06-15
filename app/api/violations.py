import os
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.core import config
from app.services.inference.pipeline import pipeline_manager

router = APIRouter(prefix="/api/violations", tags=["violations"])


@router.get("")
async def list_violations(video_id: str = None):
    violations = pipeline_manager.sheet_writer.list_violations(video_id)
    return {"violations": violations, "total": len(violations)}


@router.get("/{violation_id}/evidence/{evidence_type}")
async def get_evidence(violation_id: str, evidence_type: str = "vehicle"):
    suffix_map = {
        "vehicle": "_vehicle.jpg",
        "plate": "_plate.jpg",
        "frame": "_frame.jpg",
    }
    suffix = suffix_map.get(evidence_type, "_vehicle.jpg")
    filepath = os.path.join(config.EVIDENCE_DIR, f"{violation_id}{suffix}")

    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Evidence not found")

    return FileResponse(filepath, media_type="image/jpeg")
