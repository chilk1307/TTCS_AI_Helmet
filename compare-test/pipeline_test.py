#!/usr/bin/env python3
"""Compare system pipeline on a video: with async queues vs sequential (no queue)."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(ROOT)
sys.path.insert(0, PROJECT_ROOT)

import pipeline_config as pt_cfg


def _setup_app_config(backend: str) -> None:
    import app.core.config as config

    config.INFERENCE_BACKEND = backend
    config.EVIDENCE_DIR = pt_cfg.EVIDENCE_DIR
    config.VIOLATIONS_DIR = pt_cfg.VIOLATIONS_DIR
    config.VIOLATIONS_FILE = pt_cfg.VIOLATIONS_FILE
    os.makedirs(config.EVIDENCE_DIR, exist_ok=True)
    os.makedirs(config.VIOLATIONS_DIR, exist_ok=True)


def _video_info(path: str) -> dict:
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return {"error": "cannot open video"}
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {
        "path": path,
        "width": w,
        "height": h,
        "fps": round(fps, 2),
        "total_frames": frames,
        "duration_s": round(frames / fps, 2) if fps else None,
    }


async def run_with_queue(video_path: str, backend: str) -> dict:
    """Run production VideoPipeline (multi-task + asyncio queues)."""
    from app.core import config
    from app.services.inference.pipeline import VideoPipeline, pipeline_manager
    from app.services.storage.evidence_writer import EvidenceWriter
    from app.services.storage.sheet_writer import SheetWriter

    pipeline_manager.load_models()
    video_id = f"queue_{uuid.uuid4().hex[:8]}"
    pipeline = VideoPipeline(
        video_id=video_id,
        video_path=video_path,
        stage1=pipeline_manager.stage1,
        stage2=pipeline_manager.stage2,
        stage3=pipeline_manager.stage3,
        sheet_writer=SheetWriter(),
        evidence_writer=EvidenceWriter(),
    )

    max_queues = {
        "frame_queue": 0,
        "stage2_queue": 0,
        "display_queue": 0,
        "ocr_queue": 0,
    }

    t0 = time.perf_counter()
    await pipeline.start()

    while pipeline.status not in ("completed", "error", "stopped"):
        max_queues["frame_queue"] = max(max_queues["frame_queue"], pipeline.frame_queue.qsize())
        max_queues["stage2_queue"] = max(max_queues["stage2_queue"], pipeline.stage2_queue.qsize())
        max_queues["display_queue"] = max(max_queues["display_queue"], pipeline.display_queue.qsize())
        max_queues["ocr_queue"] = max(max_queues["ocr_queue"], pipeline.ocr_queue.qsize())
        await asyncio.sleep(0.05)

    elapsed = time.perf_counter() - t0
    status = pipeline.get_status()

    return {
        "mode": "with_queue",
        "backend": backend,
        "video_id": video_id,
        "wall_time_s": round(elapsed, 2),
        "status": pipeline.status,
        "violations_count": pipeline.violations_count,
        "input_fps": status["input_fps"],
        "stage1_fps": status["stage1_fps"],
        "stream_fps": status["stream_fps"],
        "violation_fps": status["violation_fps"],
        "active_tracks_at_end": status["active_tracks"],
        "queue_max_size": config.QUEUE_MAX_SIZE,
        "queue_peak": max_queues,
        "queue_final": {
            "frame_queue": pipeline.frame_queue.qsize(),
            "stage2_queue": pipeline.stage2_queue.qsize(),
            "display_queue": pipeline.display_queue.qsize(),
            "ocr_queue": pipeline.ocr_queue.qsize(),
        },
        "throughput_fps": round(status["input_fps"], 2) if elapsed > 0 else 0,
    }


def run_no_queue(video_path: str, backend: str) -> dict:
    """Run same pipeline logic sequentially in one loop (no asyncio queues)."""
    import uuid as _uuid
    from datetime import datetime as _dt

    import numpy as np

    from app.core import config
    from app.services.inference.pipeline import pipeline_manager
    from app.services.inference.types import ViolationRecord
    from app.services.realtime.tracker import IoUTracker
    from app.services.storage.evidence_writer import EvidenceWriter
    from app.services.storage.sheet_writer import SheetWriter
    from app.services.video.frame_utils import crop_bbox
    from app.services.video.reader import VideoReader

    try:
        from faker import Faker

        fake = Faker("vi_VN")
    except Exception:
        from faker import Faker

        fake = Faker()

    pipeline_manager.load_models()
    stage1 = pipeline_manager.stage1
    stage2 = pipeline_manager.stage2
    stage3 = pipeline_manager.stage3
    sheet_writer = SheetWriter()
    evidence_writer = EvidenceWriter()

    tracker = IoUTracker(iou_threshold=0.3, max_age_seconds=2.0)
    dedup_cache: Dict[str, float] = {}
    violations_count = 0
    frame_counter = 0
    frames_read = 0
    stage1_calls = 0
    stage2_batches = 0
    ocr_calls = 0

    reader = VideoReader(video_path)
    if not reader.open():
        return {"mode": "no_queue", "error": "cannot open video"}

    target_fps = min(config.TARGET_FPS, reader.fps)
    frame_interval = 1.0 / target_fps if target_fps else 0

    t0 = time.perf_counter()

    for frame_idx, frame in reader.read_frames():
        loop_start = time.monotonic()
        frames_read += 1

        result = stage1.detect(frame, frame_idx)
        stage1_calls += 1
        tracks = tracker.update(result.bboxes)
        frame_counter += 1

        if frame_counter % config.VIOLATION_PROCESS_EVERY_N_FRAMES != 0:
            elapsed = time.monotonic() - loop_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            continue

        unprocessed = tracker.get_unprocessed(tracks)
        if not unprocessed:
            elapsed = time.monotonic() - loop_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            continue

        crops = []
        valid_tracks = []
        for track in unprocessed:
            crop = crop_bbox(frame, track.bbox)
            if crop.size > 0:
                crops.append(crop)
                valid_tracks.append(track)

        if not crops:
            elapsed = time.monotonic() - loop_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            continue

        vehicle_bboxes = [t.bbox for t in valid_tracks]
        stage2_results = stage2.detect_batch(crops, vehicle_bboxes)
        stage2_batches += 1

        for track, s2_result, crop in zip(valid_tracks, stage2_results, crops):
            if not s2_result.nohelmet_bboxes:
                tracker.mark_processed(track.track_id, "safe")
                continue

            tracker.mark_processed(track.track_id, "violation")
            plate_text = None
            plate_crop = None

            if s2_result.plate_bboxes:
                best_plate = max(s2_result.plate_bboxes, key=lambda b: b.confidence)
                plate_crop = crop_bbox(crop, best_plate)

            if plate_crop is not None and plate_crop.size > 0:
                try:
                    plate_text = stage3.ocr_plate(plate_crop)
                    ocr_calls += 1
                except Exception:
                    pass

            now = time.time()
            expired = [k for k, t in dedup_cache.items() if now - t > config.DEDUP_SECONDS]
            for k in expired:
                del dedup_cache[k]

            is_dup = False
            if plate_text and plate_text != "Không rõ":
                if plate_text in dedup_cache:
                    is_dup = True
                else:
                    dedup_cache[plate_text] = now
            else:
                bbox = track.bbox
                bbox_key = f"{int(bbox.x1)}_{int(bbox.y1)}_{int(bbox.x2)}_{int(bbox.y2)}"
                if bbox_key in dedup_cache:
                    is_dup = True
                else:
                    dedup_cache[bbox_key] = now

            if is_dup:
                continue

            violation_id = str(_uuid.uuid4())[:8]
            vehicle_path = evidence_writer.save_vehicle(violation_id, crop)
            plate_path = None
            if plate_crop is not None and plate_crop.size > 0:
                plate_path = evidence_writer.save_plate(violation_id, plate_crop)
            frame_path = evidence_writer.save_frame(violation_id, frame)

            record = ViolationRecord(
                id=violation_id,
                ten=fake.name(),
                bien_so=plate_text or "Không rõ",
                minh_chung=vehicle_path or "",
                thoi_gian=_dt.now().strftime("%Y-%m-%d %H:%M:%S"),
                frame_index=frame_idx,
                confidence=float(s2_result.nohelmet_bboxes[0].confidence),
                video_id="no_queue_test",
                vehicle_image_path=vehicle_path,
                plate_image_path=plate_path,
                frame_image_path=frame_path,
            )
            sheet_writer.save_violation(record)
            violations_count += 1

        elapsed = time.monotonic() - loop_start
        sleep_time = frame_interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    reader.release()
    elapsed = time.perf_counter() - t0

    return {
        "mode": "no_queue",
        "backend": backend,
        "wall_time_s": round(elapsed, 2),
        "status": "completed",
        "violations_count": violations_count,
        "frames_read": frames_read,
        "stage1_calls": stage1_calls,
        "stage2_batches": stage2_batches,
        "ocr_calls": ocr_calls,
        "effective_fps": round(frames_read / elapsed, 2) if elapsed > 0 else 0,
        "queue_peak": None,
    }


def _compare(with_q: dict, no_q: dict) -> dict:
    t_with = with_q.get("wall_time_s") or 0
    t_no = no_q.get("wall_time_s") or 0
    return {
        "time_ratio_queue_over_no_queue": round(t_with / t_no, 3) if t_no > 0 else None,
        "time_saved_s": round(t_no - t_with, 2),
        "violations_delta": (with_q.get("violations_count") or 0) - (no_q.get("violations_count") or 0),
        "faster_mode": "with_queue" if t_with < t_no else "no_queue",
    }


async def main_async(args: argparse.Namespace) -> dict:
    _setup_app_config(args.code)
    os.makedirs(pt_cfg.RESULTS_DIR, exist_ok=True)

    if not os.path.isfile(pt_cfg.VIDEO_PATH):
        raise FileNotFoundError(f"Video not found: {pt_cfg.VIDEO_PATH}")

    info = _video_info(pt_cfg.VIDEO_PATH)
    print(f"Video: {info['width']}x{info['height']} @ {info['fps']}fps, {info['total_frames']} frames")
    print(f"Backend: {args.code}\n")

    report: Dict[str, Any] = {
        "started_at": datetime.now().isoformat(),
        "video": info,
        "backend": args.code,
    }

    if args.mode in ("both", "queue"):
        print("=" * 60)
        print("MODE 1: WITH QUEUE (async pipeline)")
        print("=" * 60)
        with_q = await run_with_queue(pt_cfg.VIDEO_PATH, args.code)
        report["with_queue"] = with_q
        print(
            f"  time={with_q['wall_time_s']}s  violations={with_q['violations_count']}  "
            f"stage1_fps={with_q['stage1_fps']}  peak_queues={with_q['queue_peak']}"
        )

    if args.mode in ("both", "no_queue"):
        print("\n" + "=" * 60)
        print("MODE 2: NO QUEUE (sequential)")
        print("=" * 60)
        no_q = run_no_queue(pt_cfg.VIDEO_PATH, args.code)
        report["no_queue"] = no_q
        print(
            f"  time={no_q['wall_time_s']}s  violations={no_q['violations_count']}  "
            f"effective_fps={no_q['effective_fps']}"
        )

    if "with_queue" in report and "no_queue" in report:
        report["comparison"] = _compare(report["with_queue"], report["no_queue"])
        c = report["comparison"]
        print("\n" + "=" * 60)
        print("SO SANH")
        print("=" * 60)
        print(f"  Nhanh hon: {c['faster_mode']}")
        print(f"  time ratio (queue/no_queue): {c['time_ratio_queue_over_no_queue']}")
        print(f"  chenh violations: {c['violations_delta']:+d}")

    report["finished_at"] = datetime.now().isoformat()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(pt_cfg.RESULTS_DIR, f"pipeline_{ts}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport: {out}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline queue vs no-queue test")
    parser.add_argument("--code", choices=["python", "cpp"], default="python")
    parser.add_argument("--mode", choices=["both", "queue", "no_queue"], default="both")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
