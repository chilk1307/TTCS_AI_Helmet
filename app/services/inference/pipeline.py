import asyncio
import time
import uuid
import numpy as np
from typing import Any, Optional, Dict, List
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from app.core import config
from app.core.logger import get_logger
from app.services.inference.types import Stage1Result, ViolationRecord, BBox
from app.services.video.reader import VideoReader
from app.services.video.frame_utils import draw_bboxes_tracked, crop_bbox, resize_for_stream
from app.services.storage.sheet_writer import SheetWriter
from app.services.storage.evidence_writer import EvidenceWriter
from app.services.realtime.fps_counter import FPSCounter
from app.services.realtime.queues import put_drop_oldest
from app.services.realtime.tracker import IoUTracker, Track

logger = get_logger(__name__)

try:
    from faker import Faker
    fake = Faker("vi_VN")
except Exception:
    from faker import Faker
    fake = Faker()

_stage1_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stage1")
_violation_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="violation")


class PipelineManager:

    def __init__(self):
        self.pipelines: Dict[str, "VideoPipeline"] = {}
        self._models_loaded = False
        self.stage1: Optional[Stage1MotorbikeDetector] = None
        self.stage2: Optional[Stage2PartDetector] = None
        self.stage3: Optional[Stage3HelmetOCR] = None
        self.sheet_writer = SheetWriter()
        self.evidence_writer = EvidenceWriter()

    def load_models(self):
        if self._models_loaded:
            return

        backend = config.INFERENCE_BACKEND
        logger.info(f"Loading AI models (backend={backend})...")

        if backend == "cpp":
            from app.services.inference.onnx_backend import (
                Stage1OnnxDetector, Stage2OnnxDetector, Stage3OnnxOCR,
            )
            self.stage1 = Stage1OnnxDetector()
            self.stage2 = Stage2OnnxDetector()
            self.stage3 = Stage3OnnxOCR()
        else:
            from app.services.inference.stage1_motorbike_detector import Stage1MotorbikeDetector
            from app.services.inference.stage2_part_detector import Stage2PartDetector
            from app.services.inference.stage3_helmet_ocr import Stage3HelmetOCR
            self.stage1 = Stage1MotorbikeDetector()
            self.stage2 = Stage2PartDetector()
            self.stage3 = Stage3HelmetOCR()

        self._models_loaded = True
        logger.info(f"All models loaded ({backend}).")

    def create_pipeline(self, video_id: str, video_path: str) -> "VideoPipeline":
        if not self._models_loaded:
            self.load_models()
        pipeline = VideoPipeline(
            video_id=video_id,
            video_path=video_path,
            stage1=self.stage1,
            stage2=self.stage2,
            stage3=self.stage3,
            sheet_writer=self.sheet_writer,
            evidence_writer=self.evidence_writer,
        )
        self.pipelines[video_id] = pipeline
        return pipeline

    def get_pipeline(self, video_id: str) -> Optional["VideoPipeline"]:
        return self.pipelines.get(video_id)

    def stop_pipeline(self, video_id: str):
        pipeline = self.pipelines.get(video_id)
        if pipeline:
            pipeline.stop()


class VideoPipeline:

    def __init__(
        self,
        video_id: str,
        video_path: str,
        stage1: Any,
        stage2: Any,
        stage3: Any,
        sheet_writer: SheetWriter,
        evidence_writer: EvidenceWriter,
    ):
        self.video_id = video_id
        self.video_path = video_path
        self.stage1 = stage1
        self.stage2 = stage2
        self.stage3 = stage3
        self.sheet_writer = sheet_writer
        self.evidence_writer = evidence_writer

        self.frame_queue: asyncio.Queue = asyncio.Queue(maxsize=config.QUEUE_MAX_SIZE)
        self.stage2_queue: asyncio.Queue = asyncio.Queue(maxsize=config.QUEUE_MAX_SIZE)
        self.display_queue: asyncio.Queue = asyncio.Queue(maxsize=config.QUEUE_MAX_SIZE)
        self.ocr_queue: asyncio.Queue = asyncio.Queue(maxsize=config.QUEUE_MAX_SIZE)

        self.status = "idle"
        self.running = False
        self._reader_done = False
        self._stage1_done = False
        self.show_roi = False

        self.input_fps = FPSCounter()
        self.stage1_fps = FPSCounter()
        self.stream_fps = FPSCounter()
        self.violation_fps = FPSCounter()

        self.current_frame: Optional[np.ndarray] = None
        self.violations_count = 0
        self._tasks = []

        # Tracking
        self.tracker = IoUTracker(iou_threshold=0.3, max_age_seconds=2.0)
        self._processed_track_ids: set = set()
        self._frame_counter = 0

        # Dedup
        self._dedup_cache: Dict[str, float] = {}

    async def start(self):
        if self.running:
            return
        self.running = True
        self.status = "running"

        self._tasks = [
            asyncio.create_task(self._frame_reader_task()),
            asyncio.create_task(self._stage1_task()),
            asyncio.create_task(self._stage2_task()),
            asyncio.create_task(self._display_task()),
            asyncio.create_task(self._ocr_task()),
        ]
        logger.info(f"Pipeline started: {self.video_id}")

    def stop(self):
        self.running = False
        self.status = "stopped"
        for task in self._tasks:
            task.cancel()
        self._tasks.clear()
        logger.info(f"Pipeline stopped: {self.video_id}")

    def get_status(self) -> dict:
        return {
            "video_id": self.video_id,
            "status": self.status,
            "input_fps": round(self.input_fps.fps, 1),
            "stage1_fps": round(self.stage1_fps.fps, 1),
            "stream_fps": round(self.stream_fps.fps, 1),
            "violation_fps": round(self.violation_fps.fps, 1),
            "frame_queue_size": self.frame_queue.qsize(),
            "stage2_queue_size": self.stage2_queue.qsize(),
            "display_queue_size": self.display_queue.qsize(),
            "ocr_queue_size": self.ocr_queue.qsize(),
            "violations_count": self.violations_count,
            "active_tracks": len(self.tracker.tracks),
            "roi_enabled": config.ENABLE_ROI,
            "detection_line": config.DETECTION_LINE_RATIO if config.ENABLE_DETECTION_LINE else None,
            "roi": {
                "x_start": config.ROI_X_START_RATIO,
                "x_end": config.ROI_X_END_RATIO,
                "y_start": config.ROI_Y_START_RATIO,
                "y_end": config.ROI_Y_END_RATIO,
            },
        }

    async def _frame_reader_task(self):
        reader = VideoReader(self.video_path)
        if not reader.open():
            self.status = "error"
            self.running = False
            self._reader_done = True
            return

        target_fps = min(config.TARGET_FPS, reader.fps)
        frame_interval = 1.0 / target_fps

        try:
            for frame_idx, frame in reader.read_frames():
                if not self.running:
                    break

                t0 = time.monotonic()
                self.input_fps.tick()
                await put_drop_oldest(self.frame_queue, (frame_idx, frame))

                elapsed = time.monotonic() - t0
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
        except asyncio.CancelledError:
            pass
        finally:
            reader.release()
            self._reader_done = True
            logger.info(f"Frame reader done: {self.video_id}")

    async def _stage1_task(self):
        """Stage1 detect + tracking. Only send UNPROCESSED tracks to violation queue."""
        loop = asyncio.get_event_loop()
        try:
            while True:
                if self._reader_done and self.frame_queue.empty():
                    break

                try:
                    frame_idx, frame = await asyncio.wait_for(
                        self.frame_queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    if self._reader_done and self.frame_queue.empty():
                        break
                    continue

                # Stage1 detect
                result: Stage1Result = await loop.run_in_executor(
                    _stage1_executor, self.stage1.detect, frame, frame_idx
                )
                self.stage1_fps.tick()

                # Update tracker with new detections
                tracks = self.tracker.update(result.bboxes)

                # Draw all tracked objects (processed or not)
                annotated = draw_bboxes_tracked(result.frame, tracks, show_roi=self.show_roi)
                self.current_frame = resize_for_stream(annotated)
                self.stream_fps.tick()

                # Only send unprocessed tracks to stage2 queue
                self._frame_counter += 1
                if self._frame_counter % config.VIOLATION_PROCESS_EVERY_N_FRAMES == 0:
                    unprocessed = self.tracker.get_unprocessed(tracks)
                    if unprocessed:
                        await put_drop_oldest(
                            self.stage2_queue,
                            (frame, frame_idx, unprocessed)
                        )

        except asyncio.CancelledError:
            pass
        finally:
            self._stage1_done = True
            logger.info(f"Stage1 done: {self.video_id}")

    async def _stage2_task(self):
        """Run stage2 once, split output to display and OCR queues."""
        loop = asyncio.get_event_loop()
        try:
            while True:
                if self._stage1_done and self.stage2_queue.empty():
                    break

                try:
                    item = await asyncio.wait_for(
                        self.stage2_queue.get(), timeout=2.0
                    )
                except asyncio.TimeoutError:
                    if self._stage1_done and self.stage2_queue.empty():
                        break
                    continue

                frame, frame_idx, unprocessed_tracks = item
                crops = []
                valid_tracks = []
                for track in unprocessed_tracks:
                    crop = crop_bbox(frame, track.bbox)
                    if crop.size > 0:
                        crops.append(crop)
                        valid_tracks.append(track)

                if not crops:
                    continue

                try:
                    vehicle_bboxes = [t.bbox for t in valid_tracks]
                    stage2_results = await loop.run_in_executor(
                        _violation_executor,
                        self.stage2.detect_batch, crops, vehicle_bboxes
                    )
                except Exception as e:
                    logger.error(f"Stage2 error: {e}")
                    continue

                # Đẩy vào display queue để hiển thị ngay
                await put_drop_oldest(self.display_queue, (frame, frame_idx, valid_tracks, stage2_results))

                # Đẩy những cái no-helmet vào ocr queue
                ocr_items = []
                for i, (track, result) in enumerate(zip(valid_tracks, stage2_results)):
                    if result.nohelmet_bboxes:
                        ocr_items.append((track, result, crops[i]))
                    elif result.helmet_bboxes:
                        self.tracker.mark_processed(track.track_id, "safe")

                if ocr_items:
                    await put_drop_oldest(self.ocr_queue, (frame, frame_idx, ocr_items))

        except asyncio.CancelledError:
            pass

    async def _display_task(self):
        """Show stage2 results realtime without waiting for OCR."""
        try:
            while True:
                if self._stage1_done and self.stage2_queue.empty() and self.display_queue.empty():
                    break

                try:
                    item = await asyncio.wait_for(
                        self.display_queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    if self._stage1_done and self.stage2_queue.empty() and self.display_queue.empty():
                        break
                    continue

                frame, frame_idx, valid_tracks, stage2_results = item
                annotated = draw_bboxes_tracked(frame, valid_tracks, show_roi=self.show_roi)
                self.current_frame = resize_for_stream(annotated)
                self.stream_fps.tick()

        except asyncio.CancelledError:
            pass

    async def _ocr_task(self):
        """Process only no-helmet tracks through OCR/3 in background."""
        loop = asyncio.get_event_loop()
        try:
            while True:
                if self._stage1_done and self.ocr_queue.empty():
                    break

                try:
                    item = await asyncio.wait_for(
                        self.ocr_queue.get(), timeout=2.0
                    )
                except asyncio.TimeoutError:
                    if self._stage1_done and self.ocr_queue.empty():
                        break
                    continue

                frame, frame_idx, ocr_items = item
                await loop.run_in_executor(
                    _violation_executor,
                    self._process_ocr_batch, frame, frame_idx, ocr_items
                )
                self.violation_fps.tick()

        except asyncio.CancelledError:
            pass
        finally:
            self.status = "completed"
            self.running = False
            logger.info(f"Pipeline completed: {self.video_id} | violations={self.violations_count}")

    def _process_ocr_batch(self, frame: np.ndarray, frame_idx: int, ocr_items):
        """Process no-helmet tracks through OCR/3."""
        for track, s2_result, crop in ocr_items:
            self.tracker.mark_processed(track.track_id, "violation")

            plate_text = None
            plate_crop = None

            if s2_result.plate_bboxes:
                best_plate = max(s2_result.plate_bboxes, key=lambda b: b.confidence)
                plate_crop = crop_bbox(crop, best_plate)

            if plate_crop is not None and plate_crop.size > 0:
                try:
                    plate_text = self.stage3.ocr_plate(plate_crop)
                except Exception as e:
                    logger.error(f"OCR error: {e}")

            if self._is_duplicate(plate_text, track.bbox):
                continue

            # Save violation
            violation_id = str(uuid.uuid4())[:8]
            vehicle_path = self.evidence_writer.save_vehicle(violation_id, crop)
            plate_path = None
            if plate_crop is not None and plate_crop.size > 0:
                plate_path = self.evidence_writer.save_plate(violation_id, plate_crop)
            frame_path = self.evidence_writer.save_frame(violation_id, frame)

            record = ViolationRecord(
                id=violation_id,
                ten=fake.name(),
                bien_so=plate_text or "Không rõ",
                minh_chung=vehicle_path or "",
                thoi_gian=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                frame_index=frame_idx,
                confidence=float(s2_result.nohelmet_bboxes[0].confidence),
                video_id=self.video_id,
                vehicle_image_path=vehicle_path,
                plate_image_path=plate_path,
                frame_image_path=frame_path,
            )

            self.sheet_writer.save_violation(record)
            self.violations_count += 1

    def _is_duplicate(self, plate_text: Optional[str], bbox: BBox) -> bool:
        now = time.time()

        expired = [k for k, t in self._dedup_cache.items() if now - t > config.DEDUP_SECONDS]
        for k in expired:
            del self._dedup_cache[k]

        if plate_text and plate_text != "Không rõ":
            if plate_text in self._dedup_cache:
                return True
            self._dedup_cache[plate_text] = now
            return False

        bbox_key = f"{int(bbox.x1)}_{int(bbox.y1)}_{int(bbox.x2)}_{int(bbox.y2)}"
        if bbox_key in self._dedup_cache:
            return True
        self._dedup_cache[bbox_key] = now
        return False


pipeline_manager = PipelineManager()
