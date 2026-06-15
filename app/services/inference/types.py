from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str


@dataclass
class Stage1Result:
    frame: np.ndarray
    frame_index: int
    bboxes: List[BBox] = field(default_factory=list)
    roi_offset: tuple = (0, 0)


@dataclass
class Stage2Result:
    vehicle_bbox: BBox
    helmet_bboxes: List[BBox] = field(default_factory=list)
    nohelmet_bboxes: List[BBox] = field(default_factory=list)
    plate_bboxes: List[BBox] = field(default_factory=list)


@dataclass
class ViolationRecord:
    id: str
    ten: str
    bien_so: str
    minh_chung: str
    thoi_gian: str
    frame_index: int
    confidence: float
    video_id: str
    vehicle_image_path: Optional[str] = None
    plate_image_path: Optional[str] = None
    frame_image_path: Optional[str] = None
