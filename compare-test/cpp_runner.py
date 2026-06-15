"""C++ ONNX inference via libtraffic_inference.so (ctypes)."""
from __future__ import annotations

import ctypes
import glob
import os
import sys
from typing import List, Optional

import numpy as np

from metrics import Box


class CDetection(ctypes.Structure):
    _fields_ = [
        ("x1", ctypes.c_float),
        ("y1", ctypes.c_float),
        ("x2", ctypes.c_float),
        ("y2", ctypes.c_float),
        ("confidence", ctypes.c_float),
        ("class_id", ctypes.c_int),
    ]


class CDetectionResult(ctypes.Structure):
    _fields_ = [
        ("detections", CDetection * 512),
        ("count", ctypes.c_int),
    ]


_lib: Optional[ctypes.CDLL] = None
_initialized = False


def _preload_ort_libs() -> None:
    try:
        import onnxruntime as ort

        capi_dir = os.path.join(os.path.dirname(ort.__file__), "capi")
    except ImportError:
        return
    if not os.path.isdir(capi_dir):
        return
    os.environ["LD_LIBRARY_PATH"] = capi_dir + os.pathsep + os.environ.get(
        "LD_LIBRARY_PATH", ""
    )
    for pattern in (
        "libonnxruntime_providers_shared.so",
        "libonnxruntime_providers_cuda.so",
        "libonnxruntime.so*",
    ):
        for lib_path in sorted(glob.glob(os.path.join(capi_dir, pattern))):
            try:
                ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass


def _preload_cuda_libs() -> None:
    nvidia_base = os.path.join(
        sys.prefix,
        "lib",
        f"python{sys.version_info.major}.{sys.version_info.minor}",
        "site-packages",
        "nvidia",
    )
    if not os.path.isdir(nvidia_base):
        return
    for lib_dir in glob.glob(os.path.join(nvidia_base, "*/lib")):
        os.environ["LD_LIBRARY_PATH"] = lib_dir + os.pathsep + os.environ.get(
            "LD_LIBRARY_PATH", ""
        )
        for name in ("libcublas.so.12", "libcublasLt.so.12", "libcudart.so.12"):
            path = os.path.join(lib_dir, name)
            if os.path.isfile(path):
                try:
                    ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass


def _get_lib(lib_path: str) -> ctypes.CDLL:
    global _lib
    if _lib is not None:
        return _lib
    if not os.path.isfile(lib_path):
        raise FileNotFoundError(f"C++ library not found: {lib_path}")
    _preload_ort_libs()
    _preload_cuda_libs()
    _lib = ctypes.CDLL(lib_path)

    _lib.engine_init.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_int,
    ]
    _lib.engine_init.restype = ctypes.c_int
    _lib.engine_shutdown.argtypes = []
    _lib.engine_shutdown.restype = None

    _lib.stage1_detect.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(CDetectionResult),
    ]
    _lib.stage1_detect.restype = ctypes.c_int

    _lib.stage2_detect.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.POINTER(CDetectionResult),
    ]
    _lib.stage2_detect.restype = ctypes.c_int
    return _lib


def init_engine(
    lib_path: str,
    stage1_onnx: str,
    stage2_onnx: str,
    stage3_onnx: str,
    imgsz: int,
    use_cuda: bool = True,
) -> None:
    global _initialized
    lib = _get_lib(lib_path)
    s1 = stage1_onnx.encode()
    s2 = stage2_onnx.encode()
    s3 = stage3_onnx.encode()
    ret = lib.engine_init(s1, s2, s3, imgsz, 1 if use_cuda else 0)
    if ret != 0:
        raise RuntimeError("engine_init failed")
    _initialized = True


def shutdown_engine() -> None:
    global _initialized, _lib
    if _lib is not None and _initialized:
        _lib.engine_shutdown()
        _initialized = False


def _result_to_boxes(result: CDetectionResult) -> List[Box]:
    boxes: List[Box] = []
    for i in range(result.count):
        d = result.detections[i]
        boxes.append(
            Box(
                class_id=int(d.class_id),
                x1=float(d.x1),
                y1=float(d.y1),
                x2=float(d.x2),
                y2=float(d.y2),
                confidence=float(d.confidence),
            )
        )
    return boxes


class CppStage1Runner:
    def __init__(self, conf: float):
        self.conf = conf

    def predict(self, bgr: np.ndarray) -> List[Box]:
        lib = _get_lib(_lib_path)
        h, w = bgr.shape[:2]
        buf = np.ascontiguousarray(bgr)
        out = CDetectionResult()
        lib.stage1_detect(
            buf.ctypes.data,
            w,
            h,
            ctypes.c_float(self.conf),
            0,
            h,
            0,
            w,
            0,
            ctypes.byref(out),
        )
        return _result_to_boxes(out)


class CppStage2Runner:
    def __init__(self, conf: float):
        self.conf = conf

    def predict(self, bgr: np.ndarray) -> List[Box]:
        lib = _get_lib(_lib_path)
        h, w = bgr.shape[:2]
        buf = np.ascontiguousarray(bgr)
        out = CDetectionResult()
        lib.stage2_detect(
            buf.ctypes.data,
            w,
            h,
            ctypes.c_float(self.conf),
            ctypes.byref(out),
        )
        return _result_to_boxes(out)


_lib_path = ""


def setup_cpp(lib_path: str, stage1_onnx: str, stage2_onnx: str, stage3_onnx: str, imgsz: int) -> None:
    global _lib_path
    _lib_path = lib_path
    init_engine(lib_path, stage1_onnx, stage2_onnx, stage3_onnx, imgsz)
