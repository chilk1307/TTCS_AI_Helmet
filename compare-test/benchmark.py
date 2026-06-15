#!/usr/bin/env python3
"""Benchmark PT vs ONNX (C++) on full yolo_1 / yolo_2 test sets."""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime

import cv2

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import config
from cpp_runner import CppStage1Runner, CppStage2Runner, setup_cpp, shutdown_engine
from metrics import list_images
from pt_runner import PtRunner


def _run_eval(runner, image_paths: list[str], labels_dir: str, class_names: dict, warmup: int) -> dict:
    """Single pass: accuracy metrics + per-image timing after warmup."""
    from metrics import Box, load_yolo_labels, match_image, average_precision

    total_tp = total_fp = total_fn = 0
    per_class_scored = {cid: [] for cid in class_names}
    per_class_gt = {cid: 0 for cid in class_names}
    times_ms: list[float] = []
    images = 0

    for i, path in enumerate(image_paths):
        img = cv2.imread(path)
        if img is None:
            continue
        h, w = img.shape[:2]
        stem = os.path.splitext(os.path.basename(path))[0]
        gts = load_yolo_labels(os.path.join(labels_dir, stem + ".txt"), w, h)

        if i < warmup:
            runner.predict(img)
            continue

        t0 = time.perf_counter()
        preds = runner.predict(img)
        times_ms.append((time.perf_counter() - t0) * 1000.0)

        tp, fp, fn, scored = match_image(preds, gts, config.IOU_THRESH)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        images += 1
        for gt in gts:
            if gt.class_id in per_class_gt:
                per_class_gt[gt.class_id] += 1
        for cls_id, conf, is_tp in scored:
            if cls_id in per_class_scored:
                per_class_scored[cls_id].append((cls_id, conf, is_tp))

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    per_class_ap = {}
    for cid, name in class_names.items():
        per_class_ap[name] = round(average_precision(per_class_scored[cid], per_class_gt[cid]), 4)
    map50 = sum(per_class_ap.values()) / len(per_class_ap) if per_class_ap else 0.0

    total_s = sum(times_ms) / 1000.0 if times_ms else 0.0
    avg_ms = sum(times_ms) / len(times_ms) if times_ms else 0.0
    return {
        "images": images,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "map50": round(map50, 4),
        "per_class_ap50": per_class_ap,
        "avg_ms": round(avg_ms, 3),
        "total_s": round(total_s, 2),
        "fps": round(len(times_ms) / total_s, 2) if total_s > 0 else 0.0,
        "timed_images": len(times_ms),
    }


def run_stage(stage_cfg: dict, stage_num: int) -> dict:
    images = list_images(stage_cfg["images_dir"])
    print(f"\n{'='*60}")
    print(f"{stage_cfg['name']} — {len(images)} images")
    print(f"{'='*60}")

    results = {}

    # --- PT ---
    print("\n[PT] Loading model...")
    pt = PtRunner(stage_cfg["pt_model"], stage_cfg["conf"], config.IMGSZ)
    print("[PT] Running benchmark...")
    pt_res = _run_eval(pt, images, stage_cfg["labels_dir"], stage_cfg["classes"], config.WARMUP_IMAGES)
    results["pt"] = pt_res
    print(
        f"  PT  mAP@0.5={pt_res['map50']:.4f}  "
        f"P={pt_res['precision']:.4f} R={pt_res['recall']:.4f}  "
        f"{pt_res['avg_ms']:.2f} ms/img"
    )

    # --- ONNX C++ ---
    print("\n[ONNX C++] Loading engine...")
    setup_cpp(config.CPP_LIB, config.STAGE1_ONNX, config.STAGE2_ONNX, config.STAGE3_ONNX, config.IMGSZ)
    if stage_num == 1:
        cpp = CppStage1Runner(stage_cfg["conf"])
    else:
        cpp = CppStage2Runner(stage_cfg["conf"])

    print("[ONNX C++] Running benchmark...")
    onnx_res = _run_eval(cpp, images, stage_cfg["labels_dir"], stage_cfg["classes"], config.WARMUP_IMAGES)
    shutdown_engine()
    results["onnx_cpp"] = onnx_res
    print(
        f"  ONNX mAP@0.5={onnx_res['map50']:.4f}  "
        f"P={onnx_res['precision']:.4f} R={onnx_res['recall']:.4f}  "
        f"{onnx_res['avg_ms']:.2f} ms/img"
    )

    # --- Delta ---
    map_delta = onnx_res["map50"] - pt_res["map50"]
    time_ratio = onnx_res["avg_ms"] / pt_res["avg_ms"] if pt_res["avg_ms"] > 0 else 0.0
    speedup = pt_res["avg_ms"] / onnx_res["avg_ms"] if onnx_res["avg_ms"] > 0 else 0.0
    results["comparison"] = {
        "map50_delta_onnx_minus_pt": round(map_delta, 4),
        "onnx_vs_pt_time_ratio": round(time_ratio, 3),
        "pt_speedup_over_onnx_x": round(speedup, 3),
        "onnx_faster": onnx_res["avg_ms"] < pt_res["avg_ms"],
    }
    print(
        f"\n  Δ mAP@0.5 (ONNX-PT): {map_delta:+.4f}  |  "
        f"time ratio ONNX/PT: {time_ratio:.2f}x  |  "
        f"{'ONNX nhanh hon' if onnx_res['avg_ms'] < pt_res['avg_ms'] else 'PT nhanh hon'}"
    )
    return results


def main() -> None:
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    started = datetime.now()
    print(f"Benchmark started: {started.isoformat()}")
    print(f"conf stage1={config.YOLO_1['conf']} stage2={config.YOLO_2['conf']} imgsz={config.IMGSZ}")

    report = {
        "started_at": started.isoformat(),
        "config": {
            "imgsz": config.IMGSZ,
            "iou_thresh": config.IOU_THRESH,
            "warmup": config.WARMUP_IMAGES,
            "yolo_1_conf": config.YOLO_1["conf"],
            "yolo_2_conf": config.YOLO_2["conf"],
        },
        "yolo_1": run_stage(config.YOLO_1, stage_num=1),
        "yolo_2": run_stage(config.YOLO_2, stage_num=2),
    }
    report["finished_at"] = datetime.now().isoformat()

    ts = started.strftime("%Y%m%d_%H%M%S")
    out_json = os.path.join(config.RESULTS_DIR, f"benchmark_{ts}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    _print_summary(report)
    print(f"\nFull report: {out_json}")


def _print_summary(report: dict) -> None:
    print("\n" + "=" * 60)
    print("TONG KET PT vs ONNX (C++)")
    print("=" * 60)
    for key, label in (("yolo_1", "YOLO_1 stage1"), ("yolo_2", "YOLO_2 stage2")):
        r = report[key]
        pt, ox = r["pt"], r["onnx_cpp"]
        cmp_ = r["comparison"]
        print(f"\n{label}:")
        print(
            f"  PT   : mAP@0.5={pt['map50']:.4f}  "
            f"P={pt['precision']:.4f} R={pt['recall']:.4f}  "
            f"{pt['avg_ms']:.2f} ms/img ({pt['fps']:.1f} fps)"
        )
        print(
            f"  ONNX : mAP@0.5={ox['map50']:.4f}  "
            f"P={ox['precision']:.4f} R={ox['recall']:.4f}  "
            f"{ox['avg_ms']:.2f} ms/img ({ox['fps']:.1f} fps)"
        )
        print(
            f"  Chenh  : mAP {cmp_['map50_delta_onnx_minus_pt']:+.4f}  |  "
            f"time ONNX/PT = {cmp_['onnx_vs_pt_time_ratio']:.2f}x"
        )
        if pt.get("per_class_ap50"):
            print(f"  PT per-class AP@0.5: {pt['per_class_ap50']}")
            print(f"  ONNX per-class AP@0.5: {ox['per_class_ap50']}")


if __name__ == "__main__":
    main()
