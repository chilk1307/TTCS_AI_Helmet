"""YOLO label I/O and detection metrics (IoU, P/R/F1, mAP@0.5)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import cv2


@dataclass
class Box:
    class_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = 1.0


def load_yolo_labels(label_path: str, img_w: int, img_h: int) -> List[Box]:
    boxes: List[Box] = []
    if not os.path.isfile(label_path):
        return boxes
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            cx, cy, bw, bh = map(float, parts[1:5])
            x1 = (cx - bw / 2.0) * img_w
            y1 = (cy - bh / 2.0) * img_h
            x2 = (cx + bw / 2.0) * img_w
            y2 = (cy + bh / 2.0) * img_h
            boxes.append(Box(cls, x1, y1, x2, y2, 1.0))
    return boxes


def box_iou(a: Box, b: Box) -> float:
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
    area_b = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_image(
    preds: Sequence[Box],
    gts: Sequence[Box],
    iou_thresh: float,
) -> Tuple[int, int, int, List[Tuple[int, float, bool]]]:
    """Return tp, fp, fn and scored preds: (class_id, conf, is_tp)."""
    sorted_preds = sorted(preds, key=lambda p: p.confidence, reverse=True)
    gt_used = [False] * len(gts)
    scored: List[Tuple[int, float, bool]] = []
    tp = fp = 0

    for pred in sorted_preds:
        best_iou = 0.0
        best_j = -1
        for j, gt in enumerate(gts):
            if gt_used[j] or gt.class_id != pred.class_id:
                continue
            iou = box_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j >= 0 and best_iou >= iou_thresh:
            gt_used[best_j] = True
            tp += 1
            scored.append((pred.class_id, pred.confidence, True))
        else:
            fp += 1
            scored.append((pred.class_id, pred.confidence, False))

    fn = sum(1 for u in gt_used if not u)
    return tp, fp, fn, scored


def average_precision(scored: List[Tuple[int, float, bool]], num_gt: int) -> float:
    if num_gt == 0:
        return 0.0
    scored = sorted(scored, key=lambda x: x[1], reverse=True)
    tp_cum = 0
    precisions: List[float] = []
    for i, (_, _, is_tp) in enumerate(scored, start=1):
        if is_tp:
            tp_cum += 1
        precisions.append(tp_cum / i)
    if not precisions:
        return 0.0
    # 11-point interpolation (VOC)
    recalls = [tp_cum / num_gt for _ in precisions]
    ap = 0.0
    for t in [i / 10.0 for i in range(11)]:
        p = 0.0
        for r, pr in zip(recalls, precisions):
            if r >= t:
                p = max(p, pr)
        ap += p / 11.0
    return ap


def evaluate_dataset(
    image_paths: Sequence[str],
    labels_dir: str,
    predict_fn,
    class_names: Dict[int, str],
    iou_thresh: float = 0.5,
) -> dict:
    total_tp = total_fp = total_fn = 0
    per_class_scored: Dict[int, List[Tuple[int, float, bool]]] = {
        cid: [] for cid in class_names
    }
    per_class_gt: Dict[int, int] = {cid: 0 for cid in class_names}
    images = 0

    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        stem = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(labels_dir, stem + ".txt")
        gts = load_yolo_labels(label_path, w, h)
        preds = predict_fn(img)

        tp, fp, fn, scored = match_image(preds, gts, iou_thresh)
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
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    per_class_ap = {}
    for cid, name in class_names.items():
        ap = average_precision(per_class_scored[cid], per_class_gt[cid])
        per_class_ap[name] = round(ap, 4)

    map50 = (
        sum(per_class_ap.values()) / len(per_class_ap) if per_class_ap else 0.0
    )

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
    }


def list_images(images_dir: str) -> List[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = []
    for name in sorted(os.listdir(images_dir)):
        if os.path.splitext(name.lower())[1] in exts:
            paths.append(os.path.join(images_dir, name))
    return paths
