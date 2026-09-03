# SPDX-License-Identifier: AGPL-3.0-only
"""Precision gate: compare FP32 vs FP16 PPE detections against frozen-test labels.

Runs the same end-to-end PPE graph at FP32 and FP16 through ONNX Runtime over a
frozen, labeled test split, and reports per-class precision/recall deltas. This
measures the FP16 quantization effect of the exact graph the TensorRT engines
are built from; the mixed-precision TensorRT engine keeps sensitive layers in
FP32, so a pass here is a conservative gate.

The result does not measure association person-EPP or fall detection.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnx
import onnxruntime as ort

IOU_MATCH = 0.5
IMGSZ = 640
PADDING = 114


def letterbox(image: np.ndarray) -> tuple[np.ndarray, float, int, int]:
    scale = min(IMGSZ / image.shape[1], IMGSZ / image.shape[0])
    resized_w, resized_h = round(image.shape[1] * scale), round(image.shape[0] * scale)
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    left = (IMGSZ - resized_w) // 2
    top = (IMGSZ - resized_h) // 2
    canvas = np.full((IMGSZ, IMGSZ, 3), PADDING, dtype=np.uint8)
    canvas[top:top + resized_h, left:left + resized_w] = resized
    return canvas, scale, left, top


def nchw(image: np.ndarray) -> np.ndarray:
    tensor = image[:, :, ::-1].astype(np.float32) / 255.0
    return tensor.transpose(2, 0, 1)[None]


def restore_box(box: list[float], scale: float, left: int, top: int) -> list[float]:
    return [
        (box[0] - left) / scale,
        (box[1] - top) / scale,
        (box[2] - left) / scale,
        (box[3] - top) / scale,
    ]


def decode(output: np.ndarray, confidence: float) -> list[dict[str, Any]]:
    rows = output[0]
    detections = []
    for x1, y1, x2, y2, score, class_value in rows.tolist():
        class_id = int(class_value)
        if score < confidence or not 0 <= class_id < 8:
            continue
        detections.append({"box": [x1, y1, x2, y2], "score": score, "class_id": class_id})
    return detections


def iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match(detections: list[dict[str, Any]], ground_truth: list[dict[str, Any]]) -> dict[int, list[int]]:
    order = sorted(range(len(detections)), key=lambda i: -detections[i]["score"])
    taken = [False] * len(ground_truth)
    result: dict[int, list[int]] = {}
    for index in order:
        detection = detections[index]
        best, best_iou = None, IOU_MATCH
        for gt_index, gt in enumerate(ground_truth):
            if taken[gt_index] or gt["class_id"] != detection["class_id"]:
                continue
            value = iou(detection["box"], gt["box"])
            if value >= best_iou:
                best, best_iou = gt_index, value
        if best is not None:
            taken[best] = True
            result.setdefault(detection["class_id"], []).append(1)
        else:
            result.setdefault(detection["class_id"], []).append(0)
    return result


def counts(matches: dict[int, list[int]], ground_truth: list[dict[str, Any]], class_count: int) -> dict[int, dict[str, int]]:
    stats: dict[int, dict[str, int]] = {
        class_id: {"tp": 0, "fp": 0, "fn": 0} for class_id in range(class_count)
    }
    for class_id, values in matches.items():
        stats[class_id]["tp"] += sum(values)
        stats[class_id]["fp"] += len(values) - sum(values)
    for gt in ground_truth:
        stats[gt["class_id"]]["fn"] += 1
    return stats


def precision_recall(stats: dict[str, int]) -> tuple[float, float]:
    precision = stats["tp"] / (stats["tp"] + stats["fp"]) if stats["tp"] + stats["fp"] else 1.0
    recall = stats["tp"] / (stats["tp"] + stats["fn"]) if stats["tp"] + stats["fn"] else 1.0
    return precision, recall


def convert_to_fp16(source: Path, target: Path) -> None:
    # onnxruntime.transformers handles cast wiring around block-listed ops
    # (Resize, TopK...) correctly, unlike onnxconverter_common on this graph.
    from onnxruntime.transformers.onnx_model import OnnxModel

    wrapped = OnnxModel(onnx.load(str(source)))
    wrapped.convert_float_to_float16(keep_io_types=True)
    wrapped.save_model_to_file(str(target), use_external_data_format=False)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="FP32 vs FP16 precision gate on a frozen, labeled test split."
    )
    parser.add_argument("--onnx", type=Path, required=True, help="End-to-end FP32 PPE ONNX baseline.")
    parser.add_argument(
        "--fp16-onnx",
        type=Path,
        help="Pre-converted FP16 candidate; converted from --onnx when omitted.",
    )
    parser.add_argument("--data", type=Path, required=True, help="Frozen split data.yaml.")
    parser.add_argument("--confidence", type=float, default=0.30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data_root = args.data.resolve().parent
    images_dir = data_root / "images" / "test"
    labels_dir = data_root / "labels" / "test"
    image_paths = sorted(images_dir.glob("*.jpg"))
    if not image_paths:
        raise SystemExit(f"No frozen test images found under {images_dir}")

    fp16_path = (args.fp16_onnx.resolve() if args.fp16_onnx
                 else args.output.parent / (args.onnx.stem + "-fp16.onnx"))
    fp16_path.parent.mkdir(parents=True, exist_ok=True)
    if not args.fp16_onnx:
        convert_to_fp16(args.onnx.resolve(), fp16_path)

    def session(model_path: Path) -> ort.InferenceSession:
        return ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    sessions = {"fp32": session(args.onnx.resolve()), "fp16": session(fp16_path)}
    input_name = sessions["fp32"].get_inputs()[0].name

    names = [
        "Gloves", "Person", "Safety_boots", "Vest",
        "respirador", "tapaorejas", "Hard_hat", "lentes_protectores",
    ]
    matched: dict[str, dict[int, list[int]]] = {"fp32": {}, "fp16": {}}
    ground_truth_total: list[dict[str, Any]] = []

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            raise SystemExit(f"Unreadable frozen test image: {image_path}")
        height, width = image.shape[:2]
        canvas, scale, left, top = letterbox(image)
        tensor = nchw(canvas)
        label_path = labels_dir / (image_path.stem + ".txt")
        ground_truth = []
        if label_path.is_file():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                cx, cy, w, h = (float(v) for v in parts[1:])
                box = [
                    (cx - w / 2) * width,
                    (cy - h / 2) * height,
                    (cx + w / 2) * width,
                    (cy + h / 2) * height,
                ]
                ground_truth.append({"class_id": int(parts[0]), "box": box})
        ground_truth_total.extend(ground_truth)

        for name, session_value in sessions.items():
            output = session_value.run(None, {input_name: tensor})[0]
            detections = [
                {**detection, "box": restore_box(detection["box"], scale, left, top)}
                for detection in decode(output, args.confidence)
            ]
            for class_id, values in match(detections, ground_truth).items():
                matched[name].setdefault(class_id, []).extend(values)

    class_count = len(names)
    stats = {
        name: counts(values, ground_truth_total, class_count)
        for name, values in matched.items()
    }
    per_class = []
    for class_id, name in enumerate(names):
        fp32_p, fp32_r = precision_recall(stats["fp32"][class_id])
        fp16_p, fp16_r = precision_recall(stats["fp16"][class_id])
        per_class.append({
            "class_id": class_id,
            "name": name,
            "fp32": {"precision": round(fp32_p, 4), "recall": round(fp32_r, 4), **stats["fp32"][class_id]},
            "fp16": {"precision": round(fp16_p, 4), "recall": round(fp16_r, 4), **stats["fp16"][class_id]},
            "delta": {
                "precision": round(fp16_p - fp32_p, 4),
                "recall": round(fp16_r - fp32_r, 4),
            },
        })

    report = {
        "schema_version": 1,
        "scope": "precision-gate",
        "disclaimer": "No mide asociación persona-EPP ni detección de caídas.",
        "onnx": str(args.onnx.resolve()),
        "fp16_onnx": str(fp16_path),
        "frozen_split": str(images_dir.parent.parent),
        "images": len(image_paths),
        "confidence_threshold": args.confidence,
        "iou_match": IOU_MATCH,
        "imgsz": IMGSZ,
        "per_class": per_class,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"{'Class':<22} {'P32':>7} {'P16':>7} {'dP':>8} {'R32':>7} {'R16':>7} {'dR':>8}")
    for item in per_class:
        print(
            f"{item['name']:<22} {item['fp32']['precision']:>7.4f} {item['fp16']['precision']:>7.4f} "
            f"{item['delta']['precision']:>+8.4f} {item['fp32']['recall']:>7.4f} "
            f"{item['fp16']['recall']:>7.4f} {item['delta']['recall']:>+8.4f}"
        )
    print(f"\nJSON report: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
