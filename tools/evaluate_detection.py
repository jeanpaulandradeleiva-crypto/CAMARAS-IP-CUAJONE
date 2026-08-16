# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


DISCLAIMER = (
    "El mAP del detector no mide asociación persona-EPP, cumplimiento por persona "
    "ni detección de caídas."
)
SEMANTIC_ALIASES = {
    "helmet": {"hard_hat", "hardhat", "helmet", "safety_helmet", "casco"},
    "vest": {"vest", "safety_vest", "reflective_vest", "chaleco", "chaleco_reflectivo"},
    "safety_boots": {"safety_boots", "safety_boot", "boots", "boot", "botas", "botas_de_seguridad"},
}


def normalize_label(label: str) -> str:
    return label.strip().lower().replace("-", "_").replace(" ", "_")


def names_by_id(names: Any) -> dict[int, str]:
    items = names.items() if isinstance(names, Mapping) else enumerate(names)
    return {int(class_id): str(name) for class_id, name in items}


def validate_required_classes(names: Any) -> dict[str, int]:
    available = names_by_id(names)
    normalized = {class_id: normalize_label(name) for class_id, name in available.items()}
    resolved: dict[str, int] = {}
    for semantic, aliases in SEMANTIC_ALIASES.items():
        match = next((class_id for class_id, name in normalized.items() if name in aliases), None)
        if match is not None:
            resolved[semantic] = match
    missing = sorted(set(SEMANTIC_ALIASES) - set(resolved))
    if missing:
        raise ValueError(
            "Faltan clases semánticas requeridas: "
            f"{', '.join(missing)}. Clases disponibles: {available}"
        )
    return resolved


def metric_value(values: Any, position: int) -> float:
    return float(values[position])


def normalize_metrics(metrics: Any, names: Any) -> dict[str, Any]:
    available = names_by_id(names)
    required = validate_required_classes(available)
    box = metrics.box
    class_indices = [int(value) for value in box.ap_class_index]
    per_class = []
    for position, class_id in enumerate(class_indices):
        per_class.append(
            {
                "class_id": class_id,
                "name": available[class_id],
                "precision": metric_value(box.p, position),
                "recall": metric_value(box.r, position),
                "map50": metric_value(box.ap50, position),
                "map50_95": metric_value(box.ap, position),
            }
        )
    return {
        "schema_version": 1,
        "scope": "object_detection",
        "disclaimer": DISCLAIMER,
        "required_classes": required,
        "class_schema": [
            {"class_id": class_id, "name": available[class_id]}
            for class_id in sorted(available)
        ],
        "metrics": {
            "precision": float(box.mp),
            "recall": float(box.mr),
            "map50": float(box.map50),
            "map50_95": float(box.map),
        },
        "per_class": sorted(per_class, key=lambda item: item["class_id"]),
    }


def evaluate(
    model_path: str,
    data: str,
    split: str = "test",
    device: str | None = None,
    yolo_factory: Any | None = None,
    validation_options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if Path(model_path).suffix.lower() not in {".pt", ".engine"}:
        raise ValueError("El modelo debe tener extensión .pt o .engine.")
    if yolo_factory is None:
        from ultralytics import YOLO

        yolo_factory = YOLO
    model = yolo_factory(model_path, task="detect")
    validation_options = dict(validation_options or {})
    reserved = sorted({"data", "split", "device"} & validation_options.keys())
    if reserved:
        raise ValueError(
            "Validation options cannot override common inputs: " + ", ".join(reserved)
        )
    kwargs: dict[str, Any] = {"data": data, "split": split, **validation_options}
    if device:
        kwargs["device"] = device
    metrics = model.val(**kwargs)
    names = getattr(metrics, "names", None) or getattr(model, "names", None)
    if names is None:
        raise ValueError("La validación no devolvió nombres de clases.")
    return normalize_metrics(metrics, names)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Evalúa un detector YOLO. {DISCLAIMER}")
    parser.add_argument("model", help="Detector .pt o .engine.")
    parser.add_argument("--data", required=True, help="Dataset YAML etiquetado.")
    parser.add_argument("--split", default="test", help="Split del dataset (predeterminado: test).")
    parser.add_argument("--device")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)
    try:
        report = evaluate(args.model, args.data, args.split, args.device)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
