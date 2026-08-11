# SPDX-License-Identifier: AGPL-3.0-only
"""Export the fixed ONNX artifacts and manifests consumed by ppe_reportev2."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from ultralytics import YOLO


MANIFEST_VERSION = 1


def _shape(value_info: Any) -> list[int]:
    tensor = value_info.type.tensor_type
    dimensions: list[int] = []
    for dimension in tensor.shape.dim:
        if not dimension.HasField("dim_value") or dimension.dim_value <= 0:
            raise RuntimeError(f"ONNX tensor {value_info.name!r} has a dynamic dimension")
        dimensions.append(int(dimension.dim_value))
    return dimensions


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _export_options(task: str) -> dict[str, Any]:
    options: dict[str, Any] = {
        "format": "onnx",
        "imgsz": 640,
        "batch": 1,
        "dynamic": False,
        "simplify": True,
        "opset": 12,
        "nms": False,
        "device": 0,
    }
    if task == "detect":
        options["end2end"] = False
    return options


def export_one(source: Path, target: Path, role: str, task: str) -> None:
    try:
        import onnx
    except ModuleNotFoundError as exc:
        if exc.name != "onnx":
            raise
        raise RuntimeError(
            "ONNX export validation requires the optional 'onnx' package. "
            "Install it with `uv pip install onnx` and retry."
        ) from exc

    print(f"Exportando {role}: {source.name} -> {target}")
    model = YOLO(str(source), task=task)
    if task == "detect" and hasattr(model.model.model[-1], "end2end"):
        # The native decoder expects raw YOLO channels, not YOLO26 end-to-end [N, 6].
        model.model.model[-1].end2end = False
    exported = model.export(**_export_options(task))
    exported_path = Path(exported).resolve()
    if not exported_path.is_file():
        raise RuntimeError(f"Ultralytics no produjo el ONNX esperado: {exported_path}")

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(exported_path, target)
    model = onnx.load(str(target), load_external_data=False)
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        raise RuntimeError(
            f"{target.name} debe tener exactamente un input y un output; "
            f"obtenidos {len(model.graph.input)} y {len(model.graph.output)}"
        )

    input_info = model.graph.input[0]
    output_info = model.graph.output[0]
    payload = {
        "schema_version": MANIFEST_VERSION,
        "artifact_type": "onnx",
        "role": role,
        "model_file": target.name,
        "model_sha256": _sha256(target),
        "model_size_bytes": target.stat().st_size,
        "external_data": False,
        "custom_operators": False,
        "input": {
            "name": input_info.name,
            "element_type": "float32",
            "shape": _shape(input_info),
        },
        "output": {
            "name": output_info.name,
            "element_type": "float32",
            "shape": _shape(output_info),
        },
        "provenance": {
            "source_uri": f"urn:cuajone:model:{role}:ultralytics-yolo26",
            "exporter": "Ultralytics YOLO ONNX export",
            "license": "AGPL-3.0-only",
        },
    }
    manifest = target.with_name(f"{target.name}.manifest.json")
    manifest.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"  ONNX: {target} ({target.stat().st_size} bytes)")
    print(f"  Manifest: {manifest}")
    print(f"  Input: {payload['input']['name']} {payload['input']['shape']}")
    print(f"  Output: {payload['output']['name']} {payload['output']['shape']}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ppe", type=Path, required=True)
    parser.add_argument("--pose", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    export_one(args.ppe.resolve(), args.output_dir.resolve() / "ppe.onnx", "ppe", "detect")
    export_one(args.pose.resolve(), args.output_dir.resolve() / "pose.onnx", "pose", "pose")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
