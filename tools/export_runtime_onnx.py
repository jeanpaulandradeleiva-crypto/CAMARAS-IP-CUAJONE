# SPDX-License-Identifier: AGPL-3.0-only
"""Export bounded dynamic ONNX artifacts and manifests consumed by the native runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from ultralytics import YOLO

from cuajone_qa.ppe import validate_ppe_labels


MANIFEST_VERSION = 3
ALLOWED_IMAGE_SIZES = (640, 768, 960, 1280)
MAXIMUM_OUTPUT_ELEMENTS = 16 * 1024 * 1024


def _shape(value_info: Any, symbolic_names: tuple[str, ...]) -> list[int | str]:
    tensor = value_info.type.tensor_type
    if len(tensor.shape.dim) != len(symbolic_names):
        raise RuntimeError(f"ONNX tensor {value_info.name!r} has an unexpected rank")
    dimensions: list[int | str] = []
    for dimension, symbolic_name in zip(tensor.shape.dim, symbolic_names, strict=True):
        if dimension.HasField("dim_value") and dimension.dim_value > 0:
            dimensions.append(int(dimension.dim_value))
        elif dimension.HasField("dim_param") and dimension.dim_param:
            dimensions.append(symbolic_name)
        else:
            raise RuntimeError(f"ONNX tensor {value_info.name!r} has an unbound dimension")
    return dimensions


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_provenance(source: Path) -> dict[str, str]:
    return {
        "filename": source.name,
        "sha256": _sha256(source),
    }


def _export_options(task: str) -> dict[str, Any]:
    options: dict[str, Any] = {
        "format": "onnx",
        "imgsz": 640,
        "batch": 1,
        "dynamic": True,
        "simplify": True,
        "opset": 12,
        "nms": False,
        "device": "cpu",
    }
    if task == "detect":
        options["end2end"] = False
    return options


def _prediction_count(image_size: int) -> int:
    return sum((image_size // stride) ** 2 for stride in (8, 16, 32))


def _validate_dynamic_model(target: Path, role: str, input_name: str, output_name: str) -> None:
    try:
        import numpy as np
        import onnxruntime as ort
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Dynamic ONNX validation requires numpy and onnxruntime."
        ) from exc

    session = ort.InferenceSession(str(target), providers=["CPUExecutionProvider"])
    for image_size in ALLOWED_IMAGE_SIZES:
        output = session.run(
            [output_name],
            {input_name: np.zeros((1, 3, image_size, image_size), dtype=np.float32)},
        )[0]
        expected_shape = (
            (1, 12, _prediction_count(image_size))
            if role == "ppe"
            else (1, 300, 57)
        )
        if output.dtype != np.float32 or tuple(output.shape) != expected_shape:
            raise RuntimeError(
                f"{role} ONNX at imgsz {image_size} returned {output.dtype} {tuple(output.shape)}, "
                f"expected float32 {expected_shape}"
            )
        if output.size > MAXIMUM_OUTPUT_ELEMENTS:
            raise RuntimeError(f"{role} ONNX output exceeds the runtime element bound")


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

    source_checkpoint = _checkpoint_provenance(source)
    print(f"Exportando {role}: {source.name} -> {target}")
    model = YOLO(str(source), task=task)
    exported_labels: list[str] | None = None
    if role == "ppe":
        names = {int(class_id): str(name) for class_id, name in dict(model.names).items()}
        validate_ppe_labels(names)
        exported_labels = [names[index] for index in range(len(names))]
    if task == "detect" and hasattr(model.model.model[-1], "end2end"):
        # The native decoder expects raw YOLO channels, not YOLO26 end-to-end [N, 6].
        model.model.model[-1].end2end = False
    exported = model.export(**_export_options(task))
    exported_path = Path(exported).resolve()
    if not exported_path.is_file():
        raise RuntimeError(f"Ultralytics no produjo el ONNX esperado: {exported_path}")

    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(exported_path, target)
    finally:
        if exported_path != target.resolve():
            exported_path.unlink(missing_ok=True)
    model = onnx.load(str(target), load_external_data=False)
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        raise RuntimeError(
            f"{target.name} debe tener exactamente un input y un output; "
            f"obtenidos {len(model.graph.input)} y {len(model.graph.output)}"
        )

    # Ultralytics marks batch symbolic whenever dynamic=True. The runtime contract
    # permits dynamic spatial axes only, so bind both public I/O batch axes to one.
    for value_info in (model.graph.input[0], model.graph.output[0]):
        batch = value_info.type.tensor_type.shape.dim[0]
        batch.ClearField("dim_param")
        batch.dim_value = 1
    onnx.checker.check_model(model)
    onnx.save(model, str(target))
    model = onnx.load(str(target), load_external_data=False)

    input_info = model.graph.input[0]
    output_info = model.graph.output[0]
    input_shape = _shape(input_info, ("batch", "channels", "height", "width"))
    output_shape = _shape(output_info, ("batch", "channels", "predictions"))
    if input_shape != [1, 3, "height", "width"]:
        raise RuntimeError(f"{target.name} must expose dynamic [1,3,height,width] input")
    expected_output_shape: list[int | str] = (
        [1, 12, "predictions"] if role == "ppe" else [1, 300, 57]
    )
    if output_shape != expected_output_shape:
        raise RuntimeError(
            f"{target.name} must expose the approved {role} output {expected_output_shape}"
        )
    _validate_dynamic_model(target, role, input_info.name, output_info.name)
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
            "shape": input_shape,
        },
        "output": {
            "name": output_info.name,
            "element_type": "float32",
            "shape": output_shape,
        },
        "dynamic_shape": {
            "batch": 1,
            "channels": 3,
            "allowed_image_sizes": list(ALLOWED_IMAGE_SIZES),
            "minimum_image_size": 640,
            "optimum_image_size": 640,
            "maximum_image_size": 1280,
        },
        "provenance": {
            "source_uri": f"urn:cuajone:model:{role}:ultralytics-yolo26",
            "exporter": "Ultralytics YOLO ONNX export",
            "license": "AGPL-3.0-only",
            "source_checkpoint": source_checkpoint,
        },
    }
    if exported_labels is not None:
        payload["labels"] = exported_labels
        payload["label_contract"] = "always-all-seven-v2"
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
