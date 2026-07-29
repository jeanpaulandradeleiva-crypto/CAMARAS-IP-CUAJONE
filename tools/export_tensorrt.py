# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
from typing import Any, Sequence


PLAN_VERSION = 2
MANIFEST_VERSION = 1


class ExportError(RuntimeError):
    pass


def parse_imgsz(values: list[int]) -> int | tuple[int, int]:
    if len(values) == 1 and values[0] > 0:
        return values[0]
    if len(values) == 2 and all(value > 0 for value in values):
        return values[0], values[1]
    raise ValueError("--imgsz requiere uno o dos enteros positivos.")


def build_export_plan(args: argparse.Namespace) -> dict[str, Any]:
    model = Path(args.model)
    if model.suffix.lower() != ".pt":
        raise ValueError("El modelo de entrada debe ser un archivo .pt.")
    if args.batch < 1:
        raise ValueError("--batch debe ser un entero positivo.")
    if args.quantize == 8 and not args.data:
        raise ValueError("La exportación INT8 requiere --data con el YAML del dataset.")
    if args.quantize != 8 and args.data:
        raise ValueError("--data sólo es válido con --quantize 8 (INT8).")

    imgsz = parse_imgsz(args.imgsz)
    export: dict[str, Any] = {
        "format": "engine",
        "device": args.device,
        "imgsz": imgsz,
        "batch": args.batch,
        "dynamic": args.dynamic,
        "quantize": args.quantize,
    }
    if args.data:
        export["data"] = args.data
    artifact = model.with_suffix(".engine")
    return {
        "plan_version": PLAN_VERSION,
        "model": str(model),
        "task": args.task,
        "export": export,
        "artifact": str(artifact),
        "manifest": str(artifact.with_suffix(".engine.manifest.json")),
    }


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Construye y ejecuta un plan versionado de exportación TensorRT.",
    )
    parser.add_argument("model", help="Modelo detect o pose en formato .pt.")
    parser.add_argument("--task", choices=("detect", "pose"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--imgsz", nargs="+", type=int, default=[640])
    parser.add_argument("--batch", type=int, default=1)
    shape = parser.add_mutually_exclusive_group()
    shape.add_argument("--dynamic", action="store_true", help="Habilita formas dinámicas.")
    shape.add_argument("--fixed", action="store_false", dest="dynamic", help="Usa formas fijas (predeterminado).")
    parser.set_defaults(dynamic=False)
    parser.add_argument("--quantize", type=int, choices=(8, 16, 32), default=16)
    parser.add_argument("--data", help="Dataset YAML obligatorio para calibración INT8.")
    parser.add_argument("--dry-run", action="store_true", help="Imprime el plan sin importar ni cargar YOLO.")
    return parser


def validate_export_prerequisites(
    plan: dict[str, Any],
    importer: Any | None = None,
) -> dict[str, Any]:
    model = Path(plan["model"])
    if not model.is_file():
        raise ExportError(f"No existe el modelo de entrada: {model}")
    data = plan["export"].get("data")
    if data and not Path(data).is_file():
        raise ExportError(f"No existe el YAML de calibración INT8: {data}")

    device = str(plan["export"]["device"]).lower()
    if not (device == "cuda" or device.startswith("cuda:") or device.isdigit()):
        raise ExportError("La exportación TensorRT requiere un dispositivo CUDA.")

    importer = importer or importlib.import_module
    modules: dict[str, Any] = {}
    for module_name in ("torch", "tensorrt", "ultralytics"):
        try:
            modules[module_name] = importer(module_name)
        except Exception as exc:
            raise ExportError(
                f"No se puede importar {module_name}; instala y valida este "
                "prerrequisito antes de exportar."
            ) from exc
    if not modules["torch"].cuda.is_available():
        raise ExportError(
            "PyTorch no detecta CUDA. Instala una compilación CUDA compatible y "
            "verifica el controlador NVIDIA antes de exportar."
        )
    return modules


def execute_plan(plan: dict[str, Any], yolo_factory: Any | None = None) -> Any:
    if yolo_factory is None:
        # Ultralytics must not mutate deployment hosts by installing dependencies.
        os.environ["YOLO_AUTOINSTALL"] = "false"
        modules = validate_export_prerequisites(plan)
        yolo_factory = modules["ultralytics"].YOLO
    model = yolo_factory(plan["model"], task=plan["task"])
    return model.export(**plan["export"])


def validate_export_artifact(plan: dict[str, Any], exported: Any) -> Path:
    if not isinstance(exported, (str, Path)):
        raise ExportError(
            "Ultralytics no devolvió una ruta de artefacto TensorRT verificable."
        )
    expected = Path(plan["artifact"]).resolve()
    actual = Path(exported).resolve()
    if actual != expected:
        raise ExportError(
            f"La exportación devolvió un artefacto inesperado: {actual}. "
            f"Se esperaba: {expected}"
        )
    if not actual.is_file() or actual.stat().st_size == 0:
        raise ExportError(f"El artefacto TensorRT falta o está vacío: {actual}")
    return actual


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(plan: dict[str, Any], artifact: Path) -> Path:
    source = Path(plan["model"])
    manifest = Path(plan["manifest"])
    payload = {
        "manifest_version": MANIFEST_VERSION,
        "plan": plan,
        "source": {
            "file": source.name,
            "sha256": sha256_file(source),
        },
        "artifact": {
            "file": artifact.name,
            "sha256": sha256_file(artifact),
            "size_bytes": artifact.stat().st_size,
        },
    }
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    temporary = manifest.with_suffix(manifest.suffix + ".tmp")
    temporary.write_text(serialized, encoding="utf-8")
    temporary.replace(manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)
    try:
        plan = build_export_plan(args)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(plan, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    if not args.dry_run:
        try:
            exported = execute_plan(plan)
            artifact = validate_export_artifact(plan, exported)
            manifest = write_manifest(plan, artifact)
        except ExportError as exc:
            parser.exit(1, f"ERROR: {exc}\n")
        print(f"Artefacto TensorRT: {artifact}")
        print(f"Manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
