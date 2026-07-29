# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tools import export_tensorrt


def parse_args(*args: str) -> Any:
    return export_tensorrt.create_parser().parse_args(args)


def test_default_export_plan_is_fixed_batch_one_fp16() -> None:
    plan = export_tensorrt.build_export_plan(parse_args("ppe.pt", "--task", "detect"))

    assert plan == {
        "plan_version": 2,
        "model": "ppe.pt",
        "task": "detect",
        "export": {
            "format": "engine",
            "device": "cuda:0",
            "imgsz": 640,
            "batch": 1,
            "dynamic": False,
            "quantize": 16,
        },
        "artifact": "ppe.engine",
        "manifest": "ppe.engine.manifest.json",
    }


def test_int8_requires_dataset_yaml() -> None:
    with pytest.raises(ValueError, match="INT8 requiere --data"):
        export_tensorrt.build_export_plan(
            parse_args("pose.pt", "--task", "pose", "--quantize", "8")
        )


def test_dataset_is_rejected_for_non_int8_export() -> None:
    with pytest.raises(ValueError, match="sólo es válido"):
        export_tensorrt.build_export_plan(
            parse_args("ppe.pt", "--task", "detect", "--data", "ppe.yaml")
        )


def test_dry_run_prints_deterministic_plan_without_execution(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        export_tensorrt,
        "execute_plan",
        lambda _plan: (_ for _ in ()).throw(AssertionError("must not execute")),
    )

    assert export_tensorrt.main(["ppe.pt", "--task", "detect", "--dry-run"]) == 0

    output = capsys.readouterr().out.strip()
    assert json.loads(output)["export"]["quantize"] == 16
    assert output == (
        '{"artifact":"ppe.engine","export":{"batch":1,"device":"cuda:0",'
        '"dynamic":false,"format":"engine","imgsz":640,"quantize":16},'
        '"manifest":"ppe.engine.manifest.json","model":"ppe.pt",'
        '"plan_version":2,"task":"detect"}'
    )


def test_real_plan_uses_explicit_task_and_export_arguments() -> None:
    calls: list[tuple[str, str]] = []
    exports: list[dict[str, Any]] = []

    class FakeModel:
        def export(self, **kwargs: Any) -> str:
            exports.append(kwargs)
            return "ppe.engine"

    def factory(path: str, *, task: str) -> FakeModel:
        calls.append((path, task))
        return FakeModel()

    plan = export_tensorrt.build_export_plan(parse_args("ppe.pt", "--task", "detect"))

    assert export_tensorrt.execute_plan(plan, factory) == "ppe.engine"
    assert calls == [("ppe.pt", "detect")]
    assert exports == [plan["export"]]


def test_real_export_requires_existing_model_before_importing(
    tmp_path: Path,
) -> None:
    plan = export_tensorrt.build_export_plan(
        parse_args(str(tmp_path / "missing.pt"), "--task", "detect")
    )

    with pytest.raises(export_tensorrt.ExportError, match="No existe el modelo"):
        export_tensorrt.validate_export_prerequisites(
            plan,
            lambda _name: (_ for _ in ()).throw(AssertionError("must not import")),
        )


def test_real_export_requires_importable_tensorrt(
    tmp_path: Path,
) -> None:
    model = tmp_path / "ppe.pt"
    model.write_bytes(b"synthetic-model")
    plan = export_tensorrt.build_export_plan(
        parse_args(str(model), "--task", "detect")
    )

    def importer(name: str) -> Any:
        if name == "torch":
            return SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
        raise ImportError(name)

    with pytest.raises(export_tensorrt.ExportError, match="importar tensorrt"):
        export_tensorrt.validate_export_prerequisites(plan, importer)


def test_real_export_rejects_cpu_only_pytorch(
    tmp_path: Path,
) -> None:
    model = tmp_path / "ppe.pt"
    model.write_bytes(b"synthetic-model")
    plan = export_tensorrt.build_export_plan(
        parse_args(str(model), "--task", "detect")
    )

    modules = {
        "torch": SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
        "tensorrt": SimpleNamespace(),
        "ultralytics": SimpleNamespace(YOLO=object()),
    }

    with pytest.raises(export_tensorrt.ExportError, match="no detecta CUDA"):
        export_tensorrt.validate_export_prerequisites(plan, modules.__getitem__)


def test_export_artifact_and_manifest_are_verified_deterministically(
    tmp_path: Path,
) -> None:
    model = tmp_path / "ppe.pt"
    artifact = tmp_path / "ppe.engine"
    model.write_bytes(b"synthetic-model")
    artifact.write_bytes(b"synthetic-engine")
    plan = export_tensorrt.build_export_plan(
        parse_args(str(model), "--task", "detect")
    )

    verified = export_tensorrt.validate_export_artifact(plan, artifact)
    manifest = export_tensorrt.write_manifest(plan, verified)
    first_content = manifest.read_bytes()
    export_tensorrt.write_manifest(plan, verified)
    payload = json.loads(manifest.read_text(encoding="utf-8"))

    assert manifest == tmp_path / "ppe.engine.manifest.json"
    assert manifest.read_bytes() == first_content
    assert payload["manifest_version"] == 1
    assert payload["artifact"]["file"] == "ppe.engine"
    assert payload["artifact"]["size_bytes"] == len(b"synthetic-engine")
    assert payload["source"]["file"] == "ppe.pt"
    assert len(payload["artifact"]["sha256"]) == 64
    assert not manifest.with_suffix(manifest.suffix + ".tmp").exists()


def test_export_artifact_must_match_deterministic_output_path(
    tmp_path: Path,
) -> None:
    model = tmp_path / "ppe.pt"
    unexpected = tmp_path / "other.engine"
    model.write_bytes(b"synthetic-model")
    unexpected.write_bytes(b"synthetic-engine")
    plan = export_tensorrt.build_export_plan(
        parse_args(str(model), "--task", "detect")
    )

    with pytest.raises(export_tensorrt.ExportError, match="artefacto inesperado"):
        export_tensorrt.validate_export_artifact(plan, unexpected)
