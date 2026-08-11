# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import importlib
import sys

import pytest


MODULE_NAME = "tools.export_runtime_onnx"


def test_module_import_does_not_require_onnx(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "onnx", None)
    monkeypatch.delitem(sys.modules, MODULE_NAME, raising=False)

    module = importlib.import_module(MODULE_NAME)

    assert module._export_options("detect")["end2end"] is False


def test_export_reports_missing_onnx_before_export(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    module = importlib.import_module(MODULE_NAME)
    monkeypatch.setitem(sys.modules, "onnx", None)
    monkeypatch.setattr(
        module,
        "YOLO",
        lambda *_args, **_kwargs: pytest.fail("Ultralytics export must not start"),
    )

    with pytest.raises(RuntimeError, match=r"uv pip install onnx"):
        module.export_one(tmp_path / "source.pt", tmp_path / "target.onnx", "ppe", "detect")


def test_detect_export_is_raw_and_pose_keeps_ultralytics_default() -> None:
    module = importlib.import_module(MODULE_NAME)
    detect_options = module._export_options("detect")
    pose_options = module._export_options("pose")

    assert detect_options["end2end"] is False
    assert "end2end" not in pose_options
    assert detect_options["nms"] is False
    assert pose_options["nms"] is False
    assert detect_options["dynamic"] is True
    assert pose_options["dynamic"] is True
    assert detect_options["device"] == "cpu"
    assert pose_options["device"] == "cpu"


def test_dynamic_contract_uses_exact_bounded_sizes() -> None:
    module = importlib.import_module(MODULE_NAME)

    assert module.MANIFEST_VERSION == 2
    assert module.ALLOWED_IMAGE_SIZES == (640, 768, 960, 1280)
    assert [module._prediction_count(size) for size in module.ALLOWED_IMAGE_SIZES] == [
        8400,
        12096,
        18900,
        33600,
    ]
