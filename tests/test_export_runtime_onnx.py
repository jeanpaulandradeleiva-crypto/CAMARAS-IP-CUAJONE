# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import hashlib
import importlib
import sys

import pytest


MODULE_NAME = "tools.export_runtime_onnx"


def test_module_import_does_not_require_onnx(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "onnx", None)
    monkeypatch.delitem(sys.modules, MODULE_NAME, raising=False)

    module = importlib.import_module(MODULE_NAME)

    assert module._export_options("detect")["end2end"] is True


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


def test_detect_export_is_end2end_and_pose_keeps_ultralytics_default() -> None:
    module = importlib.import_module(MODULE_NAME)
    detect_options = module._export_options("detect")
    pose_options = module._export_options("pose")

    assert detect_options["end2end"] is True
    assert "end2end" not in pose_options
    assert detect_options["nms"] is False
    assert pose_options["nms"] is False
    assert detect_options["dynamic"] is True
    assert pose_options["dynamic"] is True
    assert detect_options["device"] == "cpu"
    assert pose_options["device"] == "cpu"


def test_dynamic_contract_uses_exact_bounded_sizes() -> None:
    module = importlib.import_module(MODULE_NAME)

    assert module.MANIFEST_VERSION == 3
    assert module.ALLOWED_IMAGE_SIZES == (640, 768, 960, 1280)
    assert [module._prediction_count(size) for size in module.ALLOWED_IMAGE_SIZES] == [
        8400,
        12096,
        18900,
        33600,
    ]


def test_checkpoint_provenance_is_calculated_from_the_selected_file(tmp_path) -> None:
    module = importlib.import_module(MODULE_NAME)
    checkpoint = tmp_path / "selected.pt"
    checkpoint.write_bytes(b"checkpoint-contents")

    assert module._checkpoint_provenance(checkpoint) == {
        "filename": "selected.pt",
        "sha256": hashlib.sha256(b"checkpoint-contents").hexdigest(),
    }

def test_raw_export_is_opt_in_and_pose_is_untouched() -> None:
    module = importlib.import_module(MODULE_NAME)

    assert module._export_options("detect", end2end=False)["end2end"] is False
    assert module._export_options("detect")["end2end"] is True
    assert "end2end" not in module._export_options("pose", end2end=False)


def test_expected_output_shapes_cover_raw_and_end2end() -> None:
    module = importlib.import_module(MODULE_NAME)

    assert module._expected_output_shape("ppe", end2end=False) == [1, 12, "predictions"]
    assert module._expected_output_shape("ppe", end2end=True) == [1, 300, 6]
    assert module._expected_output_shape("pose", end2end=False) == [1, 300, 57]


def test_main_exports_end2end_by_default_and_raw_only_when_requested(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    module = importlib.import_module(MODULE_NAME)
    calls: list[tuple[str, str, bool]] = []

    def fake_export_one(source, target, role, task, *, end2end=True, half=False):
        # Mirror export_one's normalization: end-to-end only applies to detect.
        calls.append((role, target.name, end2end and task == "detect", half))

    monkeypatch.setattr(module, "export_one", fake_export_one)
    base_argv = [
        "export_runtime_onnx.py",
        "--ppe",
        "best_ppe.pt",
        "--pose",
        "yolo26s-pose.pt",
        "--output-dir",
        str(tmp_path),
    ]

    monkeypatch.setattr(sys, "argv", list(base_argv))
    assert module.main() == 0
    assert calls == [("ppe", "ppe.onnx", True, False), ("pose", "pose.onnx", False, False)]

    calls.clear()
    monkeypatch.setattr(sys, "argv", [*base_argv, "--ppe-raw"])
    assert module.main() == 0
    assert calls == [
        ("ppe", "ppe.onnx", True, False),
        ("ppe", "ppe-raw.onnx", False, False),
        ("pose", "pose.onnx", False, False),
    ]


def test_half_option_wires_conversion_through_main(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    module = importlib.import_module(MODULE_NAME)
    calls: list[tuple[str, str, bool, bool]] = []

    def fake_export_one(source, target, role, task, *, end2end=True, half=False):
        calls.append((role, target.name, end2end and task == "detect", half))

    monkeypatch.setattr(module, "export_one", fake_export_one)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_runtime_onnx.py",
            "--ppe",
            "best_ppe.pt",
            "--pose",
            "yolo26s-pose.pt",
            "--output-dir",
            str(tmp_path),
            "--half",
        ],
    )

    assert module.main() == 0
    assert calls == [
        ("ppe", "ppe.onnx", True, True),
        ("pose", "pose.onnx", False, True),
    ]
