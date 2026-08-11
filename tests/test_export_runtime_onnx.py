# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from tools.export_runtime_onnx import _export_options


def test_detect_export_is_raw_and_pose_keeps_ultralytics_default() -> None:
    detect_options = _export_options("detect")
    pose_options = _export_options("pose")

    assert detect_options["end2end"] is False
    assert "end2end" not in pose_options
    assert detect_options["nms"] is False
    assert pose_options["nms"] is False
