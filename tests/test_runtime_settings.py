# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from pathlib import Path

from cuajone_qa.runtime import RuntimeSettings, RuntimeState


def test_runtime_settings_defaults_and_overrides_are_isolated(tmp_path: Path) -> None:
    first_runtime = tmp_path / "first-runtime"
    second_runtime = tmp_path / "second-runtime"
    first = RuntimeSettings.from_environment(
        {
            "CAMERA_ID": "CAM_TEST_01",
            "OUTPUT_DIR": "first-output",
            "PPE_MODEL_PATH": "models/custom-ppe.pt",
            "SHOW_WINDOW": "0",
            "EPP_WINDOW": "6",
            "POSE_PERSON_GATE": "1",
        },
        runtime_dir=first_runtime,
    )
    second = RuntimeSettings.from_environment({}, runtime_dir=second_runtime)

    assert first.camera_id == "CAM_TEST_01"
    assert first.base_dir == (first_runtime / "first-output").resolve()
    assert first.ppe_model_path == str((first_runtime / "models/custom-ppe.pt").resolve())
    assert not first.show_window
    assert first.epp_window == 6
    assert first.pose_person_gate

    assert second.camera_id == "CAM_P01_ADM"
    assert second.base_dir == second_runtime.resolve()
    assert second.ppe_model_path == str((second_runtime / "best_ppe.pt").resolve())
    assert second.pose_model_path == str((second_runtime / "yolo26s-pose.pt").resolve())
    assert second.ppe_onnx_path == str((second_runtime / "models/ppe.onnx").resolve())
    assert second.pose_onnx_path == str((second_runtime / "models/pose.onnx").resolve())
    assert second.ppe_labels == (
        "Gloves,Person,Safety_boots,Vest,respirador,tapaorejas,Hard_hat,lentes_protectores"
    )
    assert second.show_window
    assert second.epp_window == 20
    assert not second.pose_person_gate


def test_runtime_state_is_not_shared_between_constructions() -> None:
    first = RuntimeState()
    second = RuntimeState()

    first.stop_event.set()

    assert first.stop_event.is_set()
    assert not second.stop_event.is_set()
