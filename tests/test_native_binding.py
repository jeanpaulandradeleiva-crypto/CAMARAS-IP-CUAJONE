# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from cuajone_qa.backends.experimental import ExperimentalBackend
from cuajone_qa.backends.native import NativeBackend
from cuajone_qa.canonical import canonical_json, safe_source_id
from cuajone_qa.config import QaRuntimeConfig
from cuajone_qa.contracts import runtime_defaults
from cuajone_qa.parity import normalize_track_identities, run_synthetic_parity, synthetic_frames

native = pytest.importorskip("cuajone_native")


def binding_config() -> QaRuntimeConfig:
    values = deepcopy(runtime_defaults())
    values["analytics"]["mode"] = "ppe-fall"
    values["ppe"]["window"] = 2
    values["ppe"]["minimum_samples"] = 2
    values["fall"]["confirm_frames"] = 2
    values["fall"]["reset_frames"] = 2
    return QaRuntimeConfig(values)


def test_binding_import_versions_and_synthetic_pipeline() -> None:
    backend = NativeBackend(binding_config(), module=native)
    results = [backend.process_observations(frame) for frame in synthetic_frames()]
    assert native.CONTRACT_VERSION == "1.0.0"
    assert len(results[-1].events) == 2
    if native.ENGINE_RUNTIME_AVAILABLE:
        assert hasattr(native, "EngineConfig")
        assert hasattr(native, "EnginePipeline")
        assert hasattr(native, "InferenceProvider")
        assert native.ComputeBackend.CPU != native.ComputeBackend.CUDA
        config = native.EngineConfig()
        config.backend = native.ComputeBackend.CPU
        config.provider = native.InferenceProvider.ONNX_RUNTIME_CPU
        config.ppe_onnx = "ppe.onnx"
        config.pose_onnx = "pose.onnx"
        assert config.backend == native.ComputeBackend.CPU
        assert config.provider == native.InferenceProvider.ONNX_RUNTIME_CPU
        assert config.ppe_onnx == "ppe.onnx"
        assert config.pose_onnx == "pose.onnx"


def test_actual_binding_constructs_cpu_engine_pipelines_from_staged_onnx() -> None:
    if not native.ENGINE_RUNTIME_AVAILABLE:
        pytest.skip("binding does not include the engine runtime")
    models = Path(__file__).parents[1] / ".tools" / "native" / "installer" / "stage" / "bin" / "models"
    for role in ("ppe", "pose"):
        model = models / f"{role}.onnx"
        manifest_path = models / f"{role}.onnx.manifest.json"
        if not model.is_file() or not manifest_path.is_file():
            pytest.skip(f"staged {role} ONNX model and manifest are unavailable")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["artifact_type"] == "onnx"
        assert manifest["role"] == role
        assert manifest["model_file"] == model.name

    for mode in ("ppe-only", "ppe-fall"):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                """
import sys
from pathlib import Path
import os

dll_handles = [
    os.add_dll_directory(path)
    for path in os.environ.get("CUAJONE_NATIVE_DLL_DIRS", "").split(os.pathsep)
    if path
]
import cuajone_native as native

mode = sys.argv[1]
models = Path(sys.argv[2])
config = native.EngineConfig()
config.backend = native.ComputeBackend.CPU
config.provider = native.InferenceProvider.ONNX_RUNTIME_CPU
config.ppe_onnx = str(models / "ppe.onnx")
config.pose_onnx = str(models / "pose.onnx") if mode == "ppe-fall" else ""
config.ppe_labels = {
    0: "Gloves",
    1: "Hard_hat",
    2: "Mask",
    3: "Person",
    4: "Safety_boots",
    5: "Vest",
}
analytics = native.AnalyticsConfig()
analytics.mode = (
    native.AnalyticsMode.PPE_ONLY
    if mode == "ppe-only"
    else native.AnalyticsMode.PPE_FALL
)
config.analytics = analytics
native.EnginePipeline(config)
""",
                mode,
                str(models),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr


def test_binding_validates_numpy_without_implicit_copy() -> None:
    backend = NativeBackend(binding_config(), module=native)
    frame = synthetic_frames()[0]
    with pytest.raises(TypeError, match="numpy.uint8"):
        backend.process_frame(np.zeros((720, 640, 3), dtype=np.float32), frame)
    with pytest.raises(ValueError, match="C-contiguous"):
        backend.process_frame(np.zeros((720, 640, 3), dtype=np.uint8)[:, ::2], frame)


def test_actual_binding_staged_synthetic_parity() -> None:
    receipt = run_synthetic_parity(native_module=native, revision="a" * 40)
    assert receipt["passed"] is True
    assert receipt["full_model_parity_claimed"] is False
    assert receipt["stages"][-1]["status"] == "skipped"


def test_binding_and_python_emit_exact_canonical_json_after_identity_normalization() -> None:
    values = deepcopy(binding_config().values)
    values["analytics"]["mode"] = "ppe-only"
    values["ppe"]["window"] = 1
    values["ppe"]["minimum_samples"] = 1
    config = QaRuntimeConfig(values)
    frame = deepcopy(synthetic_frames()[0])
    frame["source_id"] = "CANONICAL_QA"
    frame["ppe_detections"] = [
        {
            "box": [10.123456789, 20.987654321, 210.333333333, 420.666666667],
            "confidence": 0.876543219,
            "class_id": 0,
        },
        {
            "box": [250.111111111, 30.222222222, 450.444444444, 430.555555556],
            "confidence": 0.765432198,
            "class_id": 0,
        },
    ]
    frame["pose_detections"] = []

    native_backend = NativeBackend(config, module=native)
    native_bundle = native_backend._pipeline.process_observations_bundle(
        native_backend._observations(frame)
    )
    experimental = ExperimentalBackend(config).process_observations(frame)

    native_frame, native_events = normalize_track_identities(
        json.loads(native_bundle[0]), [json.loads(event) for event in native_bundle[1]]
    )
    experimental_frame, experimental_events = normalize_track_identities(
        experimental.frame_result, experimental.events
    )
    assert canonical_json(native_frame) == canonical_json(experimental_frame)
    assert [canonical_json(event) for event in native_events] == [
        canonical_json(event) for event in experimental_events
    ]


@pytest.mark.parametrize("source_id", ("CAMERA_01", "ZONE/A 01", "camera:west"))
def test_binding_and_python_accept_the_same_safe_source_ids(source_id: str) -> None:
    frame = deepcopy(synthetic_frames()[0])
    frame["source_id"] = source_id
    assert safe_source_id(source_id)
    NativeBackend(binding_config(), module=native).process_observations(frame)


@pytest.mark.parametrize(
    "source_id",
    ("prefix-rtsp://camera", "prefix-rtsps://camera", "camera-password", "name@host"),
)
def test_binding_and_python_reject_the_same_secret_like_source_ids(source_id: str) -> None:
    frame = deepcopy(synthetic_frames()[0])
    frame["source_id"] = source_id
    with pytest.raises(ValueError, match="non-secret"):
        safe_source_id(source_id)
    with pytest.raises(ValueError, match="non-secret"):
        NativeBackend(binding_config(), module=native).process_observations(frame)
