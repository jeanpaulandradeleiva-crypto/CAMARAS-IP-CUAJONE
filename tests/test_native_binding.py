# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from cuajone_qa.backends.experimental import ExperimentalBackend
from cuajone_qa.backends.native import NativeBackend
from cuajone_qa.canonical import canonical_json, safe_source_id
from cuajone_qa.config import QaRuntimeConfig
from cuajone_qa.contracts import runtime_defaults
from cuajone_qa.parity import run_synthetic_parity, synthetic_frames

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


def test_binding_and_python_emit_exact_canonical_json() -> None:
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

    assert native_bundle[0] == canonical_json(experimental.frame_result)
    assert native_bundle[1] == [canonical_json(event) for event in experimental.events]


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
