# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from types import SimpleNamespace

import pytest

from cuajone_qa.backends.native import NativeBackend
from cuajone_qa.config import QaRuntimeConfig
from cuajone_qa.contracts import CONTRACT_VERSION, CONTRACT_VERSION_V2
from cuajone_qa.ppe import PPE_LABELS


FIXED_LABELS = dict(enumerate(PPE_LABELS))


class _Values:
    pass


class _EnginePipeline:
    def __init__(self, config: _Values) -> None:
        self.config = config

    def reset(self) -> None:
        pass


def native_module() -> SimpleNamespace:
    return SimpleNamespace(
        CONTRACT_VERSION=CONTRACT_VERSION,
        CONTRACT_VERSION_V2=CONTRACT_VERSION_V2,
        ENGINE_RUNTIME_AVAILABLE=True,
        ComputeBackend=SimpleNamespace(CPU="cpu", CUDA="cuda"),
        InferenceProvider=SimpleNamespace(
            ONNX_RUNTIME_CPU="onnx-runtime-cpu",
            ONNX_RUNTIME_CUDA="onnx-runtime-cuda",
            TENSORRT="tensorrt",
        ),
        AnalyticsMode=SimpleNamespace(PPE_ONLY="ppe-only", PPE_FALL="ppe-fall"),
        AnalyticsConfig=_Values,
        TrackerConfig=_Values,
        PpeConfig=_Values,
        FallConfig=_Values,
        AnalyticsPipeline=lambda config: SimpleNamespace(config=config, reset=lambda: None),
        EngineConfig=_Values,
        EnginePipeline=_EnginePipeline,
    )


def test_native_backend_maps_cpu_onnx_config() -> None:
    backend = NativeBackend(
        QaRuntimeConfig.defaults(mode="ppe-fall", backend="native"),
        module=native_module(),
        engine_config={
            "backend": "cpu",
            "ppe_onnx": "ppe.onnx",
            "pose_onnx": "pose.onnx",
            "ppe_labels": FIXED_LABELS,
        },
    )

    config = backend._engine_pipeline.config
    assert config.backend == "cpu"
    assert config.provider == "onnx-runtime-cpu"
    assert config.ppe_onnx == "ppe.onnx"
    assert config.pose_onnx == "pose.onnx"
    assert config.device is None
    assert config.image_size == 640
    assert config.ppe_class_confidences == [0.30] * 8


def test_native_backend_maps_dynamic_size_and_exact_class_thresholds() -> None:
    thresholds = {label: (index + 1) / 10 for index, label in enumerate(PPE_LABELS)}
    backend = NativeBackend(
        QaRuntimeConfig.defaults(mode="ppe-only", backend="native"),
        module=native_module(),
        engine_config={
            "backend": "cpu",
            "ppe_onnx": "ppe.onnx",
            "ppe_labels": FIXED_LABELS,
            "image_size": 960,
            "ppe_class_confidences": thresholds,
        },
    )

    config = backend._engine_pipeline.config
    assert config.image_size == 960
    assert config.ppe_class_confidences == list(thresholds.values())


@pytest.mark.parametrize(
    ("provider", "artifacts", "expected_provider"),
    (
        (
            "onnx-runtime-cuda",
            {
                "ppe_onnx": "ppe.onnx",
                "pose_onnx": "pose.onnx",
                "ppe_labels": FIXED_LABELS,
            },
            "onnx-runtime-cuda",
        ),
        (
            "tensorrt",
            {"ppe_engine": "ppe.engine", "pose_engine": "pose.engine"},
            "tensorrt",
        ),
    ),
)
def test_native_backend_maps_explicit_cuda_provider(
    provider: str,
    artifacts: dict[str, object],
    expected_provider: str,
) -> None:
    backend = NativeBackend(
        QaRuntimeConfig.defaults(mode="ppe-fall", backend="native"),
        module=native_module(),
        engine_config={"backend": "cuda", "provider": provider, **artifacts},
    )

    config = backend._engine_pipeline.config
    assert config.backend == "cuda"
    assert config.provider == expected_provider
    assert config.device == 0


@pytest.mark.parametrize(
    ("engine_config", "message"),
    (
        ({"backend": "cpu", "ppe_labels": {0: "Person"}}, "ppe_onnx"),
        ({"backend": "cpu", "ppe_onnx": "ppe.onnx", "ppe_labels": {0: "Person"}}, "pose_onnx"),
        ({"backend": "cpu", "ppe_onnx": "ppe.onnx", "pose_onnx": "pose.onnx"}, "ppe_labels"),
        ({"backend": "auto"}, "backend/provider"),
        ({"backend": "cpu", "provider": "tensorrt"}, "backend/provider"),
    ),
)
def test_native_backend_rejects_incomplete_or_unresolved_engine_config(
    engine_config: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        NativeBackend(
            QaRuntimeConfig.defaults(mode="ppe-fall", backend="native"),
            module=native_module(),
            engine_config=engine_config,
        )
