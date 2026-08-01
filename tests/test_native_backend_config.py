# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from types import SimpleNamespace

import pytest

from cuajone_qa.backends.native import NativeBackend
from cuajone_qa.config import QaRuntimeConfig
from cuajone_qa.contracts import CONTRACT_VERSION


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
        ENGINE_RUNTIME_AVAILABLE=True,
        ComputeBackend=SimpleNamespace(CPU="cpu", CUDA="cuda"),
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
            "ppe_labels": {0: "Person", 1: "Hard_hat", 2: "Vest"},
        },
    )

    config = backend._engine_pipeline.config
    assert config.backend == "cpu"
    assert config.ppe_onnx == "ppe.onnx"
    assert config.pose_onnx == "pose.onnx"
    assert config.device is None


@pytest.mark.parametrize(
    ("engine_config", "message"),
    (
        ({"backend": "cpu", "ppe_labels": {0: "Person"}}, "ppe_onnx"),
        ({"backend": "cpu", "ppe_onnx": "ppe.onnx", "ppe_labels": {0: "Person"}}, "pose_onnx"),
        ({"backend": "cpu", "ppe_onnx": "ppe.onnx", "pose_onnx": "pose.onnx"}, "ppe_labels"),
        ({"backend": "auto"}, "cpu or cuda"),
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
