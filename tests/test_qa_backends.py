# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from cuajone_qa.backends.experimental import ExperimentalBackend
from cuajone_qa.canonical import canonical_json, canonical_number_text, safe_source_id
from cuajone_qa.config import QaRuntimeConfig
from cuajone_qa.contracts import runtime_defaults
from cuajone_qa.parity import synthetic_frames


def fast_config() -> QaRuntimeConfig:
    values = deepcopy(runtime_defaults())
    values["analytics"]["mode"] = "ppe-fall"
    values["analytics"]["backend"] = "experimental"
    values["ppe"]["window"] = 2
    values["ppe"]["minimum_samples"] = 2
    values["fall"]["confirm_frames"] = 2
    values["fall"]["reset_frames"] = 2
    return QaRuntimeConfig(values)


def test_experimental_observation_pipeline_is_deterministic() -> None:
    backend = ExperimentalBackend(fast_config())
    first_run = [backend.process_observations(frame) for frame in synthetic_frames()]
    assert len(first_run[-1].events) == 2
    assert {event["type"] for event in first_run[-1].events} == {
        "com.cuajone.safety.ppe.violation.v1",
        "com.cuajone.safety.fall.possible.v1",
    }
    backend.reset()
    second_run = [backend.process_observations(frame) for frame in synthetic_frames()]
    assert second_run == first_run


def test_experimental_frame_validation() -> None:
    backend = ExperimentalBackend(fast_config())
    with pytest.raises(ValueError, match="uint8 BGR"):
        backend.process_frame(np.zeros((10, 10, 3), dtype=np.float32), synthetic_frames()[0])


def test_experimental_rejects_out_of_order_frames() -> None:
    backend = ExperimentalBackend(fast_config())
    frame = synthetic_frames()[0]
    backend.process_observations(frame)
    with pytest.raises(ValueError, match="increase strictly"):
        backend.process_observations(frame)


def test_experimental_serializes_each_persons_own_box() -> None:
    values = deepcopy(fast_config().values)
    values["analytics"]["mode"] = "ppe-only"
    frame = deepcopy(synthetic_frames()[0])
    frame["ppe_detections"] = [
        {"box": [10.25, 20.5, 110.75, 320.125], "confidence": 0.91, "class_id": 0},
        {"box": [210.5, 30.25, 410.875, 430.625], "confidence": 0.82, "class_id": 0},
    ]
    frame["pose_detections"] = []

    result = ExperimentalBackend(QaRuntimeConfig(values)).process_observations(frame)

    assert [person["box"] for person in result.frame_result["people"]] == [
        [10.25, 20.5, 110.75, 320.125],
        [210.5, 30.25, 410.875, 430.625],
    ]


def test_canonical_float32_numbers_use_six_decimal_fixed_json() -> None:
    assert canonical_number_text(0.123456789) == "0.123457"
    assert canonical_json({"small": 0.00001, "whole": 100.0}) == (
        '{"small":0.00001,"whole":100}'
    )


@pytest.mark.parametrize("value", ("CAMERA_01", "ZONE/A 01", "camera:west"))
def test_source_id_acceptance_is_explicit(value: str) -> None:
    assert safe_source_id(value)


@pytest.mark.parametrize(
    "value",
    ("prefix-rtsp://camera", "prefix-rtsps://camera", "camera-password", "name@host"),
)
def test_source_id_rejects_native_forbidden_tokens(value: str) -> None:
    with pytest.raises(ValueError, match="non-secret"):
        safe_source_id(value)
