# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from copy import deepcopy

import pytest

from cuajone_qa.contracts import (
    CONTRACT_ROOT_V3,
    ContractValidationError,
    load_json,
    validate_instance_v3,
)
from cuajone_qa.ppe import PPE_ITEMS, PPE_ITEM_LABELS


def test_v3_frame_contract_accepts_explicit_four_state_items() -> None:
    items = [
        {
            "semantic": semantic,
            "label": PPE_ITEM_LABELS[semantic],
            "enabled": semantic != "gloves",
            "wear_state": "NO_VERIFICABLE" if semantic == "gloves" else "PRESENTE_CORRECTAMENTE",
            "reason": "DISABLED_BY_POLICY" if semantic == "gloves" else "ASSOCIATED_REGION",
            "confidence": 0.9,
            "detection": None,
        }
        for semantic in PPE_ITEMS
    ]
    frame = {
        "contract_version": "3.0.0", "source_id": "CAM_01", "frame_id": 1,
        "monotonic_timestamp_ms": 1, "observed_at": "2026-08-13T00:00:00Z",
        "frame": {"width": 640, "height": 480}, "events": [],
        "people": [{"track_id": 1, "box": [0, 0, 10, 20], "confidence": 0.9,
                    "ppe_visibility_sufficient": False, "ppe": {"state": "noncompliant", "samples": 1, "items": items},
                    "fall_active": False, "keypoints": []}],
    }
    assert validate_instance_v3("frame-result", frame) == frame


def test_v3_schemas_are_loadable() -> None:
    assert load_json(CONTRACT_ROOT_V3 / "frame-result.schema.json")["$id"].endswith("frame-result.schema.json")
    assert load_json(CONTRACT_ROOT_V3 / "event.schema.json")["$id"].endswith("event.schema.json")


def test_v3_event_contract_requires_complete_four_state_ppe() -> None:
    items = [
        {
            "semantic": semantic,
            "label": PPE_ITEM_LABELS[semantic],
            "enabled": True,
            "wear_state": "AUSENTE" if semantic == "gloves" else "PRESENTE_CORRECTAMENTE",
            "reason": "NO_ASSOCIATED_DETECTION" if semantic == "gloves" else "ASSOCIATED_REGION",
            "confidence": 0.9,
            "detection": None,
        }
        for semantic in PPE_ITEMS
    ]
    event = {
        "specversion": "1.0", "id": "evt-CAM_01-1-1-0", "source": "urn:cuajone:camera:CAM_01",
        "type": "com.cuajone.safety.ppe.violation.v3", "time": "2026-08-13T00:00:00Z",
        "datacontenttype": "application/json",
        "dataschema": "https://cuajone.example/contracts/v3/event.schema.json",
        "subject": "track/1", "contractversion": "3.0.0",
        "data": {"contract_version": "3.0.0", "frame_id": 1, "monotonic_timestamp_ms": 1,
                 "track_id": 1, "status": "Falta: Gloves", "confidence": 0.9, "evidence": [],
                 "ppe": {"state": "noncompliant", "samples": 1, "items": items}},
    }
    assert validate_instance_v3("event", event) == event

    missing_reason = deepcopy(event)
    del missing_reason["data"]["ppe"]["items"][0]["reason"]
    invalid_state = deepcopy(event)
    invalid_state["data"]["ppe"]["items"][0]["wear_state"] = "missing"
    for malformed in (missing_reason, invalid_state):
        with pytest.raises(ContractValidationError):
            validate_instance_v3("event", malformed)
