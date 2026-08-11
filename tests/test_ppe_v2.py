# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np
from jsonschema import Draft202012Validator

import ppe_reportev2
from cuajone_qa.contracts import CONTRACT_ROOT_V2, ContractValidationError, load_json, load_schema_v2, validate_instance_v2
from cuajone_qa.ppe import PPE_ITEMS, PPE_LABELS, validate_ppe_labels


@pytest.mark.parametrize("name", ("frame-result", "event"))
def test_v2_contract_schema_and_fixtures(name: str) -> None:
    schema = load_schema_v2(name)
    Draft202012Validator.check_schema(schema)
    valid = load_json(CONTRACT_ROOT_V2 / "fixtures" / "valid" / f"{name}.json")
    assert validate_instance_v2(name, valid)
    invalid = load_json(CONTRACT_ROOT_V2 / "fixtures" / "invalid" / f"{name}.json")
    with pytest.raises(ContractValidationError):
        validate_instance_v2(name, invalid)


def test_fixed_label_contract_rejects_missing_wrong_and_ambiguous_semantics() -> None:
    registry = load_json(CONTRACT_ROOT_V2 / "ppe-labels.json")
    assert registry["labels"] == list(PPE_LABELS)
    assert registry["required"] == list(PPE_ITEMS)
    assert validate_ppe_labels(dict(enumerate(PPE_LABELS)))
    with pytest.raises(ValueError, match="0 through 7"):
        validate_ppe_labels({0: "Gloves", 1: "Person"})
    wrong = dict(enumerate(PPE_LABELS))
    wrong[0], wrong[1] = wrong[1], wrong[0]
    with pytest.raises(ValueError, match="ID 0"):
        validate_ppe_labels(wrong)
    ambiguous = dict(enumerate(PPE_LABELS))
    ambiguous[7] = "Hard_hat"
    with pytest.raises(ValueError, match="ID 7"):
        validate_ppe_labels(ambiguous)
    wrong_case = dict(enumerate(PPE_LABELS))
    wrong_case[0] = "gloves"
    with pytest.raises(ValueError, match="ID 0"):
        validate_ppe_labels(wrong_case)


def test_v2_report_row_exposes_every_item_and_missing_list(tmp_path: Path) -> None:
    ppe = {
        "state": "noncompliant",
        "missing": ["gloves", "safety_boots"],
        "items": [
            {"semantic": item, "present": item not in {"gloves", "safety_boots"}, "ratio": 0.8, "confidence": 0.9}
            for item in PPE_ITEMS
        ],
    }
    event = ppe_reportev2.make_event(
        "evt-X-1-1-0", 1, "INCUMPLIMIENTO_EPP", "Falta: Gloves, Safety_boots",
        ppe, 0.9, str(tmp_path / "evidence.jpg"),
    )
    assert event["Version_Contrato"] == "2.0.0"
    assert event["Guantes"] == "NO" and event["Botas_Seguridad"] == "NO"
    assert event["Casco"] == "SI" and event["Lentes_Protectores"] == "SI"
    assert event["Faltantes_EPP"] == "gloves;safety_boots"
    assert set(event) == set(ppe_reportev2.EVENT_FIELDS)


def test_python_overlay_draws_all_seven_states_and_associated_detections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels: list[str] = []
    rectangles: list[tuple[tuple[int, int], tuple[int, int]]] = []
    monkeypatch.setattr(
        ppe_reportev2.cv2,
        "putText",
        lambda _frame, text, *_args, **_kwargs: labels.append(str(text)),
    )
    monkeypatch.setattr(
        ppe_reportev2.cv2,
        "rectangle",
        lambda _frame, start, end, *_args, **_kwargs: rectangles.append((start, end)),
    )

    items = [
        {
            "semantic": semantic,
            "label": PPE_LABELS[class_id],
            "present": True,
            "ratio": 1.0,
            "confidence": 0.9,
            "detection": {"box": [10 + class_id, 20, 30 + class_id, 40], "confidence": 0.9},
        }
        for semantic, class_id in zip(PPE_ITEMS, (0, 2, 3, 4, 5, 6, 7))
    ]

    class Backend:
        def process_frame_v2(self, _frame: np.ndarray, _observations: dict[str, object]) -> SimpleNamespace:
            return SimpleNamespace(
                frame_result={"people": [{
                    "track_id": 1,
                    "box": [5, 5, 100, 200],
                    "ppe": {"state": "compliant", "missing": [], "items": items},
                    "fall_active": False,
                    "keypoints": [],
                }]},
                events=(),
            )

    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    ppe_reportev2.process_native_analytics_frame(frame, "ppe-only", Backend(), 1, 1.0)
    assert len(rectangles) == 8
    for item in items:
        assert any(text.startswith(f"{item['label']}: OK") for text in labels)
