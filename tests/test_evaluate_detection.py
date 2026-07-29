# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from tools import evaluate_detection


NAMES = {0: "Person", 1: "Hard_hat", 2: "Vest", 3: "Safety_boots"}


def fake_metrics() -> Any:
    return SimpleNamespace(
        names=NAMES,
        box=SimpleNamespace(
            mp=0.81,
            mr=0.72,
            map50=0.77,
            map=0.55,
            ap_class_index=[1, 2, 3],
            p=[0.91, 0.82, 0.73],
            r=[0.71, 0.62, 0.53],
            ap50=[0.88, 0.78, 0.68],
            ap=[0.58, 0.48, 0.38],
        ),
    )


def test_normalizes_summary_and_per_class_metrics_deterministically() -> None:
    report = evaluate_detection.normalize_metrics(fake_metrics(), NAMES)

    assert report["metrics"] == {
        "precision": 0.81,
        "recall": 0.72,
        "map50": 0.77,
        "map50_95": 0.55,
    }
    assert report["required_classes"] == {"helmet": 1, "vest": 2, "safety_boots": 3}
    assert report["per_class"][0] == {
        "class_id": 1,
        "name": "Hard_hat",
        "precision": 0.91,
        "recall": 0.71,
        "map50": 0.88,
        "map50_95": 0.58,
    }
    assert "no mide asociación persona-EPP" in report["disclaimer"]


def test_missing_required_semantic_class_reports_available_names() -> None:
    with pytest.raises(ValueError, match=r"safety_boots.*Clases disponibles.*Hard_hat"):
        evaluate_detection.validate_required_classes({0: "Hard_hat", 1: "Vest"})


def test_evaluate_constructs_detect_model_and_uses_configured_split() -> None:
    calls: list[tuple[str, str]] = []
    val_calls: list[dict[str, Any]] = []

    class FakeModel:
        names = NAMES

        def val(self, **kwargs: Any) -> Any:
            val_calls.append(kwargs)
            return fake_metrics()

    def factory(path: str, *, task: str) -> FakeModel:
        calls.append((path, task))
        return FakeModel()

    report = evaluate_detection.evaluate(
        "ppe.engine",
        "dataset.yaml",
        split="val",
        device="cuda:0",
        yolo_factory=factory,
    )

    assert calls == [("ppe.engine", "detect")]
    assert val_calls == [{"data": "dataset.yaml", "split": "val", "device": "cuda:0"}]
    assert report["scope"] == "object_detection"
