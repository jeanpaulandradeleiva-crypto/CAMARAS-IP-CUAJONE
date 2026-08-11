# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from typing import Any, Final


PPE_LABELS: Final[tuple[str, ...]] = (
    "Gloves",
    "Person",
    "Safety_boots",
    "Vest",
    "respirador",
    "tapaorejas",
    "Hard_hat",
    "lentes_protectores",
)
PPE_ITEMS: Final[tuple[str, ...]] = (
    "gloves",
    "safety_boots",
    "vest",
    "respirator",
    "hearing_protection",
    "hard_hat",
    "eye_protection",
)
PPE_ITEM_LABELS: Final[dict[str, str]] = {
    "gloves": "Gloves",
    "safety_boots": "Safety_boots",
    "vest": "Vest",
    "respirator": "respirador",
    "hearing_protection": "tapaorejas",
    "hard_hat": "Hard_hat",
    "eye_protection": "lentes_protectores",
}
PPE_ITEM_CLASS_IDS: Final[dict[str, int]] = {
    "gloves": 0,
    "safety_boots": 2,
    "vest": 3,
    "respirator": 4,
    "hearing_protection": 5,
    "hard_hat": 6,
    "eye_protection": 7,
}


def normalize_label(label: str) -> str:
    return "_".join(label.strip().lower().replace("-", "_").split()).replace("__", "_")


def validate_ppe_labels(labels: dict[int, str]) -> dict[int, str]:
    expected_ids = list(range(len(PPE_LABELS)))
    if list(labels) != expected_ids:
        raise ValueError("PPE labels must use exact contiguous IDs 0 through 7")
    for class_id, expected in enumerate(PPE_LABELS):
        if str(labels[class_id]) != expected:
            raise ValueError(f"PPE label ID {class_id} must be {expected!r}")
    return labels


def native_item_ids(module: Any) -> dict[Any, int]:
    return {
        module.PpeItem.GLOVES: PPE_ITEM_CLASS_IDS["gloves"],
        module.PpeItem.SAFETY_BOOTS: PPE_ITEM_CLASS_IDS["safety_boots"],
        module.PpeItem.VEST: PPE_ITEM_CLASS_IDS["vest"],
        module.PpeItem.RESPIRATOR: PPE_ITEM_CLASS_IDS["respirator"],
        module.PpeItem.HEARING_PROTECTION: PPE_ITEM_CLASS_IDS["hearing_protection"],
        module.PpeItem.HARD_HAT: PPE_ITEM_CLASS_IDS["hard_hat"],
        module.PpeItem.EYE_PROTECTION: PPE_ITEM_CLASS_IDS["eye_protection"],
    }
