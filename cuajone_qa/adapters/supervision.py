# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any, Sequence

import numpy as np


class SupervisionAdapter:
    """Lazy canonical conversion boundary for the optional Supervision package."""

    def __init__(self, module: ModuleType | Any | None = None) -> None:
        self._module = module

    @property
    def sv(self) -> Any:
        if self._module is None:
            # Keep Supervision out of the base QA import and the product MSI boundary.
            self._module = importlib.import_module("supervision")
        return self._module

    def to_detections(self, canonical: Sequence[dict[str, Any]]) -> Any:
        xyxy = np.asarray([item["box"] for item in canonical], dtype=np.float32).reshape((-1, 4))
        confidence = np.asarray([item["confidence"] for item in canonical], dtype=np.float32)
        class_id = np.asarray([item["class_id"] for item in canonical], dtype=int)
        tracker_values = [item.get("track_id") for item in canonical]
        tracker_id = (
            np.asarray(tracker_values, dtype=int)
            if tracker_values and all(value is not None for value in tracker_values)
            else None
        )
        data = {"label": np.asarray([item.get("label", "") for item in canonical], dtype=object)}
        return self.sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id,
            tracker_id=tracker_id,
            data=data,
        )

    def from_detections(self, detections: Any) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for index, box in enumerate(detections.xyxy):
            item: dict[str, Any] = {
                "box": [float(value) for value in box],
                "confidence": float(detections.confidence[index]) if detections.confidence is not None else 1.0,
                "class_id": int(detections.class_id[index]) if detections.class_id is not None else 0,
            }
            if detections.tracker_id is not None:
                item["track_id"] = int(detections.tracker_id[index])
            if getattr(detections, "data", None) and "label" in detections.data:
                item["label"] = str(detections.data["label"][index])
            result.append(item)
        return result

    def render(self, frame: np.ndarray, detections: Any) -> np.ndarray:
        return self.sv.BoxAnnotator().annotate(scene=frame.copy(), detections=detections)

    def mean_average_precision(self, predictions: Sequence[Any], targets: Sequence[Any]) -> Any:
        metric_type = getattr(self.sv, "MeanAveragePrecision", None)
        if metric_type is None:
            raise NotImplementedError("Installed Supervision does not expose MeanAveragePrecision")
        metric = metric_type()
        for prediction, target in zip(predictions, targets):
            metric.update(prediction, target)
        return metric.compute()

    def to_dataset(
        self,
        *,
        classes: Sequence[str],
        images: dict[str, np.ndarray],
        annotations: dict[str, Any],
    ) -> Any:
        return self.sv.DetectionDataset(
            classes=list(classes),
            images=images,
            annotations=annotations,
        )
