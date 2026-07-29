# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from cuajone_qa.adapters.cvat import CvatAdapter
from cuajone_qa.adapters.supervision import SupervisionAdapter


class FakeTask:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    def export_dataset(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append(("export", args, kwargs))

    def import_annotations(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append(("import", args, kwargs))


class FakeTasks:
    def __init__(self) -> None:
        self.task = FakeTask()
        self.created: dict[str, Any] | None = None

    def create_from_data(self, **kwargs: Any) -> FakeTask:
        self.created = kwargs
        return self.task

    def retrieve(self, task_id: int) -> FakeTask:
        assert task_id == 7
        return self.task


def test_cvat_adapter_uses_injected_sdk_boundary_without_credentials() -> None:
    tasks = FakeTasks()
    adapter = CvatAdapter(SimpleNamespace(tasks=tasks))
    adapter.create_task(
        name="Synthetic task",
        labels=[{"name": "Person"}],
        resources=[Path("synthetic.jpg")],
        resource_type="local",
    )
    adapter.export_dataset(7, format_name="COCO", output_path=Path("out.zip"))
    adapter.import_annotations(7, format_name="COCO", annotations_path=Path("in.zip"))
    assert tasks.created is not None
    assert "credentials" not in tasks.created
    assert [call[0] for call in tasks.task.calls] == ["export", "import"]


class FakeDetections:
    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


def test_supervision_adapter_round_trips_canonical_detections() -> None:
    module = SimpleNamespace(Detections=FakeDetections)
    adapter = SupervisionAdapter(module)
    canonical = [
        {"box": [1, 2, 3, 4], "confidence": 0.9, "class_id": 2, "track_id": 5, "label": "vest"}
    ]
    detections = adapter.to_detections(canonical)
    result = adapter.from_detections(detections)
    assert result == [
        {"box": [1.0, 2.0, 3.0, 4.0], "confidence": float(np.float32(0.9)), "class_id": 2, "track_id": 5, "label": "vest"}
    ]
