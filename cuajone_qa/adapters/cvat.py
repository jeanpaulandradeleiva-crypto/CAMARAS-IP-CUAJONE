# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol


class TaskRepository(Protocol):
    def create_from_data(self, *, spec: dict[str, Any], resource_type: Any, resources: Sequence[str], **kwargs: Any) -> Any: ...

    def retrieve(self, task_id: int) -> Any: ...


class CvatClient(Protocol):
    tasks: TaskRepository


def make_cvat_client(
    server_url: str,
    credential_provider: Callable[[], tuple[str, str]] | None = None,
) -> Any:
    """Create an SDK client without persisting credentials in project configuration."""
    # CVAT remains an optional extra; importing this module must not require its SDK.
    from cvat_sdk import make_client

    credentials = credential_provider() if credential_provider else None
    return make_client(host=server_url, credentials=credentials)


class CvatAdapter:
    """High-level CVAT SDK boundary; it never owns or serializes credentials."""

    def __init__(self, client: CvatClient) -> None:
        self._client = client

    def create_task(
        self,
        *,
        name: str,
        labels: Sequence[dict[str, Any]],
        resources: Sequence[Path],
        resource_type: Any,
        **kwargs: Any,
    ) -> Any:
        if not name or not resources:
            raise ValueError("CVAT task name and resources are required")
        spec = {"name": name, "labels": list(labels)}
        return self._client.tasks.create_from_data(
            spec=spec,
            resource_type=resource_type,
            resources=[str(path) for path in resources],
            **kwargs,
        )

    def export_dataset(
        self,
        task_id: int,
        *,
        format_name: str,
        output_path: Path,
        include_images: bool = False,
    ) -> None:
        task = self._client.tasks.retrieve(task_id)
        task.export_dataset(format_name, str(output_path), include_images=include_images)

    def import_annotations(
        self,
        task_id: int,
        *,
        format_name: str,
        annotations_path: Path,
    ) -> None:
        task = self._client.tasks.retrieve(task_id)
        task.import_annotations(format_name, str(annotations_path))

    def auto_annotate(
        self,
        task_id: int,
        function: Any,
        **kwargs: Any,
    ) -> Any:
        # Auto-annotation is optional even for callers that only use task import/export.
        from cvat_sdk.auto_annotation import annotate_task

        return annotate_task(self._client, task_id, function, **kwargs)
