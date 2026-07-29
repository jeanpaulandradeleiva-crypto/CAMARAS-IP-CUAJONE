# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np


class SourceKind(str, Enum):
    IMAGE = "image"
    VIDEO = "video"
    WEBCAM = "webcam"
    RTSP = "rtsp"
    FIXTURE = "fixture"


@dataclass(frozen=True)
class SourceSpec:
    kind: SourceKind
    value: str


def iter_fixture(path: Path) -> Iterator[tuple[np.ndarray, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    frames = payload.get("frames") if isinstance(payload, dict) else None
    if not isinstance(frames, list):
        raise ValueError("Deterministic fixture must contain a frames array")
    for observations in frames:
        dimensions = observations["frame"]
        frame = np.zeros(
            (int(dimensions["height"]), int(dimensions["width"]), 3),
            dtype=np.uint8,
        )
        yield frame, observations


def iter_source(spec: SourceSpec) -> Iterator[np.ndarray]:
    """Open an explicitly authorized source only when this iterator is consumed."""
    import cv2

    if spec.kind == SourceKind.IMAGE:
        frame = cv2.imread(spec.value, cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"Could not read image source: {spec.value}")
        yield frame
        return
    capture_value: int | str = int(spec.value) if spec.kind == SourceKind.WEBCAM else spec.value
    capture = cv2.VideoCapture(capture_value)
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Could not open authorized {spec.kind.value} source")
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            yield frame
    finally:
        capture.release()
