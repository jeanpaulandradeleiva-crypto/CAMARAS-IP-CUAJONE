# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class BackendResult:
    frame_result: dict[str, Any]
    events: tuple[dict[str, Any], ...]


class AnalyticsBackend(Protocol):
    def process_observations(self, observations: dict[str, Any]) -> BackendResult: ...

    def process_frame(
        self,
        frame: np.ndarray,
        observations: dict[str, Any],
    ) -> BackendResult: ...

    def reset(self) -> None: ...
