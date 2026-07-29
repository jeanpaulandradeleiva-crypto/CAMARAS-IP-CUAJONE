# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .contracts import load_and_validate, runtime_defaults, validate_instance

AnalyticsMode = Literal["ppe-only", "ppe-fall"]
BackendName = Literal["experimental", "native"]


@dataclass(frozen=True)
class QaRuntimeConfig:
    values: dict[str, Any]

    @property
    def mode(self) -> AnalyticsMode:
        return self.values["analytics"]["mode"]

    @property
    def backend(self) -> BackendName:
        return self.values["analytics"]["backend"]

    @classmethod
    def defaults(
        cls,
        *,
        mode: AnalyticsMode | None = None,
        backend: BackendName | None = None,
    ) -> "QaRuntimeConfig":
        values = deepcopy(runtime_defaults())
        if mode is not None:
            values["analytics"]["mode"] = mode
        if backend is not None:
            values["analytics"]["backend"] = backend
        return cls(validate_instance("runtime-config", values))

    @classmethod
    def from_file(cls, path: Path) -> "QaRuntimeConfig":
        return cls(load_and_validate("runtime-config", path))
