from __future__ import annotations

import runpy
import sys
import types
from pathlib import Path

import pytest


class StopScript(Exception):
    pass


def test_loads_project_dotenv_before_requiring_rtsp_url(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project_root = Path(__file__).resolve().parents[1]
    script_path = project_root / "ppe_reporte.py"
    dotenv_calls: list[tuple[Path, bool]] = []

    def load_dotenv(*, dotenv_path: Path, override: bool) -> None:
        dotenv_calls.append((dotenv_path, override))
        monkeypatch.setenv("RTSP_URL", "rtsp://test.example/stream")

    class FakeYOLO:
        def __init__(self, _path: str) -> None:
            raise StopScript

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RTSP_URL", raising=False)
    monkeypatch.setitem(sys.modules, "dotenv", types.SimpleNamespace(load_dotenv=load_dotenv))
    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO))
    monkeypatch.setitem(sys.modules, "pandas", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "cv2", types.SimpleNamespace())

    with pytest.raises(StopScript):
        runpy.run_path(str(script_path))

    assert dotenv_calls == [(project_root / ".env", False)]
