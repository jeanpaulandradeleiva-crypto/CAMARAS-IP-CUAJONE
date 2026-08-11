# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import ppe_reportev2 as app


def test_facade_import_does_not_load_experimental_dependencies() -> None:
    environment = dict(os.environ, CUAJONE_SKIP_DOTENV="1")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import ppe_reportev2; "
                "assert 'torch' not in sys.modules; "
                "assert 'ultralytics' not in sys.modules"
            ),
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_facade_does_not_expose_legacy_ultralytics_pipeline() -> None:
    legacy_names = {
        "AnalyticsModels",
        "TrackState",
        "load_analytics_models",
        "process_analytics_frame",
        "selected_device",
    }

    assert legacy_names.isdisjoint(vars(app))


def test_default_mode_is_ppe_fall() -> None:
    assert app.resolve_analytics_mode(None, {}) == "ppe-fall"


def test_cli_mode_overrides_environment() -> None:
    args = app.parse_args(
        ["--mode", "ppe-only"],
        {"ANALYTICS_MODE": "ppe-fall"},
    )
    assert args.mode == "ppe-only"


def test_help_exits_without_runtime_startup(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        app,
        "load_native_backend",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("help must not load models")
        ),
    )

    with pytest.raises(SystemExit) as exc_info:
        app.main(["--help"])

    assert exc_info.value.code == 0
    assert "--preflight" in capsys.readouterr().out


def test_relative_model_resolution_uses_source_directory(tmp_path: Path) -> None:
    source_script = tmp_path / "source" / "ppe_reportev2.py"
    source_base = source_script.parent.resolve()

    assert app.resolve_runtime_path("models/ppe.engine", base_dir=source_base) == (
        source_script.parent / "models" / "ppe.engine"
    ).resolve()


def test_absolute_model_path_is_preserved(tmp_path: Path) -> None:
    absolute = (tmp_path / "external" / "ppe.engine").resolve()

    assert app.resolve_runtime_path(absolute, base_dir=tmp_path / "ignored") == absolute


def test_local_native_binding_is_discovered_automatically(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    build_python = tmp_path / ".tools/native/build/presets/python-bindings/python"
    onnx_lib = tmp_path / ".tools/native/onnxruntime-win-x64-1.25.0/lib"
    build_python.mkdir(parents=True)
    onnx_lib.mkdir(parents=True)
    monkeypatch.setattr(app, "RUNTIME_DIR", tmp_path)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.delenv("CUAJONE_NATIVE_DLL_DIRS", raising=False)

    app.configure_local_native_binding()

    assert sys.path[0] == str(build_python.resolve())
    assert os.environ["CUAJONE_NATIVE_DLL_DIRS"].split(os.pathsep) == [
        str(build_python.resolve()),
        str(onnx_lib.resolve()),
    ]


def test_native_preflight_reports_fixed_onnx_without_rtsp(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    onnx = tmp_path / "ppe.onnx"
    onnx.write_bytes(b"fixed-onnx")
    (tmp_path / "ppe.onnx.manifest.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(app, "PPE_ONNX_PATH", str(onnx))
    loaded_modes: list[str] = []
    monkeypatch.setattr(
        app,
        "load_native_backend",
        lambda mode: loaded_modes.append(mode) or object(),
    )
    monkeypatch.setattr(
        app,
        "LatestFrameCapture",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("preflight must not open RTSP")
        ),
    )
    assert app.main(["--mode", "ppe-only", "--preflight"]) == 0
    assert loaded_modes == ["ppe-only"]

    output = capsys.readouterr()
    assert "Modo de analítica: ppe-only" in output.out
    assert f"Modelo EPP ONNX: {onnx}" in output.out
    assert "Binding cuajone_native: OK" in output.out
    assert "Preflight: OK" in output.out
    assert output.err == ""


def test_native_preflight_propagates_engine_startup_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    onnx = tmp_path / "ppe.onnx"
    onnx.write_bytes(b"fixed-onnx")
    (tmp_path / "ppe.onnx.manifest.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(app, "PPE_ONNX_PATH", str(onnx))

    def failed_engine_startup(_mode: str) -> object:
        raise app.RuntimePrerequisiteError("engine construction failed")

    monkeypatch.setattr(app, "load_native_backend", failed_engine_startup)

    assert app.main(["--mode", "ppe-only", "--preflight"]) == 1

    output = capsys.readouterr()
    assert "Binding cuajone_native: FALTA" in output.out
    assert "Binding cuajone_native no disponible" in output.err
    assert "engine construction failed" in output.err
    assert "Preflight: ERROR" in output.err


def test_native_preflight_rejects_missing_onnx_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    onnx = tmp_path / "ppe.onnx"
    onnx.write_bytes(b"fixed-onnx")
    monkeypatch.setattr(app, "PPE_ONNX_PATH", str(onnx))
    monkeypatch.setattr(app, "load_native_backend", lambda _mode: object())

    assert app.main(["--mode", "ppe-only", "--preflight"]) == 1

    output = capsys.readouterr()
    assert "Manifest: FALTA" in output.out
    assert "Falta el modelo ONNX o manifest EPP ONNX" in output.err


def test_native_preflight_requires_pose_onnx_only_for_ppe_fall(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    ppe_onnx = tmp_path / "ppe.onnx"
    ppe_onnx.write_bytes(b"fixed-onnx")
    (tmp_path / "ppe.onnx.manifest.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(app, "PPE_ONNX_PATH", str(ppe_onnx))
    monkeypatch.setattr(app, "POSE_ONNX_PATH", str(tmp_path / "missing-pose.onnx"))
    monkeypatch.setattr(app, "load_native_backend", lambda _mode: object())

    assert app.main(["--mode", "ppe-only", "--preflight"]) == 0
    assert app.main(["--mode", "ppe-fall", "--preflight"]) == 1

    output = capsys.readouterr()
    assert "Falta el modelo ONNX o manifest pose ONNX" in output.err


def test_native_engine_config_uses_fixed_cpu_onnx_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(app, "PPE_ONNX_PATH", "C:/models/ppe.onnx")
    monkeypatch.setattr(app, "POSE_ONNX_PATH", "C:/models/pose.onnx")
    monkeypatch.setattr(app, "PPE_LABELS", "0:Person,1:Hard_hat,2:Vest")

    config = app.native_engine_config("ppe-fall")

    assert config == {
        "backend": "cpu",
        "ppe_onnx": "C:/models/ppe.onnx",
        "pose_onnx": "C:/models/pose.onnx",
        "ppe_labels": {0: "Person", 1: "Hard_hat", 2: "Vest"},
    }


def test_native_frame_translates_canonical_events_for_existing_report() -> None:
    calls: list[dict[str, Any]] = []

    class FakeNativeBackend:
        def process_frame(self, frame: np.ndarray, observations: dict[str, Any]) -> Any:
            calls.append(observations)
            return SimpleNamespace(
                frame_result={"people": []},
                events=(
                    {
                        "id": "evt-CAM_P01-1-3-0",
                        "type": "com.cuajone.safety.ppe.violation.v1",
                        "data": {
                            "track_id": 3,
                            "status": "Falta Chaleco",
                            "confidence": 0.8,
                        },
                    },
                ),
            )

    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    annotated, events = app.process_native_analytics_frame(
        frame,
        "ppe-only",
        FakeNativeBackend(),
        frame_id=7,
        now_monotonic=12.5,
    )

    assert annotated is frame
    assert calls[0]["contract_version"] == "1.0.0"
    assert calls[0]["frame_id"] == 7
    assert calls[0]["monotonic_timestamp_ms"] == 12500
    assert events == [
        {
            "event_id": "evt-CAM_P01-1-3-0",
            "track_id": 3,
            "type": "INCUMPLIMIENTO_EPP",
            "epp_status": "Falta Chaleco",
            "helmet": True,
            "vest": False,
            "confidence": 0.8,
        }
    ]


@pytest.mark.parametrize(
    ("argv", "environ"),
    [([], {"ANALYTICS_MODE": "unknown"}), (["--mode", "unknown"], {})],
)
def test_invalid_mode_is_rejected(
    argv: list[str],
    environ: dict[str, str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        app.parse_args(argv, environ)
    assert exc_info.value.code == 2


def test_inference_throttle_is_unlimited_at_zero() -> None:
    throttle = app.InferenceThrottle(0)
    assert throttle.ready(0.0)
    assert throttle.ready(0.0001)


def test_inference_throttle_skips_until_monotonic_deadline() -> None:
    throttle = app.InferenceThrottle(2)
    assert throttle.ready(10.0)
    assert not throttle.ready(10.49)
    assert throttle.ready(10.5)
    assert throttle.ready(12.0)
    assert not throttle.ready(12.49)


@pytest.mark.parametrize("target_fps", [-1, float("nan"), float("inf")])
def test_inference_throttle_rejects_invalid_values(target_fps: float) -> None:
    with pytest.raises(ValueError, match="TARGET_INFERENCE_FPS"):
        app.InferenceThrottle(target_fps)


def test_latest_frame_capture_returns_stable_reference_without_copy() -> None:
    capture = app.LatestFrameCapture("rtsp://example.invalid/stream")
    first_frame = np.zeros((4, 5, 3), dtype=np.uint8)
    second_frame = np.ones((4, 5, 3), dtype=np.uint8)

    with capture.lock:
        capture.frame = first_frame
        capture.frame_number = 1

    has_frame, returned_frame, frame_number = capture.read_latest(0)

    assert has_frame
    assert returned_frame is first_frame
    assert frame_number == 1

    with capture.lock:
        capture.frame = second_frame
        capture.frame_number = 2

    assert returned_frame is first_frame
    assert np.count_nonzero(returned_frame) == 0
    has_newer_frame, newer_frame, newer_frame_number = capture.read_latest(1)
    assert has_newer_frame
    assert newer_frame is second_frame
    assert newer_frame_number == 2


def configure_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    failure: BaseException | None,
) -> tuple[Any, Any]:
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    camera = SimpleNamespace(stopped=False)
    camera.start = lambda: True
    camera.read_latest = lambda _previous: (True, frame, 1)
    camera.is_disconnected = lambda: False
    camera.stop = lambda: setattr(camera, "stopped", True)
    logger = SimpleNamespace(exported=False, windows_destroyed=False)
    logger.export_excel = lambda: setattr(logger, "exported", True)
    backend = SimpleNamespace(reset=lambda: None)

    app.STOP_EVENT.clear()
    monkeypatch.setattr(app, "RTSP_URL", "rtsp://example.invalid/stream")
    monkeypatch.setattr(app, "EVIDENCE_DIR", tmp_path)
    monkeypatch.setattr(app, "SHOW_WINDOW", False)
    monkeypatch.setattr(app, "TARGET_INFERENCE_FPS", 0)
    monkeypatch.setattr(app, "parse_args", lambda _argv: SimpleNamespace(mode="ppe-only"))
    monkeypatch.setattr(app, "EventLogger", lambda *_args: logger)
    monkeypatch.setattr(app, "diagnose_rtsp_endpoint", lambda _url: None)
    monkeypatch.setattr(app, "validate_native_runtime_prerequisites", lambda _mode: None)
    monkeypatch.setattr(app, "load_native_backend", lambda _mode: backend)
    monkeypatch.setattr(app, "install_signal_handlers", lambda: None)
    monkeypatch.setattr(app, "LatestFrameCapture", lambda _url: camera)
    monkeypatch.setattr(
        app.cv2,
        "destroyAllWindows",
        lambda: setattr(logger, "windows_destroyed", True),
    )

    def run_analytics(**kwargs: Any) -> tuple[np.ndarray, list[dict[str, Any]]]:
        if failure is not None:
            raise failure
        app.request_stop()
        return kwargs["frame"], []

    monkeypatch.setattr(app, "process_native_analytics_frame", run_analytics)
    return camera, logger


def test_normal_exit_stops_camera_and_reports_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    capsys: pytest.CaptureFixture[str],
) -> None:
    camera, logger = configure_runtime(monkeypatch, tmp_path, None)

    assert app.main([]) == 0

    output = capsys.readouterr()
    assert camera.stopped
    assert logger.exported
    assert logger.windows_destroyed
    assert "Monitoreo finalizado correctamente" in output.out
    assert "memoria insuficiente" not in output.err


def test_memory_error_stops_camera_and_reports_failed_shutdown(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    capsys: pytest.CaptureFixture[str],
) -> None:
    camera, logger = configure_runtime(
        monkeypatch,
        tmp_path,
        MemoryError("sin memoria"),
    )

    assert app.main([]) == 1

    output = capsys.readouterr()
    assert camera.stopped
    assert logger.exported
    assert logger.windows_destroyed
    assert "ERROR: memoria insuficiente durante el monitoreo" in output.err
    assert "reduce la resolución del stream o los FPS de inferencia" in output.err
    assert "Monitoreo finalizado correctamente" not in output.out


def test_runtime_prerequisite_error_is_reported_without_starting_camera(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(app, "RTSP_URL", "rtsp://example.invalid/stream")
    monkeypatch.setattr(
        app,
        "validate_native_runtime_prerequisites",
        lambda _mode: (_ for _ in ()).throw(
            app.RuntimePrerequisiteError("falta ppe.onnx")
        ),
    )
    monkeypatch.setattr(
        app,
        "LatestFrameCapture",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("camera must not start")
        ),
    )

    assert app.main(["--mode", "ppe-only"]) == 2

    output = capsys.readouterr()
    assert output.out == ""
    assert output.err.strip() == "ERROR: falta ppe.onnx"


def test_unrelated_runtime_error_propagates_after_stopping_camera(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    capsys: pytest.CaptureFixture[str],
) -> None:
    camera, logger = configure_runtime(
        monkeypatch,
        tmp_path,
        RuntimeError("fallo analítico"),
    )

    with pytest.raises(RuntimeError, match="fallo analítico"):
        app.main([])

    assert camera.stopped
    assert logger.exported
    assert logger.windows_destroyed
    assert "Monitoreo finalizado correctamente" not in capsys.readouterr().out
