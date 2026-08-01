# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import ppe_reportev2 as app


class FakeTensor:
    def __init__(self, values: Any) -> None:
        self.values = np.asarray(values)

    def cpu(self) -> "FakeTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self.values

    def int(self) -> "FakeTensor":
        return FakeTensor(self.values.astype(int))

    def tolist(self) -> list[Any]:
        return self.values.tolist()


class FakeBoxes:
    def __init__(
        self,
        boxes: list[list[float]],
        class_ids: list[int],
        track_ids: list[int],
        confidences: list[float],
    ) -> None:
        self.xyxy = FakeTensor(boxes)
        self.cls = FakeTensor(class_ids)
        self.id = FakeTensor(track_ids)
        self.conf = FakeTensor(confidences)


class FakeResult:
    def __init__(
        self,
        boxes: FakeBoxes,
        names: dict[int, str] | None = None,
    ) -> None:
        self.boxes = boxes
        self.names = names or {0: "Person", 1: "Hard_hat", 2: "Vest"}
        self.keypoints = None


class FakePPEModel:
    names = {0: "Person", 1: "Hard_hat", 2: "Vest"}

    def __init__(self, result: FakeResult) -> None:
        self.result = result
        self.track_calls: list[dict[str, Any]] = []

    def track(self, **kwargs: Any) -> list[FakeResult]:
        self.track_calls.append(kwargs)
        return [self.result]

    def predict(self, **_kwargs: Any) -> list[FakeResult]:
        raise AssertionError("PPE-only must use track(), not predict().")


class ExplodingPoseModel:
    def track(self, **_kwargs: Any) -> None:
        raise AssertionError("PPE-only must not call the pose model.")


def ppe_result_with_people() -> FakeResult:
    return FakeResult(
        FakeBoxes(
            boxes=[[10, 10, 50, 80], [12, 8, 25, 25]],
            class_ids=[0, 1],
            track_ids=[73, 800],
            confidences=[0.91, 0.88],
        )
    )


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
        "load_analytics_models",
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


def test_native_preflight_reports_fixed_onnx_without_startup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    onnx = tmp_path / "ppe.onnx"
    onnx.write_bytes(b"fixed-onnx")
    (tmp_path / "ppe.onnx.manifest.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(app, "PPE_ONNX_PATH", str(onnx))
    monkeypatch.setattr(app, "NativeBackend", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        app,
        "LatestFrameCapture",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("preflight must not open RTSP")
        ),
    )
    assert app.main(["--mode", "ppe-only", "--preflight"]) == 0

    output = capsys.readouterr()
    assert "Modo de analítica: ppe-only" in output.out
    assert f"Modelo EPP ONNX: {onnx}" in output.out
    assert "Binding cuajone_native: OK" in output.out
    assert "Preflight: OK" in output.out
    assert output.err == ""


def test_native_preflight_fails_clearly_when_binding_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    onnx = tmp_path / "ppe.onnx"
    onnx.write_bytes(b"fixed-onnx")
    (tmp_path / "ppe.onnx.manifest.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(app, "PPE_ONNX_PATH", str(onnx))

    def missing_binding(*_args: Any, **_kwargs: Any) -> object:
        raise ImportError("not installed")

    monkeypatch.setattr(app, "NativeBackend", missing_binding)

    assert app.main(["--mode", "ppe-only", "--preflight"]) == 1

    output = capsys.readouterr()
    assert "Binding cuajone_native: FALTA" in output.out
    assert "Binding cuajone_native no disponible" in output.err
    assert "Preflight: ERROR" in output.err


def test_native_preflight_rejects_missing_onnx_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    onnx = tmp_path / "ppe.onnx"
    onnx.write_bytes(b"fixed-onnx")
    monkeypatch.setattr(app, "PPE_ONNX_PATH", str(onnx))
    monkeypatch.setattr(app, "NativeBackend", lambda *_args, **_kwargs: object())

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
    monkeypatch.setattr(app, "NativeBackend", lambda *_args, **_kwargs: object())

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


def test_ppe_only_does_not_construct_pose_and_ignores_invalid_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ppe_model = FakePPEModel(ppe_result_with_people())
    constructed_paths: list[str] = []

    def factory(path: str, *, task: str) -> FakePPEModel:
        constructed_paths.append(path)
        assert task == "detect"
        if path != app.PPE_MODEL_PATH:
            raise AssertionError("Pose model was constructed.")
        return ppe_model

    monkeypatch.setattr(app, "POSE_MODEL_PATH", "Z:/missing/invalid-pose.pt")
    models = app.load_analytics_models("ppe-only", yolo_factory=factory)

    assert models.pose is None
    assert constructed_paths == [app.PPE_MODEL_PATH]


def test_model_without_recognized_person_fails_early() -> None:
    class NoPersonModel:
        names = {0: "Hard_hat", 1: "Vest"}

    with pytest.raises(RuntimeError, match="clase Person reconocida"):
        app.load_analytics_models(
            "ppe-only",
            yolo_factory=lambda _path, *, task: NoPersonModel(),
        )


def test_ppe_fall_constructs_explicit_detect_and_pose_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []

    def factory(path: str, *, task: str) -> Any:
        calls.append((path, task))
        return SimpleNamespace(names={0: "Person", 1: "Hard_hat", 2: "Vest"})

    monkeypatch.setattr(app, "PPE_MODEL_PATH", "ppe.pt")
    monkeypatch.setattr(app, "POSE_MODEL_PATH", "pose.pt")

    app.load_analytics_models("ppe-fall", yolo_factory=factory)

    assert calls == [("ppe.pt", "detect"), ("pose.pt", "pose")]


def test_engine_defers_names_until_first_result_and_caches_person_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class LazyEngine:
        def __getattr__(self, name: str) -> Any:
            if name == "names":
                raise AssertionError("Engine names must not be read at construction.")
            raise AttributeError(name)

    monkeypatch.setattr(app, "PPE_MODEL_PATH", "ppe.engine")
    monkeypatch.setattr(app, "YOLO_DEVICE", "cuda:0")
    monkeypatch.setattr(app.torch.cuda, "is_available", lambda: True)
    models = app.load_analytics_models(
        "ppe-only",
        yolo_factory=lambda _path, *, task: LazyEngine(),
    )
    result = ppe_result_with_people()

    app.cache_ppe_result_names(models, result)
    result.names = {0: "Hard_hat"}
    app.cache_ppe_result_names(models, result)

    assert models.person_class_ids == (0,)
    assert models.ppe_names == {0: "Person", 1: "Hard_hat", 2: "Vest"}


def test_engine_missing_person_fails_on_first_result_not_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(app, "PPE_MODEL_PATH", "ppe.engine")
    monkeypatch.setattr(app, "YOLO_DEVICE", "cuda:0")
    monkeypatch.setattr(app.torch.cuda, "is_available", lambda: True)
    models = app.load_analytics_models(
        "ppe-only",
        yolo_factory=lambda _path, *, task: SimpleNamespace(),
    )

    with pytest.raises(RuntimeError, match="clase Person reconocida"):
        app.cache_ppe_result_names(
            models,
            FakeResult(FakeBoxes([], [], [], []), names={0: "Hard_hat"}),
        )


@pytest.mark.parametrize(
    ("device", "cuda_available"),
    [("cpu", True), ("cuda:0", False), (None, False)],
)
def test_engine_rejects_runtime_without_actual_cuda(
    monkeypatch: pytest.MonkeyPatch,
    device: str | None,
    cuda_available: bool,
) -> None:
    monkeypatch.setattr(app, "PPE_MODEL_PATH", "ppe.engine")
    monkeypatch.setattr(app, "YOLO_DEVICE", device)
    monkeypatch.setattr(app.torch.cuda, "is_available", lambda: cuda_available)

    with pytest.raises(RuntimeError, match="requieren una GPU NVIDIA con CUDA activa"):
        app.load_analytics_models(
            "ppe-only",
            yolo_factory=lambda _path, *, task: SimpleNamespace(),
        )


@pytest.mark.parametrize(
    ("configured_device", "cuda_available", "expected_device"),
    [
        (None, True, "cuda:0"),
        ("0", True, "cuda:0"),
        ("cuda:1", True, "cuda:1"),
        ("0", False, "cpu"),
        ("cuda:0", False, "cpu"),
        ("cpu", False, "cpu"),
    ],
)
def test_selected_device_uses_explicit_cuda_or_safe_cpu_fallback(
    monkeypatch: pytest.MonkeyPatch,
    configured_device: str | None,
    cuda_available: bool,
    expected_device: str,
) -> None:
    monkeypatch.setattr(app, "YOLO_DEVICE", configured_device)
    monkeypatch.setattr(app.torch.cuda, "is_available", lambda: cuda_available)

    assert app.selected_device() == expected_device


def test_inference_kwargs_pass_explicit_cuda_device_to_tracking_and_prediction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(app, "YOLO_DEVICE", None)
    monkeypatch.setattr(app.torch.cuda, "is_available", lambda: True)

    ppe_kwargs, pose_kwargs = app.inference_kwargs_for_mode("ppe-fall")

    assert ppe_kwargs["device"] == "cuda:0"
    assert pose_kwargs is not None
    assert pose_kwargs["device"] == "cuda:0"


def test_compiled_engines_do_not_receive_runtime_quantization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(app, "PPE_MODEL_PATH", "ppe.engine")
    monkeypatch.setattr(app, "POSE_MODEL_PATH", "pose.engine")
    monkeypatch.setattr(app, "USE_FP16", True)
    monkeypatch.setattr(app.torch.cuda, "is_available", lambda: True)

    ppe_kwargs, pose_kwargs = app.inference_kwargs_for_mode("ppe-fall")

    assert "quantize" not in ppe_kwargs
    assert pose_kwargs is not None
    assert "quantize" not in pose_kwargs


def test_ppe_only_uses_tracking_ids_from_person_detections() -> None:
    ppe_model = FakePPEModel(ppe_result_with_people())
    models = app.AnalyticsModels(
        ppe=ppe_model,
        pose=ExplodingPoseModel(),
        person_class_ids=(0,),
    )
    ppe_kwargs, pose_kwargs = app.inference_kwargs_for_mode("ppe-only")
    people, _ = app.infer_people_and_ppe(
        frame=np.zeros((100, 100, 3), dtype=np.uint8),
        mode="ppe-only",
        models=models,
        ppe_kwargs=ppe_kwargs,
        pose_kwargs=pose_kwargs,
    )

    assert [person["track_id"] for person in people] == [73]
    assert pose_kwargs is None
    assert ppe_model.track_calls[0]["persist"] is True
    assert ppe_model.track_calls[0]["tracker"] == "bytetrack.yaml"


def test_ppe_only_skips_fall_evaluation_and_pose_drawing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("Pose/fall function called in PPE-only mode.")

    monkeypatch.setattr(app, "evaluate_fall", forbidden)
    monkeypatch.setattr(app, "draw_valid_pose", forbidden)
    ppe_model = FakePPEModel(ppe_result_with_people())
    models = app.AnalyticsModels(ppe_model, ExplodingPoseModel(), (0,))

    _, events = app.process_analytics_frame(
        frame=np.zeros((100, 100, 3), dtype=np.uint8),
        mode="ppe-only",
        models=models,
        states={},
        ppe_kwargs={"persist": True, "tracker": "bytetrack.yaml"},
        pose_kwargs=None,
        now_monotonic=1.0,
    )

    assert all(event["type"] != "POSIBLE_CAIDA" for event in events)


def test_process_analytics_frame_annotates_consumer_frame_in_place(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    person = {
        "track_id": 73,
        "box": np.array([10, 10, 50, 80], dtype=float),
        "confidence": 0.91,
        "keypoints": None,
        "epp_evaluable": False,
    }
    monkeypatch.setattr(
        app,
        "infer_people_and_ppe",
        lambda *_args, **_kwargs: ([person], []),
    )

    annotated, _ = app.process_analytics_frame(
        frame=frame,
        mode="ppe-only",
        models=SimpleNamespace(),
        states={},
        ppe_kwargs={},
        pose_kwargs=None,
        now_monotonic=1.0,
    )

    assert annotated is frame
    assert np.count_nonzero(frame) > 0


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
