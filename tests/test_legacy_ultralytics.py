# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from cuajone_qa.experimental import legacy_ultralytics as app


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
