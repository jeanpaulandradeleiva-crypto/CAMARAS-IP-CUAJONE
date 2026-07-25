from __future__ import annotations

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

    def factory(path: str) -> FakePPEModel:
        constructed_paths.append(path)
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
        app.load_analytics_models("ppe-only", yolo_factory=lambda _path: NoPersonModel())


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
