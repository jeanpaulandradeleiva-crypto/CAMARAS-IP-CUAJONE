# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..canonical import (
    FALL_EVENT_TYPE,
    PPE_EVENT_TYPE,
    canonical_number,
    float32_value,
    make_event,
    validate_source_id,
)
from ..config import QaRuntimeConfig
from ..contracts import CONTRACT_VERSION, validate_instance
from .base import BackendResult


@dataclass
class _State:
    helmet: deque[bool]
    vest: deque[bool]
    centers: deque[tuple[int, float]] = field(default_factory=deque)
    last_status: str = ""
    last_ppe_alert: int | None = None
    fall_candidate_frames: int = 0
    upright_frames: int = 0
    fall_active: bool = False
    last_fall_alert: int | None = None
    recent_descent_until: int = 0


def _box_area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _iou(left: list[float], right: list[float]) -> float:
    intersection = max(0.0, min(left[2], right[2]) - max(left[0], right[0])) * max(
        0.0, min(left[3], right[3]) - max(left[1], right[1])
    )
    union = _box_area(left) + _box_area(right) - intersection
    return intersection / union if union > 0 else 0.0


def _valid_pose_person(
    pose: dict[str, Any],
    ppe_person_boxes: list[list[float]],
    pose_confidence: float,
    nms_iou: float,
) -> bool:
    threshold = min(0.5, max(0.25, pose_confidence))
    keypoints = pose.get("keypoints", [])
    visible = sum(point[2] >= threshold for point in keypoints)
    group_visible = lambda indices: any(
        index < len(keypoints) and keypoints[index][2] >= threshold for index in indices
    )
    if visible < 5 or not group_visible((5, 6)) or not group_visible((11, 12)):
        return False
    if float(pose["confidence"]) >= min(0.85, pose_confidence + 0.25):
        return True
    pose_box = pose["box"]
    minimum_iou = min(0.3, max(0.1, nms_iou / 2.0))
    for candidate in ppe_person_boxes:
        if _iou(pose_box, candidate) >= minimum_iou:
            return True
        center_x = (candidate[0] + candidate[2]) / 2.0
        center_y = (candidate[1] + candidate[3]) / 2.0
        if pose_box[0] <= center_x <= pose_box[2] and pose_box[1] <= center_y <= pose_box[3]:
            return True
    return False


class ExperimentalBackend:
    """Python reference for QA, not the production runtime or a ByteTrack parity claim."""

    def __init__(self, config: QaRuntimeConfig) -> None:
        self.config = config
        self._states: dict[int, _State] = {}
        self._tracks: dict[int, tuple[list[float], int]] = {}
        self._next_track_id = 1
        self._last_frame_id: int | None = None
        self._last_timestamp = 0

    def reset(self) -> None:
        self._states.clear()
        self._tracks.clear()
        self._next_track_id = 1
        self._last_frame_id = None
        self._last_timestamp = 0

    def _track(
        self,
        boxes: list[list[float]],
        provided_ids: list[int | None] | None = None,
    ) -> list[int]:
        if provided_ids and len(provided_ids) == len(boxes) and all(
            track_id is not None and track_id > 0 for track_id in provided_ids
        ):
            return [int(track_id) for track_id in provided_ids if track_id is not None]
        maximum_age = self.config.values["tracker"]["maximum_age"]
        minimum_iou = self.config.values["tracker"]["minimum_iou"]
        maximum_tracks = self.config.values["tracker"]["maximum_tracks"]
        tracks = {track_id: (box, missed + 1) for track_id, (box, missed) in self._tracks.items()}
        candidates = sorted(
            (
                (-_iou(track_box, box), track_id, index)
                for track_id, (track_box, _missed) in tracks.items()
                for index, box in enumerate(boxes)
                if _iou(track_box, box) >= minimum_iou
            )
        )
        result = [-1] * len(boxes)
        used: set[int] = set()
        for _negative_score, track_id, index in candidates:
            if track_id in used or result[index] != -1:
                continue
            tracks[track_id] = (boxes[index], 0)
            result[index] = track_id
            used.add(track_id)
        tracks = {
            track_id: value for track_id, value in tracks.items() if value[1] <= maximum_age
        }
        for index, box in enumerate(boxes):
            if result[index] != -1 or len(tracks) >= maximum_tracks:
                continue
            result[index] = self._next_track_id
            tracks[self._next_track_id] = (box, 0)
            self._next_track_id += 1
        self._tracks = dict(sorted(tracks.items()))
        return result

    @staticmethod
    def _region(box: list[float], helmet: bool) -> list[float]:
        width = max(1.0, box[2] - box[0])
        height = max(1.0, box[3] - box[1])
        if helmet:
            return [box[0] - 0.1 * width, box[1] - 0.15 * height, box[2] + 0.1 * width, box[1] + 0.38 * height]
        return [box[0] + 0.02 * width, box[1] + 0.15 * height, box[2] - 0.02 * width, box[1] + 0.78 * height]

    @staticmethod
    def _intersection_over_item(item: list[float], region: list[float]) -> float:
        intersection = max(0.0, min(item[2], region[2]) - max(item[0], region[0])) * max(
            0.0, min(item[3], region[3]) - max(item[1], region[1])
        )
        return intersection / max(1.0, _box_area(item))

    def _associate(
        self,
        people: list[dict[str, Any]],
        detections: list[dict[str, Any]],
        helmet_ids: set[int],
        vest_ids: set[int],
    ) -> dict[int, dict[str, bool]]:
        result = {person["track_id"]: {"helmet": False, "vest": False} for person in people}
        scores: dict[tuple[int, str], float] = {}
        for item in detections:
            class_id = int(item["class_id"])
            semantic = "helmet" if class_id in helmet_ids else "vest" if class_id in vest_ids else None
            if semantic is None:
                continue
            box = [float32_value(value) for value in item["box"]]
            center = ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
            best: tuple[float, int] | None = None
            for person in people:
                if not person["ppe_evaluable"]:
                    continue
                region = self._region(person["box"], semantic == "helmet")
                inside = region[0] <= center[0] <= region[2] and region[1] <= center[1] <= region[3]
                score = self._intersection_over_item(box, region) + (0.5 if inside else 0.0)
                candidate = (score, -person["track_id"])
                if best is None or candidate > best:
                    best = candidate
            if best is not None and best[0] >= 0.35:
                track_id = -best[1]
                key = (track_id, semantic)
                confidence = float32_value(item["confidence"])
                if confidence > scores.get(key, -1.0):
                    scores[key] = confidence
                    result[track_id][semantic] = True
        return result

    def _fall(
        self,
        person: dict[str, Any],
        state: _State,
        frame_height: int,
        timestamp: int,
    ) -> tuple[bool, bool, float]:
        config = self.config.values["fall"]
        box = person["box"]
        width = max(1.0, box[2] - box[0])
        height = max(1.0, box[3] - box[1])
        center_y = (box[1] + box[3]) / 2.0
        state.centers.append((timestamp, center_y))
        while state.centers and timestamp - state.centers[0][0] > 1000:
            state.centers.popleft()
        previous = [value for _, value in list(state.centers)[:-1]]
        if previous and center_y - min(previous) >= config["descent_ratio"] * frame_height:
            state.recent_descent_until = timestamp + 1500
        keypoints = person.get("keypoints", [])
        torso_angle = 0.0
        shoulders = [keypoints[index] for index in (5, 6) if index < len(keypoints) and keypoints[index][2] >= 0.35]
        hips = [keypoints[index] for index in (11, 12) if index < len(keypoints) and keypoints[index][2] >= 0.35]
        if shoulders and hips:
            shoulder = np.mean(np.asarray(shoulders)[:, :2], axis=0)
            hip = np.mean(np.asarray(hips)[:, :2], axis=0)
            torso_angle = math.degrees(math.atan2(abs(float(hip[0] - shoulder[0])), abs(float(hip[1] - shoulder[1])) + 1e-6))
        horizontal = width / height >= config["aspect_ratio"] or torso_angle >= config["torso_angle_degrees"]
        near_floor = box[3] >= config["near_floor_ratio"] * frame_height
        recent_descent = timestamp <= state.recent_descent_until
        candidate = horizontal and (near_floor or recent_descent)
        if candidate:
            state.fall_candidate_frames += 1
            state.upright_frames = 0
        else:
            state.fall_candidate_frames = max(0, state.fall_candidate_frames - 2)
            state.upright_frames = state.upright_frames + 1 if width / height < 0.8 and torso_angle < 35 else 0
        cooldown = state.last_fall_alert is None or timestamp - state.last_fall_alert >= config["alert_cooldown_ms"]
        confirmed = state.fall_candidate_frames >= config["confirm_frames"] and not state.fall_active and cooldown
        if confirmed:
            state.fall_active = True
            state.last_fall_alert = timestamp
        if state.fall_active and state.upright_frames >= config["reset_frames"]:
            state.fall_active = False
            state.fall_candidate_frames = 0
        score = min(1.0, 0.4 * horizontal + 0.35 * recent_descent + 0.25 * near_floor)
        return confirmed, state.fall_active, score

    def process_observations(self, observations: dict[str, Any]) -> BackendResult:
        if observations.get("contract_version", CONTRACT_VERSION) != CONTRACT_VERSION:
            raise ValueError("Unsupported contract version")
        validate_source_id(observations["source_id"])
        frame_id = int(observations["frame_id"])
        timestamp = int(observations["monotonic_timestamp_ms"])
        if self._last_frame_id is not None and frame_id <= self._last_frame_id:
            raise ValueError("frame_id must increase strictly until reset")
        if self._last_frame_id is not None and timestamp < self._last_timestamp:
            raise ValueError("monotonic_timestamp_ms must not move backwards until reset")
        dimensions = observations["frame"]
        width, height = int(dimensions["width"]), int(dimensions["height"])
        classes = observations.get("ppe_classes", {"person_ids": [0], "helmet_ids": [1], "vest_ids": [2]})
        person_ids = set(map(int, classes["person_ids"]))
        ppe_detections = observations.get("ppe_detections", [])
        ppe_person_boxes = [
            [float32_value(value) for value in item["box"]]
            for item in ppe_detections
            if int(item["class_id"]) in person_ids
        ]
        candidates = (
            [item for item in ppe_detections if int(item["class_id"]) in person_ids]
            if self.config.mode == "ppe-only"
            else observations.get("pose_detections", [])
        )
        track_ids = self._track(
            [[float32_value(value) for value in item["box"]] for item in candidates],
            [item.get("track_id") for item in candidates],
        )
        threshold = min(0.5, max(0.25, self.config.values["thresholds"]["pose_confidence"]))
        people: list[dict[str, Any]] = []
        for item, track_id in zip(candidates, track_ids):
            if track_id < 0:
                continue
            box = [float32_value(value) for value in item["box"]]
            keypoints = [
                [float32_value(value) for value in point]
                for point in item.get("keypoints", [])
            ]
            candidate = {
                "box": box,
                "confidence": float32_value(item["confidence"]),
                "keypoints": keypoints,
            }
            if self.config.mode == "ppe-fall" and not _valid_pose_person(
                candidate,
                ppe_person_boxes,
                float32_value(self.config.values["thresholds"]["pose_confidence"]),
                float32_value(self.config.values["thresholds"]["nms_iou"]),
            ):
                continue
            box_evaluable = box[3] < height - max(8, int(min(width, height) * 0.01)) and (box[3] - box[1]) / height >= 0.12
            visible = lambda indices: any(index < len(keypoints) and keypoints[index][2] >= threshold for index in indices)
            evaluable = box_evaluable and (
                self.config.mode == "ppe-only" or (visible((0, 1, 2, 3, 4)) and visible((5, 6)) and visible((11, 12)))
            )
            people.append({"track_id": track_id, "box": box, "confidence": candidate["confidence"], "keypoints": keypoints, "ppe_evaluable": evaluable})
        associations = self._associate(people, ppe_detections, set(classes["helmet_ids"]), set(classes["vest_ids"]))
        canonical_people: list[dict[str, Any]] = []
        events: list[dict[str, Any]] = []
        ppe_config = self.config.values["ppe"]
        for person in people:
            track_id = person["track_id"]
            state = self._states.setdefault(track_id, _State(deque(maxlen=ppe_config["window"]), deque(maxlen=ppe_config["window"])))
            association = associations[track_id]
            status = "Evaluating PPE" if person["ppe_evaluable"] else "PPE not evaluable"
            if person["ppe_evaluable"]:
                state.helmet.append(association["helmet"])
                state.vest.append(association["vest"])
            if min(len(state.helmet), len(state.vest)) >= ppe_config["minimum_samples"]:
                helmet_ratio = sum(state.helmet) / len(state.helmet)
                vest_ratio = sum(state.vest) / len(state.vest)
                has_helmet = helmet_ratio >= ppe_config["present_ratio"]
                has_vest = vest_ratio >= ppe_config["present_ratio"]
                status = "EPP Completo" if has_helmet and has_vest else "Falta Chaleco" if has_helmet else "Falta Casco" if has_vest else "Sin Casco y Chaleco"
                cooldown = state.last_ppe_alert is None or timestamp - state.last_ppe_alert >= ppe_config["alert_cooldown_ms"]
                if status != "EPP Completo" and (status != state.last_status or cooldown):
                    events.append(make_event(source_id=observations["source_id"], frame_id=frame_id, monotonic_timestamp_ms=timestamp, observed_at=observations["observed_at"], track_id=track_id, event_index=len(events), event_type=PPE_EVENT_TYPE, status=status, confidence=min(1.0, max(1.0 - helmet_ratio, 1.0 - vest_ratio))))
                    state.last_ppe_alert = timestamp
                state.last_status = status
            fall_active = False
            if self.config.mode == "ppe-fall":
                confirmed, fall_active, score = self._fall(person, state, height, timestamp)
                if confirmed:
                    events.append(make_event(source_id=observations["source_id"], frame_id=frame_id, monotonic_timestamp_ms=timestamp, observed_at=observations["observed_at"], track_id=track_id, event_index=len(events), event_type=FALL_EVENT_TYPE, status=status if person["ppe_evaluable"] else "En evaluación", confidence=score))
            canonical_people.append({
                "track_id": track_id,
                "box": [canonical_number(value) for value in person["box"]],
                "confidence": canonical_number(person["confidence"]),
                "ppe_evaluable": person["ppe_evaluable"],
                "ppe_status": status,
                "fall_active": fall_active,
                "keypoints": [
                    [canonical_number(value) for value in point]
                    for point in person["keypoints"]
                ],
            })
        result = {
            "contract_version": CONTRACT_VERSION,
            "source_id": observations["source_id"],
            "frame_id": frame_id,
            "monotonic_timestamp_ms": timestamp,
            "observed_at": observations["observed_at"],
            "frame": {"width": width, "height": height},
            "people": sorted(canonical_people, key=lambda person: person["track_id"]),
            "events": sorted(event["id"] for event in events),
        }
        self._last_frame_id, self._last_timestamp = frame_id, timestamp
        return BackendResult(validate_instance("frame-result", result), tuple(sorted(events, key=lambda event: event["id"])))

    def process_frame(self, frame: np.ndarray, observations: dict[str, Any]) -> BackendResult:
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be a uint8 BGR array with shape (height, width, 3)")
        updated = dict(observations)
        updated["frame"] = {"width": int(frame.shape[1]), "height": int(frame.shape[0])}
        return self.process_observations(updated)

    def load_ultralytics_models(self, ppe_model: str, pose_model: str | None = None) -> tuple[Any, Any | None]:
        """Lazy live/demo hook. Paths are external and no model is loaded at import time."""
        from ultralytics import YOLO

        ppe = YOLO(ppe_model, task="detect")
        pose = YOLO(pose_model, task="pose") if self.config.mode == "ppe-fall" and pose_model else None
        return ppe, pose

    @staticmethod
    def _tensor_values(value: Any) -> np.ndarray:
        return np.asarray(value.cpu().numpy())

    def process_ultralytics_frame(
        self,
        frame: np.ndarray,
        metadata: dict[str, Any],
        models: tuple[Any, Any | None],
    ) -> BackendResult:
        """Run the lazy experimental model path for an explicitly authorized frame."""
        ppe_model, pose_model = models
        common = {
            "conf": self.config.values["thresholds"]["ppe_confidence"],
            "iou": self.config.values["thresholds"]["nms_iou"],
            "verbose": False,
        }
        ppe_result = (
            ppe_model.track(source=frame, persist=True, tracker="bytetrack.yaml", **common)[0]
            if self.config.mode == "ppe-only"
            else ppe_model.predict(source=frame, **common)[0]
        )
        names = ppe_result.names
        normalized = {
            int(class_id): str(name).strip().lower().replace("-", "_").replace(" ", "_")
            for class_id, name in (names.items() if isinstance(names, dict) else enumerate(names))
        }
        person_aliases = {"person", "persona"}
        helmet_aliases = {"hard_hat", "hardhat", "helmet", "safety_helmet", "casco"}
        vest_aliases = {"vest", "safety_vest", "reflective_vest", "chaleco", "chaleco_reflectivo"}
        classes = {
            "person_ids": [class_id for class_id, name in normalized.items() if name in person_aliases],
            "helmet_ids": [class_id for class_id, name in normalized.items() if name in helmet_aliases],
            "vest_ids": [class_id for class_id, name in normalized.items() if name in vest_aliases],
        }
        if any(not values for values in classes.values()):
            raise RuntimeError("Experimental PPE labels must include person, helmet, and vest semantics")
        boxes = self._tensor_values(ppe_result.boxes.xyxy)
        confidences = self._tensor_values(ppe_result.boxes.conf)
        class_ids = self._tensor_values(ppe_result.boxes.cls).astype(int)
        ppe_track_ids = (
            self._tensor_values(ppe_result.boxes.id).astype(int)
            if getattr(ppe_result.boxes, "id", None) is not None
            else None
        )
        ppe_detections = []
        for index, (box, confidence, class_id) in enumerate(zip(boxes, confidences, class_ids)):
            item = {"box": box.tolist(), "confidence": float(confidence), "class_id": int(class_id)}
            if ppe_track_ids is not None:
                item["track_id"] = int(ppe_track_ids[index])
            ppe_detections.append(item)
        poses: list[dict[str, Any]] = []
        if self.config.mode == "ppe-fall":
            if pose_model is None:
                raise RuntimeError("ppe-fall experimental mode requires an external pose model")
            pose_result = pose_model.track(
                source=frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=self.config.values["thresholds"]["pose_confidence"],
                iou=self.config.values["thresholds"]["nms_iou"],
                verbose=False,
            )[0]
            pose_boxes = self._tensor_values(pose_result.boxes.xyxy)
            pose_confidences = self._tensor_values(pose_result.boxes.conf)
            pose_ids = self._tensor_values(pose_result.boxes.id).astype(int)
            keypoints = self._tensor_values(pose_result.keypoints.data)
            poses = [
                {
                    "box": box.tolist(),
                    "confidence": float(confidence),
                    "class_id": 0,
                    "track_id": int(track_id),
                    "keypoints": points.tolist(),
                }
                for box, confidence, track_id, points in zip(
                    pose_boxes, pose_confidences, pose_ids, keypoints
                )
            ]
        observations = {
            **metadata,
            "frame": {"width": int(frame.shape[1]), "height": int(frame.shape[0])},
            "ppe_classes": classes,
            "ppe_detections": ppe_detections,
            "pose_detections": poses,
        }
        return self.process_observations(observations)
