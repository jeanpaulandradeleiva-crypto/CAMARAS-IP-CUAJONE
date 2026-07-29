# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import importlib
import json
import os
from types import ModuleType
from typing import Any

import numpy as np

from ..config import QaRuntimeConfig
from ..contracts import CONTRACT_VERSION, validate_instance
from .base import BackendResult


class NativeBackend:
    """Thin conversion layer over the shared C++ analytics implementation."""

    def __init__(
        self,
        config: QaRuntimeConfig,
        *,
        module: ModuleType | Any | None = None,
        engine_config: dict[str, Any] | None = None,
    ) -> None:
        self._dll_handles = []
        if module is None and os.name == "nt":
            # Keep handles alive; closing one removes that directory from DLL resolution.
            for directory in os.getenv("CUAJONE_NATIVE_DLL_DIRS", "").split(os.pathsep):
                if directory:
                    self._dll_handles.append(os.add_dll_directory(directory))
        self._module = module or importlib.import_module("cuajone_native")
        # Refuse mixed package/binding versions before translating any observations.
        if self._module.CONTRACT_VERSION != CONTRACT_VERSION:
            raise RuntimeError(
                f"Native contract {self._module.CONTRACT_VERSION} does not match {CONTRACT_VERSION}"
            )
        self._pipeline = self._module.AnalyticsPipeline(self._native_config(config))
        self._engine_pipeline = None
        if engine_config is not None:
            if not self._module.ENGINE_RUNTIME_AVAILABLE:
                raise RuntimeError("This cuajone_native build does not include the external engine runtime")
            native_engine = self._module.EngineConfig()
            native_engine.ppe_engine = str(engine_config["ppe_engine"])
            native_engine.pose_engine = str(engine_config.get("pose_engine", ""))
            native_engine.ppe_labels = engine_config.get("ppe_labels")
            native_engine.pose_class_count = int(engine_config.get("pose_class_count", 1))
            native_engine.pose_keypoint_shape = engine_config.get("pose_keypoint_shape", [17, 3])
            native_engine.allow_nonperson_pose_class = bool(engine_config.get("allow_nonperson_pose_class", False))
            native_engine.device = int(engine_config.get("device", 0))
            native_engine.ppe_confidence = config.values["thresholds"]["ppe_confidence"]
            native_engine.pose_confidence = config.values["thresholds"]["pose_confidence"]
            native_engine.nms_iou = config.values["thresholds"]["nms_iou"]
            native_engine.maximum_detections = config.values["thresholds"]["maximum_detections"]
            native_engine.analytics = self._native_config(config)
            self._engine_pipeline = self._module.EnginePipeline(native_engine)

    def _native_config(self, config: QaRuntimeConfig) -> Any:
        values = config.values
        result = self._module.AnalyticsConfig()
        result.mode = (
            self._module.AnalyticsMode.PPE_ONLY
            if config.mode == "ppe-only"
            else self._module.AnalyticsMode.PPE_FALL
        )
        tracker = self._module.TrackerConfig()
        tracker.minimum_iou = values["tracker"]["minimum_iou"]
        tracker.maximum_age = values["tracker"]["maximum_age"]
        tracker.maximum_tracks = values["tracker"]["maximum_tracks"]
        result.tracker = tracker
        ppe = self._module.PpeConfig()
        ppe.window = values["ppe"]["window"]
        ppe.minimum_samples = values["ppe"]["minimum_samples"]
        ppe.present_ratio = values["ppe"]["present_ratio"]
        ppe.alert_cooldown_ms = values["ppe"]["alert_cooldown_ms"]
        ppe.track_ttl_ms = values["ppe"]["track_ttl_ms"]
        result.ppe = ppe
        fall = self._module.FallConfig()
        fall.confirm_frames = values["fall"]["confirm_frames"]
        fall.reset_frames = values["fall"]["reset_frames"]
        fall.alert_cooldown_ms = values["fall"]["alert_cooldown_ms"]
        fall.track_ttl_ms = values["fall"]["track_ttl_ms"]
        fall.aspect_ratio = values["fall"]["aspect_ratio"]
        fall.torso_angle_degrees = values["fall"]["torso_angle_degrees"]
        fall.descent_ratio = values["fall"]["descent_ratio"]
        fall.near_floor_ratio = values["fall"]["near_floor_ratio"]
        result.fall = fall
        result.pose_confidence = values["thresholds"]["pose_confidence"]
        result.nms_iou = values["thresholds"]["nms_iou"]
        return result

    def _box(self, values: list[float]) -> Any:
        if len(values) != 4:
            raise ValueError("box must contain x1, y1, x2, y2")
        box = self._module.Box()
        box.x1, box.y1, box.x2, box.y2 = map(float, values)
        return box

    def _detection(self, value: dict[str, Any]) -> Any:
        detection = self._module.Detection()
        detection.box = self._box(value["box"])
        detection.confidence = float(value["confidence"])
        detection.class_id = int(value["class_id"])
        return detection

    def _pose(self, value: dict[str, Any]) -> Any:
        pose = self._module.PoseDetection()
        pose.box = self._box(value["box"])
        pose.confidence = float(value["confidence"])
        pose.class_id = int(value.get("class_id", 0))
        keypoints = []
        for raw in value.get("keypoints", []):
            if len(raw) != 3:
                raise ValueError("keypoint must contain x, y, confidence")
            point = self._module.Keypoint()
            point.x, point.y, point.confidence = map(float, raw)
            keypoints.append(point)
        pose.keypoints = keypoints
        return pose

    def _observations(self, value: dict[str, Any]) -> Any:
        frame = self._module.ObservationFrame()
        frame.contract_version = value.get("contract_version", CONTRACT_VERSION)
        frame.source_id = value["source_id"]
        frame.frame_id = int(value["frame_id"])
        frame.monotonic_timestamp_ms = int(value["monotonic_timestamp_ms"])
        frame.observed_at = value["observed_at"]
        dimensions = value.get("frame", {})
        frame.frame_width = int(dimensions.get("width", 0))
        frame.frame_height = int(dimensions.get("height", 0))
        frame.ppe_detections = [self._detection(item) for item in value.get("ppe_detections", [])]
        frame.pose_detections = [self._pose(item) for item in value.get("pose_detections", [])]
        classes = value.get("ppe_classes", {"person_ids": [0], "helmet_ids": [1], "vest_ids": [2]})
        class_map = self._module.PpeClassMap()
        class_map.person_ids = list(map(int, classes["person_ids"]))
        class_map.helmet_ids = list(map(int, classes["helmet_ids"]))
        class_map.vest_ids = list(map(int, classes["vest_ids"]))
        frame.ppe_classes = class_map
        return frame

    @staticmethod
    def _result(bundle: tuple[str, list[str]]) -> BackendResult:
        frame = validate_instance("frame-result", json.loads(bundle[0]))
        events = tuple(validate_instance("event", json.loads(value)) for value in bundle[1])
        return BackendResult(frame, events)

    def process_observations(self, observations: dict[str, Any]) -> BackendResult:
        return self._result(
            self._pipeline.process_observations_bundle(self._observations(observations))
        )

    def process_frame(
        self,
        frame: np.ndarray,
        observations: dict[str, Any],
    ) -> BackendResult:
        if self._engine_pipeline is not None:
            bundle = self._engine_pipeline.process_frame(
                frame,
                observations["source_id"],
                int(observations["frame_id"]),
                int(observations["monotonic_timestamp_ms"]),
                observations["observed_at"],
            )
            return self._result(bundle)
        return self._result(
            self._pipeline.process_frame_bundle(frame, self._observations(observations))
        )

    def reset(self) -> None:
        self._pipeline.reset()
        if self._engine_pipeline is not None:
            self._engine_pipeline.reset()
