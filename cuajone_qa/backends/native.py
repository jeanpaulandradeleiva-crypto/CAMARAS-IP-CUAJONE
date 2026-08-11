# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import importlib
import json
import os
from types import ModuleType
from typing import Any

import numpy as np

from ..config import QaRuntimeConfig
from ..contracts import CONTRACT_VERSION, CONTRACT_VERSION_V2, validate_instance, validate_instance_v2
from ..ppe import native_item_ids, validate_ppe_labels
from .base import BackendResult


_INFERENCE_PROVIDERS = {
    ("cpu", "onnx-runtime-cpu"): "ONNX_RUNTIME_CPU",
    ("cuda", "onnx-runtime-cuda"): "ONNX_RUNTIME_CUDA",
    ("cuda", "tensorrt"): "TENSORRT",
}


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
        if self._module.CONTRACT_VERSION_V2 != CONTRACT_VERSION_V2:
            raise RuntimeError(
                f"Native v2 contract {self._module.CONTRACT_VERSION_V2} does not match {CONTRACT_VERSION_V2}"
            )
        self._pipeline = self._module.AnalyticsPipeline(self._native_config(config))
        self._engine_pipeline = None
        if engine_config is not None:
            if not self._module.ENGINE_RUNTIME_AVAILABLE:
                raise RuntimeError("This cuajone_native build does not include the external engine runtime")
            native_engine = self._module.EngineConfig()
            backend = str(engine_config.get("backend", "cuda")).lower()
            default_provider = {
                "cpu": "onnx-runtime-cpu",
                "cuda": "tensorrt",
            }.get(backend)
            provider = str(engine_config.get("provider", default_provider)).lower()
            provider_key = (backend, provider)
            try:
                native_provider = _INFERENCE_PROVIDERS[provider_key]
            except KeyError as exc:
                raise ValueError(
                    "Native engine backend/provider must be cpu/onnx-runtime-cpu, "
                    "cuda/onnx-runtime-cuda, or cuda/tensorrt"
                ) from exc
            native_engine.provider = getattr(self._module.InferenceProvider, native_provider)
            if provider in {"onnx-runtime-cpu", "onnx-runtime-cuda"}:
                if not engine_config.get("ppe_onnx"):
                    raise ValueError("Native ONNX engine config requires ppe_onnx")
                if config.mode == "ppe-fall" and not engine_config.get("pose_onnx"):
                    raise ValueError("Native ONNX ppe-fall config requires pose_onnx")
                if not engine_config.get("ppe_labels"):
                    raise ValueError("Native ONNX engine config requires ppe_labels")
                native_engine.backend = (
                    self._module.ComputeBackend.CPU
                    if backend == "cpu"
                    else self._module.ComputeBackend.CUDA
                )
                native_engine.ppe_onnx = str(engine_config["ppe_onnx"])
                native_engine.pose_onnx = str(engine_config.get("pose_onnx", ""))
                native_engine.device = None if backend == "cpu" else int(engine_config.get("device", 0))
            else:
                if not engine_config.get("ppe_engine"):
                    raise ValueError("Native TensorRT engine config requires ppe_engine")
                if config.mode == "ppe-fall" and not engine_config.get("pose_engine"):
                    raise ValueError("Native TensorRT ppe-fall config requires pose_engine")
                native_engine.backend = self._module.ComputeBackend.CUDA
                native_engine.ppe_engine = str(engine_config["ppe_engine"])
                native_engine.pose_engine = str(engine_config.get("pose_engine", ""))
                native_engine.device = int(engine_config.get("device", 0))
            native_engine.ppe_labels = engine_config.get("ppe_labels")
            if engine_config.get("ppe_labels") is not None:
                validate_ppe_labels(dict(engine_config["ppe_labels"]))
            native_engine.pose_class_count = int(engine_config.get("pose_class_count", 1))
            native_engine.pose_keypoint_shape = engine_config.get("pose_keypoint_shape", [17, 3])
            native_engine.allow_nonperson_pose_class = bool(engine_config.get("allow_nonperson_pose_class", False))
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
        tracker.high_confidence_threshold = values["tracker"]["high_confidence_threshold"]
        tracker.low_confidence_threshold = values["tracker"]["low_confidence_threshold"]
        tracker.match_threshold = values["tracker"]["match_threshold"]
        tracker.maximum_age = values["tracker"]["maximum_age"]
        tracker.maximum_tracks = values["tracker"]["maximum_tracks"]
        tracker.frame_rate = values["tracker"]["frame_rate"]
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
        class_map = self._module.PpeClassMap()
        class_map.person_ids = [1]
        class_map.item_ids = native_item_ids(self._module)
        frame.ppe_classes = class_map
        return frame

    @staticmethod
    def _result(bundle: tuple[str, list[str]]) -> BackendResult:
        frame = validate_instance("frame-result", json.loads(bundle[0]))
        events = tuple(validate_instance("event", json.loads(value)) for value in bundle[1])
        return BackendResult(frame, events)

    @staticmethod
    def _result_v2(bundle: tuple[str, list[str]]) -> BackendResult:
        frame = validate_instance_v2("frame-result", json.loads(bundle[0]))
        events = tuple(validate_instance_v2("event", json.loads(value)) for value in bundle[1])
        return BackendResult(frame, events)

    def process_observations(self, observations: dict[str, Any]) -> BackendResult:
        return self._result(
            self._pipeline.process_observations_bundle(self._observations(observations))
        )

    def process_observations_v2(self, observations: dict[str, Any]) -> BackendResult:
        return self._result_v2(
            self._pipeline.process_observations_bundle_v2(self._observations(observations))
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

    def process_frame_v2(self, frame: np.ndarray, observations: dict[str, Any]) -> BackendResult:
        if self._engine_pipeline is not None:
            bundle = self._engine_pipeline.process_frame_v2(
                frame,
                observations["source_id"],
                int(observations["frame_id"]),
                int(observations["monotonic_timestamp_ms"]),
                observations["observed_at"],
            )
            return self._result_v2(bundle)
        return self._result_v2(
            self._pipeline.process_frame_bundle_v2(frame, self._observations(observations))
        )

    def reset(self) -> None:
        self._pipeline.reset()
        if self._engine_pipeline is not None:
            self._engine_pipeline.reset()
