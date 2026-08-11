# SPDX-License-Identifier: AGPL-3.0-only

"""Runtime configuration and process-local state for the monitoring facade."""

from __future__ import annotations

import os
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


def resolve_runtime_path(configured_path: str | Path, *, base_dir: Path) -> Path:
    """Resolve an external runtime path relative to the application directory."""
    path = Path(configured_path).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve(strict=False)


@dataclass(frozen=True)
class RuntimeSettings:
    """Environment-derived values used by the operational monitoring runtime."""

    runtime_dir: Path
    camera_id: str
    rtsp_url: str | None
    ppe_model_path: str
    pose_model_path: str
    ppe_onnx_path: str
    pose_onnx_path: str
    ppe_labels: str
    yolo_device: str | None
    target_inference_fps: float
    base_dir: Path
    show_window: bool
    show_temporary_track_id: bool
    tracker: str
    pose_imgsz: int
    ppe_imgsz: int
    use_fp16: bool
    pose_conf: float
    ppe_conf: float
    iou_threshold: float
    epp_window: int
    epp_min_samples: int
    epp_present_ratio: float
    epp_alert_cooldown_s: float
    fall_confirm_frames: int
    fall_reset_frames: int
    fall_alert_cooldown_s: float
    fall_aspect_ratio: float
    fall_torso_angle_deg: float
    fall_descent_ratio: float
    fall_near_floor_ratio: float
    track_ttl_s: float
    reconnect_delay_s: float
    rtsp_open_timeout_ms: int
    rtsp_read_timeout_ms: int
    rtsp_transport: str
    rtsp_socket_timeout_s: float
    excel_export_every_events: int

    @property
    def env_path(self) -> Path:
        return self.runtime_dir / ".env"

    @property
    def evidence_dir(self) -> Path:
        return self.base_dir / "Evidencias"

    @property
    def csv_path(self) -> Path:
        return self.base_dir / "Reporte_Eventos_Seguridad.csv"

    @property
    def excel_path(self) -> Path:
        return self.base_dir / "Reporte_Eventos_Seguridad.xlsx"

    @classmethod
    def load(cls, *, runtime_dir: Path) -> "RuntimeSettings":
        """Load the optional local .env once, preserving process environment overrides."""
        resolved_runtime_dir = runtime_dir.resolve(strict=False)
        if os.getenv("CUAJONE_SKIP_DOTENV") != "1":
            load_dotenv(dotenv_path=resolved_runtime_dir / ".env", override=False)
        return cls.from_environment(os.environ, runtime_dir=resolved_runtime_dir)

    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str],
        *,
        runtime_dir: Path,
    ) -> "RuntimeSettings":
        """Construct settings from a supplied mapping without mutating process state."""
        resolved_runtime_dir = runtime_dir.resolve(strict=False)

        def value(name: str, default: str | None = None) -> str | None:
            return environ.get(name, default)

        base_dir = resolve_runtime_path(
            value("OUTPUT_DIR", str(resolved_runtime_dir)) or "",
            base_dir=resolved_runtime_dir,
        )
        return cls(
            runtime_dir=resolved_runtime_dir,
            camera_id=value("CAMERA_ID", "CAM_P01_ADM"),
            rtsp_url=value("RTSP_URL"),
            ppe_model_path=str(
                resolve_runtime_path(
                    value("PPE_MODEL_PATH", "best_ppe.pt") or "",
                    base_dir=resolved_runtime_dir,
                )
            ),
            pose_model_path=str(
                resolve_runtime_path(
                    value("POSE_MODEL_PATH", "yolo26s-pose.pt") or "",
                    base_dir=resolved_runtime_dir,
                )
            ),
            ppe_onnx_path=str(
                resolve_runtime_path(
                    value("PPE_ONNX_PATH", "models/ppe.onnx") or "",
                    base_dir=resolved_runtime_dir,
                )
            ),
            pose_onnx_path=str(
                resolve_runtime_path(
                    value("POSE_ONNX_PATH", "models/pose.onnx") or "",
                    base_dir=resolved_runtime_dir,
                )
            ),
            ppe_labels=value(
                "PPE_LABELS",
                "Gloves,Person,Safety_boots,Vest,respirador,tapaorejas,Hard_hat,lentes_protectores",
            ) or "",
            yolo_device=value("YOLO_DEVICE"),
            target_inference_fps=float(value("TARGET_INFERENCE_FPS", "0") or ""),
            base_dir=base_dir,
            show_window=value("SHOW_WINDOW", "1") == "1",
            show_temporary_track_id=value("SHOW_TEMPORARY_TRACK_ID", "0") == "1",
            tracker=value("YOLO_TRACKER", "bytetrack.yaml") or "",
            pose_imgsz=int(value("POSE_IMGSZ", "640") or ""),
            ppe_imgsz=int(value("PPE_IMGSZ", "640") or ""),
            use_fp16=value("USE_FP16", "1") == "1",
            pose_conf=float(value("POSE_CONF", "0.35") or ""),
            ppe_conf=float(value("PPE_CONF", "0.30") or ""),
            iou_threshold=float(value("IOU_THRESHOLD", "0.45") or ""),
            epp_window=int(value("EPP_WINDOW", "20") or ""),
            epp_min_samples=int(value("EPP_MIN_SAMPLES", "12") or ""),
            epp_present_ratio=float(value("EPP_PRESENT_RATIO", "0.35") or ""),
            epp_alert_cooldown_s=float(value("EPP_ALERT_COOLDOWN_S", "60") or ""),
            fall_confirm_frames=int(value("FALL_CONFIRM_FRAMES", "12") or ""),
            fall_reset_frames=int(value("FALL_RESET_FRAMES", "20") or ""),
            fall_alert_cooldown_s=float(value("FALL_ALERT_COOLDOWN_S", "120") or ""),
            fall_aspect_ratio=float(value("FALL_ASPECT_RATIO", "1.05") or ""),
            fall_torso_angle_deg=float(value("FALL_TORSO_ANGLE_DEG", "55") or ""),
            fall_descent_ratio=float(value("FALL_DESCENT_RATIO", "0.12") or ""),
            fall_near_floor_ratio=float(value("FALL_NEAR_FLOOR_RATIO", "0.65") or ""),
            track_ttl_s=float(value("TRACK_TTL_S", "5") or ""),
            reconnect_delay_s=float(value("RECONNECT_DELAY_S", "5") or ""),
            rtsp_open_timeout_ms=int(value("RTSP_OPEN_TIMEOUT_MS", "20000") or ""),
            rtsp_read_timeout_ms=int(value("RTSP_READ_TIMEOUT_MS", "10000") or ""),
            rtsp_transport=(value("RTSP_TRANSPORT", "tcp") or "").strip().lower(),
            rtsp_socket_timeout_s=float(value("RTSP_SOCKET_TIMEOUT_S", "3") or ""),
            excel_export_every_events=int(value("EXCEL_EXPORT_EVERY_EVENTS", "10") or ""),
        )


@dataclass
class RuntimeState:
    """Mutable process-local coordination state for one monitoring runtime."""

    stop_event: threading.Event = field(default_factory=threading.Event)
