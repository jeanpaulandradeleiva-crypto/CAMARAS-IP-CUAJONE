# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import argparse
import csv
import math
import os
import signal
import socket
import sys
import threading
import time
from copy import deepcopy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import cv2
import numpy as np
import pandas as pd

from cuajone_qa.backends.native import NativeBackend
from cuajone_qa.config import QaRuntimeConfig
from cuajone_qa.contracts import CONTRACT_VERSION, runtime_defaults, validate_instance
from cuajone_qa.runtime import RuntimeSettings, RuntimeState, resolve_runtime_path as _resolve_runtime_path


class RuntimePrerequisiteError(RuntimeError):
    pass


def resolve_runtime_path(
    configured_path: str | Path,
    *,
    base_dir: Path | None = None,
) -> Path:
    """Resolve relative external files without depending on the working directory."""
    return _resolve_runtime_path(configured_path, base_dir=base_dir or RUNTIME_DIR)


# ============================================================
# CONFIGURACIÓN
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
RUNTIME_DIR = SCRIPT_DIR
RUNTIME_SETTINGS = RuntimeSettings.load(runtime_dir=RUNTIME_DIR)
ENV_PATH = RUNTIME_SETTINGS.env_path

# These aliases preserve the entrypoint's established monkeypatch surface.
CAMERA_ID = RUNTIME_SETTINGS.camera_id
RTSP_URL = RUNTIME_SETTINGS.rtsp_url
PPE_ONNX_PATH = RUNTIME_SETTINGS.ppe_onnx_path
POSE_ONNX_PATH = RUNTIME_SETTINGS.pose_onnx_path
PPE_LABELS = RUNTIME_SETTINGS.ppe_labels
TARGET_INFERENCE_FPS = RUNTIME_SETTINGS.target_inference_fps

DEFAULT_ANALYTICS_MODE = "ppe-fall"
VALID_ANALYTICS_MODES = (DEFAULT_ANALYTICS_MODE, "ppe-only")

BASE_DIR = RUNTIME_SETTINGS.base_dir
EVIDENCE_DIR = RUNTIME_SETTINGS.evidence_dir
CSV_PATH = RUNTIME_SETTINGS.csv_path
EXCEL_PATH = RUNTIME_SETTINGS.excel_path

SHOW_WINDOW = RUNTIME_SETTINGS.show_window
SHOW_TEMPORARY_TRACK_ID = RUNTIME_SETTINGS.show_temporary_track_id
WINDOW_NAME = "Monitoreo EPP y caidas"


POSE_CONF = RUNTIME_SETTINGS.pose_conf
PPE_CONF = RUNTIME_SETTINGS.ppe_conf
IOU_THRESHOLD = RUNTIME_SETTINGS.iou_threshold

# Votación temporal para evitar decidir por un solo frame.
EPP_WINDOW = RUNTIME_SETTINGS.epp_window
EPP_MIN_SAMPLES = RUNTIME_SETTINGS.epp_min_samples
EPP_PRESENT_RATIO = RUNTIME_SETTINGS.epp_present_ratio
EPP_ALERT_COOLDOWN_S = RUNTIME_SETTINGS.epp_alert_cooldown_s

# Heurística inicial de caída. Debe calibrarse con videos reales de la cámara.
FALL_CONFIRM_FRAMES = RUNTIME_SETTINGS.fall_confirm_frames
FALL_RESET_FRAMES = RUNTIME_SETTINGS.fall_reset_frames
FALL_ALERT_COOLDOWN_S = RUNTIME_SETTINGS.fall_alert_cooldown_s
FALL_ASPECT_RATIO = RUNTIME_SETTINGS.fall_aspect_ratio
FALL_TORSO_ANGLE_DEG = RUNTIME_SETTINGS.fall_torso_angle_deg
FALL_DESCENT_RATIO = RUNTIME_SETTINGS.fall_descent_ratio
FALL_NEAR_FLOOR_RATIO = RUNTIME_SETTINGS.fall_near_floor_ratio

TRACK_TTL_S = RUNTIME_SETTINGS.track_ttl_s
RECONNECT_DELAY_S = RUNTIME_SETTINGS.reconnect_delay_s
RTSP_OPEN_TIMEOUT_MS = RUNTIME_SETTINGS.rtsp_open_timeout_ms
RTSP_READ_TIMEOUT_MS = RUNTIME_SETTINGS.rtsp_read_timeout_ms
RTSP_TRANSPORT = RUNTIME_SETTINGS.rtsp_transport
RTSP_SOCKET_TIMEOUT_S = RUNTIME_SETTINGS.rtsp_socket_timeout_s
EXCEL_EXPORT_EVERY_EVENTS = RUNTIME_SETTINGS.excel_export_every_events

PERSON_LABELS = {
    "person",
    "persona",
}

EVENT_FIELDS = [
    "Evento_ID",
    "Camara",
    "Fecha",
    "Hora",
    "Tipo_Evento",
    "Casco",
    "Chaleco",
    "Estado_EPP",
    "Confianza_Evento",
    "ID_Seguimiento_Temporal",
    "Estado_Revision",
    "Identificacion_Humana",
    "Observaciones_Revision",
    "Foto",
]

# El tracker NO identifica personas. Este evento permite correlacionar evidencia,
# cámara y hora; la identidad real debe ser completada por una persona autorizada.
RUNTIME_STATE = RuntimeState()
STOP_EVENT = RUNTIME_STATE.stop_event


def resolve_analytics_mode(
    cli_mode: str | None,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Resuelve una sola frontera de recursos; CLI prevalece sobre el entorno."""
    environment = os.environ if environ is None else environ
    mode = cli_mode or environment.get("ANALYTICS_MODE", DEFAULT_ANALYTICS_MODE)
    normalized = mode.strip().lower()
    if normalized not in VALID_ANALYTICS_MODES:
        valid = ", ".join(VALID_ANALYTICS_MODES)
        raise ValueError(f"Modo de analítica inválido '{mode}'. Valores válidos: {valid}.")
    return normalized


def parse_ppe_labels(value: str) -> dict[int, str]:
    """Parse fixed ONNX labels without relying on an Ultralytics model."""
    labels: dict[int, str] = {}
    for index, raw_label in enumerate(value.split(",")):
        raw_label = raw_label.strip()
        if not raw_label:
            raise RuntimePrerequisiteError("PPE_LABELS no puede contener etiquetas vacías.")
        class_id_text, separator, label = raw_label.partition(":")
        if separator:
            try:
                class_id = int(class_id_text.strip())
            except ValueError as exc:
                raise RuntimePrerequisiteError(
                    f"PPE_LABELS tiene un ID de clase inválido: {class_id_text!r}."
                ) from exc
            label = label.strip()
        else:
            class_id = index
            label = raw_label
        if class_id < 0 or not label or class_id in labels:
            raise RuntimePrerequisiteError("PPE_LABELS debe definir IDs únicos y etiquetas no vacías.")
        labels[class_id] = label
    if list(labels) != list(range(len(labels))):
        raise RuntimePrerequisiteError("PPE_LABELS debe usar IDs consecutivos desde cero.")
    if not recognized_person_class_ids(labels):
        raise RuntimePrerequisiteError("PPE_LABELS no contiene una clase Person reconocida.")
    return labels


def native_runtime_config(mode: str) -> QaRuntimeConfig:
    """Keep configuration thresholds aligned with the shared C++ pipeline."""
    values = deepcopy(runtime_defaults())
    values["analytics"].update(mode=mode, backend="native")
    values["thresholds"].update(
        ppe_confidence=PPE_CONF,
        pose_confidence=POSE_CONF,
        nms_iou=IOU_THRESHOLD,
    )
    values["ppe"].update(
        window=EPP_WINDOW,
        minimum_samples=EPP_MIN_SAMPLES,
        present_ratio=EPP_PRESENT_RATIO,
        alert_cooldown_ms=round(EPP_ALERT_COOLDOWN_S * 1000),
        track_ttl_ms=round(TRACK_TTL_S * 1000),
    )
    values["fall"].update(
        confirm_frames=FALL_CONFIRM_FRAMES,
        reset_frames=FALL_RESET_FRAMES,
        alert_cooldown_ms=round(FALL_ALERT_COOLDOWN_S * 1000),
        track_ttl_ms=round(TRACK_TTL_S * 1000),
        aspect_ratio=FALL_ASPECT_RATIO,
        torso_angle_degrees=FALL_TORSO_ANGLE_DEG,
        descent_ratio=FALL_DESCENT_RATIO,
        near_floor_ratio=FALL_NEAR_FLOOR_RATIO,
    )
    return QaRuntimeConfig(validate_instance("runtime-config", values))


def native_engine_config(mode: str) -> dict[str, Any]:
    config: dict[str, Any] = {
        "backend": "cpu",
        "ppe_onnx": PPE_ONNX_PATH,
        "ppe_labels": parse_ppe_labels(PPE_LABELS),
    }
    if mode == DEFAULT_ANALYTICS_MODE:
        config["pose_onnx"] = POSE_ONNX_PATH
    return config


def validate_native_runtime_prerequisites(mode: str) -> None:
    config = native_engine_config(mode)
    paths = [("EPP ONNX", config["ppe_onnx"])]
    if mode == DEFAULT_ANALYTICS_MODE:
        paths.append(("pose ONNX", config["pose_onnx"]))
    missing = [
        f"{name}: {path}"
        for name, path in paths
        if not Path(path).is_file() or not Path(f"{path}.manifest.json").is_file()
    ]
    if missing:
        raise RuntimePrerequisiteError(
            "Faltan modelos ONNX fijos o sus manifests requeridos: " + "; ".join(missing)
        )


def configure_local_native_binding() -> None:
    """Discover the local QA binding without requiring a shell-specific setup."""
    build_python = (
        RUNTIME_DIR
        / ".tools"
        / "native"
        / "build"
        / "presets"
        / "python-bindings"
        / "python"
    )
    if not build_python.is_dir():
        return

    build_python_text = str(build_python.resolve())
    if build_python_text not in sys.path:
        sys.path.insert(0, build_python_text)

    candidates = [
        build_python,
        RUNTIME_DIR / ".tools" / "native" / "onnxruntime-win-x64-1.25.0" / "lib",
        RUNTIME_DIR / ".tools" / "native" / "opencv" / "opencv" / "build" / "x64" / "vc16" / "bin",
        RUNTIME_DIR / ".tools" / "native" / "cuda-runtime" / "nvidia" / "cuda_runtime" / "bin",
        RUNTIME_DIR / ".tools" / "native" / "tensorrt" / "TensorRT-11.1.0.106" / "bin",
    ]
    configured = [
        value
        for value in os.environ.get("CUAJONE_NATIVE_DLL_DIRS", "").split(os.pathsep)
        if value
    ]
    for candidate in candidates:
        if candidate.is_dir():
            candidate_text = str(candidate.resolve())
            if candidate_text not in configured:
                configured.append(candidate_text)
    if configured:
        os.environ["CUAJONE_NATIVE_DLL_DIRS"] = os.pathsep.join(configured)


def load_native_backend(mode: str) -> NativeBackend:
    configure_local_native_binding()
    try:
        return NativeBackend(native_runtime_config(mode), engine_config=native_engine_config(mode))
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        raise RuntimePrerequisiteError(
            "No se pudo iniciar cuajone_native con los modelos ONNX configurados: "
            f"{exc}"
        ) from exc


def run_native_preflight(mode: str) -> int:
    """Validate static ONNX inputs and the compiled binding without opening RTSP."""
    errors: list[str] = []
    print("Modo de ejecución: cuajone_native (.pyd)")
    print(f"Modo de analítica: {mode}")
    for name, path in (("EPP ONNX", PPE_ONNX_PATH), ("pose ONNX", POSE_ONNX_PATH)):
        if name == "pose ONNX" and mode != DEFAULT_ANALYTICS_MODE:
            continue
        onnx_exists = Path(path).is_file()
        manifest_exists = Path(f"{path}.manifest.json").is_file()
        print(
            f"Modelo {name}: {path} | Archivo: {'OK' if onnx_exists else 'FALTA'} | "
            f"Manifest: {'OK' if manifest_exists else 'FALTA'}"
        )
        if not onnx_exists or not manifest_exists:
            errors.append(f"Falta el modelo ONNX o manifest {name}: {path}")
    try:
        load_native_backend(mode)
        print("Binding cuajone_native: OK")
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        errors.append(f"Binding cuajone_native no disponible: {exc}")
        print("Binding cuajone_native: FALTA")
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        print("Preflight: ERROR", file=sys.stderr)
        return 1
    print("Preflight: OK")
    return 0


def parse_args(
    argv: Sequence[str] | None = None,
    environ: Mapping[str, str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitoreo de EPP y caídas")
    parser.add_argument(
        "--mode",
        choices=VALID_ANALYTICS_MODES,
        help="Sobrescribe ANALYTICS_MODE para esta ejecución.",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Valida configuración y prerrequisitos sin abrir la cámara ni inferir.",
    )
    args = parser.parse_args(argv)
    try:
        args.mode = resolve_analytics_mode(args.mode, environ)
    except ValueError as exc:
        parser.error(str(exc))
    return args


# ============================================================
# ESTADO Y REGISTRO
# ============================================================
@dataclass
class InferenceThrottle:
    """Limita inicios de inferencia sin pausar el hilo que drena el RTSP."""

    target_fps: float
    next_inference_at: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.target_fps) or self.target_fps < 0:
            raise ValueError("TARGET_INFERENCE_FPS debe ser 0 o un número positivo.")

    def ready(self, now_monotonic: float) -> bool:
        if self.target_fps == 0:
            return True
        if now_monotonic < self.next_inference_at:
            return False
        # No intenta recuperar plazos perdidos: después de una inferencia lenta se
        # programa desde "ahora" para evitar ráfagas sobre frames ya obsoletos.
        self.next_inference_at = now_monotonic + (1.0 / self.target_fps)
        return True


class EventLogger:
    """Escribe eventos inmediatamente en CSV y genera un Excel periódicamente."""

    def __init__(self, csv_path: Path, excel_path: Path) -> None:
        self.csv_path = csv_path
        self.excel_path = excel_path
        self.events_since_export = 0
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._rotate_incompatible_csv()

    def _rotate_incompatible_csv(self) -> None:
        """Evita mezclar filas si una versión anterior tenía otras columnas."""
        if not self.csv_path.exists() or self.csv_path.stat().st_size == 0:
            return

        try:
            with self.csv_path.open("r", newline="", encoding="utf-8-sig") as file:
                current_header = next(csv.reader(file), [])
        except Exception as exc:
            print(f"No se pudo validar el encabezado CSV existente: {exc}")
            return

        if current_header == EVENT_FIELDS:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = self.csv_path.with_name(
            f"{self.csv_path.stem}_esquema_anterior_{timestamp}{self.csv_path.suffix}"
        )
        self.csv_path.replace(backup)
        print(f"CSV anterior movido a: {backup}")

    def append(self, event: dict[str, Any]) -> None:
        file_exists = self.csv_path.exists() and self.csv_path.stat().st_size > 0

        with self.csv_path.open("a", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=EVENT_FIELDS)
            if not file_exists:
                writer.writeheader()
            writer.writerow({key: event.get(key, "") for key in EVENT_FIELDS})
            file.flush()
            os.fsync(file.fileno())

        self.events_since_export += 1
        if self.events_since_export >= EXCEL_EXPORT_EVERY_EVENTS:
            self.export_excel()

    def export_excel(self) -> None:
        if not self.csv_path.exists() or self.csv_path.stat().st_size == 0:
            return

        try:
            dataframe = pd.read_csv(self.csv_path, dtype=str).fillna("")

            # Conserva la revisión humana ya escrita en el Excel entre exportaciones.
            review_columns = [
                "Estado_Revision",
                "Identificacion_Humana",
                "Observaciones_Revision",
            ]
            if self.excel_path.exists():
                previous = pd.read_excel(self.excel_path, dtype=str).fillna("")
                if "Evento_ID" in previous.columns:
                    previous = previous.drop_duplicates("Evento_ID", keep="last")
                    previous = previous.set_index("Evento_ID")
                    for column in review_columns:
                        if column not in previous.columns:
                            continue
                        reviewed_values = dataframe["Evento_ID"].map(previous[column])
                        has_review = reviewed_values.notna() & reviewed_values.ne("")
                        dataframe.loc[has_review, column] = reviewed_values[has_review]

            temporary_path = self.excel_path.with_suffix(".tmp.xlsx")
            dataframe.to_excel(temporary_path, index=False)
            temporary_path.replace(self.excel_path)
            self.events_since_export = 0
            print(f"Reporte Excel actualizado: {self.excel_path}")
        except MemoryError:
            raise
        except Exception as exc:
            # El CSV conserva los eventos; el Excel puede estar abierto o bloqueado.
            print(f"No se pudo actualizar Excel: {exc}")


# ============================================================
# UTILIDADES DE DETECCIÓN
# ============================================================
def normalize_label(label: str) -> str:
    return (
        label.strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
        .replace("__", "_")
    )


def keypoint_confidence_threshold() -> float:
    """Reutiliza POSE_CONF sin añadir otra variable de configuración."""
    return max(0.25, min(0.50, POSE_CONF))


def draw_valid_pose(frame: np.ndarray, person: dict[str, Any]) -> None:
    """Dibuja solo la pose que ya pasó la validación."""
    keypoints = person["keypoints"]
    if keypoints is None:
        return

    threshold = keypoint_confidence_threshold()
    skeleton = (
        (0, 1), (0, 2), (1, 3), (2, 4),
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (5, 11), (6, 12), (11, 12),
        (11, 13), (13, 15), (12, 14), (14, 16),
    )

    for start_index, end_index in skeleton:
        if start_index >= len(keypoints) or end_index >= len(keypoints):
            continue
        start = keypoints[start_index]
        end = keypoints[end_index]
        start_conf = float(start[2]) if start.shape[0] >= 3 else 0.0
        end_conf = float(end[2]) if end.shape[0] >= 3 else 0.0
        if start_conf < threshold or end_conf < threshold:
            continue
        cv2.line(
            frame,
            tuple(start[:2].astype(int)),
            tuple(end[:2].astype(int)),
            (255, 140, 0),
            2,
            cv2.LINE_AA,
        )

    for point in keypoints:
        confidence = float(point[2]) if point.shape[0] >= 3 else 0.0
        if confidence < threshold:
            continue
        cv2.circle(frame, tuple(point[:2].astype(int)), 4, (0, 255, 0), -1)


def save_evidence(
    annotated_frame: np.ndarray,
    event_id: str,
    event_type: str,
) -> str:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    safe_event = normalize_label(event_type).upper()
    filename = f"{CAMERA_ID}_{safe_event}_{timestamp}_{event_id[-8:]}.jpg"
    path = EVIDENCE_DIR / filename

    if not cv2.imwrite(str(path), annotated_frame):
        raise IOError(f"OpenCV no pudo escribir la evidencia en {path}")

    return str(path)


def make_event(
    event_id: str,
    temporary_track_id: int,
    event_type: str,
    epp_status: str,
    helmet: bool | None,
    vest: bool | None,
    confidence: float,
    photo_path: str,
) -> dict[str, Any]:
    now = datetime.now()

    return {
        "Evento_ID": event_id,
        "Camara": CAMERA_ID,
        "Fecha": now.strftime("%Y-%m-%d"),
        "Hora": now.strftime("%H:%M:%S.%f")[:-3],
        "Tipo_Evento": event_type,
        "Casco": "SI" if helmet is True else "NO" if helmet is False else "N/D",
        "Chaleco": "SI" if vest is True else "NO" if vest is False else "N/D",
        "Estado_EPP": epp_status,
        "Confianza_Evento": round(confidence, 3),
        "ID_Seguimiento_Temporal": temporary_track_id,
        "Estado_Revision": "PENDIENTE",
        "Identificacion_Humana": "",
        "Observaciones_Revision": "",
        "Foto": photo_path,
    }


def recognized_person_class_ids(model_names: Any) -> tuple[int, ...]:
    if isinstance(model_names, dict):
        names_by_id = model_names.items()
    else:
        names_by_id = enumerate(model_names)
    return tuple(
        int(class_id)
        for class_id, name in names_by_id
        if normalize_label(str(name)) in PERSON_LABELS
    )


def masked_rtsp_url(rtsp_url: str) -> str:
    """Oculta la contraseña al imprimir diagnósticos."""
    try:
        parsed = urlsplit(rtsp_url)
        if parsed.username is None:
            return rtsp_url
        host = parsed.hostname or ""
        port = f":{parsed.port}" if parsed.port else ""
        return f"{parsed.scheme}://{parsed.username}:***@{host}{port}{parsed.path}" + (
            f"?{parsed.query}" if parsed.query else ""
        )
    except Exception:
        return "RTSP_URL inválida o no interpretable"


def diagnose_rtsp_endpoint(rtsp_url: str) -> bool:
    """Comprueba únicamente alcance TCP al host/puerto; no valida credenciales."""
    try:
        parsed = urlsplit(rtsp_url)
        if parsed.scheme.lower() != "rtsp" or not parsed.hostname:
            print("ERROR: RTSP_URL no tiene formato rtsp://usuario:clave@host/ruta")
            return False
        port = parsed.port or 554
        with socket.create_connection(
            (parsed.hostname, port), timeout=RTSP_SOCKET_TIMEOUT_S
        ):
            print(f"Red OK: {parsed.hostname}:{port} responde por TCP.")
        return True
    except OSError as exc:
        print(f"Red/puerto RTSP no accesible: {exc}")
        print("Verifica VPN/VLAN, firewall, IP de cámara y puerto 554.")
        return False


def configure_ffmpeg_rtsp() -> None:
    """Configura FFmpeg desde el propio programa, sin comandos externos."""
    if RTSP_TRANSPORT not in {"tcp", "udp"}:
        raise ValueError("RTSP_TRANSPORT debe ser tcp o udp en .env")

    # OpenCV usa pares clave;valor separados por | para el backend FFmpeg.
    options = f"rtsp_transport;{RTSP_TRANSPORT}"
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = options
    print(f"Transporte RTSP: {RTSP_TRANSPORT.upper()} | FFmpeg: {options}")


def open_capture(rtsp_url: str) -> cv2.VideoCapture:
    """Abre RTSP con FFmpeg y timeouts configurables desde .env."""
    configure_ffmpeg_rtsp()

    parameters: list[int] = []
    if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
        parameters.extend([cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, RTSP_OPEN_TIMEOUT_MS])
    if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
        parameters.extend([cv2.CAP_PROP_READ_TIMEOUT_MSEC, RTSP_READ_TIMEOUT_MS])

    try:
        if parameters:
            capture = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG, parameters)
        else:
            capture = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    except (TypeError, cv2.error) as exc:
        print(f"OpenCV no aceptó parámetros de timeout ({exc}); usando apertura simple.")
        capture = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

    capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    return capture


def request_stop(signum: int | None = None, _frame: Any = None) -> None:
    if not STOP_EVENT.is_set():
        source = f"señal {signum}" if signum is not None else "interfaz"
        print(f"Cierre solicitado desde {source}.")
    STOP_EVENT.set()


def install_signal_handlers() -> None:
    for signal_name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        signal_value = getattr(signal, signal_name, None)
        if signal_value is not None:
            try:
                signal.signal(signal_value, request_stop)
            except (ValueError, OSError):
                pass


def window_was_closed() -> bool:
    try:
        return cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


class LatestFrameCapture:
    """
    Lee el RTSP continuamente en un hilo independiente.

    Este objeto es el único propietario del VideoCapture y de la reconexión local.
    El lector publica bajo lock una referencia numerada; el consumidor nunca toca
    el capture directamente. Después de publicar un ndarray, el productor no lo
    modifica ni reutiliza: cada lectura nueva reemplaza el slot con otro ndarray.
    Así, el consumidor conserva una referencia estable sin copiar el frame completo.

    No conserva una cola de frames:
    siempre reemplaza el frame anterior por el más reciente.
    Esto evita que el backend procese video atrasado cuando la inferencia
    es más lenta que los FPS enviados por la cámara.
    """

    def __init__(self, rtsp_url: str):
        self.rtsp_url = rtsp_url
        self.capture: cv2.VideoCapture | None = None

        self.frame: np.ndarray | None = None
        self.frame_number = 0

        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.disconnected_event = threading.Event()

        self.thread: threading.Thread | None = None

    def start(self) -> bool:
        self.capture = open_capture(self.rtsp_url)

        if not self.capture.isOpened():
            self.capture.release()
            self.capture = None
            return False

        # Puede ayudar dependiendo del backend de OpenCV.
        # El hilo sigue siendo la protección principal contra el retraso.
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.stop_event.clear()
        self.disconnected_event.clear()

        self.thread = threading.Thread(
            target=self._reader_loop,
            name="RTSP-LatestFrame",
            daemon=True,
        )
        self.thread.start()

        return True

    def _reader_loop(self) -> None:
        assert self.capture is not None

        while not self.stop_event.is_set() and not STOP_EVENT.is_set():
            ok, frame = self.capture.read()

            if not ok or frame is None:
                self.disconnected_event.set()
                break

            with self.lock:
                # Publicar cede el contenido de este ndarray al consumidor. El
                # productor sólo reemplaza la referencia; nunca muta un frame
                # publicado, por lo que no se necesita copiarlo bajo el lock.
                self.frame = frame
                self.frame_number += 1

        self.disconnected_event.set()

    def read_latest(
        self,
        previous_frame_number: int,
    ) -> tuple[bool, np.ndarray | None, int]:
        """
        Devuelve una referencia estable solamente cuando existe un frame más nuevo.

        La referencia y su número se capturan juntos bajo el lock. El productor
        puede reemplazar el slot después, pero no muta el ndarray ya publicado.
        """

        with self.lock:
            if self.frame is None:
                return False, None, previous_frame_number

            if self.frame_number == previous_frame_number:
                return False, None, previous_frame_number

            return True, self.frame, self.frame_number

    def is_disconnected(self) -> bool:
        return self.disconnected_event.is_set()

    def stop(self) -> None:
        self.stop_event.set()

        if self.capture is not None:
            self.capture.release()

        if (
            self.thread is not None
            and self.thread.is_alive()
            and self.thread is not threading.current_thread()
        ):
            self.thread.join(timeout=2.0)

        self.capture = None
        self.thread = None

        with self.lock:
            self.frame = None


def native_event_details(event: dict[str, Any]) -> dict[str, Any]:
    data = event["data"]
    status = data["status"]
    helmet, vest = {
        "EPP Completo": (True, True),
        "Falta Chaleco": (True, False),
        "Falta Casco": (False, True),
        "Sin Casco y Chaleco": (False, False),
    }.get(status, (None, None))
    return {
        "event_id": event["id"],
        "track_id": int(data["track_id"]),
        "type": (
            "INCUMPLIMIENTO_EPP"
            if event["type"] == "com.cuajone.safety.ppe.violation.v1"
            else "POSIBLE_CAIDA"
        ),
        "epp_status": "En evaluación" if status == "Evaluating PPE" else status,
        "helmet": helmet,
        "vest": vest,
        "confidence": float(data["confidence"]),
    }


def process_native_analytics_frame(
    frame: np.ndarray,
    mode: str,
    backend: NativeBackend,
    frame_id: int,
    now_monotonic: float,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    result = backend.process_frame(
        frame,
        {
            "contract_version": CONTRACT_VERSION,
            "source_id": CAMERA_ID,
            "frame_id": frame_id,
            "monotonic_timestamp_ms": round(now_monotonic * 1000),
            "observed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        },
    )
    for person in result.frame_result["people"]:
        x1, y1, x2, y2 = map(int, person["box"])
        status = person["ppe_status"]
        display_status = {
            "Evaluating PPE": "Evaluando EPP",
            "PPE not evaluable": "EPP no evaluable: persona parcial",
        }.get(status, status)
        track_text = f"T{person['track_id']} | " if SHOW_TEMPORARY_TRACK_ID else ""
        fall_text = " | POSIBLE CAIDA" if person["fall_active"] else ""
        if mode == DEFAULT_ANALYTICS_MODE and person["keypoints"]:
            draw_valid_pose(
                frame,
                {"keypoints": np.asarray(person["keypoints"], dtype=float)},
            )
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.putText(
            frame,
            f"{track_text}{display_status}{fall_text}",
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return frame, [native_event_details(event) for event in result.events]


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================
def _run_monitoring(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    mode = args.mode
    if getattr(args, "preflight", False):
        return run_native_preflight(mode)
    throttle = InferenceThrottle(TARGET_INFERENCE_FPS)
    if not RTSP_URL:
        raise RuntimePrerequisiteError(
            f"Falta RTSP_URL. Configúrala en el archivo {ENV_PATH}. "
            "Puedes copiar .env.example como .env y completar la URL."
        )
    validate_native_runtime_prerequisites(mode)

    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    logger = EventLogger(CSV_PATH, EXCEL_PATH)

    print(f"Configuración cargada desde: {ENV_PATH}")
    print(f"RTSP: {masked_rtsp_url(RTSP_URL)}")
    diagnose_rtsp_endpoint(RTSP_URL)
    backend: NativeBackend | None = None
    backend = load_native_backend(mode)
    loaded_models = "PPE ONNX" if mode == "ppe-only" else "PPE ONNX, pose ONNX"
    print(f"Modo efectivo: {mode} | Modelos cargados: {loaded_models}")
    print(
        "Backend: cuajone_native + ONNX Runtime CPU | "
        f"FPS objetivo: {TARGET_INFERENCE_FPS or 'sin límite'}"
    )
    print(f"EPP ONNX fijo: {PPE_ONNX_PATH}")
    if mode == DEFAULT_ANALYTICS_MODE:
        print(f"Pose ONNX fijo: {POSE_ONNX_PATH}")
    install_signal_handlers()

    if SHOW_WINDOW:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    camera: LatestFrameCapture | None = None
    native_frame_id = 0
    try:
        while not STOP_EVENT.is_set():
            if SHOW_WINDOW and window_was_closed():
                request_stop()
                break

            print(f"Conectando a {CAMERA_ID}...")
            camera = LatestFrameCapture(RTSP_URL)

            if not camera.start():
                print(
                    f"No se pudo abrir el stream. "
                    f"Reintento en {RECONNECT_DELAY_S}s."
                )

                if STOP_EVENT.wait(RECONNECT_DELAY_S):
                    break

                continue

            print("Stream conectado. Presiona Q, ESC o cierra la ventana.")

            last_frame_number = -1

            while not STOP_EVENT.is_set():
                # Procesa primero los eventos de la interfaz. Es importante hacerlo
                # antes de imshow(), porque imshow recrea una ventana ya cerrada.
                if SHOW_WINDOW:
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27) or window_was_closed():
                        request_stop()
                        break

                has_new_frame, frame, last_frame_number = camera.read_latest(
                    last_frame_number
                )

                if not has_new_frame:
                    if camera.is_disconnected():
                        print("Se perdió el stream RTSP; se intentará reconectar.")
                        break

                    # Evita consumir CPU mientras se espera un frame nuevo.
                    time.sleep(0.002)
                    continue
                now_monotonic = time.monotonic()
                if not throttle.ready(now_monotonic):
                    # El número ya avanzó; al continuar se descarta la referencia
                    # y la captura sigue reemplazando su único slot.
                    continue

                native_frame_id += 1
                annotated, pending_events = process_native_analytics_frame(
                    frame=frame,
                    mode=mode,
                    backend=backend,
                    frame_id=native_frame_id,
                    now_monotonic=now_monotonic,
                )

                # Solo frames con eventos llegan a disco. Se espera a terminar
                # todas las anotaciones para que cada evidencia muestre el contexto
                # completo del instante, no una persona parcialmente procesada.
                for pending in pending_events:
                    try:
                        event_id = pending["event_id"]
                        photo_path = save_evidence(
                            annotated,
                            event_id=event_id,
                            event_type=pending["type"],
                        )
                        event = make_event(
                            event_id=event_id,
                            temporary_track_id=pending["track_id"],
                            event_type=pending["type"],
                            epp_status=pending["epp_status"],
                            helmet=pending["helmet"],
                            vest=pending["vest"],
                            confidence=pending["confidence"],
                            photo_path=photo_path,
                        )
                        logger.append(event)
                        print("EVENTO:", event)
                    except MemoryError:
                        raise
                    except Exception as exc:
                        print(f"Error guardando evento: {exc}")

                if SHOW_WINDOW:
                    # Segunda comprobación justo antes de dibujar para impedir que
                    # OpenCV vuelva a crear una ventana que el usuario cerró.
                    if window_was_closed():
                        request_stop()
                        break
                    cv2.imshow(WINDOW_NAME, annotated)

            camera.stop()
            camera = None

            if not STOP_EVENT.is_set():
                STOP_EVENT.wait(RECONNECT_DELAY_S)

    except KeyboardInterrupt:
        request_stop()
    finally:
        if camera is not None:
            camera.stop()
        if backend is not None:
            backend.reset()
        try:
            logger.export_excel()
        finally:
            cv2.destroyAllWindows()
    print("Monitoreo finalizado correctamente.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return _run_monitoring(argv)
    except RuntimePrerequisiteError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except MemoryError:
        print(
            "ERROR: memoria insuficiente durante el monitoreo. "
            "Cierra otras aplicaciones y reinicia; si persiste, reduce la "
            "resolución del stream o los FPS de inferencia.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
