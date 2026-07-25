from __future__ import annotations

import csv
import math
import os
import signal
import socket
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import cv2
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from ultralytics import YOLO


# ============================================================
# CONFIGURACIÓN
# ============================================================
# Carga el archivo .env ubicado junto a este script.
SCRIPT_DIR = Path(__file__).resolve().parent
ENV_PATH = SCRIPT_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=False)

CAMERA_ID = os.getenv("CAMERA_ID", "CAM_P01_ADM")
RTSP_URL = os.getenv("RTSP_URL")
PPE_MODEL_PATH = os.getenv("PPE_MODEL_PATH", "best_ppe.pt")
POSE_MODEL_PATH = os.getenv("POSE_MODEL_PATH", "yolo26s-pose.pt")
YOLO_DEVICE = os.getenv("YOLO_DEVICE")  # Vacío: selecciona CUDA si existe.

BASE_DIR = Path(os.getenv("OUTPUT_DIR", str(SCRIPT_DIR))).expanduser()
EVIDENCE_DIR = BASE_DIR / "Evidencias"
CSV_PATH = BASE_DIR / "Reporte_Eventos_Seguridad.csv"
EXCEL_PATH = BASE_DIR / "Reporte_Eventos_Seguridad.xlsx"

SHOW_WINDOW = os.getenv("SHOW_WINDOW", "1") == "1"
SHOW_TEMPORARY_TRACK_ID = os.getenv("SHOW_TEMPORARY_TRACK_ID", "0") == "1"
WINDOW_NAME = "Monitoreo EPP y caidas"
TRACKER = os.getenv("YOLO_TRACKER", "bytetrack.yaml")

POSE_IMGSZ = int(os.getenv("POSE_IMGSZ", "640"))
PPE_IMGSZ = int(os.getenv("PPE_IMGSZ", "640"))
USE_FP16 = os.getenv("USE_FP16", "1") == "1"

POSE_CONF = float(os.getenv("POSE_CONF", "0.35"))
PPE_CONF = float(os.getenv("PPE_CONF", "0.30"))
IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", "0.45"))

# Votación temporal para evitar decidir por un solo frame.
EPP_WINDOW = int(os.getenv("EPP_WINDOW", "20"))
EPP_MIN_SAMPLES = int(os.getenv("EPP_MIN_SAMPLES", "12"))
EPP_PRESENT_RATIO = float(os.getenv("EPP_PRESENT_RATIO", "0.35"))
EPP_ALERT_COOLDOWN_S = float(os.getenv("EPP_ALERT_COOLDOWN_S", "60"))

# Heurística inicial de caída. Debe calibrarse con videos reales de la cámara.
FALL_CONFIRM_FRAMES = int(os.getenv("FALL_CONFIRM_FRAMES", "12"))
FALL_RESET_FRAMES = int(os.getenv("FALL_RESET_FRAMES", "20"))
FALL_ALERT_COOLDOWN_S = float(os.getenv("FALL_ALERT_COOLDOWN_S", "120"))
FALL_ASPECT_RATIO = float(os.getenv("FALL_ASPECT_RATIO", "1.05"))
FALL_TORSO_ANGLE_DEG = float(os.getenv("FALL_TORSO_ANGLE_DEG", "55"))
FALL_DESCENT_RATIO = float(os.getenv("FALL_DESCENT_RATIO", "0.12"))
FALL_NEAR_FLOOR_RATIO = float(os.getenv("FALL_NEAR_FLOOR_RATIO", "0.65"))

TRACK_TTL_S = float(os.getenv("TRACK_TTL_S", "5"))
RECONNECT_DELAY_S = float(os.getenv("RECONNECT_DELAY_S", "5"))
RTSP_OPEN_TIMEOUT_MS = int(os.getenv("RTSP_OPEN_TIMEOUT_MS", "20000"))
RTSP_READ_TIMEOUT_MS = int(os.getenv("RTSP_READ_TIMEOUT_MS", "10000"))
RTSP_TRANSPORT = os.getenv("RTSP_TRANSPORT", "tcp").strip().lower()
RTSP_SOCKET_TIMEOUT_S = float(os.getenv("RTSP_SOCKET_TIMEOUT_S", "3"))
EXCEL_EXPORT_EVERY_EVENTS = int(os.getenv("EXCEL_EXPORT_EVERY_EVENTS", "10"))

# Ajusta estos alias a los nombres exactos de las clases de best_ppe.pt.
HELMET_LABELS = {
    "hard_hat",
    "hardhat",
    "helmet",
    "safety_helmet",
    "casco",
}
VEST_LABELS = {
    "vest",
    "safety_vest",
    "reflective_vest",
    "chaleco",
    "chaleco_reflectivo",
}
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
STOP_EVENT = threading.Event()

# ============================================================
# ESTADO Y REGISTRO
# ============================================================
@dataclass
class TrackState:
    helmet_history: deque[bool] = field(
        default_factory=lambda: deque(maxlen=EPP_WINDOW)
    )
    vest_history: deque[bool] = field(
        default_factory=lambda: deque(maxlen=EPP_WINDOW)
    )
    center_history: deque[tuple[float, float]] = field(
        default_factory=lambda: deque(maxlen=60)
    )

    last_seen: float = 0.0
    last_epp_status: str = ""
    last_epp_alert: float = -1e9

    fall_candidate_frames: int = 0
    upright_frames: int = 0
    fall_active: bool = False
    last_fall_alert: float = -1e9
    recent_descent_until: float = 0.0


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


def intersection_over_item(item_box: np.ndarray, region_box: np.ndarray) -> float:
    """Intersección dividida por el área del EPP detectado."""
    ix1 = max(float(item_box[0]), float(region_box[0]))
    iy1 = max(float(item_box[1]), float(region_box[1]))
    ix2 = min(float(item_box[2]), float(region_box[2]))
    iy2 = min(float(item_box[3]), float(region_box[3]))

    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    item_area = max(1.0, float(item_box[2] - item_box[0])) * max(
        1.0, float(item_box[3] - item_box[1])
    )
    return intersection / item_area



def box_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """IoU entre dos cajas xyxy."""
    ax1, ay1, ax2, ay2 = box_a.astype(float)
    bx1, by1, bx2, by2 = box_b.astype(float)

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)

    area_a = max(1.0, ax2 - ax1) * max(1.0, ay2 - ay1)
    area_b = max(1.0, bx2 - bx1) * max(1.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def keypoint_confidence_threshold() -> float:
    """Reutiliza POSE_CONF sin añadir otra variable de configuración."""
    return max(0.25, min(0.50, POSE_CONF))


def visible_keypoint_count(keypoints: np.ndarray | None) -> int:
    if keypoints is None:
        return 0

    threshold = keypoint_confidence_threshold()
    return sum(
        1
        for point in keypoints
        if point.shape[0] >= 3 and float(point[2]) >= threshold
    )


def keypoint_group_visible(
    keypoints: np.ndarray | None,
    indices: tuple[int, ...],
) -> bool:
    if keypoints is None:
        return False

    threshold = keypoint_confidence_threshold()
    return any(
        index < len(keypoints)
        and keypoints[index].shape[0] >= 3
        and float(keypoints[index][2]) >= threshold
        for index in indices
    )


def pose_confirmed_by_ppe_person(
    person_box: np.ndarray,
    ppe_person_boxes: list[np.ndarray],
) -> bool:
    """Usa la clase Person de best_ppe.pt como confirmación cruzada."""
    minimum_iou = max(0.10, min(0.30, IOU_THRESHOLD / 2.0))
    px1, py1, px2, py2 = person_box.astype(float)

    for candidate in ppe_person_boxes:
        if box_iou(person_box, candidate) >= minimum_iou:
            return True

        cx = float(candidate[0] + candidate[2]) / 2.0
        cy = float(candidate[1] + candidate[3]) / 2.0
        if px1 <= cx <= px2 and py1 <= cy <= py2:
            return True

    return False


def valid_pose_person(
    person: dict[str, Any],
    ppe_person_boxes: list[np.ndarray],
) -> bool:
    """Descarta poses anatómicamente pobres o no confirmadas."""
    keypoints = person["keypoints"]

    if visible_keypoint_count(keypoints) < 5:
        return False
    if not keypoint_group_visible(keypoints, (5, 6)):
        return False
    if not keypoint_group_visible(keypoints, (11, 12)):
        return False

    strong_pose_threshold = min(0.85, POSE_CONF + 0.25)
    if person["confidence"] >= strong_pose_threshold:
        return True

    return pose_confirmed_by_ppe_person(person["box"], ppe_person_boxes)


def epp_evaluable_for_person(
    person: dict[str, Any],
    frame_width: int,
    frame_height: int,
) -> bool:
    """No evalúa EPP cuando la persona está entrando/saliendo o muy recortada."""
    x1, y1, x2, y2 = person["box"].astype(float)
    margin = max(8, int(min(frame_width, frame_height) * 0.01))

    # Para esta cámara la entrada principal ocurre por el borde inferior.
    if y2 >= frame_height - margin:
        return False

    person_height = max(0.0, y2 - y1)
    if person_height / max(1.0, frame_height) < 0.12:
        return False

    keypoints = person["keypoints"]
    head_visible = keypoint_group_visible(keypoints, (0, 1, 2, 3, 4))
    shoulders_visible = keypoint_group_visible(keypoints, (5, 6))
    hips_visible = keypoint_group_visible(keypoints, (11, 12))
    return head_visible and shoulders_visible and hips_visible


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


def draw_associated_ppe(frame: np.ndarray, item: dict[str, Any]) -> None:
    """Dibuja únicamente EPP asociado a una persona válida."""
    x1, y1, x2, y2 = item["box"].astype(int)
    label = "casco" if item["type"] == "helmet" else "chaleco"
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 255), 2)
    cv2.putText(
        frame,
        f"{label} {item['confidence']:.2f}",
        (x1, max(20, y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (0, 220, 255),
        2,
        cv2.LINE_AA,
    )


def region_for_person(person_box: np.ndarray, item_type: str) -> np.ndarray:
    """Crea una región anatómica aproximada para casco o chaleco."""
    x1, y1, x2, y2 = person_box.astype(float)
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)

    if item_type == "helmet":
        return np.array(
            [
                x1 - 0.10 * width,
                y1 - 0.15 * height,
                x2 + 0.10 * width,
                y1 + 0.38 * height,
            ]
        )

    return np.array(
        [
            x1 + 0.02 * width,
            y1 + 0.15 * height,
            x2 - 0.02 * width,
            y1 + 0.78 * height,
        ]
    )


def associate_ppe_to_people(
    people: list[dict[str, Any]],
    ppe_detections: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    """Asocia cada casco/chaleco a una sola persona y conserva la caja asociada."""
    associations: dict[int, dict[str, Any]] = {
        person["track_id"]: {
            "helmet": False,
            "vest": False,
            "helmet_item": None,
            "vest_item": None,
        }
        for person in people
    }

    for item in ppe_detections:
        if item["type"] not in {"helmet", "vest"}:
            continue

        item_box = item["box"]
        item_type = item["type"]
        center_x = float(item_box[0] + item_box[2]) / 2.0
        center_y = float(item_box[1] + item_box[3]) / 2.0

        best_track_id: int | None = None
        best_score = 0.0

        for person in people:
            if not person.get("epp_evaluable", False):
                continue

            region = region_for_person(person["box"], item_type)
            center_inside = (
                region[0] <= center_x <= region[2]
                and region[1] <= center_y <= region[3]
            )
            overlap = intersection_over_item(item_box, region)
            score = overlap + (0.50 if center_inside else 0.0)

            if score > best_score:
                best_score = score
                best_track_id = person["track_id"]

        if best_track_id is not None and best_score >= 0.35:
            item_key = f"{item_type}_item"
            current = associations[best_track_id][item_key]
            if current is None or item["confidence"] > current["confidence"]:
                associations[best_track_id][item_type] = True
                associations[best_track_id][item_key] = item

    return associations

def stable_epp_status(state: TrackState) -> tuple[str | None, float, float]:
    sample_count = min(len(state.helmet_history), len(state.vest_history))
    if sample_count < EPP_MIN_SAMPLES:
        return None, 0.0, 0.0

    helmet_ratio = sum(state.helmet_history) / len(state.helmet_history)
    vest_ratio = sum(state.vest_history) / len(state.vest_history)

    has_helmet = helmet_ratio >= EPP_PRESENT_RATIO
    has_vest = vest_ratio >= EPP_PRESENT_RATIO

    if has_helmet and has_vest:
        status = "EPP Completo"
    elif has_helmet:
        status = "Falta Chaleco"
    elif has_vest:
        status = "Falta Casco"
    else:
        status = "Sin Casco y Chaleco"

    return status, helmet_ratio, vest_ratio


def average_visible_keypoints(
    keypoints: np.ndarray,
    indices: tuple[int, ...],
    minimum_confidence: float = 0.35,
) -> np.ndarray | None:
    selected: list[np.ndarray] = []

    for index in indices:
        if index >= len(keypoints):
            continue

        point = keypoints[index]
        confidence = float(point[2]) if point.shape[0] >= 3 else 1.0
        if confidence >= minimum_confidence:
            selected.append(point[:2].astype(float))

    if not selected:
        return None

    return np.mean(np.stack(selected), axis=0)


def evaluate_fall(
    person_box: np.ndarray,
    keypoints: np.ndarray | None,
    state: TrackState,
    frame_height: int,
    now_monotonic: float,
) -> dict[str, Any]:
    x1, y1, x2, y2 = person_box.astype(float)
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    center_y = (y1 + y2) / 2.0
    aspect_ratio = width / height

    state.center_history.append((now_monotonic, center_y))
    while state.center_history and now_monotonic - state.center_history[0][0] > 1.0:
        state.center_history.popleft()

    previous_centers = [value for _, value in list(state.center_history)[:-1]]
    descent = center_y - min(previous_centers) if previous_centers else 0.0
    rapid_descent = descent >= FALL_DESCENT_RATIO * frame_height
    if rapid_descent:
        state.recent_descent_until = now_monotonic + 1.5

    torso_angle = 0.0
    torso_horizontal = False
    if keypoints is not None:
        shoulders = average_visible_keypoints(keypoints, (5, 6))
        hips = average_visible_keypoints(keypoints, (11, 12))

        if shoulders is not None and hips is not None:
            dx = float(hips[0] - shoulders[0])
            dy = float(hips[1] - shoulders[1])
            torso_angle = math.degrees(math.atan2(abs(dx), abs(dy) + 1e-6))
            torso_horizontal = torso_angle >= FALL_TORSO_ANGLE_DEG

    bbox_horizontal = aspect_ratio >= FALL_ASPECT_RATIO
    near_floor = y2 >= FALL_NEAR_FLOOR_RATIO * frame_height
    recent_descent = now_monotonic <= state.recent_descent_until
    horizontal = bbox_horizontal or torso_horizontal

    candidate = horizontal and (near_floor or recent_descent)

    if candidate:
        state.fall_candidate_frames += 1
        state.upright_frames = 0
    else:
        state.fall_candidate_frames = max(0, state.fall_candidate_frames - 2)

        clearly_upright = aspect_ratio < 0.80 and torso_angle < 35.0
        if clearly_upright:
            state.upright_frames += 1
        else:
            state.upright_frames = 0

    confirmed_now = (
        state.fall_candidate_frames >= FALL_CONFIRM_FRAMES
        and not state.fall_active
        and now_monotonic - state.last_fall_alert >= FALL_ALERT_COOLDOWN_S
    )

    if confirmed_now:
        state.fall_active = True
        state.last_fall_alert = now_monotonic

    if state.fall_active and state.upright_frames >= FALL_RESET_FRAMES:
        state.fall_active = False
        state.fall_candidate_frames = 0

    confidence_score = min(
        1.0,
        0.40 * float(horizontal)
        + 0.35 * float(recent_descent)
        + 0.25 * float(near_floor),
    )

    return {
        "confirmed_now": confirmed_now,
        "active": state.fall_active,
        "score": confidence_score,
        "aspect_ratio": aspect_ratio,
        "torso_angle": torso_angle,
        "rapid_descent": recent_descent,
    }


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


def new_event_id() -> str:
    return f"EVT-{uuid.uuid4().hex.upper()}"


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


def selected_device() -> str:
    if YOLO_DEVICE:
        return YOLO_DEVICE
    return "0" if torch.cuda.is_available() else "cpu"


def model_kwargs() -> dict[str, Any]:
    device = selected_device()

    kwargs: dict[str, Any] = {
        "device": device,
    }

    using_cuda = (
        device not in {"cpu", "mps"}
        and torch.cuda.is_available()
    )

    if USE_FP16 and using_cuda:
        kwargs["quantize"] = 16

    return kwargs


def extract_people(pose_result: Any) -> list[dict[str, Any]]:
    if pose_result.boxes is None or pose_result.boxes.id is None:
        return []

    boxes = pose_result.boxes.xyxy.cpu().numpy()
    track_ids = pose_result.boxes.id.int().cpu().tolist()
    confidences = pose_result.boxes.conf.cpu().numpy()

    keypoint_data: np.ndarray | None = None
    if pose_result.keypoints is not None:
        keypoint_data = pose_result.keypoints.data.cpu().numpy()

    people: list[dict[str, Any]] = []
    for index, (box, track_id, confidence) in enumerate(
        zip(boxes, track_ids, confidences)
    ):
        keypoints = None
        if keypoint_data is not None and index < len(keypoint_data):
            keypoints = keypoint_data[index]

        people.append(
            {
                "track_id": int(track_id),
                "box": box.astype(float),
                "confidence": float(confidence),
                "keypoints": keypoints,
            }
        )

    return people


def extract_ppe_detections(ppe_result: Any) -> list[dict[str, Any]]:
    if ppe_result.boxes is None:
        return []

    boxes = ppe_result.boxes.xyxy.cpu().numpy()
    classes = ppe_result.boxes.cls.int().cpu().tolist()
    confidences = ppe_result.boxes.conf.cpu().numpy()

    detections: list[dict[str, Any]] = []
    for box, class_id, confidence in zip(boxes, classes, confidences):
        label = normalize_label(str(ppe_result.names[int(class_id)]))

        if label in HELMET_LABELS:
            item_type = "helmet"
        elif label in VEST_LABELS:
            item_type = "vest"
        elif label in PERSON_LABELS:
            item_type = "person"
        else:
            continue

        detections.append(
            {
                "box": box.astype(float),
                "type": item_type,
                "label": label,
                "confidence": float(confidence),
            }
        )

    return detections

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

    No conserva una cola de frames:
    siempre reemplaza el frame anterior por el más reciente.
    Esto evita que YOLO procese video atrasado cuando la inferencia
    es más lenta que los FPS enviados por la cámara.
    """

    def __init__(self, rtsp_url: str):
        self.rtsp_url = rtsp_url
        self.capture: cv2.VideoCapture | None = None

        self.frame = None
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
                # El nuevo frame reemplaza al anterior.
                # No se forma una cola.
                self.frame = frame
                self.frame_number += 1

        self.disconnected_event.set()

    def read_latest(
        self,
        previous_frame_number: int,
    ) -> tuple[bool, object | None, int]:
        """
        Devuelve un frame solamente cuando existe uno más reciente
        que el último procesado.
        """

        with self.lock:
            if self.frame is None:
                return False, None, previous_frame_number

            if self.frame_number == previous_frame_number:
                return False, None, previous_frame_number

            return True, self.frame.copy(), self.frame_number

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

# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================
def main() -> None:
    if not RTSP_URL:
        raise RuntimeError(
            f"Falta RTSP_URL. Configúrala en el archivo {ENV_PATH}. "
            "Puedes copiar .env.example como .env y completar la URL."
        )

    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    logger = EventLogger(CSV_PATH, EXCEL_PATH)

    print(f"Configuración cargada desde: {ENV_PATH}")
    print(f"RTSP: {masked_rtsp_url(RTSP_URL)}")
    diagnose_rtsp_endpoint(RTSP_URL)
    print(f"Cargando modelo EPP: {PPE_MODEL_PATH}")
    ppe_model = YOLO(PPE_MODEL_PATH)

    print(f"Cargando modelo pose: {POSE_MODEL_PATH}")
    pose_model = YOLO(POSE_MODEL_PATH)

    print("Clases del modelo EPP:")
    print(ppe_model.names)
    device = selected_device()
    kwargs = model_kwargs()

    fp16_enabled = kwargs.get("quantize") in {16, "fp16"}

    print(
        f"Dispositivo: {device} | "
        f"FP16: {fp16_enabled}"
    )
    print(f"Tracker temporal: {TRACKER}")

    model_names = (
        ppe_model.names.values()
        if isinstance(ppe_model.names, dict)
        else ppe_model.names
    )
    normalized_model_names = {normalize_label(str(name)) for name in model_names}
    if not normalized_model_names.intersection(HELMET_LABELS):
        print("ADVERTENCIA: no se reconoció ninguna clase de casco. Ajusta HELMET_LABELS.")
    if not normalized_model_names.intersection(VEST_LABELS):
        print("ADVERTENCIA: no se reconoció ninguna clase de chaleco. Ajusta VEST_LABELS.")

    states: dict[int, TrackState] = {}
    install_signal_handlers()

    if SHOW_WINDOW:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

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
                frame_height, _ = frame.shape[:2]

                pose_result = pose_model.track(
                    source=frame,
                    persist=True,
                    tracker=TRACKER,
                    conf=POSE_CONF,
                    iou=IOU_THRESHOLD,
                    classes=[0],
                    imgsz=POSE_IMGSZ,
                    verbose=False,
                    **model_kwargs(),
                )[0]

                ppe_result = ppe_model.predict(
                    source=frame,
                    conf=PPE_CONF,
                    iou=IOU_THRESHOLD,
                    imgsz=PPE_IMGSZ,
                    verbose=False,
                    **model_kwargs(),
                )[0]

                raw_people = extract_people(pose_result)
                all_ppe_detections = extract_ppe_detections(ppe_result)

                ppe_person_boxes = [
                    item["box"]
                    for item in all_ppe_detections
                    if item["type"] == "person"
                ]
                people = [
                    person
                    for person in raw_people
                    if valid_pose_person(person, ppe_person_boxes)
                ]

                for person in people:
                    person["epp_evaluable"] = epp_evaluable_for_person(
                        person,
                        frame_width=frame.shape[1],
                        frame_height=frame.shape[0],
                    )

                ppe_detections = [
                    item
                    for item in all_ppe_detections
                    if item["type"] in {"helmet", "vest"}
                ]
                associations = associate_ppe_to_people(people, ppe_detections)

                # Se dibuja desde el frame original para no mostrar detecciones descartadas.
                annotated = frame.copy()
                pending_events: list[dict[str, Any]] = []

                for person in people:
                    track_id = person["track_id"]
                    box = person["box"]
                    state = states.setdefault(track_id, TrackState())
                    state.last_seen = now_monotonic

                    current_ppe = associations.get(
                        track_id,
                        {
                            "helmet": False,
                            "vest": False,
                            "helmet_item": None,
                            "vest_item": None,
                        },
                    )

                    epp_status: str | None = None
                    helmet_ratio = 0.0
                    vest_ratio = 0.0
                    stable_helmet: bool | None = None
                    stable_vest: bool | None = None

                    if person["epp_evaluable"]:
                        state.helmet_history.append(current_ppe["helmet"])
                        state.vest_history.append(current_ppe["vest"])
                        epp_status, helmet_ratio, vest_ratio = stable_epp_status(state)

                    if epp_status is not None:
                        stable_helmet = helmet_ratio >= EPP_PRESENT_RATIO
                        stable_vest = vest_ratio >= EPP_PRESENT_RATIO

                        violation = epp_status != "EPP Completo"
                        status_changed = epp_status != state.last_epp_status
                        cooldown_elapsed = (
                            now_monotonic - state.last_epp_alert
                            >= EPP_ALERT_COOLDOWN_S
                        )

                        if violation and (status_changed or cooldown_elapsed):
                            pending_events.append(
                                {
                                    "track_id": track_id,
                                    "type": "INCUMPLIMIENTO_EPP",
                                    "epp_status": epp_status,
                                    "helmet": stable_helmet,
                                    "vest": stable_vest,
                                    "confidence": min(
                                        1.0,
                                        max(1.0 - helmet_ratio, 1.0 - vest_ratio),
                                    ),
                                }
                            )
                            state.last_epp_alert = now_monotonic

                        state.last_epp_status = epp_status

                    fall = evaluate_fall(
                        person_box=box,
                        keypoints=person["keypoints"],
                        state=state,
                        frame_height=frame_height,
                        now_monotonic=now_monotonic,
                    )

                    if fall["confirmed_now"]:
                        pending_events.append(
                            {
                                "track_id": track_id,
                                "type": "POSIBLE_CAIDA",
                                "epp_status": epp_status or "En evaluación",
                                "helmet": stable_helmet,
                                "vest": stable_vest,
                                "confidence": fall["score"],
                            }
                        )

                    x1, y1, x2, y2 = box.astype(int)
                    display_epp = (
                        epp_status or "Evaluando EPP"
                        if person["epp_evaluable"]
                        else "EPP no evaluable: persona parcial"
                    )
                    display_fall = " | POSIBLE CAIDA" if fall["active"] else ""
                    track_text = (
                        f"T{track_id} | " if SHOW_TEMPORARY_TRACK_ID else ""
                    )
                    label = f"{track_text}{display_epp}{display_fall}"

                    draw_valid_pose(annotated, person)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 255, 255), 2)
                    cv2.putText(
                        annotated,
                        label,
                        (x1, max(25, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                    if current_ppe["helmet_item"] is not None:
                        draw_associated_ppe(annotated, current_ppe["helmet_item"])
                    if current_ppe["vest_item"] is not None:
                        draw_associated_ppe(annotated, current_ppe["vest_item"])

                # Guardar evidencias después de dibujar todas las anotaciones del frame.
                for pending in pending_events:
                    try:
                        event_id = new_event_id()
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
                    except Exception as exc:
                        print(f"Error guardando evento: {exc}")

                # Elimina estados de IDs que ya desaparecieron para evitar crecimiento infinito.
                stale_ids = [
                    track_id
                    for track_id, state in states.items()
                    if now_monotonic - state.last_seen > TRACK_TTL_S
                ]
                for track_id in stale_ids:
                    del states[track_id]

                if SHOW_WINDOW:
                    # Segunda comprobación justo antes de dibujar para impedir que
                    # OpenCV vuelva a crear una ventana que el usuario cerró.
                    if window_was_closed():
                        request_stop()
                        break
                    cv2.imshow(WINDOW_NAME, annotated)

            camera.stop()

            if not STOP_EVENT.is_set():
                STOP_EVENT.wait(RECONNECT_DELAY_S)

    except KeyboardInterrupt:
        request_stop()
    finally:
        logger.export_excel()
        cv2.destroyAllWindows()
        print("Monitoreo finalizado correctamente.")


if __name__ == "__main__":
    main()
