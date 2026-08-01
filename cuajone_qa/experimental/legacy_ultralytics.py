# SPDX-License-Identifier: AGPL-3.0-only

"""Legacy Ultralytics analytics retained only for local experiments and QA."""

from __future__ import annotations

import importlib
import math
import platform
import sys
import uuid
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from ppe_reportev2 import (
    DEFAULT_ANALYTICS_MODE,
    EPP_ALERT_COOLDOWN_S,
    EPP_MIN_SAMPLES,
    EPP_PRESENT_RATIO,
    EPP_WINDOW,
    FALL_ALERT_COOLDOWN_S,
    FALL_ASPECT_RATIO,
    FALL_CONFIRM_FRAMES,
    FALL_DESCENT_RATIO,
    FALL_NEAR_FLOOR_RATIO,
    FALL_RESET_FRAMES,
    FALL_TORSO_ANGLE_DEG,
    IOU_THRESHOLD,
    PERSON_LABELS,
    POSE_CONF,
    PPE_CONF,
    RUNTIME_SETTINGS,
    SHOW_TEMPORARY_TRACK_ID,
    RuntimePrerequisiteError,
    draw_valid_pose,
    keypoint_confidence_threshold,
    normalize_label,
    recognized_person_class_ids,
    resolve_analytics_mode,
)


PPE_MODEL_PATH = RUNTIME_SETTINGS.ppe_model_path
POSE_MODEL_PATH = RUNTIME_SETTINGS.pose_model_path
YOLO_DEVICE = RUNTIME_SETTINGS.yolo_device
TRACKER = RUNTIME_SETTINGS.tracker
POSE_IMGSZ = RUNTIME_SETTINGS.pose_imgsz
PPE_IMGSZ = RUNTIME_SETTINGS.ppe_imgsz
USE_FP16 = RUNTIME_SETTINGS.use_fp16

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


@dataclass
class AnalyticsModels:
    """Modelos realmente disponibles; `pose=None` garantiza aislamiento PPE-only."""

    ppe: Any
    pose: Any | None
    person_class_ids: tuple[int, ...] | None
    ppe_names: Any | None = None

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


def box_epp_evaluable_for_person(
    person: dict[str, Any],
    frame_width: int,
    frame_height: int,
) -> bool:
    """Filtro geométrico compartido que no presupone la existencia de keypoints."""
    x1, y1, x2, y2 = person["box"].astype(float)
    margin = max(8, int(min(frame_width, frame_height) * 0.01))

    # Para esta cámara la entrada principal ocurre por el borde inferior.
    if y2 >= frame_height - margin:
        return False

    person_height = max(0.0, y2 - y1)
    if person_height / max(1.0, frame_height) < 0.12:
        return False

    return True


def pose_keypoints_evaluable_for_person(person: dict[str, Any]) -> bool:
    """Añade al filtro de caja la visibilidad anatómica exclusiva de ppe-fall."""
    keypoints = person.get("keypoints")
    head_visible = keypoint_group_visible(keypoints, (0, 1, 2, 3, 4))
    shoulders_visible = keypoint_group_visible(keypoints, (5, 6))
    hips_visible = keypoint_group_visible(keypoints, (11, 12))
    return head_visible and shoulders_visible and hips_visible

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
    """Aproxima cabeza o torso para no asignar EPP solo por cercanía de cajas."""
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
    """Asocia cada EPP a una sola región anatómica y conserva la mejor detección."""
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
            # El centro dentro de la zona pesa más que un roce entre cajas, algo
            # frecuente cuando dos trabajadores aparecen juntos.
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
    """Vota sobre una ventana temporal para no alertar por una omisión aislada."""
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
    """Mantiene confirmación y recuperación separadas para evitar oscilaciones."""
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
        # La evidencia negativa reduce gradualmente el candidato; un único frame
        # ruidoso no borra de inmediato una secuencia de caída coherente.
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
        # Se exige una secuencia erguida independiente antes de permitir una nueva
        # caída, evitando eventos repetidos mientras la persona sigue en el suelo.
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

def new_event_id() -> str:
    return f"EVT-{uuid.uuid4().hex.upper()}"

def selected_device() -> str:
    requested_device = YOLO_DEVICE.strip() if YOLO_DEVICE else "cuda:0"
    normalized_device = requested_device.lower()
    cuda_requested = (
        normalized_device == "cuda"
        or normalized_device.startswith("cuda:")
        or requested_device.isdigit()
    )

    if cuda_requested:
        if not torch.cuda.is_available():
            return "cpu"
        if requested_device.isdigit():
            return f"cuda:{requested_device}"
        return "cuda:0" if normalized_device == "cuda" else requested_device

    return requested_device


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

def validate_ppe_names(model_names: Any) -> tuple[int, ...]:
    person_class_ids = recognized_person_class_ids(model_names)
    if not person_class_ids:
        recognized = ", ".join(sorted(PERSON_LABELS))
        raise RuntimeError(
            "El modelo EPP no contiene una clase Person reconocida. "
            f"Nombres aceptados: {recognized}. Clases encontradas: {model_names}"
        )
    return person_class_ids


def is_engine_model(path: str) -> bool:
    return Path(path).suffix.lower() == ".engine"


def active_model_paths(mode: str) -> tuple[tuple[str, str], ...]:
    models = [("EPP", PPE_MODEL_PATH)]
    if mode == DEFAULT_ANALYTICS_MODE:
        models.append(("pose", POSE_MODEL_PATH))
    return tuple(models)


def model_backend(path: str) -> str:
    return "TensorRT" if is_engine_model(path) else "PyTorch"


def validate_model_files(mode: str) -> None:
    missing = [
        f"{name}: {path}"
        for name, path in active_model_paths(mode)
        if not Path(path).is_file()
    ]
    if missing:
        raise RuntimePrerequisiteError(
            "No se encontraron los archivos de modelo requeridos: " + "; ".join(missing)
        )


def ensure_tensorrt_importable(
    model_paths: Sequence[str],
    importer: Any | None = None,
) -> None:
    if not any(is_engine_model(path) for path in model_paths):
        return
    importer = importer or importlib.import_module
    try:
        importer("tensorrt")
    except Exception as exc:
        raise RuntimePrerequisiteError(
            "Hay un modelo .engine configurado, pero TensorRT no se puede importar. "
            "Instala una versión de TensorRT compatible con el engine, CUDA, el "
            "controlador NVIDIA y esta distribución."
        ) from exc


def ensure_engine_cuda(model_paths: Sequence[str]) -> None:
    engine_paths = [path for path in model_paths if is_engine_model(path)]
    if not engine_paths:
        return

    device = selected_device()
    if not torch.cuda.is_available() or not device.lower().startswith("cuda:"):
        paths = ", ".join(engine_paths)
        raise RuntimePrerequisiteError(
            "Los modelos TensorRT .engine requieren una GPU NVIDIA con CUDA activa; "
            "no se permite continuar en CPU. Verifica el driver/CUDA, instala una "
            "versión de PyTorch con CUDA y configura YOLO_DEVICE=cuda:0. "
            f"Modelos: {paths}"
        )


def validate_runtime_prerequisites(mode: str) -> None:
    paths = [path for _name, path in active_model_paths(mode)]
    validate_model_files(mode)
    ensure_engine_cuda(paths)
    ensure_tensorrt_importable(paths)


def run_preflight(mode: str, importer: Any | None = None) -> int:
    """Print safe diagnostics without constructing models or touching RTSP."""
    models = active_model_paths(mode)
    importer = importer or importlib.import_module
    engine_paths = [path for _name, path in models if is_engine_model(path)]
    errors: list[str] = []

    print("Modo de ejecución: source")
    print(f"Python: {platform.python_version()} ({sys.executable})")
    print(f"Plataforma: {platform.platform()}")
    print(f"Modo de analítica: {mode}")
    for name, path in models:
        exists = Path(path).is_file()
        print(
            f"Modelo {name}: {path} | Backend: {model_backend(path)} | "
            f"Archivo: {'OK' if exists else 'FALTA'}"
        )
        if not exists:
            errors.append(f"Falta el modelo {name}: {path}")

    try:
        cuda_available = bool(torch.cuda.is_available())
    except Exception as exc:
        cuda_available = False
        errors.append(f"No se pudo consultar CUDA: {type(exc).__name__}.")
    print(f"CUDA disponible: {'sí' if cuda_available else 'no'}")
    device = selected_device()
    print(f"Dispositivo seleccionado: {device}")
    if engine_paths and not cuda_available:
        errors.append("TensorRT .engine requiere CUDA disponible en PyTorch.")
    if engine_paths and not device.lower().startswith("cuda:"):
        errors.append(
            "TensorRT .engine requiere YOLO_DEVICE en un dispositivo CUDA, "
            "por ejemplo cuda:0."
        )

    if engine_paths:
        try:
            ensure_tensorrt_importable(engine_paths, importer)
            print("Importación TensorRT: OK (requerida)")
        except RuntimeError as exc:
            print("Importación TensorRT: FALTA (requerida)")
            errors.append(str(exc))
    else:
        print("Importación TensorRT: no requerida")

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        print("Preflight: ERROR", file=sys.stderr)
        return 1
    print("Preflight: OK")
    return 0


def cache_ppe_result_names(models: AnalyticsModels, ppe_result: Any) -> None:
    """Valida una sola vez los nombres que un backend exportado entrega al inferir."""
    if models.person_class_ids is not None:
        return
    model_names = getattr(ppe_result, "names", None)
    if model_names is None:
        raise RuntimeError(
            "El resultado del modelo EPP no incluye nombres de clases; no se puede "
            "identificar Person ni asociar el EPP."
        )
    models.person_class_ids = validate_ppe_names(model_names)
    models.ppe_names = model_names


def load_analytics_models(
    mode: str,
    yolo_factory: Any = YOLO,
) -> AnalyticsModels:
    """Carga pose solo en ppe-fall; PPE-only nunca toca su ruta ni constructor."""
    mode = resolve_analytics_mode(mode, {})
    active_paths = [PPE_MODEL_PATH]
    if mode == DEFAULT_ANALYTICS_MODE:
        active_paths.append(POSE_MODEL_PATH)
    ensure_engine_cuda(active_paths)

    print(f"Cargando modelo EPP: {PPE_MODEL_PATH}")
    ppe_model = yolo_factory(PPE_MODEL_PATH, task="detect")
    ppe_names = None
    person_class_ids = None
    if not is_engine_model(PPE_MODEL_PATH):
        ppe_names = getattr(ppe_model, "names", None)
        if ppe_names is not None:
            person_class_ids = validate_ppe_names(ppe_names)

    pose_model = None
    if mode == DEFAULT_ANALYTICS_MODE:
        print(f"Cargando modelo pose: {POSE_MODEL_PATH}")
        pose_model = yolo_factory(POSE_MODEL_PATH, task="pose")

    return AnalyticsModels(
        ppe=ppe_model,
        pose=pose_model,
        person_class_ids=person_class_ids,
        ppe_names=ppe_names,
    )


def inference_kwargs_for_mode(
    mode: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Construye una vez los argumentos inmutables que reutiliza el bucle."""
    common_kwargs = model_kwargs()
    ppe_runtime_kwargs = dict(common_kwargs)
    if is_engine_model(PPE_MODEL_PATH):
        ppe_runtime_kwargs.pop("quantize", None)
    ppe_kwargs: dict[str, Any] = {
        "conf": PPE_CONF,
        "iou": IOU_THRESHOLD,
        "imgsz": PPE_IMGSZ,
        "verbose": False,
        **ppe_runtime_kwargs,
    }
    pose_kwargs: dict[str, Any] | None = None

    if mode == "ppe-only":
        ppe_kwargs.update({"persist": True, "tracker": TRACKER})
    else:
        pose_runtime_kwargs = dict(common_kwargs)
        if is_engine_model(POSE_MODEL_PATH):
            pose_runtime_kwargs.pop("quantize", None)
        pose_kwargs = {
            "persist": True,
            "tracker": TRACKER,
            "conf": POSE_CONF,
            "iou": IOU_THRESHOLD,
            "classes": [0],
            "imgsz": POSE_IMGSZ,
            "verbose": False,
            **pose_runtime_kwargs,
        }
    return ppe_kwargs, pose_kwargs


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


def extract_tracked_ppe_people(
    ppe_result: Any,
    person_class_ids: tuple[int, ...],
) -> list[dict[str, Any]]:
    """Obtiene IDs exclusivamente de las cajas Person rastreadas por el modelo EPP."""
    if ppe_result.boxes is None or ppe_result.boxes.id is None:
        return []

    boxes = ppe_result.boxes.xyxy.cpu().numpy()
    track_ids = ppe_result.boxes.id.int().cpu().tolist()
    classes = ppe_result.boxes.cls.int().cpu().tolist()
    confidences = ppe_result.boxes.conf.cpu().numpy()
    people: list[dict[str, Any]] = []

    for box, track_id, class_id, confidence in zip(
        boxes, track_ids, classes, confidences
    ):
        if int(class_id) not in person_class_ids:
            continue
        people.append(
            {
                "track_id": int(track_id),
                "box": box.astype(float),
                "confidence": float(confidence),
                "keypoints": None,
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

def infer_people_and_ppe(
    frame: np.ndarray,
    mode: str,
    models: AnalyticsModels,
    ppe_kwargs: dict[str, Any],
    pose_kwargs: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Separa los grafos de inferencia para sostener la garantía de recursos."""
    if mode == "ppe-only":
        # En este modo los IDs nacen del tracker del modelo EPP, no de pose ni de
        # una asociación espacial posterior.
        ppe_result = models.ppe.track(source=frame, **ppe_kwargs)[0]
        cache_ppe_result_names(models, ppe_result)
        assert models.person_class_ids is not None
        people = extract_tracked_ppe_people(ppe_result, models.person_class_ids)
    else:
        if models.pose is None or pose_kwargs is None:
            raise RuntimeError("El modo ppe-fall requiere el modelo de pose.")
        pose_result = models.pose.track(source=frame, **pose_kwargs)[0]
        ppe_result = models.ppe.predict(source=frame, **ppe_kwargs)[0]
        cache_ppe_result_names(models, ppe_result)
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

    if mode == "ppe-only":
        all_ppe_detections = extract_ppe_detections(ppe_result)
    frame_height, frame_width = frame.shape[:2]
    for person in people:
        box_evaluable = box_epp_evaluable_for_person(
            person,
            frame_width=frame_width,
            frame_height=frame_height,
        )
        person["epp_evaluable"] = box_evaluable and (
            mode == "ppe-only" or pose_keypoints_evaluable_for_person(person)
        )

    ppe_detections = [
        item
        for item in all_ppe_detections
        if item["type"] in {"helmet", "vest"}
    ]
    return people, ppe_detections

def process_analytics_frame(
    frame: np.ndarray,
    mode: str,
    models: AnalyticsModels,
    states: dict[int, TrackState],
    ppe_kwargs: dict[str, Any],
    pose_kwargs: dict[str, Any] | None,
    now_monotonic: float,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Anota en sitio el frame del consumidor; la persistencia ocurre después.

    ``LatestFrameCapture`` conserva el ndarray publicado sólo como referencia de
    último frame; su hilo productor nunca vuelve a acceder a su contenido. ``main``
    es el único consumidor: tras completar la inferencia y extraer sus resultados,
    este frame le pertenece para anotarlo sin reservar otro buffer de resolución
    completa.
    """
    people, ppe_detections = infer_people_and_ppe(
        frame,
        mode,
        models,
        ppe_kwargs,
        pose_kwargs,
    )
    associations = associate_ppe_to_people(people, ppe_detections)
    annotated = frame
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
            # El historial suaviza oclusiones breves. El cambio de estado permite
            # alertar pronto y el cooldown limita repeticiones del mismo incumplimiento.
            state.helmet_history.append(current_ppe["helmet"])
            state.vest_history.append(current_ppe["vest"])
            epp_status, helmet_ratio, vest_ratio = stable_epp_status(state)

        if epp_status is not None:
            stable_helmet = helmet_ratio >= EPP_PRESENT_RATIO
            stable_vest = vest_ratio >= EPP_PRESENT_RATIO
            violation = epp_status != "EPP Completo"
            status_changed = epp_status != state.last_epp_status
            cooldown_elapsed = (
                now_monotonic - state.last_epp_alert >= EPP_ALERT_COOLDOWN_S
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

        fall: dict[str, Any] | None = None
        if mode == DEFAULT_ANALYTICS_MODE:
            fall = evaluate_fall(
                person_box=box,
                keypoints=person["keypoints"],
                state=state,
                frame_height=frame.shape[0],
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
        display_fall = " | POSIBLE CAIDA" if fall and fall["active"] else ""
        track_text = f"T{track_id} | " if SHOW_TEMPORARY_TRACK_ID else ""
        label = f"{track_text}{display_epp}{display_fall}"

        if mode == DEFAULT_ANALYTICS_MODE:
            # Dibujar pose es parte del producto ppe-fall y también representa
            # trabajo evitable; PPE-only no entra a esta ruta.
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

    return annotated, pending_events
