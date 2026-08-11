# SPDX-License-Identifier: AGPL-3.0-only
"""Entrena el detector de EPP usando YOLO26 con split 90/10 aleatorio.

Uso rapido:
    python tools\\train_ppe.py

Parametros:
    --dataset     Ruta al dataset (default: ../PPE-DATASET-CUAJONE/final_antes-de-aument2)
    --base-model  Modelo base YOLO26 (default: yolo26n.pt)
    --output      Nombre del modelo de salida (default: best_ppe_final_antes-de-aument2.pt)
    --epochs      Numero de epocas (default: 100)
    --imgsz       Tamano de imagen (default: 640)
    --batch       Batch size (default: 16; subir solo tras comprobar estabilidad)
    --device      Dispositivo de entrenamiento (default: 0 = GPU CUDA)
    --workers     Workers del DataLoader (default: 2 en Windows)
    --cache       Cache de imagenes: ram, disk o none (default: ram)
    --patience    Early stopping patience (default: 15)
    --seed        Semilla para el split aleatorio (default: 42)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import cv2
import numpy as np
import yaml

DISCLAIMER = (
    "Este script realiza un split 90/10 aleatorio del dataset de EPP, "
    "entrena YOLO26 desde yolo26n.pt y guarda el mejor modelo con el nombre solicitado."
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
PPE_CLASS_NAMES = (
    "Gloves",
    "Person",
    "Safety_boots",
    "Vest",
    "respirador",
    "tapaorejas",
    "Hard_hat",
    "lentes_protectores",
)
HARD_HAT_CLASS_ID = 6
AUGMENTATION_PREFIX = "__ppeaug_v1__"
AUGMENTATION_MANIFEST = "labels/train/.ppe_offline_aug_v1.json"
PHOTOMETRIC_TRANSFORMS = (
    "brightness",
    "contrast",
    "shadow",
    "gamma",
    "noise",
    "blur",
    "compression",
    "saturation",
)


@dataclass(frozen=True)
class ImageRecord:
    image_path: Path
    label_path: Path
    label_bytes: bytes
    instance_counts: tuple[int, ...]


@dataclass(frozen=True)
class ClassSupport:
    image_counts: tuple[int, ...]
    instance_counts: tuple[int, ...]


@dataclass(frozen=True)
class AugmentationResult:
    before: ClassSupport
    after: ClassSupport
    targets: tuple[int, ...]
    rare_class_ids: tuple[int, ...]
    zero_support_class_ids: tuple[int, ...]
    generated_count: int
    cap: int


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------

def find_images(images_dir: Path) -> list[Path]:
    """Devuelve todas las imagenes, tanto planas como anidadas."""
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted(
        f for f in images_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in extensions
    )


def _link_or_copy(source: Path, destination: Path) -> None:
    """Crea un hardlink para evitar duplicar el dataset; copia si no es posible."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    try:
        source.hardlink_to(destination)
    except OSError:
        shutil.copy2(source, destination)


def validate_class_names(names: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    """Require the deployed eight-class PPE label contract in exact ID order."""
    normalized = tuple(str(name) for name in names)
    if normalized != PPE_CLASS_NAMES:
        raise ValueError(
            "Dataset classes must exactly match the eight-class PPE contract: "
            f"{list(PPE_CLASS_NAMES)}; received {list(normalized)}"
        )
    return normalized


def parse_yolo_label(label_path: Path) -> tuple[bytes, tuple[int, ...]]:
    """Parse one YOLO detection label and return its original bytes and class counts."""
    try:
        label_bytes = label_path.read_bytes()
        text = label_bytes.decode("utf-8-sig")
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"Cannot read YOLO label {label_path}: {exc}") from exc

    counts = [0] * len(PPE_CLASS_NAMES)
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(
                f"Malformed YOLO label {label_path}:{line_number}: expected 5 fields, got {len(fields)}"
            )
        try:
            class_id = int(fields[0])
            coordinates = tuple(float(value) for value in fields[1:])
        except ValueError as exc:
            raise ValueError(
                f"Malformed YOLO label {label_path}:{line_number}: non-numeric value"
            ) from exc
        if not 0 <= class_id < len(PPE_CLASS_NAMES):
            raise ValueError(
                f"Malformed YOLO label {label_path}:{line_number}: class ID {class_id} is outside 0-7"
            )
        if not all(math.isfinite(value) for value in coordinates):
            raise ValueError(
                f"Malformed YOLO label {label_path}:{line_number}: coordinates must be finite"
            )
        x_center, y_center, width, height = coordinates
        if not (0.0 <= x_center <= 1.0 and 0.0 <= y_center <= 1.0):
            raise ValueError(
                f"Malformed YOLO label {label_path}:{line_number}: center coordinates must be in [0, 1]"
            )
        if not (0.0 < width <= 1.0 and 0.0 < height <= 1.0):
            raise ValueError(
                f"Malformed YOLO label {label_path}:{line_number}: width and height must be in (0, 1]"
            )
        counts[class_id] += 1
    return label_bytes, tuple(counts)


def collect_training_records(split_dir: Path) -> list[ImageRecord]:
    """Load non-generated train image/label pairs without touching validation data."""
    images_dir = split_dir / "images" / "train"
    labels_dir = split_dir / "labels" / "train"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError("The generated split must contain images/train and labels/train")

    image_files = [
        path for path in find_images(images_dir)
        if not path.name.startswith(AUGMENTATION_PREFIX)
    ]
    label_files = [
        path for path in labels_dir.rglob("*.txt")
        if not path.name.startswith(AUGMENTATION_PREFIX)
    ]
    images_by_stem: dict[str, Path] = {}
    labels_by_stem: dict[str, Path] = {}
    for image_path in image_files:
        if image_path.stem in images_by_stem:
            raise ValueError(f"Duplicate train image stem: {image_path.stem}")
        images_by_stem[image_path.stem] = image_path
    for label_path in label_files:
        if label_path.stem in labels_by_stem:
            raise ValueError(f"Duplicate train label stem: {label_path.stem}")
        labels_by_stem[label_path.stem] = label_path

    missing_labels = sorted(set(images_by_stem) - set(labels_by_stem))
    orphan_labels = sorted(set(labels_by_stem) - set(images_by_stem))
    if missing_labels or orphan_labels:
        raise ValueError(
            "Inconsistent train split: "
            f"{len(missing_labels)} images without labels, {len(orphan_labels)} labels without images"
        )

    records = []
    for stem, image_path in sorted(images_by_stem.items()):
        label_path = labels_by_stem[stem]
        label_bytes, counts = parse_yolo_label(label_path)
        records.append(ImageRecord(image_path, label_path, label_bytes, counts))
    return records


def compute_class_support(records: list[ImageRecord]) -> ClassSupport:
    image_counts = [0] * len(PPE_CLASS_NAMES)
    instance_counts = [0] * len(PPE_CLASS_NAMES)
    for record in records:
        for class_id, count in enumerate(record.instance_counts):
            instance_counts[class_id] += count
            if count:
                image_counts[class_id] += 1
    return ClassSupport(tuple(image_counts), tuple(instance_counts))


def plan_class_aware_augmentations(
    records: list[ImageRecord],
    *,
    rare_target_ratio: float,
    hard_hat_boost: float,
    max_augmented_ratio: float,
    max_copies_per_source: int,
    seed: int,
) -> tuple[list[ImageRecord], ClassSupport, tuple[int, ...], tuple[int, ...], int]:
    """Greedily cover image-support deficits within global and per-source limits."""
    if not 0.0 < rare_target_ratio <= 5.0:
        raise ValueError("rare_target_ratio must be in (0, 5]")
    if not 0.0 <= hard_hat_boost <= 5.0:
        raise ValueError("hard_hat_boost must be in [0, 5]")
    if not 0.0 <= max_augmented_ratio <= 5.0:
        raise ValueError("max_augmented_ratio must be in [0, 5]")
    if max_copies_per_source < 0:
        raise ValueError("max_copies_per_source must be non-negative")
    if seed < 0:
        raise ValueError("seed must be non-negative")

    support = compute_class_support(records)
    nonzero = [count for count in support.image_counts if count > 0]
    support_median = float(median(nonzero)) if nonzero else 0.0
    rare_ids = tuple(
        class_id
        for class_id, count in enumerate(support.image_counts)
        if 0 < count < support_median
    )
    targets = list(support.image_counts)
    rare_target = math.ceil(support_median * rare_target_ratio)
    for class_id in rare_ids:
        targets[class_id] = max(targets[class_id], rare_target)
    if support.image_counts[HARD_HAT_CLASS_ID] > 0 and hard_hat_boost > 0.0:
        targets[HARD_HAT_CLASS_ID] += math.ceil(support_median * hard_hat_boost)

    cap = min(
        math.floor(len(records) * max_augmented_ratio),
        len(records) * max_copies_per_source,
    )
    deficits = [max(0, target - current) for target, current in zip(targets, support.image_counts)]
    copies = [0] * len(records)
    plan: list[ImageRecord] = []
    rng = random.Random(seed)

    while len(plan) < cap and any(deficits):
        candidates: list[tuple[float, float, int]] = []
        for index, record in enumerate(records):
            if copies[index] >= max_copies_per_source:
                continue
            score = 0.0
            for class_id, count in enumerate(record.instance_counts):
                if count and deficits[class_id] > 0:
                    weight = 2.0 + hard_hat_boost if class_id == HARD_HAT_CLASS_ID else 1.0
                    score += weight * deficits[class_id] / max(targets[class_id], 1)
            if score > 0.0:
                candidates.append((score, rng.random(), index))
        if not candidates:
            break
        _, _, selected = max(candidates)
        record = records[selected]
        plan.append(record)
        copies[selected] += 1
        for class_id, count in enumerate(record.instance_counts):
            if count and deficits[class_id] > 0:
                deficits[class_id] -= 1

    return plan, support, tuple(targets), rare_ids, cap


def apply_photometric_transform(
    image: np.ndarray,
    transform_name: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply one bounded photometric transform without changing image geometry."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Photometric augmentation requires a three-channel BGR image")

    height, width = image.shape[:2]
    if transform_name == "brightness":
        factor = rng.uniform(0.65, 0.90)
        output = image.astype(np.float32) * factor
    elif transform_name == "contrast":
        factor = rng.uniform(0.75, 0.95) if rng.random() < 0.5 else rng.uniform(1.05, 1.25)
        output = (image.astype(np.float32) - 127.5) * factor + 127.5
    elif transform_name == "shadow":
        x0 = int(rng.uniform(0.0, 0.55) * width)
        y0 = int(rng.uniform(0.0, 0.55) * height)
        x1 = min(width - 1, x0 + max(2, int(rng.uniform(0.20, 0.55) * width)))
        y1 = min(height - 1, y0 + max(2, int(rng.uniform(0.20, 0.55) * height)))
        polygon = np.array(
            [[x0, y0], [x1, y0], [max(x0, x1 - int(0.15 * width)), y1], [x0, y1]],
            dtype=np.int32,
        )
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], 255)
        output = image.astype(np.float32)
        output[mask > 0] *= rng.uniform(0.55, 0.82)
    elif transform_name == "gamma":
        gamma = rng.uniform(0.80, 0.95) if rng.random() < 0.5 else rng.uniform(1.05, 1.25)
        lookup = np.clip(((np.arange(256) / 255.0) ** gamma) * 255.0, 0, 255).astype(np.uint8)
        output = cv2.LUT(image, lookup)
    elif transform_name == "noise":
        sigma = rng.uniform(2.0, 10.0)
        output = image.astype(np.float32) + rng.normal(0.0, sigma, image.shape)
    elif transform_name == "blur":
        kernel = 3 if rng.random() < 0.8 else 5
        output = cv2.GaussianBlur(image, (kernel, kernel), rng.uniform(0.3, 1.0))
    elif transform_name == "compression":
        scale = rng.uniform(0.55, 0.85)
        down_width = max(1, int(round(width * scale)))
        down_height = max(1, int(round(height * scale)))
        reduced = cv2.resize(image, (down_width, down_height), interpolation=cv2.INTER_AREA)
        restored = cv2.resize(reduced, (width, height), interpolation=cv2.INTER_LINEAR)
        quality = int(rng.integers(55, 86))
        encoded, buffer = cv2.imencode(".jpg", restored, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not encoded:
            raise RuntimeError("OpenCV failed to encode JPEG compression simulation")
        output = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    elif transform_name == "saturation":
        factor = rng.uniform(0.70, 0.92) if rng.random() < 0.5 else rng.uniform(1.08, 1.30)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= factor
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        output = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    else:
        raise ValueError(f"Unknown photometric transform: {transform_name}")

    if output.shape != image.shape:
        raise RuntimeError(f"Transform {transform_name} changed image dimensions")
    return np.clip(output, 0, 255).astype(np.uint8)


def select_photometric_transforms(rng: np.random.Generator) -> tuple[str, ...]:
    """Select a non-empty deterministic random subset of available transforms."""
    selected = tuple(name for name in PHOTOMETRIC_TRANSFORMS if rng.random() < 0.5)
    if selected:
        return selected
    return (PHOTOMETRIC_TRANSFORMS[int(rng.integers(0, len(PHOTOMETRIC_TRANSFORMS)))],)


def create_photometric_variant(
    image: np.ndarray,
    rng: np.random.Generator,
    transform_names: tuple[str, ...] | None = None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    selected = transform_names or select_photometric_transforms(rng)
    output = image.copy()
    for transform_name in selected:
        output = apply_photometric_transform(output, transform_name, rng)
    return output, selected


def cleanup_generated_augmentations(split_dir: Path) -> int:
    """Delete only files recorded by this augmenter in its private manifest."""
    manifest_path = split_dir / AUGMENTATION_MANIFEST
    if not manifest_path.is_file():
        return 0
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        relative_files = manifest["files"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"Invalid augmentation manifest {manifest_path}: {exc}") from exc
    if not isinstance(relative_files, list):
        raise ValueError(f"Invalid augmentation manifest {manifest_path}: files must be a list")
    if manifest.get("version") != 1 or manifest.get("prefix") != AUGMENTATION_PREFIX:
        raise ValueError(f"Invalid augmentation manifest identity: {manifest_path}")

    validated: list[Path] = []
    allowed_parents = {Path("images/train"), Path("labels/train")}
    for value in relative_files:
        relative = Path(value)
        if (
            relative.is_absolute()
            or relative.parent not in allowed_parents
            or not relative.name.startswith(AUGMENTATION_PREFIX)
        ):
            raise ValueError(f"Unsafe augmentation manifest entry: {value!r}")
        validated.append(split_dir / relative)
    for generated_path in validated:
        generated_path.unlink(missing_ok=True)
    manifest_path.unlink()
    return len(validated) // 2


def _write_variant_pair(
    image_path: Path,
    label_path: Path,
    image: np.ndarray,
    label_bytes: bytes,
) -> None:
    encoded, buffer = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not encoded:
        raise RuntimeError(f"OpenCV failed to encode augmented image {image_path}")
    image_temp = image_path.with_name(f".{image_path.name}.tmp")
    label_temp = label_path.with_name(f".{label_path.name}.tmp")
    try:
        image_temp.write_bytes(buffer.tobytes())
        label_temp.write_bytes(label_bytes)
        label_temp.replace(label_path)
        image_temp.replace(image_path)
    except Exception:
        for path in (image_temp, label_temp, image_path, label_path):
            path.unlink(missing_ok=True)
        raise


def _write_augmentation_manifest(split_dir: Path, files: list[str], generated: list[dict[str, object]]) -> None:
    manifest_path = split_dir / AUGMENTATION_MANIFEST
    temporary = manifest_path.with_suffix(".tmp")
    payload = {"version": 1, "prefix": AUGMENTATION_PREFIX, "files": files, "generated": generated}
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(manifest_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def print_class_support_report(result: AugmentationResult) -> None:
    print("\n  Offline augmentation class support (train only):")
    print("  ID  Class                 Before images/instances  Target  After images/instances  Status")
    for class_id, name in enumerate(PPE_CLASS_NAMES):
        status = []
        if class_id in result.rare_class_ids:
            status.append("rare")
        if class_id == HARD_HAT_CLASS_ID:
            status.append("hard-hat boost")
        if class_id in result.zero_support_class_ids:
            status.append("zero source; not generated")
        print(
            f"  {class_id:<3} {name:<21} "
            f"{result.before.image_counts[class_id]:>5}/{result.before.instance_counts[class_id]:<9} "
            f"{result.targets[class_id]:>6}  "
            f"{result.after.image_counts[class_id]:>5}/{result.after.instance_counts[class_id]:<9} "
            f"{', '.join(status) or '-'}"
        )
    print(f"  Generated variants: {result.generated_count} (cap: {result.cap})")


def augment_training_split(
    split_dir: Path,
    *,
    rare_target_ratio: float = 1.0,
    hard_hat_boost: float = 0.25,
    max_augmented_ratio: float = 0.5,
    max_copies_per_source: int = 2,
    seed: int = 42,
) -> AugmentationResult:
    """Generate deterministic class-aware photometric variants in train only."""
    records = collect_training_records(split_dir)
    plan, before, targets, rare_ids, cap = plan_class_aware_augmentations(
        records,
        rare_target_ratio=rare_target_ratio,
        hard_hat_boost=hard_hat_boost,
        max_augmented_ratio=max_augmented_ratio,
        max_copies_per_source=max_copies_per_source,
        seed=seed,
    )
    cleanup_generated_augmentations(split_dir)

    images_dir = split_dir / "images" / "train"
    labels_dir = split_dir / "labels" / "train"
    rng = np.random.default_rng(seed)
    source_copies: dict[Path, int] = {}
    generated_files: list[str] = []
    generated_metadata: list[dict[str, object]] = []
    written_pairs: list[tuple[Path, Path]] = []
    after_images = list(before.image_counts)
    after_instances = list(before.instance_counts)
    try:
        for record in plan:
            image = cv2.imread(str(record.image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"OpenCV cannot decode train image {record.image_path}")
            source_copies[record.image_path] = source_copies.get(record.image_path, 0) + 1
            copy_number = source_copies[record.image_path]
            source_key = hashlib.sha256(record.image_path.name.encode("utf-8")).hexdigest()[:12]
            generated_stem = f"{AUGMENTATION_PREFIX}{source_key}_{copy_number:02d}"
            generated_image = images_dir / f"{generated_stem}.png"
            generated_label = labels_dir / f"{generated_stem}.txt"
            variant, selected = create_photometric_variant(image, rng)
            written_pairs.append((generated_image, generated_label))
            _write_variant_pair(generated_image, generated_label, variant, record.label_bytes)
            image_relative = generated_image.relative_to(split_dir).as_posix()
            label_relative = generated_label.relative_to(split_dir).as_posix()
            generated_files.extend((image_relative, label_relative))
            generated_metadata.append(
                {
                    "image": image_relative,
                    "label": label_relative,
                    "source": record.image_path.name,
                    "transforms": list(selected),
                }
            )
            for class_id, count in enumerate(record.instance_counts):
                after_instances[class_id] += count
                if count:
                    after_images[class_id] += 1
        _write_augmentation_manifest(split_dir, generated_files, generated_metadata)
    except Exception:
        for generated_image, generated_label in written_pairs:
            generated_image.unlink(missing_ok=True)
            generated_label.unlink(missing_ok=True)
        raise

    result = AugmentationResult(
        before=before,
        after=ClassSupport(tuple(after_images), tuple(after_instances)),
        targets=targets,
        rare_class_ids=rare_ids,
        zero_support_class_ids=tuple(
            class_id for class_id, count in enumerate(before.image_counts) if count == 0
        ),
        generated_count=len(plan),
        cap=cap,
    )
    print_class_support_report(result)
    return result


def create_split_yaml(
    dataset_dir: Path,
    train_ratio: float,
    seed: int,
    split_dir: Path,
    epochs: int,
    imgsz: int,
    batch: int,
    patience: int,
) -> Path:
    """Crea un split YOLO estandar y normaliza imagenes anidadas sin tocar el origen."""
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"

    if not images_dir.is_dir():
        raise FileNotFoundError(f"Directorio de imagenes no encontrado: {images_dir}")

    image_files = find_images(images_dir)
    if not image_files:
        raise ValueError(f"No se encontraron imagenes en {images_dir}")

    # Leer classes.txt o data.yaml original para los nombres de clases
    classes_file = dataset_dir / "classes.txt"
    if classes_file.is_file():
        names = [line.strip() for line in classes_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        # Fallback: leer data.yaml original
        orig_yaml = dataset_dir / "data.yaml"
        if orig_yaml.is_file():
            with open(orig_yaml, encoding="utf-8") as f:
                orig = yaml.safe_load(f)
            names = orig.get("names", [])
            if isinstance(names, dict):
                try:
                    names = [names[index] if index in names else names[str(index)] for index in range(len(names))]
                except (KeyError, TypeError) as exc:
                    raise ValueError("data.yaml names must use consecutive class IDs starting at 0") from exc
        else:
            raise FileNotFoundError("No se encontro classes.txt ni data.yaml en el dataset")
    names = list(validate_class_names(names))

    label_files = sorted(labels_dir.rglob("*.txt")) if labels_dir.is_dir() else []
    images_by_stem: dict[str, Path] = {}
    labels_by_stem: dict[str, Path] = {}
    duplicate_images: set[str] = set()
    duplicate_labels: set[str] = set()

    for image in image_files:
        stem = image.stem
        if stem in images_by_stem:
            duplicate_images.add(stem)
        images_by_stem[stem] = image
    for label in label_files:
        stem = label.stem
        if stem in labels_by_stem:
            duplicate_labels.add(stem)
        labels_by_stem[stem] = label

    if duplicate_images:
        raise ValueError(f"Stems de imagen duplicados: {sorted(duplicate_images)[:5]}")
    if duplicate_labels:
        raise ValueError(f"Stems de label duplicados: {sorted(duplicate_labels)[:5]}")

    missing_labels = sorted(set(images_by_stem) - set(labels_by_stem))
    orphan_labels = sorted(set(labels_by_stem) - set(images_by_stem))
    if missing_labels or orphan_labels:
        raise ValueError(
            "Dataset inconsistente: "
            f"{len(missing_labels)} imagenes sin label, {len(orphan_labels)} labels sin imagen"
        )

    # Shuffle determinista, manteniendo cada imagen junto a su label.
    rng = random.Random(seed)
    ordered = sorted(images_by_stem.values(), key=lambda p: p.stem)
    rng.shuffle(ordered)

    n_train = int(len(ordered) * train_ratio)
    train_files = ordered[:n_train]
    val_files = ordered[n_train:]

    # Limpiar solamente el area generada del split anterior.
    for generated_dir in (split_dir / "images", split_dir / "labels"):
        if generated_dir.exists():
            shutil.rmtree(generated_dir)

    # Crear layout YOLO estandar. Los hardlinks no duplican los bytes del dataset.
    for split_name, split_files in (("train", train_files), ("val", val_files)):
        for image in split_files:
            _link_or_copy(image, split_dir / "images" / split_name / image.name)
            _link_or_copy(
                labels_by_stem[image.stem],
                split_dir / "labels" / split_name / f"{image.stem}.txt",
            )

    # data_split.yaml apunta al layout normalizado y no depende de rutas anidadas.
    yaml_path = split_dir / "data_split.yaml"
    data = {
        "path": str(split_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(names),
        "names": names,
    }
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    # Tambien escribir un config de entrenamiento
    hyp = {
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "patience": patience,
        "seed": seed,
    }

    print(f"  Split completado: {len(train_files)} train, {len(val_files)} val")
    print(f"  Imagenes normalizadas: {len(ordered)} (origen intacto)")
    print(f"  data_split.yaml: {yaml_path}")
    print(f"  Clases: {names}")

    return yaml_path


# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------

def train(
    data_yaml: Path,
    base_model: Path,
    output_name: str,
    epochs: int,
    imgsz: int,
    batch: int,
    patience: int,
    seed: int,
    device: str = "0",
    workers: int = 2,
    cache: str = "ram",
    run_name: str = "ppe_train",
) -> Path:
    """Ejecuta el entrenamiento con Ultralytics YOLO y devuelve el path del mejor modelo."""
    from ultralytics import YOLO

    if not base_model.is_file():
        raise FileNotFoundError(f"Modelo base no encontrado: {base_model}")

    print(f"\n{'='*60}")
    print(f"  ENTRENANDO EPP DETECTOR")
    print(f"  Modelo base: {base_model.name}")
    print(f"  Epochs: {epochs} | imgsz: {imgsz} | batch: {batch} | patience: {patience}")
    print(f"  Device: {device} | workers: {workers} | cache: {cache}")
    print(f"{'='*60}\n")

    model = YOLO(str(base_model))

    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=patience,
        seed=seed,
        device=device,
        workers=workers,
        cache=None if cache == "none" else cache,
        amp=True,
        deterministic=False,
        hsv_h=0.01,
        hsv_s=0.30,
        hsv_v=0.18,
        mosaic=0.25,
        close_mosaic=10,
        mixup=0.0,
        cutmix=0.0,
        copy_paste=0.0,
        name=run_name,
        project=str(PROJECT_ROOT / "runs" / "ppe"),
        exist_ok=True,
    )

    # El mejor modelo se guarda dentro del run especifico.
    best_pt = PROJECT_ROOT / "runs" / "ppe" / run_name / "weights" / "best.pt"
    if not best_pt.is_file():
        # Fallback: buscar best.pt en cualquier subdirectorio
        candidates = list((PROJECT_ROOT / "runs" / "ppe" / run_name).rglob("weights/best.pt"))
        if candidates:
            best_pt = candidates[0]
        else:
            raise FileNotFoundError("No se encontro best.pt despues del entrenamiento")

    # Copiar a la ubicacion solicitada
    output_path = PROJECT_ROOT / output_name
    shutil.copy2(best_pt, output_path)
    print(f"\n  Modelo guardado en: {output_path}")

    # Metrics resumidas
    try:
        metrics = results.val
        if metrics is not None:
            print(f"\n  Metricas de validacion:")
            print(f"    mAP50:      {metrics.get('metrics/mAP50(B)', 'N/A')}")
            print(f"    mAP50-95:   {metrics.get('metrics/mAP50-95(B)', 'N/A')}")
    except Exception:
        pass

    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Entrenamiento de detector de EPP con YOLO26. {DISCLAIMER}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(PROJECT_ROOT.parent / "PPE-DATASET-CUAJONE" / "final_antes-de-aument2"),
        help="Ruta al dataset con images/ y labels/ (default: %(default)s)",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=str(PROJECT_ROOT / "yolo26n.pt"),
        help="Modelo base YOLO26 (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="best_ppe_final_antes-de-aument2.pt",
        help="Nombre del modelo de salida (default: %(default)s)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Numero de epocas (default: %(default)s)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Resolucion de entrenamiento (default: %(default)s)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Dispositivo de entrenamiento, ej: 0, cpu, 0,1 (default: %(default)s)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Workers del DataLoader (default: %(default)s)",
    )
    parser.add_argument(
        "--cache",
        choices=("ram", "disk", "none"),
        default="ram",
        help="Cache de imagenes (default: %(default)s)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Early stopping patience (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed reused for split, planner, and photometric transforms (default: %(default)s)",
    )
    parser.add_argument(
        "--offline-augment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable class-aware offline train-only photometric augmentation (default: enabled)",
    )
    parser.add_argument(
        "--rare-target-ratio",
        type=float,
        default=1.0,
        help="Rare-class target as a ratio of median nonzero image support (default: %(default)s)",
    )
    parser.add_argument(
        "--hard-hat-boost",
        type=float,
        default=0.25,
        help="Additional Hard_hat image support as a ratio of the median (default: %(default)s)",
    )
    parser.add_argument(
        "--max-augmented-ratio",
        type=float,
        default=0.5,
        help="Maximum generated variants relative to original train images (default: %(default)s)",
    )
    parser.add_argument(
        "--max-copies-per-source",
        type=int,
        default=2,
        help="Maximum generated variants from one source image (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_dir = Path(args.dataset).resolve()
    base_model = Path(args.base_model).resolve()

    if not dataset_dir.is_dir():
        print(f"ERROR: Dataset no encontrado en {dataset_dir}", file=sys.stderr)
        sys.exit(1)

    # Directorio temporal para el split
    split_dir = dataset_dir / "split_90_10_gpu"
    split_dir.mkdir(parents=True, exist_ok=True)

    # 1. Crear split 90/10
    print("\n--- Paso 1: Creando split 90/10 aleatorio ---")
    data_yaml = create_split_yaml(
        dataset_dir=dataset_dir,
        train_ratio=0.9,
        seed=args.seed,
        split_dir=split_dir,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
    )

    # 2. Generate offline variants only after the split is fixed.
    print("\n--- Step 2: Class-aware offline photometric augmentation ---")
    if args.offline_augment:
        augment_training_split(
            split_dir,
            rare_target_ratio=args.rare_target_ratio,
            hard_hat_boost=args.hard_hat_boost,
            max_augmented_ratio=args.max_augmented_ratio,
            max_copies_per_source=args.max_copies_per_source,
            seed=args.seed,
        )
    else:
        removed = cleanup_generated_augmentations(split_dir)
        print(f"  Offline augmentation disabled; removed {removed} prior generated variants")

    # 3. Entrenar
    print("\n--- Paso 3: Entrenando modelo ---")
    run_name = f"ppe_train_{dataset_dir.name}_b{args.batch}_w{args.workers}"
    output_path = train(
        data_yaml=data_yaml,
        base_model=base_model,
        output_name=args.output,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        seed=args.seed,
        device=args.device,
        workers=args.workers,
        cache=args.cache,
        run_name=run_name,
    )

    print(f"\n{'='*60}")
    print(f"  ENTRENAMIENTO COMPLETADO")
    print(f"  Modelo: {output_path}")
    print(f"  Dataset: {dataset_dir}")
    print(f"  Split: {split_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
