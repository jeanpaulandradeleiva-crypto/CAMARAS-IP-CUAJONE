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
import random
import shutil
import sys
from pathlib import Path

import yaml

DISCLAIMER = (
    "Este script realiza un split 90/10 aleatorio del dataset de EPP, "
    "entrena YOLO26 desde yolo26n.pt y guarda el mejor modelo con el nombre solicitado."
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


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
        else:
            raise FileNotFoundError("No se encontro classes.txt ni data.yaml en el dataset")

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
        help="Semilla para el split y shuffle (default: %(default)s)",
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

    # 2. Entrenar
    print("\n--- Paso 2: Entrenando modelo ---")
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
