# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from tools import train_ppe


def _image(index: int, height: int = 48, width: int = 72) -> np.ndarray:
    y, x = np.indices((height, width))
    return np.stack(
        (
            (x * 5 + index * 17) % 256,
            (y * 7 + index * 29) % 256,
            ((x + y) * 3 + index * 11) % 256,
        ),
        axis=2,
    ).astype(np.uint8)


def _write_image(path: Path, index: int, height: int = 48, width: int = 72) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), _image(index, height, width))


def _label_bytes(*class_ids: int, suffix: bytes = b"\n") -> bytes:
    lines = [f"{class_id} 0.5 0.5 0.25 0.25".encode() for class_id in class_ids]
    return b"\r\n".join(lines) + suffix if lines else b""


def _make_split(tmp_path: Path, labels: list[bytes], *, with_validation: bool = True) -> Path:
    split_dir = tmp_path / "split_90_10_gpu"
    for index, label_bytes in enumerate(labels):
        _write_image(split_dir / "images" / "train" / f"train_{index}.png", index)
        label_path = split_dir / "labels" / "train" / f"train_{index}.txt"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_bytes(label_bytes)
    if with_validation:
        _write_image(split_dir / "images" / "val" / "validation.png", 99)
        val_label = split_dir / "labels" / "val" / "validation.txt"
        val_label.parent.mkdir(parents=True, exist_ok=True)
        val_label.write_bytes(_label_bytes(1))
    return split_dir


def _support_split(tmp_path: Path) -> Path:
    return _make_split(
        tmp_path,
        [
            _label_bytes(0, 1, 2, 6, suffix=b"\r\n"),
            _label_bytes(0, 1, 6),
            _label_bytes(0, 1, 6),
            _label_bytes(0, 1, 6),
            _label_bytes(1, 3),
            _label_bytes(3, 7),
            _label_bytes(3, 7),
            _label_bytes(3, 7),
            b"",
        ],
    )


def _generated_paths(split_dir: Path) -> tuple[list[Path], list[Path]]:
    images = sorted((split_dir / "images" / "train").glob(f"{train_ppe.AUGMENTATION_PREFIX}*.png"))
    labels = sorted((split_dir / "labels" / "train").glob(f"{train_ppe.AUGMENTATION_PREFIX}*.txt"))
    return images, labels


def _hashes(paths: list[Path]) -> dict[str, str]:
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def _manifest(split_dir: Path) -> dict[str, object]:
    return json.loads((split_dir / train_ppe.AUGMENTATION_MANIFEST).read_text(encoding="utf-8"))


def test_split_then_augmentation_is_train_only_and_preserves_source_and_labels(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    (dataset / "images").mkdir(parents=True)
    (dataset / "labels").mkdir()
    (dataset / "classes.txt").write_text("\n".join(train_ppe.PPE_CLASS_NAMES), encoding="utf-8")
    for index in range(10):
        _write_image(dataset / "images" / f"source_{index}.png", index, 40 + index, 60 + index)
        (dataset / "labels" / f"source_{index}.txt").write_bytes(
            _label_bytes(6, suffix=b"\r\n" if index % 2 else b"\n")
        )
    source_hashes = _hashes(sorted((dataset / "images").glob("*.png"))) | _hashes(
        sorted((dataset / "labels").glob("*.txt"))
    )

    split_dir = dataset / "split_90_10_gpu"
    train_ppe.create_split_yaml(dataset, 0.9, 42, split_dir, 1, 64, 2, 1)
    val_paths = sorted((split_dir / "images" / "val").iterdir()) + sorted(
        (split_dir / "labels" / "val").iterdir()
    )
    val_hashes = _hashes(val_paths)
    result = train_ppe.augment_training_split(split_dir, seed=42)

    generated_images, generated_labels = _generated_paths(split_dir)
    assert result.generated_count > 0
    assert len(generated_images) == len(generated_labels) == result.generated_count
    assert _hashes(val_paths) == val_hashes
    assert not list((split_dir / "images" / "val").glob(f"{train_ppe.AUGMENTATION_PREFIX}*"))
    assert source_hashes == (
        _hashes(sorted((dataset / "images").glob("*.png")))
        | _hashes(sorted((dataset / "labels").glob("*.txt")))
    )

    metadata = _manifest(split_dir)["generated"]
    for item in metadata:
        generated_image = split_dir / item["image"]
        generated_label = split_dir / item["label"]
        source_image = split_dir / "images" / "train" / item["source"]
        source_label = split_dir / "labels" / "train" / f"{Path(item['source']).stem}.txt"
        assert generated_label.read_bytes() == source_label.read_bytes()
        assert cv2.imread(str(generated_image)).shape == cv2.imread(str(source_image)).shape


def test_same_seed_reproduces_artifact_hashes_and_different_seed_changes_images(tmp_path: Path) -> None:
    split_dir = _support_split(tmp_path)
    train_ppe.augment_training_split(split_dir, hard_hat_boost=0.5, max_augmented_ratio=1.0, seed=17)
    first_images, first_labels = _generated_paths(split_dir)
    first_hashes = _hashes(first_images + first_labels)
    first_image_hashes = _hashes(first_images)

    train_ppe.augment_training_split(split_dir, hard_hat_boost=0.5, max_augmented_ratio=1.0, seed=17)
    second_images, second_labels = _generated_paths(split_dir)
    assert _hashes(second_images + second_labels) == first_hashes

    train_ppe.augment_training_split(split_dir, hard_hat_boost=0.5, max_augmented_ratio=1.0, seed=18)
    third_images, _ = _generated_paths(split_dir)
    assert _hashes(third_images) != first_image_hashes


def test_rare_classes_and_nonrare_hard_hat_gain_support(tmp_path: Path) -> None:
    result = train_ppe.augment_training_split(
        _support_split(tmp_path),
        rare_target_ratio=1.0,
        hard_hat_boost=0.5,
        max_augmented_ratio=1.0,
        max_copies_per_source=2,
        seed=5,
    )

    assert set(result.rare_class_ids) == {2, 7}
    assert result.before.image_counts[6] == 4
    assert 6 not in result.rare_class_ids
    assert result.after.image_counts[2] > result.before.image_counts[2]
    assert result.after.image_counts[7] > result.before.image_counts[7]
    assert result.after.image_counts[6] > result.before.image_counts[6]
    assert result.after.instance_counts[2] > result.before.instance_counts[2]


def test_global_cap_and_per_source_copy_limit_are_enforced(tmp_path: Path) -> None:
    split_dir = _support_split(tmp_path)
    result = train_ppe.augment_training_split(
        split_dir,
        hard_hat_boost=2.0,
        max_augmented_ratio=0.34,
        max_copies_per_source=1,
        seed=3,
    )
    generated = _manifest(split_dir)["generated"]
    sources = [item["source"] for item in generated]

    assert result.cap == 3
    assert result.generated_count <= 3
    assert len(sources) == len(set(sources))


def test_malformed_label_fails_before_cleanup_or_generation(tmp_path: Path) -> None:
    split_dir = _make_split(tmp_path, [b"6 0.5 broken 0.2 0.2\n"])
    sentinel = split_dir / "images" / "train" / f"{train_ppe.AUGMENTATION_PREFIX}sentinel.png"
    sentinel.write_bytes(b"user-owned")

    with pytest.raises(ValueError, match="Malformed YOLO label"):
        train_ppe.augment_training_split(split_dir)

    assert sentinel.read_bytes() == b"user-owned"
    assert not (split_dir / train_ppe.AUGMENTATION_MANIFEST).exists()


def test_zero_source_classes_are_reported_and_never_fabricated(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    split_dir = _support_split(tmp_path)
    result = train_ppe.augment_training_split(
        split_dir,
        hard_hat_boost=0.5,
        max_augmented_ratio=1.0,
        seed=9,
    )
    output = capsys.readouterr().out

    assert set(result.zero_support_class_ids) == {4, 5}
    assert result.after.image_counts[4:6] == (0, 0)
    assert result.after.instance_counts[4:6] == (0, 0)
    assert "zero source; not generated" in output
    for label_path in _generated_paths(split_dir)[1]:
        _, counts = train_ppe.parse_yolo_label(label_path)
        assert counts[4:6] == (0, 0)


@pytest.mark.parametrize("transform_name", train_ppe.PHOTOMETRIC_TRANSFORMS)
def test_each_photometric_transform_is_reachable_and_preserves_dimensions(transform_name: str) -> None:
    image = _image(4, 64, 80)
    output = train_ppe.apply_photometric_transform(image, transform_name, np.random.default_rng(1234))

    assert output.shape == image.shape
    assert output.dtype == np.uint8
    assert not np.array_equal(output, image)


def test_random_transform_selection_is_nonempty_subset() -> None:
    selected = train_ppe.select_photometric_transforms(np.random.default_rng(8))

    assert selected
    assert set(selected) <= set(train_ppe.PHOTOMETRIC_TRANSFORMS)
    assert len(selected) < len(train_ppe.PHOTOMETRIC_TRANSFORMS)


def test_cleanup_removes_only_manifested_generated_files(tmp_path: Path) -> None:
    split_dir = _support_split(tmp_path)
    result = train_ppe.augment_training_split(
        split_dir,
        hard_hat_boost=0.5,
        max_augmented_ratio=1.0,
        seed=11,
    )
    generated_images, generated_labels = _generated_paths(split_dir)
    source_image = split_dir / "images" / "train" / "train_0.png"
    source_label = split_dir / "labels" / "train" / "train_0.txt"
    untracked_prefixed_image = split_dir / "images" / "train" / f"{train_ppe.AUGMENTATION_PREFIX}untracked.png"
    untracked_prefixed_label = split_dir / "labels" / "train" / f"{train_ppe.AUGMENTATION_PREFIX}untracked.txt"
    untracked_prefixed_image.write_bytes(b"do-not-delete")
    untracked_prefixed_label.write_bytes(b"do-not-delete")

    removed = train_ppe.cleanup_generated_augmentations(split_dir)

    assert removed == result.generated_count
    assert all(not path.exists() for path in generated_images + generated_labels)
    assert source_image.exists() and source_label.exists()
    assert untracked_prefixed_image.read_bytes() == b"do-not-delete"
    assert untracked_prefixed_label.read_bytes() == b"do-not-delete"


def test_generation_failure_leaves_no_generated_image_without_label(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    split_dir = _support_split(tmp_path)

    def fail_manifest(*args: object, **kwargs: object) -> None:
        raise OSError("synthetic manifest failure")

    monkeypatch.setattr(train_ppe, "_write_augmentation_manifest", fail_manifest)
    with pytest.raises(OSError, match="synthetic manifest failure"):
        train_ppe.augment_training_split(
            split_dir,
            hard_hat_boost=0.5,
            max_augmented_ratio=1.0,
            seed=13,
        )

    generated_images, generated_labels = _generated_paths(split_dir)
    assert generated_images == []
    assert generated_labels == []


def test_train_uses_conservative_ultralytics_augmentation_without_running_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    project_root = tmp_path / "project"
    base_model = project_root / "base.pt"
    base_model.parent.mkdir(parents=True)
    base_model.write_bytes(b"model")

    class FakeYOLO:
        def __init__(self, model: str) -> None:
            captured["model"] = model

        def train(self, **kwargs: object) -> SimpleNamespace:
            captured.update(kwargs)
            best = project_root / "runs" / "ppe" / "unit" / "weights" / "best.pt"
            best.parent.mkdir(parents=True)
            best.write_bytes(b"best")
            return SimpleNamespace(val=None)

    monkeypatch.setattr(train_ppe, "PROJECT_ROOT", project_root)
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeYOLO))
    output = train_ppe.train(Path("data.yaml"), base_model, "output.pt", 2, 64, 2, 1, 42, run_name="unit")

    assert output.read_bytes() == b"best"
    assert captured["hsv_h"] == 0.01
    assert captured["hsv_s"] == 0.30
    assert captured["hsv_v"] == 0.18
    assert captured["mosaic"] == 0.25
    assert captured["close_mosaic"] == 10
    assert captured["mixup"] == captured["cutmix"] == captured["copy_paste"] == 0.0


def test_cli_defaults_enable_bounded_augmentation_and_support_disable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["train_ppe.py"])
    defaults = train_ppe.parse_args()
    monkeypatch.setattr(sys, "argv", ["train_ppe.py", "--no-offline-augment"])
    disabled = train_ppe.parse_args()

    assert defaults.offline_augment is True
    assert defaults.rare_target_ratio == 1.0
    assert defaults.hard_hat_boost == 0.25
    assert defaults.max_augmented_ratio == 0.5
    assert defaults.max_copies_per_source == 2
    assert disabled.offline_augment is False
