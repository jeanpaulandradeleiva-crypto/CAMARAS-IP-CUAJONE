# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

if __package__:
    from tools import evaluate_detection
else:
    import evaluate_detection


METRICS = ("precision", "recall", "map50", "map50_95")
DEFAULT_OUTPUT = Path("ppe-comparison-report.json")
Evaluator = Callable[..., dict[str, Any]]


def validate_inputs(baseline: Path, candidate: Path, data: Path) -> None:
    allowed_suffixes = {".pt", ".engine", ".onnx"}
    for label, path in (("baseline checkpoint", baseline), ("candidate checkpoint", candidate)):
        if not path.is_file():
            raise ValueError(f"The {label} does not exist: {path}")
        if path.suffix.lower() not in allowed_suffixes:
            raise ValueError(
                f"The {label} must be a .pt, .engine or .onnx model: {path}"
            )
    if baseline.resolve() == candidate.resolve() or baseline.samefile(candidate):
        raise ValueError("Baseline and candidate checkpoints must be distinct files.")
    if not data.is_file():
        raise ValueError(f"The dataset YAML does not exist: {data}")
    if data.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError(f"The dataset must be a YAML file: {data}")


def _class_index(report: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    return {int(item["class_id"]): item for item in report["per_class"]}


def _validate_matching_classes(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> list[Mapping[str, Any]]:
    baseline_schema = baseline["class_schema"]
    candidate_schema = candidate["class_schema"]
    if baseline_schema != candidate_schema:
        raise ValueError(
            "Class schemas differ between checkpoints: "
            f"baseline={baseline_schema}, candidate={candidate_schema}"
        )
    baseline_classes = _class_index(baseline)
    candidate_classes = _class_index(candidate)
    if baseline_classes.keys() != candidate_classes.keys():
        raise ValueError(
            "Per-class metric coverage differs between checkpoints: "
            f"baseline={sorted(baseline_classes)}, candidate={sorted(candidate_classes)}"
        )
    for class_id in baseline_classes:
        if baseline_classes[class_id]["name"] != candidate_classes[class_id]["name"]:
            raise ValueError(f"Class name differs for class ID {class_id}.")
    return list(baseline_schema)


def _metric_deltas(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, float]:
    return {metric: float(candidate[metric]) - float(baseline[metric]) for metric in METRICS}


def build_report(
    baseline_path: Path,
    candidate_path: Path,
    data_path: Path,
    parameters: Mapping[str, Any],
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    class_schema = _validate_matching_classes(baseline, candidate)
    baseline_classes = _class_index(baseline)
    candidate_classes = _class_index(candidate)
    per_class_deltas = []
    for class_id in sorted(baseline_classes):
        baseline_class = baseline_classes[class_id]
        candidate_class = candidate_classes[class_id]
        per_class_deltas.append(
            {
                "class_id": class_id,
                "name": baseline_class["name"],
                **_metric_deltas(baseline_class, candidate_class),
            }
        )

    return {
        "schema_version": 1,
        "task": "detect",
        "data": str(data_path.resolve()),
        "parameters": dict(parameters),
        "class_schema": class_schema,
        "models": {
            "baseline": {
                "checkpoint": str(baseline_path.resolve()),
                "metrics": dict(baseline["metrics"]),
                "per_class": list(baseline["per_class"]),
            },
            "candidate": {
                "checkpoint": str(candidate_path.resolve()),
                "metrics": dict(candidate["metrics"]),
                "per_class": list(candidate["per_class"]),
            },
        },
        "deltas": {
            "definition": "candidate_minus_baseline",
            "metrics": _metric_deltas(baseline["metrics"], candidate["metrics"]),
            "per_class": per_class_deltas,
        },
    }


def compare_models(
    baseline_path: Path,
    candidate_path: Path,
    data_path: Path,
    *,
    split: str = "val",
    imgsz: int = 640,
    batch: int = 16,
    device: str = "0",
    workers: int = 2,
    seed: int = 0,
    evaluator: Evaluator | None = None,
) -> dict[str, Any]:
    validate_inputs(baseline_path, candidate_path, data_path)
    if imgsz < 1 or batch < 1:
        raise ValueError("Image size and batch size must be positive integers.")
    if workers < 0 or seed < 0:
        raise ValueError("Workers and seed must be non-negative integers.")

    evaluator = evaluator or evaluate_detection.evaluate
    parameters = {
        "split": split,
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "workers": workers,
        "seed": seed,
        "deterministic": True,
        "plots": False,
        "save_json": False,
        "verbose": False,
    }
    validation_options = {
        key: value for key, value in parameters.items() if key not in {"split", "device"}
    }

    def evaluate_one(path: Path) -> dict[str, Any]:
        return evaluator(
            str(path.resolve()),
            str(data_path.resolve()),
            split=split,
            device=device,
            validation_options=dict(validation_options),
        )

    baseline = evaluate_one(baseline_path)
    candidate = evaluate_one(candidate_path)
    return build_report(
        baseline_path,
        candidate_path,
        data_path,
        parameters,
        baseline,
        candidate,
    )


def write_report(report: Mapping[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _format_value(value: float, *, signed: bool = False) -> str:
    return f"{value:+.4f}" if signed else f"{value:.4f}"


def format_report(report: Mapping[str, Any]) -> str:
    baseline = report["models"]["baseline"]
    candidate = report["models"]["candidate"]
    deltas = report["deltas"]
    lines = [
        "Aggregate metrics (delta = candidate - baseline)",
        f"{'Metric':<10} {'Baseline':>10} {'Candidate':>10} {'Delta':>10}",
    ]
    for metric in METRICS:
        lines.append(
            f"{metric:<10} {_format_value(baseline['metrics'][metric]):>10} "
            f"{_format_value(candidate['metrics'][metric]):>10} "
            f"{_format_value(deltas['metrics'][metric], signed=True):>10}"
        )

    baseline_classes = _class_index(baseline)
    candidate_classes = _class_index(candidate)
    delta_classes = _class_index(deltas)
    lines.extend(
        (
            "",
            "Per-class metrics (B = baseline, C = candidate, D = delta)",
            f"{'Class':<22} {'P B/C/D':>22} {'R B/C/D':>22} "
            f"{'mAP50 B/C/D':>22} {'mAP50-95 B/C/D':>22}",
        )
    )
    for class_id in sorted(baseline_classes):
        values = []
        for metric in METRICS:
            values.append(
                f"{_format_value(baseline_classes[class_id][metric])}/"
                f"{_format_value(candidate_classes[class_id][metric])}/"
                f"{_format_value(delta_classes[class_id][metric], signed=True)}"
            )
        lines.append(
            f"{baseline_classes[class_id]['name']:<22} " + " ".join(f"{value:>22}" for value in values)
        )
    return "\n".join(lines)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare two PPE detectors on one explicitly supplied holdout dataset."
    )
    parser.add_argument("--baseline", type=Path, required=True, help="Baseline .pt, .engine or .onnx model.")
    parser.add_argument("--candidate", type=Path, required=True, help="Candidate .pt, .engine or .onnx model.")
    parser.add_argument("--data", type=Path, required=True, help="Common holdout dataset YAML.")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)
    try:
        report = compare_models(
            args.baseline,
            args.candidate,
            args.data,
            split=args.split,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            seed=args.seed,
        )
        write_report(report, args.output)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(format_report(report))
    print(f"\nJSON report: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
