# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tools import compare_ppe_models


SCHEMA = [
    {"class_id": 0, "name": "Person"},
    {"class_id": 1, "name": "Hard_hat"},
]


def evaluation(offset: float = 0.0, schema: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "class_schema": schema or SCHEMA,
        "metrics": {
            "precision": 0.80 + offset,
            "recall": 0.70 + offset,
            "map50": 0.75 + offset,
            "map50_95": 0.50 + offset,
        },
        "per_class": [
            {
                "class_id": item["class_id"],
                "name": item["name"],
                "precision": 0.81 + offset + item["class_id"] * 0.01,
                "recall": 0.71 + offset + item["class_id"] * 0.01,
                "map50": 0.76 + offset + item["class_id"] * 0.01,
                "map50_95": 0.51 + offset + item["class_id"] * 0.01,
            }
            for item in schema or SCHEMA
        ],
    }


def model_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    baseline = tmp_path / "best_ppe.pt"
    candidate = tmp_path / "future_ppe.pt"
    data = tmp_path / "holdout.yaml"
    baseline.write_bytes(b"baseline")
    candidate.write_bytes(b"candidate")
    data.write_text("path: holdout\n", encoding="utf-8")
    return baseline, candidate, data


def test_compares_both_models_with_identical_explicit_validation_parameters(tmp_path: Path) -> None:
    baseline, candidate, data = model_inputs(tmp_path)
    calls: list[tuple[str, str, dict[str, Any]]] = []

    def evaluator(model: str, dataset: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((model, dataset, kwargs))
        return evaluation(0.05 if Path(model) == candidate else 0.0)

    report = compare_ppe_models.compare_models(
        baseline,
        candidate,
        data,
        imgsz=640,
        batch=16,
        device="cpu",
        workers=2,
        seed=17,
        evaluator=evaluator,
    )

    assert len(calls) == 2
    assert calls[0][1:] == calls[1][1:]
    assert calls[0][2] == {
        "split": "val",
        "device": "cpu",
        "validation_options": {
            "imgsz": 640,
            "batch": 16,
            "workers": 2,
            "seed": 17,
            "deterministic": True,
            "plots": False,
            "save_json": False,
            "verbose": False,
        },
    }
    assert report["task"] == "detect"
    assert report["deltas"]["definition"] == "candidate_minus_baseline"
    assert report["deltas"]["metrics"]["recall"] == pytest.approx(0.05)
    assert report["deltas"]["per_class"][1]["name"] == "Hard_hat"
    assert report["deltas"]["per_class"][1]["map50_95"] == pytest.approx(0.05)


def test_rejects_different_class_schemas(tmp_path: Path) -> None:
    baseline, candidate, data = model_inputs(tmp_path)

    def evaluator(model: str, _dataset: str, **_kwargs: Any) -> dict[str, Any]:
        if Path(model) == candidate:
            return evaluation(schema=[{"class_id": 0, "name": "Worker"}])
        return evaluation()

    with pytest.raises(ValueError, match="Class schemas differ"):
        compare_ppe_models.compare_models(baseline, candidate, data, evaluator=evaluator)


def test_rejects_same_or_missing_input_files_before_evaluation(tmp_path: Path) -> None:
    baseline, candidate, data = model_inputs(tmp_path)
    evaluator = lambda *_args, **_kwargs: pytest.fail("evaluation must not run")

    with pytest.raises(ValueError, match="distinct"):
        compare_ppe_models.compare_models(baseline, baseline, data, evaluator=evaluator)
    with pytest.raises(ValueError, match="candidate checkpoint does not exist"):
        compare_ppe_models.compare_models(
            baseline, tmp_path / "missing.pt", data, evaluator=evaluator
        )
    with pytest.raises(ValueError, match="dataset YAML does not exist"):
        compare_ppe_models.compare_models(
            baseline, candidate, tmp_path / "missing.yaml", evaluator=evaluator
        )


def test_writes_machine_report_and_formats_signed_human_deltas(tmp_path: Path) -> None:
    baseline, candidate, data = model_inputs(tmp_path)
    report = compare_ppe_models.build_report(
        baseline,
        candidate,
        data,
        {"split": "test"},
        evaluation(),
        evaluation(0.05),
    )
    output = tmp_path / "reports" / "comparison.json"

    compare_ppe_models.write_report(report, output)
    rendered = compare_ppe_models.format_report(report)

    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert "Aggregate metrics (delta = candidate - baseline)" in rendered
    assert "Hard_hat" in rendered
    assert "+0.0500" in rendered


def test_cli_requires_one_common_dataset_yaml() -> None:
    with pytest.raises(SystemExit):
        compare_ppe_models.create_parser().parse_args(
            ["--baseline", "old.pt", "--candidate", "new.pt"]
        )


def test_cli_uses_current_ppe_evaluation_defaults() -> None:
    args = compare_ppe_models.create_parser().parse_args(
        [
            "--baseline",
            "old.pt",
            "--candidate",
            "new.pt",
            "--data",
            "holdout.yaml",
        ]
    )

    assert (args.imgsz, args.batch, args.workers, args.device, args.seed) == (
        640,
        16,
        2,
        "0",
        0,
    )
    assert args.output == compare_ppe_models.DEFAULT_OUTPUT
    assert args.split == "val"

def test_compare_accepts_onnx_exports_for_end2end_vs_raw(tmp_path: Path) -> None:
    raw = tmp_path / "ppe.onnx"
    end2end = tmp_path / "ppe-end2end.onnx"
    _, _, data = model_inputs(tmp_path)
    raw.write_bytes(b"raw")
    end2end.write_bytes(b"end2end")
    evaluated: list[Path] = []

    def evaluator(path: str, *_args, **_kwargs) -> dict[str, Any]:
        evaluated.append(Path(path))
        return evaluation(0.05 if Path(path) == end2end else 0.0)

    report = compare_ppe_models.compare_models(
        raw, end2end, data, evaluator=evaluator
    )

    assert evaluated == [raw.resolve(), end2end.resolve()]
    assert report["models"]["candidate"]["checkpoint"] == str(end2end.resolve())
    assert report["deltas"]["metrics"]["map50"] == pytest.approx(0.05)
