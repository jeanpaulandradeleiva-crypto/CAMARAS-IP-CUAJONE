# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .backends.experimental import ExperimentalBackend
from .backends.native import NativeBackend
from .config import QaRuntimeConfig
from .contracts import SCHEMA_NAMES, load_and_validate
from .parity import run_synthetic_parity, write_receipt
from .sources import SourceKind, SourceSpec, iter_fixture, iter_source


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cuajone-qa",
        description="Development/QA coupling tools; not part of the end-user MSI.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    contracts = subcommands.add_parser("validate-contract", help="Validate one JSON contract instance")
    contracts.add_argument("schema", choices=SCHEMA_NAMES)
    contracts.add_argument("path", type=Path)

    demo = subcommands.add_parser("demo", help="Run an explicitly authorized QA source or fixture")
    demo.add_argument("--backend", choices=("experimental", "native"), required=True)
    demo.add_argument("--mode", choices=("ppe-only", "ppe-fall"), default="ppe-fall")
    source = demo.add_mutually_exclusive_group(required=True)
    source.add_argument("--fixture", type=Path)
    source.add_argument("--source")
    demo.add_argument("--source-kind", choices=("image", "video", "webcam", "rtsp"))
    demo.add_argument("--source-id", default="AUTHORIZED_QA_SOURCE")
    demo.add_argument("--ppe-model", help="External Ultralytics PPE model for the experimental backend")
    demo.add_argument("--pose-model", help="External Ultralytics pose model for ppe-fall")
    demo.add_argument("--ppe-engine", help="External TensorRT PPE engine for the native backend")
    demo.add_argument("--pose-engine", help="External TensorRT pose engine for ppe-fall")
    demo.add_argument("--headless", action="store_true")
    demo.add_argument("--annotated-output", type=Path)
    demo.add_argument("--jsonl", type=Path)

    parity = subcommands.add_parser("parity", help="Run staged synthetic parity without models or sources")
    parity.add_argument("--receipt", type=Path)
    return parser


def _annotate(frame: np.ndarray, result: dict[str, Any]) -> np.ndarray:
    import cv2

    annotated = frame.copy()
    for person in result["people"]:
        x1, y1, x2, y2 = map(int, person["box"])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.putText(annotated, person["ppe_status"], (x1, max(20, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return annotated


def _run_demo(args: argparse.Namespace) -> int:
    config = QaRuntimeConfig.defaults(mode=args.mode, backend=args.backend)
    engine_config = None
    models = None
    if args.fixture:
        backend = NativeBackend(config) if args.backend == "native" else ExperimentalBackend(config)
        frames = iter_fixture(args.fixture)
    else:
        if not args.source_kind:
            raise ValueError("--source-kind is required with --source")
        if args.backend == "native":
            if not args.ppe_engine or (args.mode == "ppe-fall" and not args.pose_engine):
                raise ValueError("Native source demo requires external --ppe-engine and --pose-engine in ppe-fall")
            engine_config = {"ppe_engine": args.ppe_engine, "pose_engine": args.pose_engine or ""}
            backend = NativeBackend(config, engine_config=engine_config)
        else:
            if not args.ppe_model or (args.mode == "ppe-fall" and not args.pose_model):
                raise ValueError("Experimental source demo requires external --ppe-model and --pose-model in ppe-fall")
            backend = ExperimentalBackend(config)
            models = backend.load_ultralytics_models(args.ppe_model, args.pose_model)

        def authorized_frames() -> Any:
            for frame_id, frame in enumerate(
                iter_source(SourceSpec(SourceKind(args.source_kind), args.source)),
                start=1,
            ):
                metadata = {
                    "contract_version": "1.0.0",
                    "source_id": args.source_id,
                    # Sequence/state use monotonic time; UTC is only the event label.
                    "frame_id": frame_id,
                    "monotonic_timestamp_ms": int(time.monotonic() * 1000),
                    "observed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                }
                yield frame, metadata

        frames = authorized_frames()
    jsonl_stream = None
    if args.jsonl:
        args.jsonl.parent.mkdir(parents=True, exist_ok=True)
        jsonl_stream = args.jsonl.open("w", encoding="utf-8")
    try:
        for frame, observations in frames:
            result = (
                backend.process_ultralytics_frame(frame, observations, models)
                if models is not None and isinstance(backend, ExperimentalBackend)
                else backend.process_frame(frame, observations)
            )
            if jsonl_stream:
                for event in result.events:
                    jsonl_stream.write(json.dumps(event, sort_keys=True) + "\n")
            annotated = _annotate(frame, result.frame_result)
            if args.annotated_output:
                import cv2

                args.annotated_output.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(args.annotated_output), annotated):
                    raise RuntimeError(f"Could not write annotated output: {args.annotated_output}")
            if not args.headless:
                import cv2

                cv2.imshow("Cuajone QA demo", annotated)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
    finally:
        if jsonl_stream:
            jsonl_stream.close()
        if not args.headless:
            import cv2

            cv2.destroyAllWindows()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "validate-contract":
        load_and_validate(args.schema, args.path)
        return 0
    if args.command == "demo":
        return _run_demo(args)
    if args.command == "parity":
        receipt = run_synthetic_parity()
        path = write_receipt(receipt, args.receipt)
        print(path)
        return 0 if receipt["passed"] else 1
    raise AssertionError(f"Unhandled command: {args.command}")
