# SPDX-License-Identifier: AGPL-3.0-only
"""Benchmark the managed dynamic ONNX models with ONNX Runtime CUDA."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


ALLOWED_IMAGE_SIZES = (640, 768, 960, 1280)


def _session(path: Path) -> ort.InferenceSession:
    session = ort.InferenceSession(str(path), providers=["CUDAExecutionProvider"])
    if not session.get_providers() or session.get_providers()[0] != "CUDAExecutionProvider":
        raise RuntimeError(f"CUDAExecutionProvider was not selected for {path.name}")
    return session


def _process_vram_mib() -> int | None:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    current_pid = str(os.getpid())
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 2 and fields[0] == current_pid:
            return int(fields[1])
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--mode", choices=("ppe-only", "ppe-fall"), required=True)
    parser.add_argument("--imgsz", type=int, choices=ALLOWED_IMAGE_SIZES, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=20)
    args = parser.parse_args()

    ppe = _session(args.models / "ppe.onnx")
    sessions = [ppe]
    if args.mode == "ppe-fall":
        sessions.append(_session(args.models / "pose.onnx"))
    inputs = [
        {
            session.get_inputs()[0].name: np.zeros(
                (1, 3, args.imgsz, args.imgsz), dtype=np.float32
            )
        }
        for session in sessions
    ]

    def run_once() -> None:
        for session, values in zip(sessions, inputs, strict=True):
            session.run(None, values)

    for _ in range(args.warmup):
        run_once()
    latencies: list[float] = []
    for _ in range(args.runs):
        started = time.perf_counter()
        run_once()
        latencies.append((time.perf_counter() - started) * 1000.0)
    ordered = sorted(latencies)
    p95_index = min(len(ordered) - 1, int(len(ordered) * 0.95))
    print(
        json.dumps(
            {
                "mode": args.mode,
                "imgsz": args.imgsz,
                "provider": "CUDAExecutionProvider",
                "warmup": args.warmup,
                "runs": args.runs,
                "median_ms": round(statistics.median(latencies), 3),
                "p95_ms": round(ordered[p95_index], 3),
                "process_vram_mib": _process_vram_mib(),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
