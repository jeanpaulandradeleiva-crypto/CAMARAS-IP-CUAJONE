# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import json
import math
import subprocess
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

from .backends.experimental import ExperimentalBackend
from .backends.native import NativeBackend
from .canonical import normalize_event
from .config import QaRuntimeConfig
from .contracts import CONTRACT_VERSION, runtime_defaults, validate_instance

RECEIPT_VERSION = 1
DEFAULT_RECEIPT_ROOT = Path(r"D:\DevTools\CuajoneNative\parity")
NUMERIC_TOLERANCES = {
    "box_absolute": 1e-4,
    "keypoint_absolute": 1e-4,
    "confidence_absolute": 1e-5,
}
EVENT_NORMALIZATION = {
    "fields": [
        "id",
        "source",
        "type",
        "time",
        "subject",
        "frame_id",
        "monotonic_timestamp_ms",
        "track_id",
        "status",
        "confidence",
        "evidence",
    ],
    "order_by": ["frame_id", "track_id", "type", "id"],
    "confidence_digits": 6,
}
EXPECTED_STAGE_NAMES = (
    "contracts-defaults",
    "preprocess-letterbox",
    "detections-keypoints-canonicalization",
    "tracking-ppe-fall-determinism",
    "canonical-events",
    "authorized-engine-video-end-to-end",
)


def source_commit(project_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    revision = result.stdout.strip()
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise RuntimeError("Git did not return a full source commit")
    return revision


def _close(left: Any, right: Any, tolerance: float = 1e-5) -> bool:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(_close(left[key], right[key], tolerance) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_close(a, b, tolerance) for a, b in zip(left, right))
    return left == right


def normalize_track_identities(
    frame_result: dict[str, Any],
    events: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compare tracker behavior without requiring process-global IDs to restart at one."""
    normalized_frame = deepcopy(frame_result)
    people = sorted(normalized_frame.get("people", []), key=lambda person: tuple(person["box"]))
    identity_map = {
        person["track_id"]: index for index, person in enumerate(people, start=1)
    }
    for person in people:
        person["track_id"] = identity_map[person["track_id"]]
    normalized_frame["people"] = people

    def event_id(value: str, track_id: int) -> str:
        parts = value.rsplit("-", 2)
        if len(parts) != 3:
            raise ValueError("Canonical event ID does not contain track and event suffixes")
        parts[1] = str(identity_map[track_id])
        return "-".join(parts)

    normalized_frame["events"] = sorted(
        event_id(value, int(value.rsplit("-", 2)[1]))
        for value in normalized_frame.get("events", [])
    )
    normalized_events = list(deepcopy(events or []))
    for event in normalized_events:
        payload = event.get("data", event)
        original = payload["track_id"]
        payload["track_id"] = identity_map[original]
        event["subject"] = f"track/{identity_map[original]}"
        event["id"] = event_id(event["id"], original)
    normalized_events.sort(key=lambda event: event["id"])
    return normalized_frame, normalized_events


def _letterbox(width: int, height: int, model_width: int, model_height: int) -> dict[str, Any]:
    ratio = min(model_width / width, model_height / height)
    resized_width = max(1, round(width * ratio))
    resized_height = max(1, round(height * ratio))
    return {
        "scale_x": resized_width / width,
        "scale_y": resized_height / height,
        "padding_left": round((model_width - resized_width) / 2.0 - 0.1),
        "padding_top": round((model_height - resized_height) / 2.0 - 0.1),
    }


def synthetic_frames() -> list[dict[str, Any]]:
    keypoints = [[0.0, 0.0, 0.0] for _ in range(17)]
    keypoints[0] = [50.0, 300.0, 0.9]
    keypoints[5] = [100.0, 300.0, 0.9]
    keypoints[6] = [100.0, 320.0, 0.9]
    keypoints[11] = [250.0, 300.0, 0.9]
    keypoints[12] = [250.0, 320.0, 0.9]
    frames = []
    for frame_id, timestamp in ((1, 100), (2, 200)):
        frames.append(
            {
                "contract_version": CONTRACT_VERSION,
                "source_id": "SYNTHETIC_QA_01",
                "frame_id": frame_id,
                "monotonic_timestamp_ms": timestamp,
                "observed_at": f"2026-01-01T00:00:00.{timestamp:03d}Z",
                "frame": {"width": 640, "height": 720},
                "ppe_classes": {
                    "person_ids": [1],
                    "item_ids": {
                        "gloves": 0, "safety_boots": 2, "vest": 3,
                        "respirator": 4, "hearing_protection": 5,
                        "hard_hat": 6, "eye_protection": 7
                    },
                },
                "ppe_detections": [
                    {"box": [100.0, 500.0, 400.0, 650.0], "confidence": 0.9, "class_id": 1}
                ],
                "pose_detections": [
                    {
                        "box": [100.0, 500.0, 400.0, 650.0],
                        "confidence": 0.9,
                        "class_id": 0,
                        "keypoints": keypoints,
                    }
                ],
            }
        )
    return frames


def parity_config() -> QaRuntimeConfig:
    values = deepcopy(runtime_defaults())
    values["analytics"]["mode"] = "ppe-fall"
    values["ppe"]["window"] = 2
    values["ppe"]["minimum_samples"] = 2
    values["fall"]["confirm_frames"] = 2
    values["fall"]["reset_frames"] = 2
    return QaRuntimeConfig(values)


def build_authorized_receipt(
    *,
    stages: list[dict[str, Any]],
    authorization_reference: str,
    approved_inputs: list[dict[str, str]],
    project_root: Path | None = None,
    revision: str | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Build a production-shaped receipt after an externally authorized parity run."""
    stage_values = deepcopy(stages)
    if tuple(stage.get("name") for stage in stage_values) != EXPECTED_STAGE_NAMES:
        raise ValueError("Authorized parity requires the exact six ordered stages")
    if any(
        stage.get("status") != "passed"
        or not isinstance(stage.get("comparisons"), int)
        or stage["comparisons"] <= 0
        or not stage.get("evidence")
        for stage in stage_values
    ):
        raise ValueError("Every authorized parity stage requires evidence and comparisons")
    if not authorization_reference.strip():
        raise ValueError("An authorization reference is required for production parity")
    identities = [item.get("identity") for item in approved_inputs]
    if len(approved_inputs) < 2 or len(set(identities)) != len(identities):
        raise ValueError("At least two uniquely identified approved inputs are required")
    end_to_end = stage_values[-1]
    if end_to_end.get("authorization_reference") != authorization_reference:
        raise ValueError("End-to-end stage authorization does not match the receipt")
    if end_to_end.get("failures") != 0:
        raise ValueError("Authorized end-to-end parity must report zero failures")

    root = project_root or Path(__file__).resolve().parents[1]
    receipt = {
        "receipt_version": RECEIPT_VERSION,
        "contract_version": CONTRACT_VERSION,
        "source_commit": revision or source_commit(root),
        "generated_at": generated_at or datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "scope": "authorized-engine-data",
        "authorization_reference": authorization_reference,
        "approved_inputs": deepcopy(approved_inputs),
        "tracker_profiles": {
            "production_sim": "byte-track-eigen",
            "experimental_live": "ultralytics-bytetrack-not-equivalent",
        },
        "numeric_tolerances": NUMERIC_TOLERANCES,
        "event_normalization": EVENT_NORMALIZATION,
        "stages": stage_values,
        "full_model_parity_claimed": True,
        "passed": True,
    }
    return validate_instance("parity-receipt", receipt)


def run_synthetic_parity(
    *,
    native_module: ModuleType | Any | None = None,
    project_root: Path | None = None,
    revision: str | None = None,
) -> dict[str, Any]:
    root = project_root or Path(__file__).resolve().parents[1]
    config = parity_config()
    native = NativeBackend(config, module=native_module)
    experimental = ExperimentalBackend(config)
    stages: list[dict[str, Any]] = []

    native_defaults = json.loads(native._module.runtime_defaults_json())
    defaults_equal = native_defaults == runtime_defaults()
    stages.append({"name": "contracts-defaults", "status": "passed" if defaults_equal else "failed", "comparisons": 1})

    native_letterbox = dict(native._module.letterbox_transform(1280, 720, 640, 640))
    letterbox_equal = _close(native_letterbox, _letterbox(1280, 720, 640, 640), 1e-6)
    stages.append({"name": "preprocess-letterbox", "status": "passed" if letterbox_equal else "failed", "comparisons": 4})

    native_results = [native.process_observations(frame) for frame in synthetic_frames()]
    experimental_results = [experimental.process_observations(frame) for frame in synthetic_frames()]
    native_first, _ = normalize_track_identities(native_results[0].frame_result)
    experimental_first, _ = normalize_track_identities(experimental_results[0].frame_result)
    first_equal = _close(native_first["people"], experimental_first["people"])
    stages.append({"name": "detections-keypoints-canonicalization", "status": "passed" if first_equal else "failed", "comparisons": len(native_results[0].frame_result["people"])})

    native_final, native_events = normalize_track_identities(
        native_results[-1].frame_result, native_results[-1].events
    )
    experimental_final, experimental_events = normalize_track_identities(
        experimental_results[-1].frame_result, experimental_results[-1].events
    )
    final_equal = _close(native_final, experimental_final)
    native.reset()
    native_repeated = [native.process_observations(frame) for frame in synthetic_frames()][-1]
    repeated_final, _ = normalize_track_identities(native_repeated.frame_result)
    deterministic = _close(repeated_final, native_final)
    stages.append({"name": "tracking-ppe-fall-determinism", "status": "passed" if final_equal and deterministic else "failed", "comparisons": 2})

    native_events = [normalize_event(event) for event in native_events]
    experimental_events = [normalize_event(event) for event in experimental_events]
    events_equal = _close(native_events, experimental_events)
    stages.append({"name": "canonical-events", "status": "passed" if events_equal else "failed", "comparisons": len(native_events)})
    # Synthetic observations prove deterministic semantics, never model/engine parity.
    stages.append({
        "name": "authorized-engine-video-end-to-end",
        "status": "skipped",
        "comparisons": 0,
        "reason": "Requires externally supplied authorized engines and video; no model parity claim is made by synthetic QA.",
    })
    passed = all(stage["status"] == "passed" for stage in stages if stage["status"] != "skipped")
    receipt = {
        "receipt_version": RECEIPT_VERSION,
        "contract_version": CONTRACT_VERSION,
        "source_commit": revision or source_commit(root),
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "scope": "synthetic-only",
        "tracker_profiles": {
            "production_sim": "byte-track-eigen",
            "experimental_live": "ultralytics-bytetrack-not-equivalent",
        },
        "numeric_tolerances": NUMERIC_TOLERANCES,
        "event_normalization": EVENT_NORMALIZATION,
        "stages": stages,
        # Only an authorized end-to-end run may change this claim in a release receipt.
        "full_model_parity_claimed": False,
        "passed": passed,
    }
    return validate_instance("parity-receipt", receipt)


def write_receipt(receipt: dict[str, Any], output_path: Path | None = None) -> Path:
    validated = validate_instance("parity-receipt", receipt)
    path = output_path or DEFAULT_RECEIPT_ROOT / "synthetic-parity-receipt.json"
    resolved = path.resolve(strict=False)
    if resolved.drive.lower() != "d:":
        raise ValueError("Parity receipts must remain on D:")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(validated, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return resolved


def run_authorized_end_to_end_stage(
    *,
    frames: Any,
    experimental_process: Any,
    native_process: Any,
    authorization_reference: str,
    evidence: list[dict[str, str]],
) -> dict[str, Any]:
    """Opt-in hook for externally supplied engines/data; callers own authorization."""
    if not authorization_reference.strip():
        raise ValueError("An authorization reference is required for end-to-end parity")
    if not evidence:
        raise ValueError("Authorized end-to-end parity requires hashed evidence")
    comparisons = 0
    failures = 0
    for frame, metadata in frames:
        experimental = experimental_process(frame, metadata)
        native = native_process(frame, metadata)
        comparisons += 1
        if not _close(experimental.frame_result, native.frame_result):
            failures += 1
        if not _close(
            [normalize_event(event) for event in experimental.events],
            [normalize_event(event) for event in native.events],
        ):
            failures += 1
    return {
        "name": "authorized-engine-video-end-to-end",
        "status": "passed" if comparisons > 0 and failures == 0 else "failed",
        "comparisons": comparisons,
        "failures": failures,
        "authorization_reference": authorization_reference,
        "evidence": deepcopy(evidence),
    }
