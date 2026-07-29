# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from typing import Any

from .contracts import CONTRACT_VERSION, validate_instance

EVENT_SCHEMA_URI = "https://cuajone.example/contracts/v1/event.schema.json"
PPE_EVENT_TYPE = "com.cuajone.safety.ppe.violation.v1"
FALL_EVENT_TYPE = "com.cuajone.safety.fall.possible.v1"
CANONICAL_DECIMAL_DIGITS = 6
FORBIDDEN_SOURCE_TOKENS = ("rtsp://", "rtsps://", "password", "@")


def canonical_json(value: dict[str, Any]) -> str:
    def encode(item: Any) -> str:
        if item is None:
            return "null"
        if item is True:
            return "true"
        if item is False:
            return "false"
        if isinstance(item, int):
            return str(item)
        if isinstance(item, float):
            return canonical_number_text(item)
        if isinstance(item, str):
            return json.dumps(item, ensure_ascii=False, separators=(",", ":"))
        if isinstance(item, list):
            return "[" + ",".join(encode(entry) for entry in item) + "]"
        if isinstance(item, tuple):
            return "[" + ",".join(encode(entry) for entry in item) + "]"
        if isinstance(item, dict):
            return "{" + ",".join(
                f"{encode(str(key))}:{encode(item[key])}" for key in sorted(item)
            ) + "}"
        raise TypeError(f"Unsupported canonical JSON value: {type(item).__name__}")

    return encode(value)


def float32_value(value: float) -> float:
    try:
        result = struct.unpack("!f", struct.pack("!f", float(value)))[0]
    except (OverflowError, struct.error) as error:
        raise ValueError("Canonical numbers must fit finite float32") from error
    if not math.isfinite(result):
        raise ValueError("Canonical numbers must fit finite float32")
    return result


def canonical_number_text(value: float) -> str:
    text = f"{float32_value(value):.{CANONICAL_DECIMAL_DIGITS}f}".rstrip("0").rstrip(".")
    return "0" if text == "-0" else text


def canonical_number(value: float) -> int | float:
    text = canonical_number_text(value)
    return int(text) if "." not in text else float(text)


def validate_source_id(value: str) -> str:
    lower = value.lower()
    if not value or any(token in lower for token in FORBIDDEN_SOURCE_TOKENS):
        raise ValueError("source_id must be non-secret and cannot contain RTSP or password material")
    return value


def safe_source_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._:-]", "_", validate_source_id(value))


def event_id(source_id: str, frame_id: int, track_id: int, event_index: int) -> str:
    return f"evt-{safe_source_id(source_id)}-{frame_id}-{track_id}-{event_index}"


def make_event(
    *,
    source_id: str,
    frame_id: int,
    monotonic_timestamp_ms: int,
    observed_at: str,
    track_id: int,
    event_index: int,
    event_type: str,
    status: str,
    confidence: float,
    evidence: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    source = safe_source_id(source_id)
    value = {
        "specversion": "1.0",
        "id": event_id(source, frame_id, track_id, event_index),
        "source": f"urn:cuajone:camera:{source}",
        "type": event_type,
        "time": observed_at,
        "datacontenttype": "application/json",
        "dataschema": EVENT_SCHEMA_URI,
        "subject": f"track/{track_id}",
        "contractversion": CONTRACT_VERSION,
        "data": {
            "contract_version": CONTRACT_VERSION,
            "frame_id": frame_id,
            "monotonic_timestamp_ms": monotonic_timestamp_ms,
            "track_id": track_id,
            "status": status,
            "confidence": canonical_number(confidence),
            "evidence": evidence or [],
        },
    }
    return validate_instance("event", value)


def evidence_reference(uri: str, content: bytes) -> dict[str, str]:
    return {"ref": uri, "sha256": hashlib.sha256(content).hexdigest()}


def normalize_event(value: dict[str, Any], *, tolerance_digits: int = 6) -> dict[str, Any]:
    validated = validate_instance("event", value)
    data = validated["data"]
    return {
        "id": validated["id"],
        "source": validated["source"],
        "type": validated["type"],
        "time": validated["time"],
        "subject": validated["subject"],
        "frame_id": data["frame_id"],
        "monotonic_timestamp_ms": data["monotonic_timestamp_ms"],
        "track_id": data["track_id"],
        "status": data["status"],
        "confidence": canonical_number(data["confidence"])
        if tolerance_digits == CANONICAL_DECIMAL_DIGITS
        else round(float(data["confidence"]), tolerance_digits),
        "evidence": sorted(data["evidence"], key=lambda item: (item["ref"], item["sha256"])),
    }
