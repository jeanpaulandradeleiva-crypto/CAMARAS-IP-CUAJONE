# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

CONTRACT_VERSION = "1.0.0"
CONTRACT_VERSION_V2 = "2.0.0"
CONTRACT_VERSION_V3 = "3.0.0"
CONTRACT_ROOT = Path(__file__).resolve().parents[1] / "contracts" / "v1"
CONTRACT_ROOT_V2 = Path(__file__).resolve().parents[1] / "contracts" / "v2"
CONTRACT_ROOT_V3 = Path(__file__).resolve().parents[1] / "contracts" / "v3"
SCHEMA_NAMES = (
    "runtime-config",
    "camera-config",
    "engine-manifest",
    "frame-result",
    "event",
    "parity-receipt",
)


class ContractValidationError(ValueError):
    """Raised when a versioned contract instance is invalid."""


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ContractValidationError(f"Expected a JSON object in {path}")
    return value


@lru_cache(maxsize=None)
def load_schema(name: str) -> dict[str, Any]:
    if name not in SCHEMA_NAMES:
        raise KeyError(f"Unknown contract schema: {name}")
    schema = load_json(CONTRACT_ROOT / f"{name}.schema.json")
    Draft202012Validator.check_schema(schema)
    return schema


@lru_cache(maxsize=None)
def validator(name: str) -> Draft202012Validator:
    return Draft202012Validator(load_schema(name), format_checker=FormatChecker())


def validate_instance(name: str, instance: dict[str, Any]) -> dict[str, Any]:
    # Reject unsupported shape/version at ingress so adapters can assume one v1 contract.
    errors = sorted(validator(name).iter_errors(instance), key=lambda error: list(error.path))
    if errors:
        details = "; ".join(
            f"{'/'.join(map(str, error.path)) or '<root>'}: {error.message}"
            for error in errors
        )
        raise ContractValidationError(f"{name} v1 validation failed: {details}")
    return instance


@lru_cache(maxsize=None)
def load_schema_v2(name: str) -> dict[str, Any]:
    if name not in {"frame-result", "event"}:
        raise KeyError(f"Unknown v2 contract schema: {name}")
    schema = load_json(CONTRACT_ROOT_V2 / f"{name}.schema.json")
    Draft202012Validator.check_schema(schema)
    return schema


@lru_cache(maxsize=None)
def validator_v2(name: str) -> Draft202012Validator:
    return Draft202012Validator(load_schema_v2(name), format_checker=FormatChecker())


def validate_instance_v2(name: str, instance: dict[str, Any]) -> dict[str, Any]:
    errors = sorted(validator_v2(name).iter_errors(instance), key=lambda error: list(error.path))
    if errors:
        details = "; ".join(
            f"{'/'.join(map(str, error.path)) or '<root>'}: {error.message}" for error in errors
        )
        raise ContractValidationError(f"{name} v2 validation failed: {details}")
    return instance


@lru_cache(maxsize=None)
def load_schema_v3(name: str) -> dict[str, Any]:
    if name not in {"frame-result", "event"}:
        raise KeyError(f"Unknown v3 contract schema: {name}")
    schema = load_json(CONTRACT_ROOT_V3 / f"{name}.schema.json")
    Draft202012Validator.check_schema(schema)
    return schema


def validate_instance_v3(name: str, instance: dict[str, Any]) -> dict[str, Any]:
    errors = sorted(
        Draft202012Validator(load_schema_v3(name), format_checker=FormatChecker()).iter_errors(instance),
        key=lambda error: list(error.path),
    )
    if errors:
        details = "; ".join(
            f"{'/'.join(map(str, error.path)) or '<root>'}: {error.message}" for error in errors
        )
        raise ContractValidationError(f"{name} v3 validation failed: {details}")
    return instance


def load_and_validate(name: str, path: Path) -> dict[str, Any]:
    return validate_instance(name, load_json(path))


def runtime_defaults() -> dict[str, Any]:
    return validate_instance("runtime-config", load_json(CONTRACT_ROOT / "runtime-defaults.json"))


def ppe_labels() -> dict[str, Any]:
    value = load_json(CONTRACT_ROOT / "ppe-labels.json")
    if value.get("contract_version") != CONTRACT_VERSION:
        raise ContractValidationError("PPE labels use an unsupported contract version")
    semantics = value.get("semantics")
    if not isinstance(semantics, dict) or set(semantics) != {"person", "helmet", "vest"}:
        raise ContractValidationError("PPE labels must define person, helmet, and vest semantics")
    for semantic, labels in semantics.items():
        if not isinstance(labels, list) or not labels or not all(
            isinstance(label, str) and label for label in labels
        ):
            raise ContractValidationError(f"PPE semantic {semantic} has invalid labels")
    return value
