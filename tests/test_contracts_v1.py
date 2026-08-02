# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from cuajone_qa.contracts import (
    CONTRACT_ROOT,
    SCHEMA_NAMES,
    ContractValidationError,
    load_and_validate,
    load_schema,
    ppe_labels,
    runtime_defaults,
)


@pytest.mark.parametrize("name", SCHEMA_NAMES)
def test_schema_is_valid_draft_2020_12(name: str) -> None:
    schema = load_schema(name)
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    Draft202012Validator.check_schema(schema)


@pytest.mark.parametrize("name", SCHEMA_NAMES)
def test_valid_synthetic_fixture(name: str) -> None:
    value = load_and_validate(name, CONTRACT_ROOT / "fixtures" / "valid" / f"{name}.json")
    assert value["contract_version" if name != "event" else "contractversion"] == "1.0.0"


@pytest.mark.parametrize("name", SCHEMA_NAMES)
def test_invalid_synthetic_fixture(name: str) -> None:
    with pytest.raises(ContractValidationError):
        load_and_validate(name, CONTRACT_ROOT / "fixtures" / "invalid" / f"{name}.json")


def test_defaults_and_semantic_labels_are_versioned() -> None:
    assert runtime_defaults()["tracker"]["profile"] == "byte-track-eigen"
    assert set(ppe_labels()["semantics"]) == {"person", "helmet", "vest"}


def test_contracts_contain_no_secret_material() -> None:
    paths = [CONTRACT_ROOT / "runtime-defaults.json", CONTRACT_ROOT / "ppe-labels.json"]
    paths.extend((CONTRACT_ROOT / "fixtures" / "valid").glob("*.json"))
    for path in paths:
        text = path.read_text(encoding="utf-8").lower()
        assert "rtsp://" not in text
        assert "password" not in text
        assert "credential" not in text
