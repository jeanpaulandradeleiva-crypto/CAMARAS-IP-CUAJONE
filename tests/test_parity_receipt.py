# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from copy import deepcopy

import pytest

from cuajone_qa.contracts import validate_instance
from cuajone_qa.parity import EXPECTED_STAGE_NAMES, build_authorized_receipt


def authorized_stages() -> list[dict[str, object]]:
    stages = [
        {
            "name": name,
            "status": "passed",
            "comparisons": 1,
            "evidence": [
                {
                    "kind": "comparison",
                    "identity": f"stage-{index}",
                    "sha256": f"{index + 1:x}" * 64,
                }
            ],
        }
        for index, name in enumerate(EXPECTED_STAGE_NAMES)
    ]
    stages[-1]["failures"] = 0
    stages[-1]["authorization_reference"] = "AUTH-QA-001"
    stages[-1]["evidence"][0]["kind"] = "authorized-input"
    return stages


def approved_inputs() -> list[dict[str, str]]:
    return [
        {"identity": "ppe-engine", "sha256": "a" * 64},
        {"identity": "authorized-video", "sha256": "b" * 64},
    ]


def test_authorized_receipt_uses_shared_production_schema() -> None:
    receipt = build_authorized_receipt(
        stages=authorized_stages(),
        authorization_reference="AUTH-QA-001",
        approved_inputs=approved_inputs(),
        revision="a" * 40,
        generated_at="2026-07-29T12:00:00.123456Z",
    )
    assert validate_instance("parity-receipt", receipt) == receipt
    assert receipt["scope"] == "authorized-engine-data"
    assert receipt["full_model_parity_claimed"] is True


@pytest.mark.parametrize("mutation", ("duplicate-stage", "zero-comparisons", "missing-evidence"))
def test_authorized_receipt_rejects_assertion_only_shapes(mutation: str) -> None:
    stages = deepcopy(authorized_stages())
    if mutation == "duplicate-stage":
        stages[-1]["name"] = stages[0]["name"]
    elif mutation == "zero-comparisons":
        stages[2]["comparisons"] = 0
    else:
        stages[3]["evidence"] = []
    with pytest.raises(ValueError):
        build_authorized_receipt(
            stages=stages,
            authorization_reference="AUTH-QA-001",
            approved_inputs=approved_inputs(),
            revision="a" * 40,
        )
