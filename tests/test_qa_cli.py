# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import pytest

from cuajone_qa.cli import main


def test_cli_help_does_not_import_native_or_open_sources(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "validate-contract" in output
    assert "parity" in output
