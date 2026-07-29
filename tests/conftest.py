# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import os

# Test collection imports the legacy entry point. Prevent local operational
# configuration from being read by development/QA tests.
os.environ["CUAJONE_SKIP_DOTENV"] = "1"

_dll_handles = []
if os.name == "nt":
    for _directory in os.getenv("CUAJONE_NATIVE_DLL_DIRS", "").split(os.pathsep):
        if _directory:
            _dll_handles.append(os.add_dll_directory(_directory))
