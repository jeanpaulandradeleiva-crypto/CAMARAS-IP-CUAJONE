# Repository layout

This document records the current repository and distribution boundaries. The
authoritative Windows production route is the approved MSI, NexoAI Vision launcher,
and `cuajone_native.exe`.

## Current layout

| Area | Current location | Responsibility |
| --- | --- | --- |
| Python native harness | `ppe_reportev2.py` | Local development/QA facade for fixed ONNX through `cuajone_native.pyd`; not a production fallback. |
| Legacy Python | `ppe_reporte.py` | Deprecated prototype; not for operations. |
| Python QA package | `cuajone_qa/` | Contracts, adapters, demos, parity, typed runtime settings, native binding wrapper, and explicit experimental modules. |
| Legacy Ultralytics analytics | `cuajone_qa/experimental/legacy_ultralytics.py` | Local compatibility/characterization only; owns the relocated `.pt`/PyTorch/ByteTrack pipeline. |
| Shared contracts | `contracts/v1/` | JSON schemas, defaults, labels, and valid/invalid fixtures. |
| Native C++ | `native/` | CMake project, public headers in `include/cuajone/`, domain-grouped implementation in `src/`, and native tests in `tests/`. |
| Python tests | `tests/` | Pytest coverage for Python behavior, contracts, QA, and the optional native binding. |
| Installer | `installer/native/` | WiX source, staging, payload policy, installer validation, signing, and release gates. |
| Documentation | `docs/` | Architecture and development documentation. |
| Tools | `tools/` | Developer utilities such as export and evaluation helpers. |

The repository root also currently contains project metadata, dependency locks,
the Python QA facade, local configuration template, and operational assets.

## Execution entrypoints

| Entrypoint | Boundary |
| --- | --- |
| NexoAI Vision launcher -> `cuajone_native.exe` | Authoritative Windows production path installed by the approved MSI. |
| `python ppe_reportev2.py` | Local `.pyd` + fixed ONNX development/QA harness. |
| `cuajone-qa` / `python -m cuajone_qa` | Python development and QA CLI. |
| `cuajone_launcher.exe` | NexoAI Vision GUI launcher built from `native/src/launcher/launcher.cpp`. |
| `cuajone_native.exe` | Authoritative native runtime built from `native/src/app/main.cpp`. |
| `installer/native/build-installer.ps1` | Installer build orchestration; not an application runtime entrypoint. |

## Distribution boundaries

`pyproject.toml` packages `cuajone_qa` and exposes `cuajone-qa`. The root facade is
not part of that package declaration. PyTorch and Ultralytics are available only
through the `experimental` extra and the development test group.

The production Windows MSI boundary is `installer/native/`. It stages and packages
the launcher, native runtime, approved runtime DLLs, documentation, and installer
metadata under its payload policy. Python source/runtime, `ppe_reportev2.py`,
`cuajone_qa`, `.pyd` files, PyTorch, Ultralytics, datasets, fixtures, and QA receipts
are outside that boundary.

`cuajone_native.pyd` is an optional pybind11 module built from
`native/src/bindings/python_bindings.cpp` only when `CUAJONE_BUILD_PYTHON_BINDINGS=ON`.
It is local development/QA tooling. It must not enter the MSI.

## Native target areas

`native/CMakeLists.txt` defines reusable C++ libraries for compute, contracts,
analytics, inference, and the Windows runtime. It also defines the separate
launcher support and GUI targets, the console runtime, an installer custom action,
and optional CPU, ONNX, and TensorRT test targets. C++ headers stay in
`native/include/cuajone/`; implementations are grouped under `native/src/`; native
tests stay in `native/tests/`.

## Native source grouping (completed)

The native implementation grouping is complete. This refactor preserves CMake
targets, public include paths, output artifacts, and installer behavior.

```text
native/
  include/cuajone/
  src/
    foundation/             # compute, contracts, shared types
    analytics/              # analytics pipeline, tracking, PPE and fall analytics
    inference/              # engine reading, preprocessing, YOLO decoding
    runtime/                # capture, evidence, model and inference runtimes
    app/                    # CLI and native executable entrypoint
    bindings/               # optional pybind11 module
    launcher/               # Windows launcher and support library
    installer/              # MSI hardware-probe custom action
  tests/
```

The `.pyd` and all Python paths remain local development/QA tooling. They do not
replace or fall back from the MSI-installed native runtime.
