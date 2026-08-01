# Repository layout

This document records the current repository boundaries and the approved future
layout. It does not move code, change execution paths, or alter distribution
contents.

## Current layout

| Area | Current location | Responsibility |
| --- | --- | --- |
| Python monitoring | `ppe_reportev2.py` | Official Python operational entrypoint. |
| Legacy Python | `ppe_reporte.py` | Deprecated prototype; not for operations. |
| Python QA package | `cuajone_qa/` | Installable package and `cuajone-qa` CLI for contracts, adapters, demos, parity, and typed runtime settings shared by the monitoring facade. |
| Shared contracts | `contracts/v1/` | JSON schemas, defaults, labels, and valid/invalid fixtures. |
| Native C++ | `native/` | CMake project, public headers in `include/cuajone/`, domain-grouped implementation in `src/`, and native tests in `tests/`. |
| Python tests | `tests/` | Pytest coverage for Python behavior, contracts, QA, and the optional native binding. |
| Installer | `installer/native/` | WiX source, staging, payload policy, installer validation, signing, and release gates. |
| Documentation | `docs/` | Architecture and development documentation. |
| Tools | `tools/` | Developer utilities such as export and evaluation helpers. |

The repository root also currently contains project metadata, dependency locks,
the active Python entrypoints, local configuration templates, and operational
assets. Those paths remain unchanged in this slice.

## Execution entrypoints

| Entrypoint | Boundary |
| --- | --- |
| `python ppe_reportev2.py` | Current Python monitoring path. |
| `cuajone-qa` / `python -m cuajone_qa` | Python development and QA CLI. |
| `cuajone_launcher.exe` | Windows GUI launcher built from `native/src/launcher/launcher.cpp`. |
| `cuajone_native.exe` | Windows native console runtime built from `native/src/app/main.cpp`. |
| `installer/native/build-installer.ps1` | Installer build orchestration; not an application runtime entrypoint. |

## Distribution boundaries

`pyproject.toml` packages `cuajone_qa` and exposes `cuajone-qa`. The root Python
monitoring scripts are not part of that package declaration.

The production Windows MSI boundary is `installer/native/`. It stages and
packages the launcher, the native runtime, approved runtime DLLs, documentation,
and installer metadata under its payload policy. Python source and runtime,
`cuajone_qa`, datasets, fixtures, model engines, and QA receipts are outside that
boundary.

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

The documented execution and distribution boundaries remain unchanged. The `.pyd`
remains local development/QA tooling and stays outside the MSI.
