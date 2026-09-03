# Analítica de seguridad para cámaras IP de Cuajone

La ruta oficial de producción en Windows es el **MSI aprobado -> NexoAI Vision
launcher -> `NexoAIVision.exe`**. El producto instalado ejecuta el runtime C++
nativo y no depende de Python, PyTorch ni Ultralytics.

`ppe_reportev2.py` es un facade local de desarrollo/QA. Usa
`cuajone_native.pyd` con modelos ONNX dinámicos acotados para ejercitar captura RTSP, binding
nativo, evidencias y reportes compatibles. No es un fallback operativo y no se
incluye en el MSI.

## Producción Windows

1. Obtén el MSI aprobado por la organización.
2. Sigue la [guía de instalación para Windows](INSTALACION_WINDOWS.md).
3. Abre **NexoAI Vision** desde Inicio.
4. Guarda los perfiles RTSP mediante el launcher, que usa Windows Credential
   Manager por usuario.
5. Valida e inicia `NexoAIVision.exe` desde el mismo launcher.

El launcher usa exclusivamente el bundle administrado instalado. Permite elegir
`imgsz` 640/768/960/1280 y una confianza para cada una de las ocho clases de salida;
idioma, tema y esos valores se guardan por usuario. Las rutas de modelos y sus
labels no son configuración del operador.

La preparación de confianza, verificación y mantenimiento corresponde a TI y se
documenta en [Para TI](PARA_TI_WINDOWS.md). La referencia de ingeniería del MSI
está en [`installer/native/README.md`](installer/native/README.md).

## QA local con Python

La ruta Python requiere 3.12 y un `cuajone_native.pyd` compilado localmente. El
binding, Python, los modelos, fixtures y resultados de QA permanecen fuera del MSI.

```powershell
uv python install 3.12
uv sync --locked
uv run python tools/export_runtime_onnx.py --ppe best_ppe.pt --pose yolo26s-pose.pt --output-dir models
Copy-Item .env.example .env
uv run python ppe_reportev2.py --help
uv run python ppe_reportev2.py --mode ppe-only --preflight
```

Configura en el `.env` local las rutas de `PPE_ONNX_PATH`, `POSE_ONNX_PATH` y sus
manifests adyacentes. `PPE_LABELS` declara el orden fijo de clases del ONNX. El
facade Python no carga modelos `.pt`, PyTorch ni Ultralytics. El backend nativo
enlaza ByteTrack-Eigen estaticamente sin importar paquetes Python.

El exportador local genera `models/ppe.onnx`, `models/pose.onnx` y sus manifests.
El detector EPP usa salida raw `[1,12,N]`, donde
`N=(S/8)^2+(S/16)^2+(S/32)^2`; pose conserva `[1,300,57]`. La entrada acepta solo
los cuatro tamaños cuadrados aprobados con batch 1. `models/` es generado y
permanece fuera de Git.

La configuración mínima de QA es:

```dotenv
CAMERA_ID=CAM_CUAJONE_01
RTSP_URL=rtsp://CAMERA_HOST/axis-media/media.amp
ANALYTICS_MODE=ppe-only
PPE_ONNX_PATH=models/ppe.onnx
POSE_ONNX_PATH=models/pose.onnx
PPE_LABELS=Gloves,Person,Safety_boots,Vest,respirador,tapaorejas,Hard_hat,lentes_protectores
TARGET_INFERENCE_FPS=0
```

La URL de ejemplo no contiene credenciales. No subas `.env` a Git. Para producción
no uses `.env`: registra la cámara con el launcher y Windows Credential Manager.
La referencia completa está en
[`docs/development/environment.md`](docs/development/environment.md).

## Binding local

`cuajone_native.pyd` se construye con
`CUAJONE_BUILD_PYTHON_BINDINGS=ON`. Su salida válida permanece bajo
`.tools\native\build\presets\python-bindings\python`; nunca debe copiarse a staging
ni al MSI. `ppe_reportev2.py` descubre automáticamente esa salida y las DLL locales
conocidas cuando existen. Consulta
[`docs/python-cpp-coupling.md`](docs/python-cpp-coupling.md) para compilarlo y
registrar sus DLL locales.

## Modos de analítica

| Modo | Modelos ONNX del harness | Comportamiento |
| --- | --- | --- |
| `ppe-only` | EPP | Seguimiento y analítica EPP nativos; no carga pose. |
| `ppe-fall` | EPP y pose | Analítica EPP y caídas mediante el backend nativo. |

`--mode` prevalece sobre `ANALYTICS_MODE`. El preflight valida archivos, manifests,
labels y construcción real del backend sin abrir RTSP:

```powershell
uv run python ppe_reportev2.py --mode ppe-only --preflight
```

La ejecución con RTSP es exclusivamente una prueba local autorizada:

```powershell
uv run python ppe_reportev2.py --mode ppe-only
```

No se ejecutan pruebas RTSP como parte de la suite automatizada.

## Inferencia GPU: TensorRT

El proveedor GPU de producción es **TensorRT 11** sobre CUDA (bundle fijo
TensorRT 11.1.0.106); el runtime instalado lo reporta como `TensorRT 11/CUDA`.
La ruta CPU es ONNX Runtime; el modo híbrido ONNX CUDA ejecuta pose en CPU por
la inestabilidad documentada del grafo pose en ORT CUDA 1.25.

Especificación de compatibilidad GPU:

| Requisito | Valor |
| --- | --- |
| Capacidad de cómputo mínima | **SM 7.5** (Turing o más nuevo) |
| Driver API CUDA mínima (probe) | 12.9 (`12090`) |
| Línea base documentada del proyecto | NVIDIA GeForce GTX 1650 Ti (SM 7.5, 4 GiB) |
| Selección de modo | `auto` / `cuda` / `cpu`; `cuda` falla cerrado sin hardware y driver listos |

Los engines `.engine` **se construyen por máquina** con `trtexec`
([`installer/native/Build-EnginesOnTarget.ps1`](installer/native/Build-EnginesOnTarget.ps1))
a partir del ONNX del bundle y quedan vinculados a la capacidad de cómputo de la
GPU donde se construyeron: **no son portables** entre arquitecturas de GPU ni
versiones mayores de TensorRT. Al cambiar de GPU o de stack, reconstruirlos.
Un engine construido en la GTX 1650 Ti (SM 7.5) funciona en cualquier GPU SM 7.5
con el mismo TensorRT; una GPU de otra arquitectura (p. ej. Ampere SM 8.6)
requiere su propia construcción.

Los engines heredan el contrato del ONNX de origen: PPE end-to-end `[1,300,6]`
(decode+NMS fusionados en el grafo) y pose `[1,300,57]`; el runtime valida el
schema al deserializar.

**Precisión FP16:** TensorRT 11 compila redes *strongly typed*: no existe flag
`--fp16`; la precisión proviene del grafo ONNX de entrada. Para un engine FP16
se convierte el ONNX a FP16 (I/O queda en float32; conversor
`onnxruntime.transformers.OnnxModel`, que resuelve correctamente los casts de
ops bloqueadas como `Resize`) y se construye con el mismo builder. El gate de
precisión por clase contra el test congelado
(`tools/compare_precision_gate.py`) es obligatorio antes de adoptar FP16:
compara FP32 vs FP16 del mismo grafo contra los labels del split congelado y
registra el reporte en `artifacts/benchmarks/`. Estado: FP16 aprobado y
desplegado en la GTX 1650 Ti (SM 7.5) el 2026-09-02 con degradación máxima
de 0.0062 de recall (Safety_boots) y 0.0000 en las clases críticas.

**Distribución FP16:** el MSI se construye en modo *on-target engine build*
con ONNX FP16 (`tools/export_runtime_onnx.py --half`) y el custom action
`BuildTensorRtEngines` compila los engines FP16 en cada máquina durante la
instalación (`-Precision fp16`). Este variante requiere GPU NVIDIA en todos
los targets: la construcción del engine falla cerrado sin GPU. Para un bundle
portable solo-ONNX (sin engines), construye sin `-PpeOnnxPath`/`-PoseOnnxPath`.

## Contrato de eventos y evidencias

La salida activa usa el [contrato EPP v2](docs/ppe-contract-v2.md). El runtime nativo
y el harness Python conservan el mismo CSV/evidencia de siete elementos. Dentro de
la carpeta de salida:

- `Reporte_Eventos_Seguridad_v2.csv`: append de eventos (síncrono por defecto);
- `Evidencias/`: imágenes JPEG anotadas.

El MSI no genera XLSX ni necesita una biblioteca Excel.
`Reporte_Eventos_Seguridad_v2.xlsx` es solo una conveniencia local/offline del harness
Python para revisión humana. El [contrato v1](docs/operator-evidence-contract-v1.md)
permanece congelado para consumidores estrictos y no se mezcla con el reporte v2.

Los IDs de seguimiento correlacionan observaciones temporales; no identifican
personas.

El flag opcional nativo `--evidence-writer-queue-capacity <1..4096>` usa un solo
writer FIFO acotado para desacoplar persistencia de inferencia. Con `0` (default)
conserva la escritura síncrona. La cola bloquea al llenarse, drena lo aceptado al
cerrar y falla el monitor si el writer entra en estado terminal.

## Ultralytics experimental

La antigua analítica `.pt`/`.engine`, selección PyTorch CUDA y ByteTrack de
Ultralytics vive
explícitamente en `cuajone_qa/experimental/legacy_ultralytics.py`. Se conserva para
compatibilidad de experimentos y caracterización, no como ruta de producción ni
como implementación del facade raíz.

El entorno de desarrollo instala ese conjunto para ejecutar toda la suite. Para
instalar el paquete sin dependencias de prueba y habilitarlo de forma explícita:

```powershell
uv sync --locked --extra experimental
```

Las utilidades `tools/export_tensorrt.py` y `tools/evaluate_detection.py` también
pertenecen a experimentación/model QA. Los resultados de mAP no demuestran
asociación persona-EPP, detección de caídas ni aprobación de producción.

## Verificación

```powershell
uv run python -m pytest -q
uv run python ppe_reportev2.py --help
uv run python ppe_reportev2.py --mode ppe-only --preflight
uv lock --check
git diff --check
```

El preflight puede fallar correctamente si el `.pyd`, sus DLL, el ONNX o su manifest
no están disponibles. `--help` no carga modelos ni abre RTSP.

## Arquitectura

- [`native/README.md`](native/README.md): runtime C++ y launcher de producción.
- [`docs/architecture/repository-layout.md`](docs/architecture/repository-layout.md): límites del repositorio y distribución.
- [`docs/python-cpp-coupling.md`](docs/python-cpp-coupling.md): binding y QA local.
- [`docs/axis-milestone-architecture.md`](docs/axis-milestone-architecture.md): integración futura con Axis/Milestone.
- [`docs/acap-integrability.md`](docs/acap-integrability.md): evaluación separada de ACAP.

## Licencia

El código fuente original se distribuye bajo [`AGPL-3.0-only`](LICENSE). Las
dependencias, modelos y datasets conservan sus propios términos. Consulta
[`LICENSES.md`](LICENSES.md) y [`SECURITY.md`](SECURITY.md).
