# Entorno local de desarrollo y QA

La configuración de este documento corresponde al facade local
`ppe_reportev2.py` con `cuajone_native.pyd` y modelos ONNX fijos. No configura el
producto Windows instalado. En producción usa el MSI aprobado, abre **NexoAI
Vision** y guarda las cámaras mediante Windows Credential Manager.

## Preparar Python

El proyecto usa Python 3.12 y `.venv`:

```powershell
uv python install 3.12
uv sync --locked
Copy-Item .env.example .env
```

`uv sync --locked` instala el grupo de pruebas, incluida la caracterización
experimental. Para instalar explícitamente el extra fuera del entorno de desarrollo:

```powershell
uv sync --locked --extra experimental
```

PyTorch y Ultralytics pertenecen exclusivamente a ese extra y a las pruebas de
`cuajone_qa.experimental`. No son dependencias del harness nativo ni del MSI.

## Compilar el binding local

`cuajone_native.pyd` es solo desarrollo/QA. Debe permanecer bajo
`.tools\native\build\presets\python-bindings\python` y nunca entrar en staging o
en el MSI.

```powershell
py -3.12 -m venv .tools\native\venvs\coupling-py312
.tools\native\venvs\coupling-py312\Scripts\python.exe -m pip install --upgrade pip
.tools\native\venvs\coupling-py312\Scripts\python.exe -m pip install "pybind11==3.0.4" "pytest>=8,<10" "numpy>=1.26,<3" "jsonschema==4.25.1"

Push-Location native
. .\activate-native.ps1
$env:CUAJONE_PYTHON_EXECUTABLE = (Resolve-Path "..\.tools\native\venvs\coupling-py312\Scripts\python.exe").Path
$env:CUAJONE_PYBIND11_ROOT = (Resolve-Path "..\.tools\native\venvs\coupling-py312\Lib\site-packages\pybind11\share\cmake\pybind11").Path
cmake --preset python-bindings
cmake --build --preset python-bindings-release
ctest --preset python-bindings-release
Pop-Location
```

`ppe_reportev2.py` descubre automáticamente el binding y las DLL en estas rutas
locales. Para importar `cuajone_native` directamente desde otro proceso, registra
las DLL y el directorio de salida:

```powershell
$buildPython = (Resolve-Path ".tools\native\build\presets\python-bindings\python").Path
$env:CUAJONE_NATIVE_DLL_DIRS = @(
    $buildPython,
    (Resolve-Path ".tools\native\onnxruntime-win-x64-1.25.0\lib").Path,
    (Resolve-Path ".tools\native\opencv\opencv\build\x64\vc16\bin").Path,
    (Resolve-Path ".tools\native\cuda-runtime\nvidia\cuda_runtime\bin").Path,
    (Resolve-Path ".tools\native\tensorrt\TensorRT-11.1.0.106\bin").Path
) -join [IO.Path]::PathSeparator
$env:PYTHONPATH = "$buildPython$([IO.Path]::PathSeparator)$PWD"

.tools\native\venvs\coupling-py312\Scripts\python.exe -c "import os; handles=[os.add_dll_directory(path) for path in os.environ['CUAJONE_NATIVE_DLL_DIRS'].split(os.pathsep) if path]; import cuajone_native; print(cuajone_native.CONTRACT_VERSION)"
```

## Archivo `.env`

`.env` vive junto a `ppe_reportev2.py`, no se versiona y solo se usa en QA local.
Reinicia el proceso después de cambiarlo. La plantilla no contiene usuario ni
contraseña RTSP.

```dotenv
CAMERA_ID=CAM_CUAJONE_01
RTSP_URL=rtsp://CAMERA_HOST/axis-media/media.amp
ANALYTICS_MODE=ppe-fall
PPE_ONNX_PATH=models/ppe.onnx
POSE_ONNX_PATH=models/pose.onnx
PPE_LABELS=Gloves,Person,Safety_boots,Vest,respirador,tapaorejas,Hard_hat,lentes_protectores
TARGET_INFERENCE_FPS=0

OUTPUT_DIR=.
SHOW_WINDOW=1
SHOW_TEMPORARY_TRACK_ID=0
EXCEL_EXPORT_EVERY_EVENTS=10

POSE_CONF=0.55
PPE_CONF=0.48
IOU_THRESHOLD=0.5

EPP_WINDOW=20
EPP_MIN_SAMPLES=12
EPP_PRESENT_RATIO=0.35
EPP_ALERT_COOLDOWN_S=60

FALL_CONFIRM_FRAMES=12
FALL_RESET_FRAMES=20
FALL_ALERT_COOLDOWN_S=120
FALL_ASPECT_RATIO=1.05
FALL_TORSO_ANGLE_DEG=55
FALL_DESCENT_RATIO=0.12
FALL_NEAR_FLOOR_RATIO=0.65

TRACK_TTL_S=5
RECONNECT_DELAY_S=5
RTSP_TRANSPORT=udp
RTSP_OPEN_TIMEOUT_MS=20000
RTSP_READ_TIMEOUT_MS=10000
RTSP_SOCKET_TIMEOUT_S=3
```

## Modelos ONNX

Genera los artefactos locales y sus manifests desde los checkpoints fuente:

```powershell
uv run python tools/export_runtime_onnx.py --ppe best_ppe.pt --pose yolo26s-pose.pt --output-dir models
```

El detector EPP se exporta raw con `end2end=False` y produce `[1,12,8400]` para las
ocho clases configuradas. Pose conserva la exportación predeterminada compatible de
Ultralytics. La carpeta `models/` completa es generada y no se versiona.

| Variable | Uso |
| --- | --- |
| `PPE_ONNX_PATH` | Detector EPP fijo requerido en ambos modos. |
| `POSE_ONNX_PATH` | Modelo pose fijo requerido solo en `ppe-fall`. |
| `PPE_LABELS` | Orden consecutivo de clases: `Gloves,Person,Safety_boots,Vest,respirador,tapaorejas,Hard_hat,lentes_protectores`. |

Cada ONNX necesita un manifest adyacente `<modelo>.onnx.manifest.json`. Las rutas
relativas se resuelven desde la raíz del repositorio. El binding valida el archivo,
manifest, contrato tensorial y proveedor antes de procesar frames.

`ppe_reportev2.py` no consulta `PPE_MODEL_PATH`, `POSE_MODEL_PATH`, `YOLO_DEVICE`,
`USE_FP16` ni `YOLO_TRACKER`. Esas opciones pertenecen únicamente al módulo
`cuajone_qa.experimental.legacy_ultralytics` y se excluyen deliberadamente de
`.env.example`.

## Analítica

| Variable | Default de plantilla | Contrato |
| --- | ---: | --- |
| `ANALYTICS_MODE` | `ppe-fall` | `ppe-only` o `ppe-fall`; `--mode` prevalece. |
| `PPE_CONF` | `0.48` | Confianza mínima EPP. |
| `POSE_CONF` | `0.55` | Confianza mínima pose; se ignora en `ppe-only`. |
| `IOU_THRESHOLD` | `0.5` | Umbral NMS compartido. |
| `TARGET_INFERENCE_FPS` | `0` | `0` procesa cada frame reciente; positivo limita inicios. |

`ppe-only` no carga ni valida pose. `ppe-fall` exige ambos ONNX. El frame se procesa
mediante `NativeBackend`; la analítica, tracking y thresholds efectivos pertenecen
al pipeline C++ compartido.

## Ventanas temporales

| Variable | Propósito |
| --- | --- |
| `EPP_WINDOW` | Tamaño de la ventana de observaciones EPP. |
| `EPP_MIN_SAMPLES` | Muestras mínimas antes de decidir. |
| `EPP_PRESENT_RATIO` | Proporción para considerar presente casco/chaleco. |
| `EPP_ALERT_COOLDOWN_S` | Separación entre alertas EPP repetidas. |
| `TRACK_TTL_S` | Tiempo de retención del estado temporal. |
| `FALL_CONFIRM_FRAMES` | Frames candidatos antes de confirmar caída. |
| `FALL_RESET_FRAMES` | Frames erguidos para restablecer el estado. |
| `FALL_ALERT_COOLDOWN_S` | Separación entre alertas de caída. |
| `FALL_ASPECT_RATIO` | Umbral de caja horizontal. |
| `FALL_TORSO_ANGLE_DEG` | Umbral angular del torso. |
| `FALL_DESCENT_RATIO` | Descenso relativo mínimo. |
| `FALL_NEAR_FLOOR_RATIO` | Posición relativa cercana al suelo. |

Los valores requieren calibración con material autorizado. Cambia una variable por
prueba y registra el resultado; no asumas que una configuración de laboratorio es
apta para producción.

## Captura RTSP

| Variable | Propósito |
| --- | --- |
| `CAMERA_ID` | Identificador no secreto usado en eventos y archivos. |
| `RTSP_URL` | Fuente autorizada para una prueba local. |
| `RTSP_TRANSPORT` | `tcp` o `udp`. |
| `RTSP_OPEN_TIMEOUT_MS` | Límite de apertura. |
| `RTSP_READ_TIMEOUT_MS` | Límite sin frames antes de reconectar. |
| `RTSP_SOCKET_TIMEOUT_S` | Timeout del diagnóstico TCP. |
| `RECONNECT_DELAY_S` | Espera entre reconexiones. |

La captura conserva solo el frame más reciente; no acumula una cola. Los
diagnósticos redactan la contraseña, pero la credencial sigue existiendo en el
proceso. No uses una URL real en tickets, documentación o Git. En producción,
Credential Manager mediante el launcher es la ruta preferida y aprobada.

## Reportes

| Variable | Propósito |
| --- | --- |
| `OUTPUT_DIR` | Base para el CSV común y `Evidencias/`; el XLSX es solo QA local/offline. |
| `EXCEL_EXPORT_EVERY_EVENTS` | Eventos nuevos entre exportaciones XLSX. |
| `SHOW_WINDOW` | `1` muestra video; `0` ejecuta sin ventana. |
| `SHOW_TEMPORARY_TRACK_ID` | Muestra IDs temporales solo para depuración. |

Los IDs temporales no son identidades humanas. El CSV usa el
[contrato v1 común](../operator-evidence-contract-v1.md) y se escribe de inmediato.
Solo el harness Python consolida XLSX periódicamente y al cierre limpio; el runtime
MSI no lo genera.

## Comprobaciones sin RTSP

```powershell
uv run python ppe_reportev2.py --help
uv run python ppe_reportev2.py --mode ppe-only --preflight
uv run python -m pytest -q
```

`--help` no construye el binding. `--preflight` no abre la cámara, pero sí valida
los modelos configurados y construye el backend real; por eso falla si el `.pyd`,
sus DLL, el ONNX o el manifest no están disponibles.
