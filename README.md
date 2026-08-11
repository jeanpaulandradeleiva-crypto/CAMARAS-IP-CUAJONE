# Analítica de seguridad para cámaras IP de Cuajone

La ruta oficial de producción en Windows es el **MSI aprobado -> NexoAI Vision
launcher -> `NexoAIVision.exe`**. El producto instalado ejecuta el runtime C++
nativo y no depende de Python, PyTorch ni Ultralytics.

`ppe_reportev2.py` es un facade local de desarrollo/QA. Usa
`cuajone_native.pyd` con modelos ONNX fijos para ejercitar captura RTSP, binding
nativo, evidencias y reportes compatibles. No es un fallback operativo y no se
incluye en el MSI.

## Producción Windows

1. Obtén el MSI aprobado por la organización.
2. Sigue la [guía de instalación para Windows](INSTALACION_WINDOWS.md).
3. Abre **NexoAI Vision** desde Inicio.
4. Guarda los perfiles RTSP mediante el launcher, que usa Windows Credential
   Manager por usuario.
5. Valida e inicia `NexoAIVision.exe` desde el mismo launcher.

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
El detector EPP usa salida raw `[1,12,8400]`; pose conserva el contrato de exportación
de Ultralytics. `models/` es generado y permanece fuera de Git.

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

## Contrato de eventos y evidencias

La salida autoritativa de producción es la del runtime nativo instalado por MSI.
El harness Python conserva el mismo contrato CSV/evidencia para QA local. Dentro de
la carpeta de salida:

- `Reporte_Eventos_Seguridad.csv`: append inmediato de eventos;
- `Evidencias/`: imágenes JPEG anotadas.

El orden v1 es:

```csv
Evento_ID,Camara,Fecha,Hora,Tipo_Evento,Casco,Chaleco,Estado_EPP,Confianza_Evento,ID_Seguimiento_Temporal,Estado_Revision,Identificacion_Humana,Observaciones_Revision,Foto
```

El MSI no genera XLSX ni necesita una biblioteca Excel.
`Reporte_Eventos_Seguridad.xlsx` es solo una conveniencia local/offline del harness
Python para revisión humana. Consulta el
[contrato v1 completo](docs/operator-evidence-contract-v1.md), incluida la migración
de `native_events.csv` y `evidence/`.

Los IDs de seguimiento correlacionan observaciones temporales; no identifican
personas.

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
