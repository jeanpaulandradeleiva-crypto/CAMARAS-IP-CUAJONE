# Runtime nativo C++ para EPP y caídas

Este directorio contiene el runtime Windows x64. Ejecuta modelos ONNX con CPU o
engines TensorRT con CUDA, sin Python, PyTorch ni Ultralytics durante la ejecución.
`Auto` prefiere CUDA solo cuando hardware, driver y engines están listos; si no,
usa ONNX CPU. `ppe_reportev2.py` continúa siendo la referencia de comportamiento.
El target WIN32 separado `cuajone_launcher.exe` ofrece la interfaz gráfica y no
enlaza `cuajone_runtime`, OpenCV, ONNX Runtime, CUDA ni TensorRT.

> Toolchain local: las herramientas y SDK se instalaron bajo
> `D:\DevTools\CuajoneNative`. La activación no modifica el PATH global y los
> builds permanecen fuera del repositorio, también en D.

## Ruta rápida

1. Instala por separado las herramientas y SDK indicados en Prerrequisitos.
2. Define `ONNXRUNTIME_ROOT`, `TENSORRT_ROOT` y `OpenCV_DIR` sin copiar binarios al repositorio.
3. Compila y ejecuta primero `cpu-tests`.
4. Compila `windows-msvc` y ejecuta `--preflight` con engines compatibles.
5. Recién después realiza una prueba controlada con un video autorizado.

El preflight no abre la fuente ni ejecuta inferencia. Sí selecciona CUDA,
deserializa ambos engines de forma secuencial y valida sus tensores.

## Arquitectura

| Módulo | Responsabilidad |
| --- | --- |
| `engine_reader` | Acepta planes raw y engines Ultralytics con prefijo reconocible; valida JSON estricto, Unicode y IDs de clase. |
| `compute` | Política Auto/CUDA/CPU, Driver API mínimo, selección multidispositivo y probe DXGI con carga dinámica de `nvcuda.dll`. |
| `model_manifest` | Verifica tipo, rol, tamaño, SHA-256, procedencia, I/O y protobuf ONNX antes de crear una sesión. |
| `onnx_session` | Sesión ONNX Runtime CPU-only creada desde los bytes ya verificados, con un input/output FP32 y shapes fijos. |
| `tensorrt_runtime` | Backend opcional: RAII para runtime, contexto, stream y buffers CUDA. |
| `preprocess` | Letterbox OpenCV, BGR a RGB, normalización y empaquetado NCHW FP32; conserva escala y padding exactos. |
| `yolo_decode` | Rechaza valores no finitos, limita candidatos, aplica NMS por clase en coordenadas del modelo y recién después restaura/recorta. |
| `iou_tracker` | IDs temporales mediante asociación IoU determinista, edad máxima y capacidad acotada. |
| `ppe_analytics` | Anclas `Person`, asociación anatómica casco/chaleco, votación temporal y cooldown. |
| `fall_analytics` | Validación de keypoints, geometría, descenso, confirmación, recuperación y cooldown. |
| `contracts` | Versiones, CloudEvents y serialización JSON canónica sin secretos. |
| `analytics_pipeline` | Composición reutilizable, timestamps/frame IDs inyectables, orden estricto y reset. |
| `engine_pipeline` | Pre/postproceso y analítica compartidos por ONNX CPU y TensorRT CUDA. |
| `capture` | Un único slot reemplazable, reinicio sin frame obsoleto, fallback de apertura y reconexión RTSP con transporte configurable. |
| `evidence` | JPEG anotado y CSV append-only; no genera Excel. |
| `launcher_support` | Matriz estructural de modelos, plan de argumentos, quoting Windows y redacción RTSP comprobables sin el runtime. |
| `launcher` | UI Win32, ejecución del CLI hermano, log ProgramData, Job Object y parada acotada. |

Los dos engines se deserializan uno después del otro y permanecen residentes. La
inferencia EPP termina y sincroniza antes de iniciar pose. No existen colas de GPU,
ejecución concurrente ni asignaciones CUDA por frame. Cada sesión reutiliza su
stream y buffers. Esto reduce picos transitorios, pero el consumo residente de los
dos engines todavía debe validarse en la GTX 1650 Ti de 4 GiB.

Los videos offline se drenan al FPS declarado por el archivo para que el productor
no salte inmediatamente al último frame. Si la inferencia es más lenta, se siguen
descartando frames intermedios y se conserva solo el más reciente.

El descarte pertenece exclusivamente a `LatestFrameCapture`. Las llamadas directas
a `AnalyticsPipeline` y el runner offline no usan ese slot: procesan cada llamada
en orden y rechazan IDs o timestamps que retrocedan hasta ejecutar `reset()`.

## Prerrequisitos

- Windows x64 y Visual Studio 2022 con MSVC C++.
- CMake 3.25 o posterior.
- ONNX Runtime 1.25.0 CPU para Windows x64.
- SDK C++ de TensorRT 11 compatible con los engines, con `NvInfer.h` e import lib.
- CUDA Toolkit/runtime compatible con TensorRT y el controlador NVIDIA.
- OpenCV C++ 4.8 o posterior con `core`, `imgproc`, `imgcodecs`, `videoio` y `highgui`.
- Driver NVIDIA cuya Driver API reporte al menos `12090` (CUDA 12.9) y GPU
  seleccionable mediante CUDA.

ONNX Runtime, TensorRT, CUDA y OpenCV son dependencias externas del build. El MSI
redistribuye únicamente runtimes aprobados con licencias y hashes. `nvcc` no es necesario para estas fuentes
porque no contienen kernels `.cu`; sí se requieren headers e import libs de CUDA.

`TENSORRT_ROOT` no puede estar vacío. CMake busca exclusivamente
`<TENSORRT_ROOT>/include/NvInfer.h` y `nvinfer_11` bajo `lib/` o `lib/x64/` con
`NO_DEFAULT_PATH`. No existe fallback a TensorRT 10, a `nvinfer` sin versión ni a
rutas globales. La configuración informa el header y la import library elegidos y
falla si el SDK no cumple ese contrato de TensorRT 11.

Este host usa el runtime oficial NVIDIA CUDA 12.9.79 extraído, sin `nvcc`, bajo
`D:\DevTools\CuajoneNative\cuda-runtime\nvidia\cuda_runtime`. Para este layout,
`CUDA_RUNTIME_ROOT` debe contener `include/cuda_runtime_api.h`,
`include/cuda_fp16.h`, `lib/x64/cudart.lib` y `bin/cudart64_12.dll`. Si la variable
queda vacía, CMake conserva el flujo normal mediante `find_package(CUDAToolkit)`.
El paquete runtime de NVIDIA separa sus headers internos `crt/`; la activación
define `CUDA_COMPILER_HEADERS_ROOT` hacia el paquete oficial
`nvidia-cuda-nvcc-cu12` 12.9.86, que aporta esos headers sin instalar `nvcc` ni un
Toolkit completo. `CUDA_CCCL_HEADERS_ROOT` apunta al paquete oficial
`nvidia-cuda-cccl-cu12` 12.9.27, que aporta `include/nv/target`.

## Compilación

PowerShell, desde `native/`, sin cambiar variables globales del sistema:

```powershell
. .\activate-native.ps1 -CpuOnly

cmake --preset cpu-tests
cmake --build --preset cpu-tests-release
ctest --preset cpu-tests-release

. .\activate-native.ps1
cmake --preset windows-msvc
cmake --build --preset windows-msvc-release
ctest --test-dir D:\DevTools\CuajoneNative\build\presets\windows-msvc -C Release --output-on-failure
```

El preset CPU compila el ejecutable y el probe con `CUAJONE_ENABLE_TENSORRT=OFF`:
no busca ni enlaza CUDA/TensorRT. El preset completo habilita ambos backends.

Las rutas activadas son TensorRT `11.1.0.106`, CUDA runtime `12.9.79`, OpenCV
`4.12.0` (`vc16`, ABI compatible con VS2022), CMake `3.31.8`, Ninja `1.13.1` y
Visual Studio Build Tools 2022 `17.14`. No copies DLL al repositorio: la activación
agrega temporalmente sus directorios externos al PATH del proceso actual. CMake sí
copia el `onnxruntime.dll` 1.25.0 fijado junto a los ejecutables externos de build;
esto evita que Windows resuelva por error otra versión instalada en `System32`.

## Launcher gráfico

`cuajone_launcher.exe` y `cuajone_native.exe` deben permanecer en la misma carpeta.
El launcher resuelve el runtime hermano mediante la ruta absoluta de su propio
módulo y usa `CreateProcessW`; no busca ejecutables mediante `PATH`. La interfaz
expone fuente RTSP o archivo, carpeta de salida, modo `PPE only`/`PPE + fall`,
cómputo `Auto`/`CUDA`/`CPU`, los cuatro artefactos de modelo, labels EPP y `Show`.
`Validate` ejecuta el mismo plan con `--preflight`; `Start` inicia el procesamiento.

Las rutas iniciales se derivan de `FOLDERID_ProgramData`:

```text
%ProgramData%\Cuajone PPE Monitor\runtime\
  models\ppe.engine
  models\pose.engine
  models\ppe.onnx
  models\pose.onnx
  output\
  logs\cuajone-<timestamp>.log
```

La validación estructural replica la matriz CLI: CUDA exige el engine EPP y, en
`PPE + fall`, el engine pose; CPU exige los ONNX equivalentes, cada manifest
`<modelo>.onnx.manifest.json` adyacente y labels EPP; Auto exige al menos un
candidato completo. `PPE only` nunca emite argumentos pose. El launcher no analiza
el contenido de engines, ONNX ni manifests: el `--preflight` del runtime conserva
esa autoridad.

Stdout y stderr pasan por un pipe y se redacta el userinfo de URLs RTSP antes de
escribir el log visible. El launcher no guarda formularios ni configuración, por
lo que las credenciales RTSP no se persisten. El proceso pertenece a un Job Object
con `KILL_ON_JOB_CLOSE`; `Stop` envía `CTRL_BREAK_EVENT`, espera hasta 30
segundos fuera del hilo UI y, si no termina, finaliza el job. El runtime continúa
siendo un ejecutable de consola y maneja `SIGBREAK` mediante su ruta normal de
apagado cooperativo.

## Preflight y ejecución

Consulta el hardware sin modelos, cámaras ni TensorRT:

```powershell
cuajone_native.exe --hardware-probe-json
```

Los exit codes estables son `0` listo, `10` sin adaptador NVIDIA, `11` driver no
usable, `12` error de probe y `13` Driver API anterior a CUDA 12.9. El JSON usa
`schema_version: 2`, publica `minimum_driver_version: 12090` y enumera el índice de
cada dispositivo. `driver_was_loaded` debe ser `false`: demuestra que
el ejecutable llegó al probe sin cargar `nvcuda.dll` de forma anticipada.
`--compute cpu` nunca ejecuta este probe. El CLI
explícito tiene prioridad sobre `HKLM\SOFTWARE\Cuajone PPE Monitor\ComputeMode`.
TensorRT 11 exige al menos SM 7.5; una GPU anterior no se declara lista aunque la
API del driver CUDA inicialice. Sin `--device`, el runtime elige el primer índice
compatible; un índice explícito inexistente o inferior a SM 7.5 falla cerrado.

Ejemplo sin secretos, usando un video local:

```powershell
build\windows-msvc\Release\cuajone_native.exe `
  --preflight `
  --compute cuda `
  --source C:\video-autorizado\turno.mp4 `
  --source-label CAM_CUAJONE_01 `
  --ppe-engine C:\models\ppe.engine `
  --pose-engine C:\models\pose.engine `
  --output C:\resultados\cuajone
```

Para CPU reemplaza engines por `--ppe-onnx`, `--pose-onnx` y declara el orden de
clases con `--ppe-labels`. CPU exige modelos FP32 con batch 1, NCHW fijo, un input
y un output raw. Shapes dinámicos, NMS fusionado o múltiples outputs fallan cerrado.

Cada `<modelo>.onnx` requiere un `<modelo>.onnx.manifest.json` adyacente. El sidecar
es parte del artefacto aprobado y usa exactamente este contrato:

```json
{
  "schema_version": 1,
  "artifact_type": "onnx",
  "role": "ppe",
  "model_file": "ppe.onnx",
  "model_sha256": "<64 hex lowercase>",
  "model_size_bytes": 123456,
  "external_data": false,
  "custom_operators": false,
  "input": { "name": "images", "element_type": "float32", "shape": [1, 3, 640, 640] },
  "output": { "name": "output0", "element_type": "float32", "shape": [1, 84, 8400] },
  "provenance": {
    "source_uri": "https://example.invalid/approved-model-record",
    "exporter": "approved-exporter-version",
    "license": "SPDX-or-approved-license-reference"
  }
}
```

Para pose, `role` debe ser `pose`. No se aceptan campos desconocidos, symlinks,
extensiones distintas, hashes/tamaños divergentes, `external_data`, funciones
locales ni dominios de operadores distintos de ONNX estándar. El runtime escanea
el protobuf y después entrega a ORT los mismos bytes verificados en memoria; no
vuelve a abrir el path ni permite resolución de sidecars.

Quita `--preflight` para procesar la fuente. `--source` acepta `rtsp://` y
`rtsps://`, exige authority y host no vacíos, acepta userinfo y requiere corchetes
para IPv6, por ejemplo `rtsp://usuario:clave@[2001:db8::1]:554/live`. La etiqueta
por defecto usa solamente el host; los diagnósticos reemplazan el userinfo por
`***`. Usa igualmente `--source-label` para una identidad estable y no secreta.
Consulta todas las opciones con:

```powershell
build\windows-msvc\Release\cuajone_native.exe --help
```

Opciones principales de calibración: `--ppe-conf`, `--pose-conf`, `--nms-iou`,
`--max-det`, `--tracker-iou`, `--tracker-max-age`, `--tracker-max-tracks`,
`--target-fps`, `--ppe-window`, `--ppe-min-samples`, `--ppe-present-ratio`,
`--ppe-cooldown`, `--ppe-track-ttl`, `--fall-confirm-frames`,
`--fall-reset-frames`, `--fall-cooldown`, `--fall-track-ttl`,
`--fall-aspect-ratio`, `--fall-torso-angle`, `--fall-descent-ratio` y
`--fall-near-floor-ratio`. Confianzas e IoU deben estar en `[0,1]`; capacidades,
edades, votos, confirmación y reset son positivos; cooldown, TTL, FPS, reconexión y
timeouts son finitos y no negativos. Los constructores vuelven a validar sus
propios contratos aunque la configuración no provenga del CLI.

### Captura RTSP

`--rtsp-transport default|tcp|udp` conserva el comportamiento de OpenCV/FFmpeg o
fuerza `rtsp_transport` mediante `OPENCV_FFMPEG_CAPTURE_OPTIONS`. La selección
explícita reemplaza las demás opciones FFmpeg de esa variable para este proceso.
Los defaults `--capture-open-timeout-ms 20000` y
`--capture-read-timeout-ms 10000` se pasan como propiedades open-only de FFmpeg.
Si esa apertura parametrizada falla, se intenta una apertura simple para builds de
OpenCV que no aceptan esas propiedades.

`stop()` interrumpe inmediatamente la espera de backoff y solicita fin al reader.
No libera `VideoCapture` desde otro hilo porque OpenCV no garantiza que eso sea
seguro. Una llamada `open/read` ya en curso solo retorna cuando lo hace el backend:
normalmente queda acotada por los timeouts anteriores, pero el fallback simple
puede conservar el límite propio del backend. Un reinicio limpia frame y secuencia
antes de crear el nuevo reader, evitando publicar datos obsoletos.

## Compatibilidad de engines

El loader admite un plan TensorRT raw o un prefijo Ultralytics de 4 bytes
little-endian, longitud JSON validada, JSON de objeto y plan no vacío. Cuando
existen, se leen únicamente `task`, `names`, `imgsz` y `kpt_shape`.

Para un engine EPP raw sin `names`, proporciona el orden exacto mediante
`--ppe-labels Person,Hard_hat,Vest`. El inicio exige clases reconocibles de persona,
casco y chaleco. Para pose raw, configura `--pose-class-count` y
`--pose-kpt-shape`; los defaults explícitos son una clase y `17,3`. Este primer
candidato exige exactamente una clase de pose. Si metadata `names` está presente,
esa clase debe normalizar a `person` o `persona`. El override explícito
`--allow-nonperson-pose-class` omite solamente esa comprobación nominal y debe
usarse únicamente tras verificar manualmente que el engine sigue representando
personas; no habilita múltiples clases.

Se admiten solamente:

- Batch 1, entrada NCHW de tres canales, FP32 o FP16. Cada dimensión `-1` se
  reemplaza individualmente; H/W fijos se conservan y H/W dinámicos requieren
  `imgsz` validado.
- Exactamente un tensor de entrada y uno de salida raw, ambos device-located,
  lineales, no vectorizados, con un componente por elemento y no shape-inference.
- Salida rank 2 o rank 3, con batch 1 cuando corresponda.
- Layout `[C,N]` o `[N,C]` cuando una sola dimensión satisface exactamente
  `4 + clases + keypoints` o `5 + clases + keypoints`.

Los límites previos a reserva son: manifest ONNX 64 KiB, modelo ONNX 256 MiB,
engine TensorRT 1 GiB, rank máximo 8, H/W máximo 4096, entrada máxima
`3 * 4096 * 4096` elementos, salida máxima 16 777 216 elementos y 256 MiB por
tensor. Una dimensión individual no puede superar 1 000 000. Las multiplicaciones
de elementos y bytes se comprueban antes de reservar memoria CPU o CUDA.

El segundo caso incluye objectness. Si ninguna fórmula coincide, ambas coinciden,
el output es dinámico o el engine contiene NMS fusionado/múltiples outputs, el
inicio falla. No se intenta adivinar el schema.

Cada predicción se descarta completa si objectness, algún score de clase, caja o
coordenada/confianza de keypoint no es finito. Se conservan como máximo los 30 000
mejores candidatos antes de NMS para acotar memoria y costo. NMS ocurre sobre las
cajas originales del modelo, antes de restaurar o recortar a la imagen, y mueve la
pose completa para mantener asociados sus keypoints. `--max-det` limita el
resultado final de cada engine a 300 por defecto y no puede superar 30 000.

El prefijo Ultralytics reconocido exige longitud máxima de 16 MiB, JSON de objeto
completo y un plan posterior no vacío. El parser aplica la gramática numérica JSON
sin ceros iniciales ni fracciones/exponentes incompletos, combina pares surrogate
UTF-16 a UTF-8, rechaza surrogates aislados, keys JSON duplicadas e IDs de clase
numéricos duplicados como `"1"`/`"01"`. Un prefijo reconocible pero truncado o
malformado falla claramente; bytes que no se reconocen como prefijo siguen siendo
tratados como un plan TensorRT raw.

Los engines no son artefactos portables. Deben corresponder a la GPU, arquitectura
SM, TensorRT, CUDA, driver, precisión, tamaño y forma usados en el host final.

## Pruebas

`cuajone_cpu_tests` cubre sin GPU el prefijo y JSON de metadata, surrogate pairs,
URLs/redacción RTSP, letterbox, rechazo no-finito, decode detect/pose, orden de NMS,
límites de detección, defensas de constructores, continuidad/expiración del
tracker, asociación/votación EPP, confirmación/recuperación de caída, las 120
combinaciones de política de cómputo y la selección multidispositivo. El target
`cuajone_onnx_tests` genera protobufs sintéticos y cubre inferencia positiva desde
memoria, manifest/hash/rol/I/O, extensión, límites, external data y dominios custom.
`cuajone_launcher_tests` cubre la matriz Auto/CUDA/CPU para ambos modos, manifests
y labels CPU, omisión pose en `PPE only`, quoting de `CreateProcessW` y redacción
de credenciales RTSP.

Baseline de esta corrección: CTest `3/3` tanto en CPU-only como en el build completo;
la suite Python ejecutada con `python -m pytest` informa `93 passed, 1 skipped` sin
binding (el único skip es el módulo nativo opcional) y `104 passed` con el binding
MSVC cargado.

La prueba TensorRT es deliberadamente opt-in y solo inspecciona un engine real:

```powershell
cmake -S . -B build\trt-smoke `
  -DCUAJONE_ENABLE_TRT_INTEGRATION_TESTS=ON `
  -DCUAJONE_TEST_ENGINE=C:\models\compatible.engine
```

No ejecutes esa ruta hasta validar toolchain, SDK, GPU y engine en el host destino.

### Binding Python de QA

`CUAJONE_BUILD_PYTHON_BINDINGS` vale `OFF` por defecto. Con `ON`, CMake exige
Python 3.12 y `pybind11` 3.0.4 desde una ruta explícita en D:. El ejecutable no
enlaza Python. Consulta la [guía de acoplamiento](../docs/python-cpp-coupling.md)
para compilación, API sintética, runtime externo y paridad.

## Salidas

```text
<output>/
  native_events.csv
  evidence/
    <source>_<event>_<track>_<timestamp>.jpg
```

El CSV contiene timestamp, etiqueta de fuente, ID temporal, tipo, confianza,
estado y ruta JPEG. El ID correlaciona observaciones; no identifica personas. La
consolidación XLSX y la revisión humana permanecen en el worker Python.

Cada append hace `flush()` y comprueba el stream. En Windows eso NO equivale a una
transacción ni garantiza durabilidad física en disco ante corte de energía: no se
invoca `_commit` sobre un descriptor nativo porque esta implementación usa
`std::ofstream` y no expone uno de forma portable y segura.

## Distribución esperada

La distribución operativa estará compuesta por el ejecutable, DLL de runtime de
MSVC, DLL de OpenCV, DLL de CUDA/TensorRT autorizadas por sus licencias, los dos
artefactos externos con sus manifests aprobados y la carpeta de salida. El tamaño depende de las builds concretas
de esos componentes; todavía no se midió y no se estima aquí.

## Brechas conocidas

- El tracker es IoU determinista. NO ofrece paridad ByteTrack y puede cambiar IDs
  ante oclusiones, cruces o movimiento rápido.
- Existe comparación sintética por etapas, pero no una comparación de modelos
  reales ni una prueba en la GTX 1650 Ti.
- Los defaults de una clase y 17 keypoints para pose raw deben confirmarse contra
  el exportador real.
- No se soportan NMS fusionado, múltiples heads/outputs ni outputs dinámicos.
- La persistencia JPEG/CSV es síncrona y puede afectar latencia durante eventos.
- El fallback de apertura simple puede quedar sujeto a un timeout interno del
  backend distinto de los valores configurados; `stop()` no puede cancelar de
  forma segura un `open/read` que OpenCV ya esté ejecutando.
- No se genera Excel. Python sigue siendo la referencia de consolidación.
- No se afirma paridad total ni mejora de rendimiento hasta compilar y medir.
