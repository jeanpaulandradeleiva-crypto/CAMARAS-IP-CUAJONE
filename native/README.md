# Runtime nativo C++ para EPP y caídas

Este directorio contiene el primer candidato nativo Windows x64. Ejecuta dos
engines TensorRT externos, uno de detección EPP y otro de pose, sin Python,
PyTorch ni Ultralytics durante la ejecución. `ppe_reportev2.py` continúa siendo la
referencia de comportamiento y el fallback operativo.

> Toolchain local: las herramientas y SDK se instalaron bajo
> `D:\DevTools\CuajoneNative`. La activación no modifica el PATH global y los
> builds permanecen fuera del repositorio, también en D.

## Ruta rápida

1. Instala por separado las herramientas y SDK indicados en Prerrequisitos.
2. Define `TENSORRT_ROOT` y `OpenCV_DIR` sin copiar DLL ni librerías al repositorio.
3. Compila y ejecuta primero `cpu-tests`.
4. Compila `windows-msvc` y ejecuta `--preflight` con engines compatibles.
5. Recién después realiza una prueba controlada con un video autorizado.

El preflight no abre la fuente ni ejecuta inferencia. Sí selecciona CUDA,
deserializa ambos engines de forma secuencial y valida sus tensores.

## Arquitectura

| Módulo | Responsabilidad |
| --- | --- |
| `engine_reader` | Acepta planes raw y engines Ultralytics con prefijo reconocible; valida JSON estricto, Unicode y IDs de clase. |
| `tensorrt_runtime` | RAII para runtime, contexto, stream y buffers CUDA; acepta solo I/O device, lineal, no vectorizado, FP32/FP16 y no-shape. |
| `preprocess` | Letterbox OpenCV, BGR a RGB, normalización y empaquetado NCHW FP32; conserva escala y padding exactos. |
| `yolo_decode` | Rechaza valores no finitos, limita candidatos, aplica NMS por clase en coordenadas del modelo y recién después restaura/recorta. |
| `iou_tracker` | IDs temporales mediante asociación IoU determinista, edad máxima y capacidad acotada. |
| `ppe_analytics` | Anclas `Person`, asociación anatómica casco/chaleco, votación temporal y cooldown. |
| `fall_analytics` | Validación de keypoints, geometría, descenso, confirmación, recuperación y cooldown. |
| `contracts` | Versiones, CloudEvents y serialización JSON canónica sin secretos. |
| `analytics_pipeline` | Composición reutilizable, timestamps/frame IDs inyectables, orden estricto y reset. |
| `engine_pipeline` | Inferencia TensorRT y analítica compartidas por ejecutable y binding QA. |
| `capture` | Un único slot reemplazable, reinicio sin frame obsoleto, fallback de apertura y reconexión RTSP con transporte configurable. |
| `evidence` | JPEG anotado y CSV append-only; no genera Excel. |

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
- SDK C++ de TensorRT 11 compatible con los engines, con `NvInfer.h` e import lib.
- CUDA Toolkit/runtime compatible con TensorRT y el controlador NVIDIA.
- OpenCV C++ 4.8 o posterior con `core`, `imgproc`, `imgcodecs`, `videoio` y `highgui`.
- Driver NVIDIA compatible y GPU seleccionable mediante CUDA.

TensorRT, CUDA y OpenCV son dependencias externas. El proyecto no descarga ni
redistribuye binarios propietarios. `nvcc` no es necesario para estas fuentes
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
. .\activate-native.ps1

cmake -S . -B D:\DevTools\CuajoneNative\build\cpu-tests -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DCUAJONE_BUILD_RUNTIME=OFF `
  -DCUAJONE_BUILD_TESTS=ON
cmake --build D:\DevTools\CuajoneNative\build\cpu-tests
ctest --test-dir D:\DevTools\CuajoneNative\build\cpu-tests --output-on-failure

cmake -S . -B D:\DevTools\CuajoneNative\build\windows-msvc -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DCUAJONE_BUILD_RUNTIME=ON `
  -DCUAJONE_BUILD_TESTS=ON
cmake --build D:\DevTools\CuajoneNative\build\windows-msvc
```

El preset CPU no busca CUDA ni TensorRT, pero requiere OpenCV C++ para probar el
letterbox. El preset completo falla de forma explícita si falta MSVC, CUDA,
TensorRT u OpenCV.

Las rutas activadas son TensorRT `11.1.0.106`, CUDA runtime `12.9.79`, OpenCV
`4.12.0` (`vc16`, ABI compatible con VS2022), CMake `3.31.8`, Ninja `1.13.1` y
Visual Studio Build Tools 2022 `17.14`. No copies DLL al repositorio: la activación
agrega temporalmente sus directorios externos al PATH del proceso actual.

## Preflight y ejecución

Ejemplo sin secretos, usando un video local:

```powershell
build\windows-msvc\Release\cuajone_native.exe `
  --preflight `
  --source C:\video-autorizado\turno.mp4 `
  --source-label CAM_CUAJONE_01 `
  --ppe-engine C:\models\ppe.engine `
  --pose-engine C:\models\pose.engine `
  --output C:\resultados\cuajone
```

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
tracker, asociación/votación EPP y confirmación/recuperación de caída.

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
engines externos y la carpeta de salida. El tamaño depende de las builds concretas
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
