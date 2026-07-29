# Analítica de seguridad para cámaras IP de Cuajone

El punto de entrada oficial es `ppe_reportev2.py`. Supervisa un flujo RTSP,
asocia cascos y chalecos con personas rastreadas, registra eventos revisables y,
de forma opcional, ejecuta heurísticas de caída basadas en pose.

`ppe_reporte.py` es un prototipo obsoleto. No debe utilizarse en operación.

## Instalar en Windows

Para instalar el MSI sin usar comandos, sigue la
[guía simple de instalación para Windows](INSTALACION_WINDOWS.md). La preparación
de confianza del equipo, cuando sea necesaria, corresponde a TI.

## Licencia

El codigo fuente original del proyecto se distribuye exclusivamente bajo
[`AGPL-3.0-only`](LICENSE). Las dependencias de terceros conservan sus propias
licencias. Los pesos de modelos, incluidos los derivados de Ultralytics, y los
datasets no se relicencian: deben usarse conforme a sus terminos de origen y
procedencia. Consulta [`LICENSES.md`](LICENSES.md) para el alcance completo y
[`SECURITY.md`](SECURITY.md) para la politica de releases firmadas.

## Inicio rápido

Utiliza Python 3.12. El entorno local del repositorio siempre es `.venv`.

1. Crea y activa `.venv` mediante una de las dos rutas de instalación siguientes.
2. Instala las dependencias bloqueadas con la misma ruta seleccionada.
3. Crea un `.env` local a partir de `.env.example` y configura `RTSP_URL`.
4. Inicia la analítica exclusiva de EPP con `python ppe_reportev2.py --mode ppe-only`.

No incluyas `.env` en un commit de Git porque contiene la credencial de la cámara.

## Instalación nativa con pip y venv

PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

Terminal POSIX:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

`requirements.txt` y `requirements-dev.txt` son exportaciones de compatibilidad
generadas desde `uv.lock`; no deben editarse manualmente. La exportación de
desarrollo incluye dependencias de ejecución y pruebas, mientras que
`requirements.txt` contiene solo las dependencias de ejecución.

## Instalación con uv

El modo de proyecto de uv utiliza `pyproject.toml` para los metadatos y las
dependencias, `.python-version` para seleccionar Python 3.12 y `uv.lock` como
archivo de bloqueo universal. No utiliza un archivo YAML de proyecto.

```powershell
uv python install 3.12
uv sync --locked
.venv\Scripts\Activate.ps1
python ppe_reportev2.py --help
```

La activación es opcional porque `uv run` ejecuta los comandos dentro del proyecto
sincronizado:

```powershell
uv run python ppe_reportev2.py --mode ppe-only
uv run pytest -q
```

La preparación y activación equivalentes en POSIX son:

```bash
uv python install 3.12
uv sync --locked
source .venv/bin/activate
```

Utiliza `uv lock` únicamente cuando cambien intencionalmente las declaraciones de
dependencias. En instalaciones y verificaciones reproducibles, utiliza
`uv sync --locked` para que un bloqueo desactualizado produzca un error en lugar
de regenerarse silenciosamente.

### PyTorch con CUDA opcional

Si el equipo tiene una GPU NVIDIA compatible, instala una compilación de PyTorch
con CUDA después de crear `.venv` y completar la ruta pip o uv. `YOLO_DEVICE=0` y
`torch.cuda.is_available()` solo seleccionan o detectan CUDA cuando la compilación
instalada de PyTorch lo admite; no instalan soporte CUDA.

Requisitos: GPU NVIDIA soportada, controlador NVIDIA actual y compatible
verificado con `nvidia-smi`, Python 3.12 y una rueda de PyTorch correspondiente a
un runtime CUDA admitido oficialmente. Obtén siempre el comando vigente del
[selector oficial de PyTorch](https://pytorch.org/get-started/locally/); la versión
compatible cambia con el tiempo.

Con `.venv` activado, la ruta pip usa este patrón:

```powershell
python -m pip install --reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

Para uv, indica explícitamente el intérprete de `.venv`:

```powershell
uv pip install --python .venv\Scripts\python.exe --reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

```bash
uv pip install --python .venv/bin/python --reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

`cuXXX` es un marcador y DEBE reemplazarse por el valor indicado por el selector
oficial; copiarlo literalmente produce un comando inválido. Verifica la instalación:

```powershell
nvidia-smi
python -c "import torch; print('CUDA disponible:', torch.cuda.is_available()); print('CUDA de PyTorch:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
uv run python -c "import torch; print('CUDA disponible:', torch.cuda.is_available()); print('CUDA de PyTorch:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

El resultado correcto muestra `True`, una versión CUDA no nula y el nombre de la
GPU. Si falla, usa `YOLO_DEVICE=cpu` y `USE_FP16=0`. Consulta la explicación y la
resolución de problemas en [`DOCUMENTACION_ENV.md`](DOCUMENTACION_ENV.md). Tras un
`uv sync --locked` o una reinstalación desde los requisitos exportados, vuelve a
aplicar y verificar la rueda CUDA: el bloqueo actual usa resolución genérica de
PyPI y no detecta automáticamente la GPU.

## Configuración de la aplicación

`.env.example` es la plantilla correspondiente a la implementación actual.
Consulta `DOCUMENTACION_ENV.md` para obtener la referencia completa de variables.

Configuración mínima:

```dotenv
CAMERA_ID=CAM_CUAJONE_01
RTSP_URL=rtsp://USUARIO:CONTRASENA@CAMARA/axis-media/media.amp
ANALYTICS_MODE=ppe-only
PPE_MODEL_PATH=best_ppe.pt
TARGET_INFERENCE_FPS=0
```

Los caracteres reservados de la credencial deben codificarse mediante porcentaje
dentro de la URL. La aplicación oculta la contraseña en los diagnósticos de
inicio, pero la credencial continúa presente en la configuración del proceso.

## Modos de ejecución

| Modo       | Modelos cargados | Seguimiento de personas                       | Comportamiento de caídas                                            |
| ---------- | ---------------- | --------------------------------------------- | ------------------------------------------------------------------- |
| `ppe-fall` | EPP y pose       | IDs del rastreador de pose                    | Evaluación de pose y caídas, con dibujo del esqueleto               |
| `ppe-only` | Solo EPP         | IDs de `Person` del rastreador del modelo EPP | Sin carga, inferencia ni dibujo de pose; no genera eventos de caída |

`ppe-fall` es el valor predeterminado compatible con el comportamiento anterior.
Configura el modo mediante `ANALYTICS_MODE` o sobrescríbelo para una ejecución con
`--mode`. El argumento CLI siempre tiene prioridad:

```powershell
$env:ANALYTICS_MODE = "ppe-fall"
python ppe_reportev2.py --mode ppe-only
```

En `ppe-only` se ignoran `POSE_MODEL_PATH`, `POSE_IMGSZ`, `POSE_CONF` y todas las
variables `FALL_*`. Por ello, una ruta de pose inválida no puede impedir el inicio
de este modo. El modelo EPP debe exponer una clase reconocida como `Person` o
`Persona`; de lo contrario, el inicio falla con un mensaje explícito.

## Controles de rendimiento

El hilo de captura drena continuamente la fuente RTSP hacia un único espacio
reemplazable para el fotograma más reciente. No crea una cola creciente.

`TARGET_INFERENCE_FPS=0` procesa cada fotograma reciente disponible. Un valor
positivo establece un límite monotónico para el inicio de inferencias. Los
fotogramas que llegan antes del siguiente plazo se omiten de la analítica mientras
la captura continúa drenando la fuente, lo que evita acumular trabajo atrasado.

Mejoras seguras implementadas:

- `ppe-only` elimina toda la carga y ejecución del modelo de pose.
- Los diccionarios inmutables de argumentos de inferencia se construyen una sola
  vez fuera del bucle de fotogramas.
- El límite opcional de FPS omite fotogramas intermedios obsoletos.

No se afirma una mejora de velocidad porque no se ejecutó una prueba comparativa
con la cámara y GPU de producción. El flujo de procesamiento mantiene un tamaño
de lote de 1. La persistencia síncrona de JPEG, CSV mediante `fsync` y la
exportación a Excel siguen siendo un cuello de botella conocido durante las
alertas; no se trasladaron a un escritor asíncrono sin pruebas de durabilidad y
cierre limpio. La eliminación de todas las copias duplicadas de fotogramas durante
la ejecución sin interfaz y sin eventos también queda pendiente hasta contar con
pruebas específicas.

Los diagnósticos de inicio muestran el modo efectivo, los modelos cargados, el
dispositivo, la configuración FP16, el objetivo de FPS de inferencia, el
rastreador y el punto de conexión RTSP con la contraseña oculta.

## Ejecución

```powershell
# Modo EPP y caídas compatible con el comportamiento anterior
python ppe_reportev2.py

# Solo EPP, independientemente de ANALYTICS_MODE
python ppe_reportev2.py --mode ppe-only

# Diagnóstico sin cámara ni inferencia
python ppe_reportev2.py --mode ppe-only --preflight

# Ejemplo de ppe-only sin interfaz gráfica
$env:SHOW_WINDOW = "0"
$env:TARGET_INFERENCE_FPS = "5"
python ppe_reportev2.py --mode ppe-only
```

Para detener el proceso, presiona `Q`, `Esc`, cierra la ventana de OpenCV o envía
una interrupción. Durante el cierre limpio, el CSV más reciente se exporta a
Excel.

## Salidas

Dentro de `OUTPUT_DIR` (de forma predeterminada, la raíz del repositorio):

- `Reporte_Eventos_Seguridad.csv`: registro persistente e inmediato de eventos.
- `Reporte_Eventos_Seguridad.xlsx`: libro periódico para operación y revisión.
- `Evidencias/`: evidencias JPEG anotadas para los eventos emitidos.

Los IDs del rastreador son correlaciones temporales, no identidades humanas.

## Advertencia de seguridad

El archivo obsoleto `ppe_reporte.py` contenía anteriormente un usuario y una
contraseña RTSP en el código fuente. La credencial literal fue eliminada, pero los
operadores DEBEN rotarla porque quitarla del árbol actual no la elimina del
historial de Git ni de otras copias. Esta corrección no reescribe el historial.

El programa obsoleto ahora lee `RTSP_URL` desde el entorno y muestra una advertencia
al iniciar. Los despliegues operativos deben utilizar `ppe_reportev2.py`.

## Verificación

Ejecuta estas comprobaciones sin necesidad de cámara, CUDA ni pesos reales:

```powershell
python -m pytest -q
python -m py_compile ppe_reportev2.py ppe_reporte.py tests\test_runtime_modes.py
python ppe_reportev2.py --help
uv lock --check
uv sync --locked
```

Las pruebas utilizan resultados sintéticos con la estructura de Ultralytics y
verifican el aislamiento de modelos, los IDs del rastreador de EPP, la omisión de
la lógica de caídas, la precedencia del modo y el límite de FPS.

## Arquitectura

Consulta [`docs/axis-milestone-architecture.md`](docs/axis-milestone-architecture.md)
para conocer la implementación actual en servidor, los criterios de viabilidad
para Axis ACAP en el dispositivo y las rutas priorizadas de integración con
Milestone XProtect.

## Exportación y evaluación TensorRT desde Python

La ruta Python continúa admitiendo modelos `.pt` y `.engine`. Genera primero un
plan verificable sin cargar modelos ni CUDA:

```powershell
python tools\export_tensorrt.py best_ppe.pt --task detect --dry-run
```

Quita `--dry-run` únicamente en un host de exportación con PyTorch CUDA, TensorRT
y Ultralytics compatibles. El exportador genera el `.engine` junto con un manifest
de hashes. Evalúa el detector `.pt` y su `.engine` contra el mismo dataset etiquetado:

```powershell
python tools\evaluate_detection.py best_ppe.pt --data PPE\data.yaml --device cuda:0
python tools\evaluate_detection.py best_ppe.engine --data PPE\data.yaml --device cuda:0
```

El mAP resultante mide detección de objetos; no demuestra asociación persona-EPP,
cumplimiento por persona ni detección de caídas. Los engines no son portables entre
combinaciones arbitrarias de GPU, TensorRT, CUDA y controlador.

## Candidato nativo C++

El primer candidato Windows x64 sin Python/PyTorch en runtime se encuentra en
[`native/`](native/README.md). Incluye EPP y pose/caídas con dos engines TensorRT
externos, pruebas CPU y preflight. La instalación para usuarios se explica en
[`INSTALACION_WINDOWS.md`](INSTALACION_WINDOWS.md). La construcción, firma,
administración y aceptación del MSI se documentan en la
[guía para TI](installer/native/README.md). `ppe_reportev2.py` continúa siendo la
referencia de comportamiento y el fallback operativo.
