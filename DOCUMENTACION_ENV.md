# Documentación de configuración (`.env`)

Este archivo explica las variables de entorno usadas por `ppe_reportev2.py` y cómo modificarlas para realizar pruebas. `ppe_reporte.py` es un prototipo obsoleto y no debe usarse en operación.

> El archivo `.env` debe estar en la misma carpeta que `ppe_reportev2.py`.
> Detén y vuelve a iniciar el programa después de cualquier cambio.

---

## Reglas básicas

Formato:

```dotenv
VARIABLE=valor
```

Evita espacios alrededor de `=`.

Para valores booleanos:

```dotenv
1 = activado
0 = desactivado
```

Las URLs deben ir entre comillas:

```dotenv
RTSP_URL="rtsp://usuario:clave@172.19.90.72:554/axis-media/media.amp"
```

No compartas ni subas `.env` a Git porque contiene credenciales.

---

# Preparación del proyecto con Python 3.12

La versión preferida y soportada por el proyecto es Python 3.12. El entorno local
se crea siempre en `.venv` y Git lo ignora.

## Ruta nativa con `venv` y pip

PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

En POSIX, usa `python3.12 -m venv .venv` y `source .venv/bin/activate`.
Los archivos `requirements.txt` y `requirements-dev.txt` son exportaciones de
compatibilidad para pip generadas desde el bloqueo de uv; no deben editarse a mano.

## Ruta de proyecto con uv

```powershell
uv python install 3.12
uv sync --locked
.venv\Scripts\Activate.ps1
```

También puedes ejecutar sin activar el entorno:

```powershell
uv run python ppe_reportev2.py --mode ppe-only
uv run pytest -q
```

uv usa `pyproject.toml` como manifiesto, `.python-version` para preferir Python
3.12 y `uv.lock` como archivo de bloqueo universal. No existe un YAML de proyecto
uv en este repositorio. `uv sync --locked` exige que el bloqueo corresponda al
manifiesto y recrea `.venv` cuando el intérprete no es compatible.

## Instalación opcional de PyTorch con CUDA

La configuración `YOLO_DEVICE=0` y la consulta `torch.cuda.is_available()` no
instalan CUDA. Solo seleccionan o detectan la GPU cuando la compilación instalada
de PyTorch incluye soporte para un runtime CUDA compatible. Una instalación
genérica puede resolver una compilación sin ese soporte aunque el equipo tenga una
GPU NVIDIA.

### Requisitos previos

Antes de reemplazar las ruedas de PyTorch, confirma:

- Una GPU NVIDIA soportada por PyTorch.
- Un controlador NVIDIA actual y compatible. `nvidia-smi` debe ejecutarse sin
  errores y reconocer la GPU.
- Python 3.12 dentro de `.venv`.
- Una rueda de PyTorch que corresponda a uno de los runtimes CUDA admitidos
  oficialmente para la versión seleccionada.

Consulta el [selector oficial de instalación de PyTorch](https://pytorch.org/get-started/locally/)
y utiliza el comando vigente que indique para tu plataforma. No existe una versión
CUDA universalmente correcta: las compilaciones soportadas cambian con las
versiones de PyTorch.

Las ruedas CUDA precompiladas de PyTorch normalmente incluyen el runtime CUDA de
espacio de usuario que necesitan. Aun así, requieren un controlador NVIDIA
compatible. Por lo general, el CUDA Toolkit instalado en el sistema solo es
necesario para compilar extensiones CUDA personalizadas, no para ejecutar estas
ruedas precompiladas.

### Reinstalación con pip

Después de crear y activar `.venv` mediante la ruta pip, reemplaza `cuXXX` por el
índice indicado en el selector oficial y reinstala `torch` y `torchvision`:

```powershell
python -m pip install --reinstall torch torchvision --index-url https://download.pytorch.org/whl/cuXXX
```

El mismo comando se aplica en POSIX con `.venv` activado:

```bash
python -m pip install --reinstall torch torchvision --index-url https://download.pytorch.org/whl/cuXXX
```

### Reinstalación con uv

Después de `uv sync --locked`, indica el intérprete de `.venv` de forma explícita.
En Windows:

```powershell
uv pip install --python .venv\Scripts\python.exe --reinstall torch torchvision --index-url https://download.pytorch.org/whl/cuXXX
```

En POSIX:

```bash
uv pip install --python .venv/Scripts/python.exe --reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

> `cuXXX` es únicamente un marcador de posición. DEBE reemplazarse por el sufijo
> CUDA que muestre el selector oficial; copiar `cuXXX` literalmente es inválido.

### Verificación

Primero comprueba el controlador:

```powershell
nvidia-smi
```

Con `.venv` activado, verifica la compilación instalada y la GPU detectada:

```powershell
python -c "import torch; print('CUDA disponible:', torch.cuda.is_available()); print('CUDA de PyTorch:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

La verificación equivalente sin activar el entorno es:

```powershell
uv run python -c "import torch; print('CUDA disponible:', torch.cuda.is_available()); print('CUDA de PyTorch:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

La instalación es correcta cuando `torch.cuda.is_available()` muestra `True`,
`torch.version.cuda` muestra una versión no nula y aparece el nombre de la GPU.

Si `torch.cuda.is_available()` muestra `False`, no configures `YOLO_DEVICE=0`.
Utiliza temporalmente:

```dotenv
YOLO_DEVICE=cpu
USE_FP16=0
```

Después, vuelve a comprobar el controlador mediante `nvidia-smi` y confirma que la
rueda elegida corresponde a una compilación CUDA ofrecida actualmente por el
selector oficial de PyTorch.

### Reproducibilidad del entorno

La reinstalación CUDA descrita aquí es una sustitución manual posterior a la
sincronización. El bloqueo actual del proyecto usa resolución genérica de PyPI y
no detecta la GPU ni conserva automáticamente una variante CUDA específica. Una
ejecución posterior de `uv sync --locked` o una reinstalación pip desde
`requirements.txt` o `requirements-dev.txt` puede reemplazarla. Después de cada
sincronización o reinstalación, vuelve a aplicar el índice CUDA seleccionado y
repite la verificación.

---

# Modo de analítica

## `ANALYTICS_MODE`

Selecciona qué recursos se cargan y qué analítica se ejecuta:

| Valor      | Modelos cargados | Comportamiento                                                 |
| ---------- | ---------------- | -------------------------------------------------------------- |
| `ppe-fall` | EPP y pose       | EPP más evaluación de caídas; valor predeterminado.            |
| `ppe-only` | Solo EPP         | Seguimiento desde la clase `Person`; no carga ni ejecuta pose. |

```dotenv
ANALYTICS_MODE=ppe-only
```

El argumento `--mode` tiene prioridad sobre `ANALYTICS_MODE`:

```powershell
python ppe_reportev2.py --mode ppe-only
```

En `ppe-only`, una ruta `POSE_MODEL_PATH` inválida no bloquea el inicio porque
`POSE_MODEL_PATH`, `POSE_IMGSZ`, `POSE_CONF` y todas las variables `FALL_*` se
ignoran. El modelo EPP debe contener una clase reconocible como `Person` o
`Persona`; de lo contrario, el proceso falla antes de abrir el monitoreo.

---

# Cámara y conexión RTSP

## `CAMERA_ID`

Identificador lógico de la cámara.

```dotenv
CAMERA_ID=CAM_CUAJONE_01
```

Se usa en reportes, evidencias y mensajes de diagnóstico. No afecta la detección.

---

## `RTSP_URL`

Dirección del stream de video.

```dotenv
RTSP_URL="rtsp://usuario:clave@172.19.90.72:554/axis-media/media.amp?videocodec=h264&resolution=1280x720&fps=12&audio=0"
```

Parámetros útiles:

- `videocodec=h264`: solicita H.264.
- `resolution=1280x720`: reduce carga y ancho de banda.
- `fps=12`: limita los FPS enviados.
- `audio=0`: desactiva audio.

Configuración recomendada inicial:

```dotenv
RTSP_URL="rtsp://usuario:clave@IP:554/axis-media/media.amp?videocodec=h264&resolution=1280x720&fps=12&audio=0"
```

Caracteres reservados en la contraseña:

| Carácter | Codificación |
| -------- | ------------ |
| `@`      | `%40`        |
| `#`      | `%23`        |
| `%`      | `%25`        |
| `:`      | `%3A`        |
| `/`      | `%2F`        |
| `?`      | `%3F`        |
| `&`      | `%26`        |

---

## `RTSP_TRANSPORT`

Transporte RTSP.

```dotenv
RTSP_TRANSPORT=tcp
```

Opciones:

```dotenv
RTSP_TRANSPORT=tcp
RTSP_TRANSPORT=udp
```

**TCP**

- Más estable.
- Tolera pérdidas mediante retransmisión.
- Puede acumular latencia.

**UDP**

- Menor latencia.
- Puede perder frames o mostrar artefactos.
- Recomendable solo en una red estable.

Prueba primero TCP y luego UDP para comparar.

---

## `RTSP_OPEN_TIMEOUT_MS`

Tiempo máximo para abrir el stream.

```dotenv
RTSP_OPEN_TIMEOUT_MS=20000
```

Valores típicos:

```dotenv
RTSP_OPEN_TIMEOUT_MS=10000
RTSP_OPEN_TIMEOUT_MS=20000
RTSP_OPEN_TIMEOUT_MS=30000
```

No reduce el lag. Solo controla cuánto espera al conectarse.

---

## `RTSP_READ_TIMEOUT_MS`

Tiempo máximo sin recibir frames antes de considerar que el stream cayó.

```dotenv
RTSP_READ_TIMEOUT_MS=10000
```

Valores típicos:

```dotenv
RTSP_READ_TIMEOUT_MS=5000
RTSP_READ_TIMEOUT_MS=10000
RTSP_READ_TIMEOUT_MS=15000
```

---

## `RTSP_SOCKET_TIMEOUT_S`

Timeout de la prueba TCP al puerto RTSP.

```dotenv
RTSP_SOCKET_TIMEOUT_S=3
```

Solo afecta diagnósticos de red.

---

## `RECONNECT_DELAY_S`

Espera antes de reconectar.

```dotenv
RECONNECT_DELAY_S=5
```

Valores recomendados:

```dotenv
RECONNECT_DELAY_S=5
RECONNECT_DELAY_S=10
```

---

# Modelos e inferencia

## `PPE_MODEL_PATH`

Ruta del modelo personalizado de EPP.

```dotenv
PPE_MODEL_PATH=best_ppe.pt
```

También puede usarse una ruta:

```dotenv
PPE_MODEL_PATH=modelos/best_ppe.pt
```

---

## `POSE_MODEL_PATH`

Modelo de pose.

Solo se consulta en `ppe-fall`. En `ppe-only` se ignora por completo y el modelo
de pose no se valida, construye ni ejecuta.

```dotenv
POSE_MODEL_PATH=yolo26s-pose.pt
```

Opciones:

```dotenv
POSE_MODEL_PATH=yolo26n-pose.pt
POSE_MODEL_PATH=yolo26s-pose.pt
```

- `n`: más rápido y ligero.
- `s`: más preciso, pero consume más GPU.

Para una Quadro T2000 de 4 GB, empieza con `yolo26s-pose.pt` a 512 px. Si falta memoria, usa `yolo26n-pose.pt`.

---

## `YOLO_DEVICE`

Dispositivo de inferencia.

```dotenv
YOLO_DEVICE=0
```

Opciones:

```dotenv
YOLO_DEVICE=0
YOLO_DEVICE=cpu
```

- `0`: primera GPU CUDA.
- `cpu`: procesador.

Para selección automática, deja `YOLO_DEVICE` sin definir: el script usa CUDA `0`
si `torch.cuda.is_available()` es verdadero y, en caso contrario, usa CPU.

Usa `0` solo si `torch.cuda.is_available()` devuelve `True`.

---

## `USE_FP16`

Activa precisión de 16 bits.

```dotenv
USE_FP16=1
```

- `1`: menor VRAM y normalmente mayor velocidad en GPU.
- `0`: necesario en CPU.

Con GPU:

```dotenv
YOLO_DEVICE=0
USE_FP16=1
```

Con CPU:

```dotenv
YOLO_DEVICE=cpu
USE_FP16=0
```

---

## `YOLO_TRACKER`

Tracker temporal.

```dotenv
YOLO_TRACKER=bytetrack.yaml
```

Opciones comunes:

```dotenv
YOLO_TRACKER=bytetrack.yaml
YOLO_TRACKER=botsort.yaml
```

El ID del tracker no identifica a una persona. Solo relaciona detecciones durante varios frames.

---

# Resolución y confianza

## `POSE_IMGSZ`

Resolución usada por el modelo pose.

Se ignora en `ppe-only`.

```dotenv
POSE_IMGSZ=512
```

Valores de prueba:

```dotenv
POSE_IMGSZ=416
POSE_IMGSZ=512
POSE_IMGSZ=640
```

- Mayor valor: mejor detección de personas pequeñas, más consumo y menor velocidad.
- Menor valor: más velocidad, pero puede perder personas lejanas.

---

## `PPE_IMGSZ`

Resolución usada por el modelo EPP.

```dotenv
PPE_IMGSZ=640
```

Valores de prueba:

```dotenv
PPE_IMGSZ=512
PPE_IMGSZ=640
```

Configuración equilibrada:

```dotenv
POSE_IMGSZ=512
PPE_IMGSZ=640
```

---

## `POSE_CONF`

Confianza mínima para aceptar una persona o pose.

Se ignora en `ppe-only`.

```dotenv
POSE_CONF=0.60
```

Prueba gradualmente:

```dotenv
POSE_CONF=0.50
POSE_CONF=0.55
POSE_CONF=0.60
POSE_CONF=0.65
```

- Aumentar reduce falsos positivos.
- Aumentar demasiado puede perder personas lejanas.

---

## `PPE_CONF`

Confianza mínima de casco, chaleco y otras clases EPP.

```dotenv
PPE_CONF=0.30
```

Valores de prueba:

```dotenv
PPE_CONF=0.30
PPE_CONF=0.40
PPE_CONF=0.50
```

---

## `IOU_THRESHOLD`

Umbral de solapamiento para eliminar cajas duplicadas.

```dotenv
IOU_THRESHOLD=0.45
```

Pruebas:

```dotenv
IOU_THRESHOLD=0.40
IOU_THRESHOLD=0.45
IOU_THRESHOLD=0.50
```

---

## `TARGET_INFERENCE_FPS`

Límite opcional para la frecuencia de inicio de inferencias.

```dotenv
TARGET_INFERENCE_FPS=0
```

- `0`: sin límite; procesa cada frame más reciente disponible.
- Valor positivo: usa un plazo basado en `time.monotonic()` para omitir frames que
  llegan antes de la siguiente inferencia permitida.
- Valor negativo, infinito o no numérico: configuración inválida.

La captura RTSP sigue drenando la fuente mientras se omiten frames intermedios. No
se forma una cola ni se recupera trabajo atrasado. Por ejemplo,
`TARGET_INFERENCE_FPS=5` limita a un máximo objetivo de cinco inicios por segundo,
pero no garantiza 5 FPS si la inferencia tarda más de 200 ms.

---

# Validación de personas

> **Estado de implementación:** las variables históricas de esta sección
> (`MIN_KEYPOINT_CONF`, `MIN_VALID_KEYPOINTS`,
> `REQUIRE_PPE_PERSON_CONFIRMATION`, `PERSON_IOU_THRESHOLD`,
> `PERSON_HISTORY_FRAMES`, `PERSON_MIN_CONFIRMATIONS` y `EXCLUSION_ZONES`) no son
> leídas actualmente por `ppe_reportev2.py`. Se conservan porque explican ajustes
> útiles para una evolución futura, pero agregarlas a `.env` no cambia la ejecución.
> La implementación actual deriva el umbral de keypoints desde `POSE_CONF`, exige
> hombros y caderas visibles y usa una confirmación geométrica fija con `Person`.

## `MIN_KEYPOINT_CONF`

Confianza mínima de cada punto corporal.

```dotenv
MIN_KEYPOINT_CONF=0.45
```

Auméntalo para reducir poses falsas sobre objetos. Redúcelo si se pierden personas lejanas.

---

## `MIN_VALID_KEYPOINTS`

Cantidad mínima de puntos corporales confiables.

```dotenv
MIN_VALID_KEYPOINTS=7
```

Pruebas:

```dotenv
MIN_VALID_KEYPOINTS=5
MIN_VALID_KEYPOINTS=7
MIN_VALID_KEYPOINTS=9
```

Más puntos reducen falsos positivos, pero pueden descartar personas parcialmente ocultas.

---

## `REQUIRE_PPE_PERSON_CONFIRMATION`

Exige que el modelo EPP también detecte una persona.

```dotenv
REQUIRE_PPE_PERSON_CONFIRMATION=1
```

- `1`: mayor precisión, menos falsos positivos.
- `0`: solo usa pose para confirmar personas.

---

## `PERSON_IOU_THRESHOLD`

Solapamiento mínimo entre la caja de pose y la caja `Person` del modelo EPP.

```dotenv
PERSON_IOU_THRESHOLD=0.30
```

Valores de prueba:

```dotenv
PERSON_IOU_THRESHOLD=0.20
PERSON_IOU_THRESHOLD=0.30
PERSON_IOU_THRESHOLD=0.40
```

---

## `PERSON_HISTORY_FRAMES`

Cantidad de observaciones recientes para confirmar una persona.

```dotenv
PERSON_HISTORY_FRAMES=10
```

---

## `PERSON_MIN_CONFIRMATIONS`

Número mínimo de detecciones válidas dentro del historial.

```dotenv
PERSON_MIN_CONFIRMATIONS=7
```

Ejemplo:

```dotenv
PERSON_HISTORY_FRAMES=10
PERSON_MIN_CONFIRMATIONS=7
```

La persona debe validarse en al menos 7 de las últimas 10 observaciones.

---

## `EXCLUSION_ZONES`

Zonas donde las detecciones de personas deben ignorarse.

Formato normalizado:

```dotenv
EXCLUSION_ZONES=0.76,0.31,0.92,0.72
```

Orden:

```text
x1,y1,x2,y2
```

Los valores van de `0.0` a `1.0`.

Varias zonas pueden separarse con `;`:

```dotenv
EXCLUSION_ZONES=0.76,0.31,0.92,0.72;0.00,0.00,0.15,0.30
```

Úsalo para ignorar tachos, equipos, letreros u objetos fijos confundidos con personas.

---

# Evaluación de EPP

## `EPP_WINDOW`

Número de muestras recientes consideradas.

```dotenv
EPP_WINDOW=20
```

Más alto:

- Mayor estabilidad.
- Más demora para alertar.

Más bajo:

- Respuesta rápida.
- Mayor riesgo de falsos positivos.

---

## `EPP_MIN_SAMPLES`

Mínimo de muestras antes de evaluar.

```dotenv
EPP_MIN_SAMPLES=12
```

Debe ser menor o igual que `EPP_WINDOW`.

---

## `EPP_PRESENT_RATIO`

Proporción de muestras donde un EPP debe aparecer para considerarse presente.

```dotenv
EPP_PRESENT_RATIO=0.35
```

Ejemplo: con 20 muestras, `0.35` requiere aproximadamente 7 detecciones.

Pruebas:

```dotenv
EPP_PRESENT_RATIO=0.30
EPP_PRESENT_RATIO=0.40
EPP_PRESENT_RATIO=0.50
```

---

## `EPP_ALERT_COOLDOWN_S`

Tiempo mínimo entre alertas EPP repetidas.

```dotenv
EPP_ALERT_COOLDOWN_S=60
```

Pruebas:

```dotenv
EPP_ALERT_COOLDOWN_S=30
EPP_ALERT_COOLDOWN_S=60
EPP_ALERT_COOLDOWN_S=120
```

---

## `TRACK_TTL_S`

Tiempo que se conserva el estado de un tracker que dejó de verse.

```dotenv
TRACK_TTL_S=5
```

Un valor muy bajo puede perder el historial durante oclusiones breves. Un valor
excesivo conserva estados ya ausentes durante más tiempo y aumenta el diccionario
de seguimiento.

---

# Detección de caídas

Toda esta sección se aplica únicamente a `ppe-fall`. En `ppe-only` no se evalúan
caídas y estas variables se ignoran.

## `FALL_CONFIRM_FRAMES`

Número de frames que deben mantener la condición de caída.

```dotenv
FALL_CONFIRM_FRAMES=12
```

Pruebas:

```dotenv
FALL_CONFIRM_FRAMES=8
FALL_CONFIRM_FRAMES=12
FALL_CONFIRM_FRAMES=18
```

Más alto reduce falsos positivos, pero retrasa la alerta.

---

## `FALL_RESET_FRAMES`

Frames normales necesarios para restablecer el estado.

```dotenv
FALL_RESET_FRAMES=20
```

---

## `FALL_ALERT_COOLDOWN_S`

Tiempo mínimo entre alertas repetidas de caída.

```dotenv
FALL_ALERT_COOLDOWN_S=120
```

---

## `FALL_ASPECT_RATIO`

Relación ancho/alto mínima para considerar una postura horizontal.

```dotenv
FALL_ASPECT_RATIO=1.05
```

Pruebas:

```dotenv
FALL_ASPECT_RATIO=1.00
FALL_ASPECT_RATIO=1.10
FALL_ASPECT_RATIO=1.20
```

Aumentarlo reduce falsos positivos, pero puede perder caídas vistas en perspectiva.

---

## `FALL_TORSO_ANGLE_DEG`

Ángulo mínimo del torso para considerarlo horizontal.

```dotenv
FALL_TORSO_ANGLE_DEG=55
```

Pruebas:

```dotenv
FALL_TORSO_ANGLE_DEG=50
FALL_TORSO_ANGLE_DEG=60
FALL_TORSO_ANGLE_DEG=70
```

---

## `FALL_DESCENT_RATIO`

Descenso mínimo del centro corporal respecto al alto del frame.

```dotenv
FALL_DESCENT_RATIO=0.12
```

Pruebas:

```dotenv
FALL_DESCENT_RATIO=0.08
FALL_DESCENT_RATIO=0.12
FALL_DESCENT_RATIO=0.18
```

---

## `FALL_NEAR_FLOOR_RATIO`

Posición vertical mínima para considerar que la persona está cerca del suelo.

```dotenv
FALL_NEAR_FLOOR_RATIO=0.65
```

Debe calibrarse según la perspectiva de cada cámara.

---

# Ventana y visualización

## `SHOW_WINDOW`

Muestra la ventana de video.

```dotenv
SHOW_WINDOW=1
```

- `1`: muestra video.
- `0`: ejecución sin interfaz.

---

## `SHOW_TEMPORARY_TRACK_ID`

Muestra el ID temporal del tracker.

```dotenv
SHOW_TEMPORARY_TRACK_ID=0
```

Actívalo solo para depuración:

```dotenv
SHOW_TEMPORARY_TRACK_ID=1
```

No debe interpretarse como identidad de una persona.

---

# Reportes y evidencias

## `OUTPUT_DIR`

Carpeta donde se guardan reportes y evidencias.

```dotenv
OUTPUT_DIR=.
```

Ruta relativa:

```dotenv
OUTPUT_DIR=Resultados
```

Ruta absoluta:

```dotenv
OUTPUT_DIR=C:/MonitoreoSeguridad/Resultados
```

Estructura esperada:

```text
Resultados/
├── Reporte_Eventos_Seguridad.csv
├── Reporte_Eventos_Seguridad.xlsx
└── Evidencias/
```

---

## `EXCEL_EXPORT_EVERY_EVENTS`

Cantidad de eventos nuevos que activa una exportación periódica del Excel.

```dotenv
EXCEL_EXPORT_EVERY_EVENTS=10
```

El CSV se escribe y sincroniza inmediatamente; el Excel se regenera cada cierta
cantidad de eventos y durante el cierre limpio. Esto reduce el costo de exportar
por cada frame o evento.

Pruebas:

```dotenv
EXCEL_EXPORT_EVERY_EVENTS=5
EXCEL_EXPORT_EVERY_EVENTS=10
EXCEL_EXPORT_EVERY_EVENTS=30
```

La variable histórica `EXCEL_EXPORT_INTERVAL_S` no está implementada. Su objetivo
de espaciar exportaciones sigue siendo válido, pero el código actual usa un conteo
de eventos, no un intervalo de segundos.

---

## `JPEG_QUALITY` (histórica, no implementada)

Calidad JPEG de las evidencias.

```dotenv
JPEG_QUALITY=92
```

Rango habitual:

```text
70 a 100
```

- Mayor valor: mejor calidad y archivos más grandes.
- Menor valor: menor tamaño y menos detalle.

Esta recomendación de calidad sigue siendo útil para una mejora futura, pero
`ppe_reportev2.py` no lee `JPEG_QUALITY`; actualmente `cv2.imwrite` usa su calidad
predeterminada.

---

# Configuraciones recomendadas para pruebas

## Perfil equilibrado para Quadro T2000

```dotenv
YOLO_DEVICE=0
USE_FP16=1

POSE_MODEL_PATH=yolo26s-pose.pt
POSE_IMGSZ=512
POSE_CONF=0.60

PPE_MODEL_PATH=best_ppe.pt
PPE_IMGSZ=640
PPE_CONF=0.35

RTSP_TRANSPORT=tcp
```

## Perfil de baja latencia

```dotenv
RTSP_TRANSPORT=udp

POSE_MODEL_PATH=yolo26n-pose.pt
POSE_IMGSZ=416
PPE_IMGSZ=512

RTSP_URL="rtsp://usuario:clave@IP:554/axis-media/media.amp?videocodec=h264&resolution=1280x720&fps=10&audio=0"
```

## Perfil para reducir falsos positivos

Las variables de validación histórica de este perfil se mantienen como propuesta
de calibración, pero no modifican el código actual. `POSE_CONF` sí está implementada.

```dotenv
POSE_CONF=0.65
MIN_KEYPOINT_CONF=0.50
MIN_VALID_KEYPOINTS=8

REQUIRE_PPE_PERSON_CONFIRMATION=1
PERSON_IOU_THRESHOLD=0.30

PERSON_HISTORY_FRAMES=12
PERSON_MIN_CONFIRMATIONS=9
```

---

# Metodología de prueba

Modifica una sola variable por prueba.

Registra:

| Prueba | Variable    | Valor | FPS | Latencia | Falsos positivos | Observaciones |
| ------ | ----------- | ----: | --: | -------: | ---------------: | ------------- |
| 1      | `POSE_CONF` |  0.55 |     |          |                  |               |
| 2      | `POSE_CONF` |  0.60 |     |          |                  |               |
| 3      | `POSE_CONF` |  0.65 |     |          |                  |               |

No cambies simultáneamente resolución, confianza, transporte y modelo, porque no podrás identificar qué cambio mejoró o empeoró el sistema.

---

# Plantilla `.env` recomendada

```dotenv
CAMERA_ID=CAM_CUAJONE_01
RTSP_URL="rtsp://usuario:clave@172.19.90.72:554/axis-media/media.amp?videocodec=h264&resolution=1280x720&fps=12&audio=0"
ANALYTICS_MODE=ppe-fall

RTSP_TRANSPORT=tcp
RTSP_OPEN_TIMEOUT_MS=20000
RTSP_READ_TIMEOUT_MS=10000
RTSP_SOCKET_TIMEOUT_S=3
RECONNECT_DELAY_S=5

PPE_MODEL_PATH=best_ppe.pt
POSE_MODEL_PATH=yolo26s-pose.pt

YOLO_DEVICE=0
USE_FP16=1
YOLO_TRACKER=bytetrack.yaml

POSE_IMGSZ=512
PPE_IMGSZ=640
POSE_CONF=0.60
PPE_CONF=0.35
IOU_THRESHOLD=0.45
TARGET_INFERENCE_FPS=0

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

SHOW_WINDOW=1
SHOW_TEMPORARY_TRACK_ID=0

OUTPUT_DIR=Resultados
EXCEL_EXPORT_EVERY_EVENTS=10
```

Para `ppe-only`, cambia `ANALYTICS_MODE=ppe-only`; las líneas `POSE_*` y `FALL_*`
pueden permanecer, pero serán ignoradas.

---

# Seguridad de credenciales

El prototipo obsoleto `ppe_reporte.py` contenía anteriormente una credencial RTSP
literal. El valor fue eliminado del árbol actual y ahora ese script exige
`RTSP_URL` desde el entorno, pero la credencial expuesta debe ROTARSE porque puede
permanecer en el historial de Git, copias o registros. Esta corrección no reescribe
el historial. Usa `ppe_reportev2.py` para toda ejecución operativa.
