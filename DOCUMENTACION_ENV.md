# Documentación de configuración (`.env`)

Este archivo explica las variables de entorno usadas por `ppe_reporte.py` y cómo modificarlas para realizar pruebas.

> El archivo `.env` debe estar en la misma carpeta que `ppe_reporte.py`.  
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
|---|---|
| `@` | `%40` |
| `#` | `%23` |
| `%` | `%25` |
| `:` | `%3A` |
| `/` | `%2F` |
| `?` | `%3F` |
| `&` | `%26` |

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
YOLO_DEVICE=auto
```

- `0`: primera GPU CUDA.
- `cpu`: procesador.
- `auto`: selección automática, si el código lo admite.

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

# Validación de personas

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

# Detección de caídas

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

## `EXCEL_EXPORT_INTERVAL_S`

Frecuencia de actualización del Excel.

```dotenv
EXCEL_EXPORT_INTERVAL_S=10
```

El CSV puede escribirse inmediatamente y el Excel cada cierto tiempo para evitar bloqueos y sobrecarga.

Pruebas:

```dotenv
EXCEL_EXPORT_INTERVAL_S=5
EXCEL_EXPORT_INTERVAL_S=10
EXCEL_EXPORT_INTERVAL_S=30
```

---

## `JPEG_QUALITY`

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

| Prueba | Variable | Valor | FPS | Latencia | Falsos positivos | Observaciones |
|---|---|---:|---:|---:|---:|---|
| 1 | `POSE_CONF` | 0.55 |  |  |  |  |
| 2 | `POSE_CONF` | 0.60 |  |  |  |  |
| 3 | `POSE_CONF` | 0.65 |  |  |  |  |

No cambies simultáneamente resolución, confianza, transporte y modelo, porque no podrás identificar qué cambio mejoró o empeoró el sistema.

---

# Plantilla `.env` recomendada

```dotenv
CAMERA_ID=CAM_CUAJONE_01
RTSP_URL="rtsp://usuario:clave@172.19.90.72:554/axis-media/media.amp?videocodec=h264&resolution=1280x720&fps=12&audio=0"

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

MIN_KEYPOINT_CONF=0.45
MIN_VALID_KEYPOINTS=7
REQUIRE_PPE_PERSON_CONFIRMATION=1
PERSON_IOU_THRESHOLD=0.30
PERSON_HISTORY_FRAMES=10
PERSON_MIN_CONFIRMATIONS=7
EXCLUSION_ZONES=

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
EXCEL_EXPORT_INTERVAL_S=10
JPEG_QUALITY=92
```
