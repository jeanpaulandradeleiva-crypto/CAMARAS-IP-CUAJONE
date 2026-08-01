# Contrato v1 de eventos y evidencias del operador

La salida autoritativa de producción es la generada por `cuajone_native.exe` desde
el MSI. El harness Python de QA conserva los mismos nombres y columnas para que las
herramientas offline puedan consumir el CSV sin mantener otro esquema.

## Salida

Dentro de la carpeta seleccionada como salida:

```text
<output>/
  Reporte_Eventos_Seguridad.csv
  Evidencias/
    <camara>_<tipo>_<YYYYMMDD_HHMMSS_mmm>_<ultimos-8-del-evento>.jpg
```

El CSV es UTF-8, append-only y usa exactamente este encabezado:

```csv
Evento_ID,Camara,Fecha,Hora,Tipo_Evento,Casco,Chaleco,Estado_EPP,Confianza_Evento,ID_Seguimiento_Temporal,Estado_Revision,Identificacion_Humana,Observaciones_Revision,Foto
```

## Semántica

| Campo | Regla v1 |
| --- | --- |
| `Evento_ID` | ID estable del evento canónico. |
| `Camara` | Etiqueta no secreta configurada para la fuente. |
| `Fecha`, `Hora` | Timestamp UTC canónico dividido como `YYYY-MM-DD` y `HH:MM:SS.mmm`. |
| `Tipo_Evento` | `INCUMPLIMIENTO_EPP` o `POSIBLE_CAIDA`. |
| `Casco`, `Chaleco` | `SI`, `NO` o `N/D`, derivados de `Estado_EPP`. |
| `Estado_EPP` | Estado EPP del evento; `Evaluating PPE` se presenta como `En evaluación`. |
| `Confianza_Evento` | Confianza redondeada a tres decimales como máximo. |
| `ID_Seguimiento_Temporal` | Correlación temporal; no identifica a una persona. |
| `Estado_Revision` | Comienza en `PENDIENTE`. |
| `Identificacion_Humana`, `Observaciones_Revision` | Comienzan vacíos para revisión autorizada. |
| `Foto` | Ruta del JPEG anotado que se escribió antes de anexar la fila. |

La derivación EPP es fija:

| `Estado_EPP` | `Casco` | `Chaleco` |
| --- | --- | --- |
| `EPP Completo` | `SI` | `SI` |
| `Falta Chaleco` | `SI` | `NO` |
| `Falta Casco` | `NO` | `SI` |
| `Sin Casco y Chaleco` | `NO` | `NO` |
| Otro estado | `N/D` | `N/D` |

Los componentes del nombre JPEG se sanejan para el sistema de archivos. Un
timestamp que no sea RFC 3339 UTC terminado en `Z`, una fecha imposible o un tipo
canónico desconocido rechazan el evento antes de escribir imagen o CSV.

## Persistencia

El runtime escribe el JPEG mediante un temporal, solicita persistencia física del
archivo y lo renombra antes de abrir el CSV. Solo entonces anexa una fila con escape
CSV de comas, comillas y saltos de línea. En Windows solicita `FlushFileBuffers`
para la imagen y para cada append del CSV. Un fallo de imagen no crea encabezado ni
fila parcial.

No existe una transacción única entre JPEG y CSV. Si el JPEG termina correctamente
y después falla el append, puede quedar una imagen huérfana que soporte debe
investigar. Tampoco se ofrece coordinación entre varios procesos que escriban el
mismo CSV.

## XLSX y migración

El runtime instalado por MSI no genera XLSX ni incluye una biblioteca Excel. El
archivo `Reporte_Eventos_Seguridad.xlsx` es una conveniencia local/offline del
harness Python para revisión humana y no es una salida de producción.

Las salidas nativas anteriores `native_events.csv` y `evidence/` se conservan sin
mover, borrar ni convertir. La versión v1 comienza a escribir
`Reporte_Eventos_Seguridad.csv` y `Evidencias/` al lado de ellas. No concatenes el
CSV antiguo con el nuevo: sus columnas tienen otra semántica. Archívalo o conviértelo
offline con trazabilidad si el propietario de los datos lo requiere.
