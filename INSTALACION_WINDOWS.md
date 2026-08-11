# Instalar NexoAI Vision en Windows

Esta guía es para la persona que usará la aplicación. No necesitas conocer
PowerShell ni escribir comandos.

## Antes de empezar

1. Obtén el archivo MSI aprobado por tu organización.
2. Pregunta a TI si el equipo ya está preparado para confiar en el instalador.
3. No continúes si Windows indica que no confía en el instalador. Contacta a TI.

## Instalar

1. Haz doble clic en el archivo MSI.
2. Si Windows pide permiso para hacer cambios, confirma solo si TI aprobó el
   instalador.
3. Deja la carpeta propuesta o selecciona **Browse** para elegir otra.

    La carpeta propuesta es `C:\Program Files\NexoAI Vision`.

4. Elige cómo procesará la aplicación:

   - **Auto (recomendado):** usa GPU NVIDIA si está lista; en caso contrario usa CPU.
   - **GPU (CUDA):** requiere GPU compatible y driver suficientemente reciente; la instalación se detiene si no están listos.
   - **CPU:** no requiere GPU NVIDIA.

   El instalador no descarga, instala ni cambia drivers.

5. Selecciona **Install**.
6. Cuando termine, selecciona **Finish**.

## Abrir y configurar el monitor

1. Abre el menú Inicio de Windows.
2. Busca **NexoAI Vision**.
3. Abre **NexoAI Vision**. Este es el iniciador gráfico para el uso normal.
4. Completa los campos con los valores aprobados por el responsable de la
   aplicación:

   - URL RTSP de la cámara autorizada (`rtsp://` o `rtsps://`);
   - modo de análisis: EPP solamente o EPP y caídas;
   - modo de cómputo: Auto, GPU o CPU;
   - tamaño de inferencia: 640, 768, 960 o 1280;
   - confianza para cada una de las ocho clases EPP;
   - carpeta donde se guardarán los resultados;
   - si deseas mostrar la ventana de análisis.

5. Revisa la configuración y usa la acción de inicio del mismo formulario. El
   iniciador valida primero la configuración y muestra el error sin cerrar la
   ventana si falta un dato.

Cuando ocurre un evento, la carpeta de salida contiene
`Reporte_Eventos_Seguridad_v2.csv` y las fotos anotadas en `Evidencias/`. El reporte muestra los siete EPP obligatorios y es la salida
de producción que debe conservarse para el procedimiento autorizado de revisión.
La aplicación instalada no crea un archivo Excel. Un XLSX que genere el harness
Python pertenece únicamente a una revisión local/offline de QA.

No escribas una contraseña RTSP en documentos, capturas o tickets. El iniciador no
la guarda como configuración permanente; deberás volver a proporcionarla cuando
corresponda.

**NexoAI Vision - Command Help** permanece en el menú Inicio para soporte y
uso avanzado. No es la forma normal de iniciar el monitor.

## Configurar cámaras después

La instalación y la configuración de cámaras son pasos separados. El instalador
 no agrega una cámara e incluye obligatoriamente el bundle ONNX de EPP y pose.
Después de instalar, el iniciador gráfico permite elegir la cámara, el modo, el
tamaño de inferencia, las confianzas y la salida sin reinstalar la aplicación.

La versión piloto tiene un alcance limitado. El bundle ONNX administrado se instala
junto al runtime y no puede sustituirse desde la interfaz. Idioma, tema, tamaño y
confianzas se guardan por usuario en LocalAppData; las credenciales RTSP permanecen
en Windows Credential Manager. La salida operativa se guarda bajo
`C:\ProgramData\NexoAI Vision`.

Al actualizar desde Cuajone PPE Monitor, NexoAI Vision usa la carpeta nueva para
archivos nuevos. No elimina ni mueve la carpeta anterior, pero tampoco carga modelos
desde la ubicación heredada.

Si una carpeta de salida conserva `native_events.csv` o `evidence/`, son resultados
de una versión nativa anterior. No los borres ni los combines con el CSV nuevo;
solicita a soporte una migración offline si deben conservarse en el flujo actual.

Se espera que versiones futuras permitan agregar más configuraciones de cámara
sin reinstalar la aplicación en otra carpeta. Esto dependerá del soporte del
producto en cada versión; no significa que la versión actual ya use varias
cámaras.

## Si algo falla

| Problema | Qué hacer |
| --- | --- |
| Windows no confía en el instalador | No cambies la seguridad del equipo. Contacta a TI. |
| La instalación falla | Pide a TI que genere el registro MSI y envíaselo. |
| El iniciador indica que falta un campo | Revisa la cámara, el modo y la carpeta de salida. |
| El iniciador indica que falta el bundle administrado | Repara o reinstala el MSI aprobado; no selecciones modelos externos. |
| GPU aparece como no disponible | Elige Auto o CPU y pide a TI revisar hardware/driver por separado. No instales drivers desde el MSI. |

## Para TI

TI debe preparar la confianza del certificado cuando corresponda y verificar el
paquete antes de instalarlo. Los pasos técnicos, el despliegue silencioso, los
registros y el mantenimiento están en la
[guía breve Para TI](PARA_TI_WINDOWS.md). La referencia de construcción y
validación avanzada permanece en
[`installer/native/README.md`](installer/native/README.md).
