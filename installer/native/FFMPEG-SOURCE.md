# Correspondencia de fuente FFmpeg de OpenCV 4.12.0

El instalador incluye `opencv_videoio_ffmpeg4120_64.dll` del paquete oficial
Windows de OpenCV 4.12.0. Esta referencia registra la procedencia comprobada del
binario y lo que debe publicarse junto a una release externa.

## Identidad comprobada

| Evidencia | Valor |
| --- | --- |
| Paquete | `opencv-4.12.0-windows.exe` de la release oficial de OpenCV 4.12.0 |
| SHA-256 publicado del paquete | `b753b14d880b9bc8d89d6acd3b665c040baec0211078435432fcae117db707af` |
| DLL | `opencv_videoio_ffmpeg4120_64.dll` |
| SHA-256 local | `a0f01e4ee5e97b4a513cd70f01fafadc0dd187ba5d1293cb7fc6b77e7d17c631` |
| MD5 local y esperado por OpenCV | `e5c6936240201064b15bcecf1816e8f4` |
| Version PE | `2025.06.0` |
| Commit de binarios OpenCV | `ea9240e39bc0d6a69d2b1f0ba4513bdc7612a41e` |
| Rama documentada | `ffmpeg/4.x_20250625` |
| Snapshot OpenCV usado para el wrapper | `e9f1da7e8e977a65b8bf8fe7ea8b92eef9171f19` |

El `ffmpeg_version.cmake` de ese commit declara `avcodec 58.134.100`,
`avformat 58.76.100`, `avutil 56.70.100`, `swscale 5.9.100` y
`avresample 4.0.0`. La salida oficial `opencv_version --verbose` del paquete
instalado coincide con esos valores.

## Fuentes fijadas por el build oficial

El script `ffmpeg/download_src.sh` del commit de binarios fija:

| Componente | Revision |
| --- | --- |
| FFmpeg | tag `n4.4.6`, commit `44b04492bfc83215e136f2a68783bff71d328692` |
| OpenH264 API | tag `v1.8.0` |
| libvpx | tag `v1.15.2` |
| AOM | tag `v3.12.1` |

Referencias oficiales:

- OpenCV 4.12.0: <https://github.com/opencv/opencv/releases/tag/4.12.0>
- Binario y scripts exactos: <https://github.com/opencv/opencv_3rdparty/tree/ea9240e39bc0d6a69d2b1f0ba4513bdc7612a41e/ffmpeg>
- Wrapper OpenCV exacto: <https://github.com/opencv/opencv/tree/e9f1da7e8e977a65b8bf8fe7ea8b92eef9171f19/modules/videoio>
- FFmpeg: <https://github.com/FFmpeg/FFmpeg/tree/n4.4.6>
- OpenH264: <https://github.com/cisco/openh264/tree/v1.8.0>
- libvpx: <https://chromium.googlesource.com/webm/libvpx/+/refs/tags/v1.15.2>
- AOM: <https://aomedia.googlesource.com/aom/+/refs/tags/v3.12.1>

## Obligacion para una release externa

Una lista de enlaces no se trata como sustituto automatico de las obligaciones
LGPL. Antes de publicar el instalador se debe crear y conservar un archivo fuente
correspondiente que contenga las revisiones anteriores, los archivos del wrapper
OpenCV, sus licencias y los scripts/configuracion usados para generar el DLL. Ese archivo debe
publicarse por HTTPS junto a la release, con SHA-256 registrado en
`SOURCE-OFFER.txt` y acceso equivalente al binario durante el plazo aplicable.

El modo `Release` de `build-installer.ps1` falla si no recibe URL y SHA-256 del
archivo fuente FFmpeg. El preview interno puede documentar la ausencia, pero no
autoriza distribucion externa.
