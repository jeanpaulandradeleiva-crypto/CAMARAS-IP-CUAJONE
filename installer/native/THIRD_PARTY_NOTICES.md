# Avisos de terceros

Este indice identifica los componentes redistribuidos por el instalador interno. No reemplaza sus licencias ni concede derechos adicionales.

| Componente | Archivo incluido | Motivo |
| --- | --- | --- |
| OpenCV 4.12.0 | `licenses/OpenCV-LICENSE.txt` | `opencv_world4120.dll` es una dependencia PE directa. |
| FFmpeg 4.4.6 incluido por OpenCV | `licenses/OpenCV-FFmpeg-LGPL-2.1.txt` y `docs/FFMPEG-SOURCE.md` | `opencv_videoio_ffmpeg4120_64.dll` se carga dinamicamente para captura RTSP/video. La correspondencia binaria y las fuentes fijadas por OpenCV se documentan por hash y revision. |
| libvpx 1.15.2, AOM 3.12.1 y API OpenH264 1.8.0 | `licenses/FFmpeg-dependencies/` | Revisiones fijadas por los scripts oficiales del plugin FFmpeg de OpenCV; conservan sus licencias upstream. |
| Terceros compilados en OpenCV | `licenses/OpenCV-third-party/` | Avisos suministrados por la distribucion oficial de OpenCV. |
| Microsoft ONNX Runtime 1.25.0 CPU | `licenses/ONNX-Runtime-LICENSE.txt` y `licenses/ONNX-Runtime-ThirdPartyNotices.txt` | `onnxruntime.dll` ejecuta modelos ONNX mediante `CPUExecutionProvider`. El asset oficial Windows x64 se fija por URL, versión y SHA-256 en `build-metadata.json`. |
| ByteTrack-Eigen 2.1.0 | `licenses/ByteTrack-Eigen-MIT.txt` | Se enlaza estaticamente desde el commit `a865158906f6138465668810a98ffd918d95f9a3` bajo MIT. El archivo fuente se fija por URL y SHA-256; no se consume el DLL upstream. |
| Eigen 3.4.0 | `licenses/Eigen-MPL-2.0.txt` y `licenses/Eigen-COPYING-README.txt` | Dependencia header-only enlazada estaticamente desde el commit `3147391d946bb4b6c68edd901f2add6ac1f31f8c`. Se define `EIGEN_MPL2_ONLY`; la fuente cubierta por MPL-2.0 debe permanecer disponible al distribuir el ejecutable. |
| NVIDIA CUDA Runtime 12.9 | `licenses/NVIDIA-CUDA-License.txt` | `cudart64_12.dll` es una dependencia PE directa. El suplemento CUDA enumera `cudart.dll` como redistribuible sujeto al acuerdo completo. |
| NVIDIA TensorRT 11.1.0.106 | `licenses/NVIDIA-TensorRT-README.txt` y `licenses/NVIDIA-TensorRT-LICENSE-REFERENCE.txt` | `nvinfer_11.dll` es una dependencia PE directa. El suplemento TensorRT identifica los runtime `.dll` como redistribuibles sujeto al acuerdo completo. |
| Microsoft Visual C++ 2022 Runtime | `licenses/Microsoft-VC-Runtime-REDISTRIBUTION-REFERENCE.txt` | Se incluyen unicamente DLL sin modificar resueltas recursivamente desde `VC/Redist/MSVC/.../x64/Microsoft.VC143.CRT`. |

No se incluye `nvinfer_plugin_11.dll`: el ejecutable no lo importa ni llama `initLibNvInferPlugins`, y la carga de plugins externos no esta implementada. Los engines que dependan de plugins no forman parte del contrato validado.

El inventario de binarios, fuentes estaticas y hashes se incluye además en
`docs/sbom.spdx.json` (SPDX 2.3). Las dependencias de desarrollo/QA `jsonschema`, `pybind11`, Supervision y
`cvat-sdk` se documentan en `LICENSES.md`, pero no se redistribuyen en este MSI.

## Estado

El codigo original del proyecto se publica como `AGPL-3.0-only`; este cambio no
relicencia componentes de terceros. El preview interno puede permanecer sin firma
y sin archivo fuente publicado. Una release externa queda bloqueada hasta contar
con firma Authenticode confiable y acceso equivalente al archivo fuente
correspondiente de FFmpeg/OpenCV, identificado por URL HTTPS y SHA-256. Consulta
`docs/SOURCE-OFFER.txt` y `docs/FFMPEG-SOURCE.md`.
