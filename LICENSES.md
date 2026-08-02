# Alcance de licencias

El codigo fuente original de este proyecto se distribuye exclusivamente bajo
la **GNU Affero General Public License, version 3** (`AGPL-3.0-only`). El texto
completo esta en [`LICENSE`](LICENSE).

La copia se obtuvo de <https://www.gnu.org/licenses/agpl-3.0.txt>. Su SHA-256 es
`0d96a4ff68ad6d4b6f1f30f713b18d5184912ba8dd389f86aa7710db079abcb0`
y se verifica byte por byte durante la revision de release.

## Material con terminos separados

- Las dependencias y los artefactos de terceros conservan sus propias licencias,
  avisos y condiciones de redistribucion. La AGPL del proyecto no los relicencia.
- Los pesos de modelos, incluidos los derivados de Ultralytics, y los datasets
  no se relicencian por este proyecto. Deben usarse conforme a sus terminos de
  origen y procedencia aplicables.
- Los reportes, evidencias, configuraciones y datos aportados o generados por
  usuarios no se convierten automaticamente en software bajo AGPL. Sus derechos
  y obligaciones dependen de su contenido, origen y normativa aplicable.
- No se concede ningun derecho sobre marcas, nombres comerciales o logotipos.

Los avisos del instalador nativo se mantienen en
[`installer/native/THIRD_PARTY_NOTICES.md`](installer/native/THIRD_PARTY_NOTICES.md).
Este documento describe el limite de licencia del repositorio; no sustituye
asesoramiento legal.

El runtime nativo redistribuye ONNX Runtime 1.25.0 CPU bajo MIT junto con sus
avisos de terceros. El paquete MSI conserva la licencia upstream, registra el
asset oficial por URL y SHA-256, e inventaría `onnxruntime.dll` en un SBOM SPDX
2.3. ONNX Runtime no relicencia los modelos ONNX aportados por el operador.

El runtime nativo enlaza estaticamente ByteTrack-Eigen 2.1.0 bajo MIT, fijado al
commit `a865158906f6138465668810a98ffd918d95f9a3`, y Eigen 3.4.0, fijado al
commit `3147391d946bb4b6c68edd901f2add6ac1f31f8c`. Eigen se compila con
`EIGEN_MPL2_ONLY` y conserva MPL-2.0. El instalador incluye ambos textos de
licencia, sus hashes de archivo fuente y relaciones `STATIC_LINK` en el SBOM. Una
distribucion de Eigen en forma ejecutable debe mantener disponible el codigo
fuente cubierto por MPL-2.0 mediante un medio razonable y oportuno.

## Runtime Python de QA y herramientas opcionales

`jsonschema` 4.25.1 (MIT) es una dependencia requerida del runtime Python de QA
porque `cuajone_qa` valida contratos en sus fronteras de entrada. `pybind11` 3.0.4
(BSD-3-Clause) se usa para construir el binding de desarrollo. Supervision 0.29.1
(MIT) y `cvat-sdk` 2.71.0 (MIT) son extras opcionales con imports diferidos. Todos
conservan sus licencias upstream; ninguno de estos paquetes, sus runtimes ni sus
archivos generados se redistribuye en el runtime nativo o el MSI.
