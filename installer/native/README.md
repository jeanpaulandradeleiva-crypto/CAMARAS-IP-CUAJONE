# Instalador Windows del runtime nativo

Este limite produce un instalador Windows x64 del proyecto abierto bajo AGPL-3.0. Empaqueta el ejecutable Release, sus DLL runtime demostradas, la licencia, el ofrecimiento de fuente y los avisos disponibles. No incluye engines, modelos, credenciales, SDK ni datos.

> Alcance validado: carga del ejecutable y `--help`. No existen engines EPP/pose reales aprobados, por lo que este instalador no demuestra inferencia operativa.

## Ruta rapida

Desde PowerShell en la raiz del repositorio:

```powershell
.\installer\native\build-installer.ps1 -BuildMode Preview -AllowUnsignedPreview

$installer = "D:\DevTools\CuajoneNative\installer\output\CuajonePPEMonitor-0.1.0-internal.3-x64-Internal-Setup.exe"
.\installer\native\test-installer.ps1 -InstallerPath $installer
```

Staging, salida, logs y aceptacion se crean bajo `D:\DevTools\CuajoneNative\installer`. El script no instala dependencias ni modifica `PATH` global.

## Contrato del paquete

| Ruta instalada | Politica |
| --- | --- |
| `bin/` | Ejecutable y DLL inmutables. SYSTEM y Administrators tienen FullControl; Users tiene Read/Execute. |
| `runtime/models/` | Engines externos persistentes. El instalador nunca los aporta. |
| `runtime/config/` | Reservado para material del operador. El CLI actual no carga archivos de configuracion. |
| `runtime/output/` | Evidencias y salidas persistentes. |
| `runtime/logs/` | Registros persistentes. |
| `docs/`, `licenses/`, `manifest/` | Guia, AGPL, avisos, ofrecimiento de fuente, hashes y metadatos de construccion. |

La instalacion es exclusivamente por maquina y requiere elevacion administrativa;
ya no existe override por usuario. El destino sigue siendo seleccionable. Para
aceptacion se usa una ruta explicita en D.

El instalador reemplaza ACL heredadas mediante SIDs conocidos. `bin/` queda limitado
a SYSTEM y Administrators con FullControl y Users con Read/Execute. Las cuatro
carpetas `runtime/` conceden Modify a Users y se conservan en upgrades y
desinstalaciones. Si una operación ACL falla, la instalación falla. La eliminación
manual es intencional para evitar pérdida de engines, configuración, evidencias o
logs.

## Dependencias

`build-installer.ps1` usa el `dumpbin.exe` oficial de MSVC para recorrer imports PE. Solo resuelve DLL desde OpenCV, CUDA Runtime, TensorRT Runtime y el directorio oficial `VC/Redist` x64. El plugin FFmpeg de OpenCV se agrega explicitamente porque `videoio` lo carga por nombre, no mediante la tabla PE.

`nvinfer_plugin_11.dll`, parsers ONNX y recursos de builder se rechazan. El runtime actual no inicializa ni carga plugins TensorRT; un engine que los requiera no esta soportado por este paquete.

## Firma y version

El modo predeterminado es `Release` y falla cerrado si falta firma confiable,
fuente exacta o correspondencia con un worktree limpio. Solo
`-BuildMode Preview -AllowUnsignedPreview` permite intencionalmente un artefacto
interno `NotSigned`.

`sign-release.ps1` firma unicamente PE propios mediante Microsoft `signtool`.
Admite un certificado del almacen de Windows o Microsoft Trusted Signing. No
acepta PFX ni contrasenas y no vuelve a firmar DLL de terceros.

Ejemplo con certificado del almacen:

```powershell
$env:CUAJONE_SIGNTOOL_PATH = "C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64\signtool.exe"
$env:CUAJONE_TIMESTAMP_URL = "http://timestamp.acs.microsoft.com"
$env:CUAJONE_CERTIFICATE_SHA1 = "<HUELLA_SHA1_DEL_CERTIFICADO_CONFIABLE>"
$env:CUAJONE_PE_SIGN_COMMAND = (Resolve-Path .\installer\native\sign-release.ps1).Path
.\installer\native\sign-release.ps1 -FilePath D:\ruta\cuajone_native.exe
```

Para Trusted Signing, define `CUAJONE_TRUSTED_SIGNING_DLIB` y
`CUAJONE_TRUSTED_SIGNING_METADATA` en lugar de la huella. La autenticacion se
resuelve fuera del repositorio mediante el proveedor aprobado.

El build llama primero a `CUAJONE_PE_SIGN_COMMAND` con la ruta del ejecutable
propio. Luego Inno usa `CUAJONE_SIGNTOOL_NAME` y
`CUAJONE_SIGNTOOL_COMMAND` para firmar setup y desinstalador. El comando Inno
debe usar `$f` como marcador de archivo y mantener `/fd SHA256`, un `/tr` RFC
3161 aprobado y `/td SHA256`.

Lista de firma:

1. Configura un certificado publico confiable o Trusted Signing; nunca generes un certificado autofirmado de produccion.
2. Firma `cuajone_native.exe` antes del staging mediante `CUAJONE_PE_SIGN_COMMAND`.
3. Configura el SignTool de Inno para setup y `SignedUninstaller`.
4. Ejecuta el build `Release` con revision, URL y SHA-256 de los archivos fuente del proyecto y FFmpeg.
5. Verifica ejecutable, instalador y desinstalador con `signtool verify /pa /all /v`.
6. Confirma que las DLL OpenCV, FFmpeg, NVIDIA y MSVC conservan sus bytes y firmas upstream.

Ejemplo de variables de correspondencia de fuente:

```powershell
$env:CUAJONE_SOURCE_REVISION = git rev-parse HEAD
$env:CUAJONE_SOURCE_ARCHIVE_URL = "https://github.com/jeanpaulandradeleiva-crypto/CAMARAS-IP-CUAJONE/releases/download/v0.1.0/source.tar.gz"
$env:CUAJONE_SOURCE_ARCHIVE_SHA256 = "<SHA256_DE_64_CARACTERES>"
$env:CUAJONE_FFMPEG_SOURCE_ARCHIVE_URL = "https://github.com/jeanpaulandradeleiva-crypto/CAMARAS-IP-CUAJONE/releases/download/v0.1.0/opencv-ffmpeg-sources.tar.xz"
$env:CUAJONE_FFMPEG_SOURCE_ARCHIVE_SHA256 = "<SHA256_DE_64_CARACTERES>"
```

El certificado, la identidad de Trusted Signing, los comandos reales y el
servidor de timestamp deben ser aprobados por la organizacion. `AppVersion`
acepta SemVer; `FileVersion` debe tener cuatro componentes numericos.
El setup se compila primero como `CuajonePPEMonitorSetup.exe` para conservar ese
`OriginalFilename` PE exacto y luego se renombra al artefacto largo versionado.

## Instalacion y operacion

El instalador exige Windows x64 y bloquea si no detecta `nvcuda.dll` y una ejecucion correcta de `nvidia-smi -L`. No descarga drivers ni realiza llamadas de red.

Coloca los dos engines compatibles manualmente en `runtime/models/`. Ejecuta el binario mediante CLI con `--ppe-engine`, `--pose-engine`, `--source` y `--output`; no existe un cargador de config y no debe inventarse uno. Consulta el acceso directo **Command Help**. No se inicia monitor, camara ni inferencia automaticamente.

## Upgrade, desinstalacion y rollback

El `AppId` es estable, por lo que una version nueva actualiza la misma aplicacion. Los binarios se reemplazan; `runtime/` se conserva.

Para rollback, reinstala un instalador interno anterior con el mismo `AppId` y verifica nuevamente `--help`. Para retirar el producto, usa el desinstalador: elimina binarios, accesos y metadatos, pero conserva `runtime/`. Borra esa carpeta manualmente solo despues de respaldar y aprobar la eliminacion.

## Aceptacion en host limpio

1. Verifica SHA-256 y Authenticode antes de ejecutar.
2. Instala en una ruta D seleccionada y sin engines.
3. Ejecuta `bin\cuajone_native.exe --help` con `PATH` limitado a `bin` y Windows.
4. Confirma las cuatro carpetas mutables y revisa ACL.
5. Desinstala silenciosamente.
6. Confirma que `bin/` desaparecio y que `runtime/` permanece.

No ejecutes `--preflight`: selecciona CUDA y deserializa engines. Tampoco abras camaras ni ejecutes inferencia durante esta aceptacion.

## Distribucion

Estado de licencia del proyecto: **codigo abierto AGPL-3.0-only**. El artefacto
actual sigue siendo un preview interno sin firma y sin validacion con engines
reales. La distribucion externa permanece bloqueada hasta publicar fuente exacta
del proyecto y del plugin FFmpeg, usar Authenticode confiable y validar la licencia
aplicable del compilador Inno Setup instalado, que informa `Non-commercial use
only`. Nada de esto convierte los modelos, datasets ni DLL de terceros a AGPL.

La firma mejora trazabilidad y reputacion, pero no garantiza cero alertas de
Defender o SmartScreen. No se recomiendan exclusiones generales de antivirus.

Consulta [`FFMPEG-SOURCE.md`](FFMPEG-SOURCE.md) para la correspondencia de fuente
y [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) para el indice de terceros.
