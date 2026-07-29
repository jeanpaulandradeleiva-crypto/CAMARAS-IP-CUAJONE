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

### Firma privada para el piloto interno

Esta ruta es **solo para equipos piloto enrolados**. Usa una CA raiz privada y un
certificado hoja separado con EKU Code Signing. Ambas claves privadas permanecen
como no exportables en `Cert:\CurrentUser\My` del usuario firmante; el repositorio
no recibe PFX, PEM, claves, contrasenas ni secretos.

En la maquina de firma:

```powershell
$certs = .\installer\native\New-InternalPilotSigningCertificates.ps1
$env:CUAJONE_CERTIFICATE_SHA1 = $certs.LeafThumbprint
$env:CUAJONE_PILOT_ROOT_CER = $certs.RootPublicCertificate
$env:CUAJONE_ALLOW_INTERNAL_PILOT_TRUST = "1"
$env:CUAJONE_SIGNTOOL_PATH = "C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64\signtool.exe"
$env:CUAJONE_TIMESTAMP_URL = "http://timestamp.acs.microsoft.com"
$env:CUAJONE_PE_SIGN_COMMAND = (Resolve-Path .\installer\native\sign-release.ps1).Path
$env:PATH = "$(Split-Path $env:CUAJONE_SIGNTOOL_PATH);$env:PATH"
$env:CUAJONE_SIGNTOOL_NAME = "CuajonePilotSign"
$env:CUAJONE_SIGNTOOL_COMMAND = "signtool.exe sign /v /fd SHA256 /tr $env:CUAJONE_TIMESTAMP_URL /td SHA256 /sha1 $env:CUAJONE_CERTIFICATE_SHA1 `$f"
.\installer\native\build-installer.ps1 -BuildMode Preview
```

El ajuste de `PATH` anterior afecta solo al proceso PowerShell actual. Permite que
Inno firme setup y desinstalador sin modificar el `PATH` global.

La creacion es idempotente: reutiliza solo certificados validos que coincidan con
la identidad piloto, la cadena y la politica no exportable. Ante ambiguedad,
expiracion proxima o un archivo CER distinto, falla sin reemplazar nada. Solo
exporta estos certificados publicos bajo `D:\DevTools\CuajoneNative\signing`:

- `Cuajone-PPE-Monitor-Internal-Pilot-Root-CA-2026.cer`
- `Cuajone-PPE-Monitor-Internal-Pilot-Code-Signing-2026.cer`

### Instalación del piloto firmado en un equipo objetivo

Este procedimiento requiere autorización previa del equipo de TI/seguridad del
cliente.

**1. Obtén y copia el paquete aprobado.** El operador debe disponer de:

- el instalador, los dos CER públicos y los valores SHA-256 y huellas digitales
  aprobados, obtenidos por un canal autorizado;
- una copia del repositorio o, como mínimo,
  `installer/native/Install-InternalPilotTrust.ps1` junto con ambos CER;
- una cuenta con permisos administrativos. Incluso `-ValidateOnly` requiere una
  sesión de PowerShell elevada.

Los valores esperados deben llegar por el medio aprobado por seguridad, de forma
independiente de los archivos.

**2. Abre PowerShell como Administrador y verifica el paquete** antes de modificar
los almacenes:

```powershell
$packageDir = "D:\<RUTA_PAQUETE_PILOTO>"
$installer = Join-Path $packageDir "CuajonePPEMonitor-0.1.0-internal.3-x64-Internal-Setup.exe"
$rootCer = Join-Path $packageDir "Cuajone-PPE-Monitor-Internal-Pilot-Root-CA-2026.cer"
$leafCer = Join-Path $packageDir "Cuajone-PPE-Monitor-Internal-Pilot-Code-Signing-2026.cer"
$trustScript = "D:\<RUTA_SCRIPT>\Install-InternalPilotTrust.ps1"

$expectedHashes = @(
    [pscustomobject]@{ Path = $installer; SHA256 = "<SHA256_INSTALADOR>" }
    [pscustomobject]@{ Path = $rootCer; SHA256 = "<SHA256_CER_RAIZ>" }
    [pscustomobject]@{ Path = $leafCer; SHA256 = "<SHA256_CER_HOJA>" }
)
foreach ($item in $expectedHashes) {
    $actual = (Get-FileHash -LiteralPath $item.Path -Algorithm SHA256).Hash
    if ($actual -cne $item.SHA256.ToUpperInvariant()) {
        throw "SHA-256 no coincide: $($item.Path)"
    }
}

$expectedRootThumbprint = "<HUELLA_SHA1_RAIZ_SIN_ESPACIOS>".ToUpperInvariant()
$expectedLeafThumbprint = "<HUELLA_SHA1_HOJA_SIN_ESPACIOS>".ToUpperInvariant()
$rootCertificate = [System.Security.Cryptography.X509Certificates.X509Certificate2]::new($rootCer)
$leafCertificate = [System.Security.Cryptography.X509Certificates.X509Certificate2]::new($leafCer)
try {
    if ($rootCertificate.Thumbprint -cne $expectedRootThumbprint) { throw "Huella raíz no coincide" }
    if ($leafCertificate.Thumbprint -cne $expectedLeafThumbprint) { throw "Huella hoja no coincide" }
}
finally {
    $rootCertificate.Dispose()
    $leafCertificate.Dispose()
}
```

**3. Valida los certificados sin importarlos:**

```powershell
& $trustScript -RootCertificatePath $rootCer -LeafCertificatePath $leafCer -ValidateOnly
```

**4. Realiza el enrolamiento:**

```powershell
& $trustScript -RootCertificatePath $rootCer -LeafCertificatePath $leafCer
```

El script valida identidad, uso y cadena, importa solo la raíz pública a
`LocalMachine\Root` y la hoja pública a `LocalMachine\TrustedPublisher`, y muestra
las huellas enroladas.

**5. Verifica la firma y el firmante antes de ejecutar:**

```powershell
$signature = Get-AuthenticodeSignature -LiteralPath $installer
if ($signature.Status -ne "Valid") { throw "Firma Authenticode no válida: $($signature.Status)" }
if ($signature.SignerCertificate.Thumbprint -cne $expectedLeafThumbprint) {
    throw "El firmante no coincide con la hoja piloto aprobada"
}
```

**6. Ejecuta el instalador:**

```powershell
Start-Process -FilePath $installer -Wait
```

**7. Realiza la comprobación posterior mínima** con la ruta seleccionada durante
la instalación:

```powershell
& "D:\<RUTA_INSTALACION>\bin\cuajone_native.exe" --help
```

No ejecutes `--preflight`, no abras cámaras y no inicies inferencia como parte de
esta validación.

El enrolamiento solo establece confianza en ese publicador para los equipos
enrolados; **no otorga confianza pública**. Tampoco garantiza silencio del
antivirus/EDR, suprime alertas del SOC, evita SmartScreen o Smart App Control, ni
invalida políticas de AppLocker, WDAC o Defender. Seguridad debe aprobar o incluir
en allowlist la instalación y el comportamiento esperado de procesos y red,
preferentemente por el certificado hoja o por el hash del artefacto cuando su
herramienta lo permita. No se deben crear exclusiones amplias ni desactivar
controles de seguridad.

#### Retiro del piloto y de la confianza

1. Desinstala **Cuajone PPE Monitor** desde Aplicaciones instaladas o mediante el
   desinstalador aprobado.
2. Confirma con TI/seguridad que ningún otro despliegue piloto en esta máquina
   depende de estos certificados.
3. En PowerShell como Administrador, inspecciona y elimina únicamente las huellas
   exactas aprobadas:

```powershell
$leafThumbprint = "<HUELLA_SHA1_HOJA_SIN_ESPACIOS>"
$rootThumbprint = "<HUELLA_SHA1_RAIZ_SIN_ESPACIOS>"

Get-Item -LiteralPath "Cert:\LocalMachine\TrustedPublisher\$leafThumbprint"
Get-Item -LiteralPath "Cert:\LocalMachine\Root\$rootThumbprint"

Remove-Item -LiteralPath "Cert:\LocalMachine\TrustedPublisher\$leafThumbprint"
Remove-Item -LiteralPath "Cert:\LocalMachine\Root\$rootThumbprint"
```

No elimines otros certificados y no retires la raíz mientras otro despliegue
autorizado todavía dependa de ella.

`CUAJONE_ALLOW_INTERNAL_PILOT_TRUST=1` es un opt-in explicito y esta desactivado
por defecto. `build-installer.ps1` lo admite solo con `-BuildMode Preview`; un
`Release` publico sigue fallando cerrado y exige confianza publica. Fuera de los
equipos enrolados, esta firma privada no es confiable y no elimina advertencias de
SmartScreen.

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

1. Configura un certificado publico confiable o Trusted Signing; la CA privada anterior es exclusiva del piloto y nunca es una credencial de produccion.
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
actual sigue siendo un preview interno sin validacion con engines reales. Una
firma privada piloto, si se aplica, solo es confiable en equipos enrolados. La
distribucion externa permanece bloqueada hasta publicar fuente exacta
del proyecto y del plugin FFmpeg, usar Authenticode confiable y validar la licencia
aplicable del compilador Inno Setup instalado, que informa `Non-commercial use
only`. Nada de esto convierte los modelos, datasets ni DLL de terceros a AGPL.

La firma mejora trazabilidad y reputacion, pero no garantiza cero alertas de
Defender o SmartScreen. No se recomiendan exclusiones generales de antivirus.

Consulta [`FFMPEG-SOURCE.md`](FFMPEG-SOURCE.md) para la correspondencia de fuente
y [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) para el indice de terceros.
