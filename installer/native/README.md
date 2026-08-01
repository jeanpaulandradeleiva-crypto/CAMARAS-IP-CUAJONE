# Referencia avanzada: MSI Windows del runtime nativo

Esta es la referencia de ingeniería para construir, firmar y validar el MSI x64,
la única distribución de producción aprobada. La ejecución normal continúa desde
NexoAI Vision hacia `cuajone_native.exe`; no existe fallback Python.
Para el piloto operativo usa primero el
[runbook breve Para TI](../../PARA_TI_WINDOWS.md). La persona usuaria debe seguir
la [guía simple de instalación](../../INSTALACION_WINDOWS.md).

## Secuencia recomendada

1. Confirma el alcance y la autorización de seguridad.
2. Prepara la confianza privada del piloto, si corresponde.
3. Verifica los hashes, los certificados y la firma antes de instalar.
4. Instala con interfaz o en modo silencioso y conserva el registro MSI.
5. Comprueba que el iniciador gráfico abre, sin configurar cámaras ni modelos.
6. Usa Windows Installer para reparar, actualizar, desinstalar o volver a una
   versión anterior.

## 1. Confirmar el alcance

El MSI está construido con WiX Toolset. Instala el iniciador gráfico
`cuajone_launcher.exe`, el runtime de consola `cuajone_native.exe`, las DLL runtime
demostradas, licencias, avisos, hashes y procedencia. Por defecto incluye el bundle
opcional de modelos ONNX para EPP y pose; no incluye credenciales, SDK, configuración
de cámaras ni datos operativos.

También falla cerrado si staging o extracción contienen Python, `.py`, `.pyc`,
`.pyd`, runtime Python, PyTorch, Ultralytics, CVAT, Supervision, datasets, fixtures,
JSONL, recibos de paridad o modelos/engines fuera de la política explícita del
instalador. Esa política permite únicamente el bundle ONNX aprobado y, cuando se
configuran expresamente, los engines TensorRT aprobados en las rutas de modelos
permitidas; cualquier otro modelo o engine se rechaza. El binding y `cuajone_qa`
son solo de desarrollo/QA.

El instalador no fija el producto a una cámara. La configuración ocurre después y
los datos mutables permanecen bajo `C:\ProgramData\NexoAI Vision`, separados
de los binarios instalados. El piloto actual tiene alcance limitado; no se afirma
soporte runtime para varias cámaras.

> Alcance validado: base MSI, extracción administrativa, subsistemas PE del
> iniciador/runtime, carga del runtime, `--help` y probe de hardware. No se ejecutaron
> instalación real, iniciador, cámaras, inferencia, engines ni `--preflight` en el
> host de desarrollo.

## 2. Preparar la confianza privada del piloto

La confianza privada es un prerrequisito independiente y requiere autorización de
TI o seguridad. **El MSI no instala certificados, raíces ni publicadores
confiables.** Tampoco desactiva Defender, EDR, SOC, SmartScreen, AppLocker o WDAC.

El script autorizado enrola únicamente la raíz pública en
`LocalMachine\Root` y la hoja pública en `LocalMachine\TrustedPublisher`.

Antes de continuar, TI debe obtener por canales aprobados e independientes:

- el MSI y los certificados públicos raíz y hoja;
- los valores SHA-256 del MSI y de ambos certificados;
- las huellas esperadas de ambos certificados;
- `Install-Pilot.ps1` e `Install-InternalPilotTrust.ps1`;
- autorización para enrolar la confianza en el equipo.

La firma mejora la trazabilidad, pero no garantiza que antivirus, EDR, SOC o
SmartScreen dejen de generar alertas. No se deben desactivar controles ni crear
exclusiones amplias.

## 3. Verificar e instalar el piloto

Desde la raíz del repositorio, abre PowerShell como administrador.
`Install-Pilot.ps1` valida primero los hashes, las huellas y la cadena sin importar
certificados. Solo después del parámetro explícito `-AuthorizeTrustEnrollment`
enrola la confianza, verifica el firmante exacto del MSI y llama a Windows
Installer.

```powershell
.\installer\native\Install-Pilot.ps1 `
  -MsiPath "D:\Paquete\NexoAIVision-0.1.0-internal.3-x64-Internal.msi" `
  -RootCertificatePath "D:\Paquete\Cuajone-PPE-Monitor-Internal-Pilot-Root-CA-2026.cer" `
  -LeafCertificatePath "D:\Paquete\Cuajone-PPE-Monitor-Internal-Pilot-Code-Signing-2026.cer" `
  -ExpectedMsiSha256 "<SHA256_MSI>" `
  -ExpectedRootCertificateSha256 "<SHA256_CER_RAIZ>" `
  -ExpectedLeafCertificateSha256 "<SHA256_CER_HOJA>" `
  -ExpectedRootThumbprint "<HUELLA_SHA1_RAIZ>" `
  -ExpectedLeafThumbprint "<HUELLA_SHA1_HOJA>" `
  -InstallFolder "D:\Apps\NexoAI Vision" `
  -ComputeMode auto `
  -LogPath "D:\Logs\cuajone-install.log" `
  -AuthorizeTrustEnrollment
```

Los valores esperados deben proceder de un canal independiente del paquete. Una
ejecución por doble clic no puede expresar de forma segura esta autorización ni
esos valores.

## 4. Elegir la forma de instalación

### Instalación interactiva

Después de que TI verifique y prepare el equipo, abre el MSI. La interfaz incluye
**Browse** para elegir el destino. El valor predeterminado es:

```text
C:\Program Files\NexoAI Vision
```

`INSTALLFOLDER` controla los binarios y archivos inmutables. Los datos operativos
se mantienen en estas carpetas, aunque se elija otro destino:

```text
C:\ProgramData\NexoAI Vision\runtime\models
C:\ProgramData\NexoAI Vision\runtime\config
C:\ProgramData\NexoAI Vision\runtime\output
C:\ProgramData\NexoAI Vision\runtime\logs
```

La carpeta `output` usa el
[contrato v1 de eventos y evidencias](../../docs/operator-evidence-contract-v1.md):
`Reporte_Eventos_Seguridad.csv` y `Evidencias/`. Esa es la salida autoritativa del
runtime MSI. El MSI no genera `Reporte_Eventos_Seguridad.xlsx`; dicho archivo es
solo una exportación local/offline del harness Python de QA.

Los usuarios estándar reciben permisos de modificación solo en esas cuatro
carpetas. Los binarios conservan los permisos endurecidos heredados de
`Program Files`. La instalación crea el acceso principal **NexoAI Vision**
para el iniciador y conserva **Command Help** y **README** en el menú Inicio.

Al actualizar desde Cuajone PPE Monitor, el MSI no mueve ni elimina
`C:\ProgramData\Cuajone PPE Monitor`. NexoAI Vision escribe los datos nuevos en su
propia carpeta y el iniciador conserva lectura de modelos desde la ruta heredada.
El valor `ComputeMode` se busca primero en `HKLM\SOFTWARE\NexoAI Vision` y después
en la clave heredada; una instalación nueva escribe solo la clave NexoAI Vision.

Dentro de una carpeta de salida reutilizada, `native_events.csv` y `evidence/` de
versiones anteriores también permanecen intactos. El runtime nuevo escribe los
nombres v1 al lado; soporte no debe concatenar ambos CSV porque sus columnas no son
compatibles.

La pantalla de cómputo ofrece Auto, GPU (CUDA) y CPU. GPU se marca como no disponible si DXGI y
la API del driver CUDA 12.9 (`12090`) no están listos o ningún dispositivo alcanza
SM 7.5. El MSI repite el gate en la secuencia de
ejecución para cubrir `/qn`; `COMPUTE_MODE=cuda` falla cerrado. Auto/CPU no instalan
ni modifican drivers y la selección queda en HKLM para el runtime.

### Despliegue silencioso y registros

Para el piloto privado, usa el comando de la sección anterior y agrega `-Silent`.
Así se conservan las validaciones, la autorización y el registro indicado con
`-LogPath`.

Cuando la confianza ya fue preparada y el MSI fue verificado mediante el proceso
aprobado, `INSTALLFOLDER` puede dirigirse a otra ruta:

```powershell
$msi = "D:\Paquetes\NexoAIVision-0.1.0-internal.3-x64-Internal.msi"
$log = "D:\Logs\cuajone-install.log"

msiexec.exe /i $msi /qn /norestart `
  INSTALLFOLDER="D:\Apps\NexoAI Vision" `
  COMPUTE_MODE=auto `
  /L*V $log
```

Para mostrar la interfaz y conservar el registro, omite `/qn`. Si la instalación
falla, entrega el archivo indicado por `/L*V` al equipo responsable.

## 5. Comprobar sin cámara

Desde el menú Inicio, abre **NexoAI Vision**. La prueba operativa es correcta
si aparece el formulario; ciérralo sin completar una fuente ni seleccionar
modelos. El acceso **NexoAI Vision - Command Help** se conserva para soporte
avanzado y ejecuta únicamente `--help`.

No uses `--preflight` como comprobación básica, porque selecciona CUDA y puede
deserializar engines.

## 6. Mantener la instalación

### Reparar

Usa el mismo MSI aprobado y conserva un registro:

```powershell
msiexec.exe /fa "D:\Paquetes\NexoAIVision.msi" /qn /L*V "D:\Logs\repair.log"
```

### Actualizar

Las versiones mayores nuevas reemplazan la versión anterior mediante el
`UpgradeCode` estable. Instala el nuevo MSI aprobado con el mismo procedimiento de
verificación y registro. Un downgrade directo se bloquea.

### Desinstalar

Puede usarse Aplicaciones instaladas o este comando:

```powershell
msiexec.exe /x "D:\Paquetes\NexoAIVision.msi" /qn /L*V "D:\Logs\uninstall.log"
```

### Volver a una versión anterior

Para rollback, desinstala la versión actual y luego instala el MSI anterior
aprobado. Los binarios, accesos y registro de la aplicación se eliminan. Las
carpetas mutables vacías también se eliminan; si contienen datos del operador,
Windows Installer las conserva. Respalda o elimina esos datos solo con aprobación.

## 7. Construir el MSI

El build predeterminado es `Release` y falla cerrado. Para un Preview interno
firmado, configura la hoja piloto no exportable existente y ejecuta:

```powershell
$certs = .\installer\native\New-InternalPilotSigningCertificates.ps1
$env:CUAJONE_CERTIFICATE_SHA1 = $certs.LeafThumbprint
$env:CUAJONE_PILOT_ROOT_CER = $certs.RootPublicCertificate
$env:CUAJONE_ALLOW_INTERNAL_PILOT_TRUST = "1"
$env:CUAJONE_SIGNTOOL_PATH = "C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64\signtool.exe"
$env:CUAJONE_TIMESTAMP_URL = "http://timestamp.acs.microsoft.com"
$env:CUAJONE_SIGN_COMMAND = (Resolve-Path .\installer\native\sign-release.ps1).Path

.\installer\native\build-installer.ps1 -BuildMode Preview
```

El candidato local usa `0.1.0-internal.4`, no está publicado ni autorizado para
instalación; `v0.1.0-internal.3` y sus assets publicados son inmutables. Un build `Release` exige además
`CUAJONE_PARITY_RECEIPT` con contrato `1.0.0`, commit exacto y paridad completa
sobre engines/video autorizados. El recibo debe cumplir el esquema compartido,
identificar y hashear al menos dos inputs aprobados, aportar evidencia hash y
comparaciones positivas para las seis etapas, y referenciar la autorización. Su
timestamp RFC3339 debe estar en UTC: se toleran cinco minutos hacia el futuro y el
recibo vence después de siete días. Un recibo sintético no atraviesa ese gate.

El MSI, su `.sha256`, staging, SBOM SPDX 2.3, temporales y evidencia quedan bajo
`.tools\native\installer`. El script firma primero
`cuajone_native.exe`, `cuajone_launcher.exe` y `CuajoneHardwareProbeCA.dll`,
construye/firma el MSI y realiza extracción administrativa en D. Ambos ejecutables
son raíces independientes del recorrido de imports PE. Nunca vuelve a firmar DLL
de terceros.

Antes de construir, los MSI/sidecars anteriores se mueven desde `output` a un
subdirectorio fechado de `installer\superseded`. Al terminar, `output` debe contener
exactamente el MSI actual y su `.sha256`; `test-installer.ps1` vuelve a comprobar
ese conjunto y el contenido del sidecar.

## 8. Controlar la firma

`sign-release.ps1` acepta únicamente `cuajone_launcher.exe`,
`cuajone_native.exe`, `CuajoneHardwareProbeCA.dll` y archivos `.msi`. Usa SignTool
con digest SHA-256, timestamp RFC 3161 y verificación Authenticode. No acepta PFX,
contraseñas ni claves exportadas.

El opt-in `CUAJONE_ALLOW_INTERNAL_PILOT_TRUST=1` solo funciona con
`-BuildMode Preview`. La verificación admite únicamente `UntrustedRoot` en una
máquina todavía no enrolada y prueba además la cadena contra el CER raíz público
indicado. `Release` nunca acepta esa ruta: exige confianza pública, worktree
limpio, revisión exacta y archivos de fuente correspondientes publicados por
HTTPS con SHA-256.

## 9. Toolchain fijado

Las herramientas locales se mantienen bajo `.tools\native`, ignoradas por Git:

| Componente | Ruta | Versión usada |
| --- | --- | --- |
| .NET SDK | `.tools\native\dotnet-sdk` | 8.0.423 |
| WiX CLI | `.tools\native\wix` | 6.0.2 |
| Extensiones UI/Util | `.tools\native\wix\.wix` | 6.0.2 |
| NuGet | `.tools\native\cache\nuget` | caché local |
| Temporales | `.tools\native\temp\wix` | proceso local |

NuGet publica WiX 7.0.0 como versión más reciente, pero su binario exige aceptar
el acuerdo OSMF y puede requerir una tarifa para uso generador de ingresos. Este
repositorio no acepta términos legales ni presupone elegibilidad: se fija WiX
6.0.2 hasta que el responsable confirme por escrito el cumplimiento OSMF o apruebe
una compilación propia de WiX 7.

`CuajonePpeMonitor.wixproj` registra las dependencias; `build-installer.ps1` usa
el CLI local fijado. `Package.wxs` define identidad, `UpgradeCode`, UI, carpetas,
ACL y accesos. El script genera `Payload.wxs` en D con un componente y GUID
determinista por ruta staged; ese archivo no se versiona.

## 10. Verificar sin instalar

`test-installer.ps1` no instala ni desinstala el producto. Realiza:

1. `wix msi validate` con temporales en D.
2. Lectura de tablas MSI en modo solo lectura.
3. Comprobación de x64, ámbito por máquina, `INSTALLFOLDER`, `UpgradeCode`,
   downgrade, componentes, feature, accesos, ARP, reparación y desinstalación.
4. Extracción administrativa con `msiexec /a` hacia D.
5. Comparación SHA-256 de todo el payload contra staging y de cada archivo copiado
   contra su fuente actual del repositorio, build o toolchain.
6. Verificación de firma de MSI, launcher y runtime; confirmación de subsistema GUI
   para el launcher y consola para el runtime; luego `--help` y probe JSON sin
   modelos.
7. Rechazo de cualquier artefacto Python/QA mediante la política compartida
   `payload-policy.ps1`.

```powershell
.\installer\native\test-installer.ps1 `
  -InstallerPath ".tools\native\installer\output\<paquete>.msi" `
  -StageDir ".tools\native\installer\stage" `
  -ExpectedSignatureStatus Signed `
  -AllowInternalPilotTrust
```

La aceptación completa de instalación, reparación, actualización y
desinstalación debe ejecutarse en una VM piloto enrolada, manteniendo Defender y
los controles corporativos activos.

## 11. Dependencias y distribución

`build-installer.ps1` recorre imports PE con `dumpbin`, incluye ONNX Runtime 1.25.0
CPU, agrega explícitamente el
plugin FFmpeg de OpenCV y rechaza plugins o builders TensorRT no aprobados. Los
hashes de cada binario se registran en `build-metadata.json` y
`SHA256SUMS.txt`; las licencias y la procedencia se incluyen como antes.

El MSI no inicia el monitor, no abre cámaras, no descarga drivers y no carga
engines. El operador aporta engines compatibles bajo sus propios términos y usa
rutas CLI explícitas. Consulta [`FFMPEG-SOURCE.md`](FFMPEG-SOURCE.md) y
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
