# Para TI: piloto Windows en cinco pasos

Esta guía permite verificar, instalar y retirar el piloto sin abrir cámaras ni
cargar engines. La [guía de usuario](INSTALACION_WINDOWS.md) sigue siendo la ruta
para quien solo necesita instalar y comprobar la ayuda.

## Primera pantalla

**Propósito:** preparar una VM Windows enrolada, verificar el paquete aprobado,
instalarlo con registro y comprobar únicamente `--help`.

**Archivos que recibe TI:**

- MSI `CuajonePPEMonitor-0.1.0-internal.3-x64-Internal.msi`;
- hash SHA-256 aprobado por un canal independiente;
- certificados públicos raíz y hoja `.cer`, con hashes y huellas aprobados;
- `Install-Pilot.ps1` e `Install-InternalPilotTrust.ps1`;
- autorización de seguridad para enrolar la confianza privada del piloto.

**Ruta de cinco pasos:**

1. Confirma que el paquete es `internal.3` y que la VM está autorizada.
2. Compara SHA-256, huellas y firma antes de modificar confianza.
3. Valida y enrola los certificados públicos con autorización explícita.
4. Instala el MSI con interfaz o en silencio y guarda `/L*V`.
5. Abre **Cuajone PPE Monitor - Command Help** y archiva la evidencia.

**Resultado esperado:** el producto aparece en Aplicaciones instaladas y `--help`
abre sin error. Esta aceptación no configura cámaras, no ejecuta RTSP, no carga
engines y no inicia inferencia.

## Versiones y alcance

| Elemento | Estado |
| --- | --- |
| `v0.1.0-internal.3` | Release interna publicada que TI puede evaluar con el paquete aprobado. |
| `0.1.0-internal.4` | Candidato de código futuro. No existe un MSI construido y no debe presentarse como instalable. |
| Builds desde fuente | Flujo de ingeniería futuro; requiere firma, revisión y sus propios gates. |

El MSI no instala Python, PyTorch, Ultralytics, CVAT, Supervision, `cuajone_qa`, el
binding `.pyd`, fixtures, modelos, engines ni recibos de paridad. Esos componentes
son exclusivamente de desarrollo y QA.

## Verificar SHA y firma

Reemplaza todos los marcadores `<...>` con valores recibidos por el canal aprobado:

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath "<RUTA_MSI>"
Get-FileHash -Algorithm SHA256 -LiteralPath "<RUTA_CER_RAIZ>"
Get-FileHash -Algorithm SHA256 -LiteralPath "<RUTA_CER_HOJA>"

Get-PfxCertificate -FilePath "<RUTA_CER_RAIZ>" | Select-Object Subject, Thumbprint
Get-PfxCertificate -FilePath "<RUTA_CER_HOJA>" | Select-Object Subject, Thumbprint
Get-AuthenticodeSignature -LiteralPath "<RUTA_MSI>" |
  Select-Object Status, StatusMessage, SignerCertificate, TimeStamperCertificate
```

Los hashes y huellas deben coincidir exactamente con valores obtenidos por un canal
independiente. Si no coinciden, detente. No agregues exclusiones ni desactives
Defender, EDR, SmartScreen, AppLocker o WDAC.

Si TI dispone de SignTool:

```powershell
& "<RUTA_SIGNTOOL_EXE>" verify /pa /all /v "<RUTA_MSI>"
```

## Enrolar certificados

Ejecuta PowerShell como administrador. Primero valida sin importar:

```powershell
& "<RUTA_INSTALL_INTERNAL_PILOT_TRUST_PS1>" `
  -RootCertificatePath "<RUTA_CER_RAIZ>" `
  -LeafCertificatePath "<RUTA_CER_HOJA>" `
  -ValidateOnly
```

Después de la aprobación de seguridad, enrola únicamente los certificados públicos:

```powershell
& "<RUTA_INSTALL_INTERNAL_PILOT_TRUST_PS1>" `
  -RootCertificatePath "<RUTA_CER_RAIZ>" `
  -LeafCertificatePath "<RUTA_CER_HOJA>"
```

El MSI no modifica almacenes de confianza. La raíz va a `LocalMachine\Root` y la
hoja a `LocalMachine\TrustedPublisher` mediante este prerrequisito separado.

## Instalar

La ruta recomendada combina validación, enrolamiento autorizado, firma e instalación:

```powershell
& "<RUTA_INSTALL_PILOT_PS1>" `
  -MsiPath "<RUTA_MSI>" `
  -RootCertificatePath "<RUTA_CER_RAIZ>" `
  -LeafCertificatePath "<RUTA_CER_HOJA>" `
  -ExpectedMsiSha256 "<SHA256_MSI>" `
  -ExpectedRootCertificateSha256 "<SHA256_CER_RAIZ>" `
  -ExpectedLeafCertificateSha256 "<SHA256_CER_HOJA>" `
  -ExpectedRootThumbprint "<HUELLA_SHA1_RAIZ>" `
  -ExpectedLeafThumbprint "<HUELLA_SHA1_HOJA>" `
  -InstallFolder "<CARPETA_INSTALACION>" `
  -LogPath "<RUTA_LOG>" `
  -AuthorizeTrustEnrollment
```

Agrega `-Silent` al comando anterior para instalación silenciosa. Si la confianza
y la firma ya fueron verificadas mediante el proceso aprobado, la instalación
interactiva con carpeta personalizada es:

```powershell
msiexec.exe /i "<RUTA_MSI>" `
  INSTALLFOLDER="<CARPETA_INSTALACION>" `
  /L*V "<RUTA_LOG>"
```

La variante silenciosa es:

```powershell
msiexec.exe /i "<RUTA_MSI>" /qn /norestart `
  INSTALLFOLDER="<CARPETA_INSTALACION>" `
  /L*V "<RUTA_LOG>"
```

`INSTALLFOLDER` cambia los binarios; los datos mutables permanecen bajo
`C:\ProgramData\Cuajone PPE Monitor`.

## Logs y aceptación

Conserva el archivo indicado por `/L*V`, el hash del MSI, la salida de firma y la
versión de Windows. La comprobación aceptada es:

1. Confirma el producto en Aplicaciones instaladas.
2. Abre **Cuajone PPE Monitor - Command Help**.
3. Confirma que aparece la ayuda sin error.
4. Cierra la ventana.

No uses `--preflight` como smoke test: puede seleccionar CUDA y deserializar engines.

## Reparar, actualizar y desinstalar

```powershell
msiexec.exe /fa "<RUTA_MSI_APROBADO>" /qn /L*V "<RUTA_LOG_REPAIR>"
msiexec.exe /i "<RUTA_MSI_NUEVO_APROBADO>" /qn /norestart /L*V "<RUTA_LOG_UPGRADE>"
msiexec.exe /x "<RUTA_MSI_INSTALADO>" /qn /L*V "<RUTA_LOG_UNINSTALL>"
```

Un downgrade directo se bloquea. Para rollback:

1. Respalda datos mutables solo si el propietario los requiere y autoriza.
2. Desinstala la versión actual.
3. Verifica el MSI anterior aprobado con sus propios hashes y firma.
4. Instálalo y repite la aceptación `--help`.

Windows Installer conserva carpetas mutables que todavía contienen datos.

## Solución de problemas

| Síntoma | Acción segura |
| --- | --- |
| Hash o huella distintos | Detener el proceso y solicitar nuevamente el paquete por el canal aprobado. |
| Firma no válida o sin timestamp | No instalar; entregar hash y salida de verificación a seguridad. |
| Acceso denegado al enrolar | Confirmar PowerShell elevado y autorización; no relajar políticas. |
| Error `msiexec` | Conservar el log `/L*V` y buscar `Return value 3` con su contexto. |
| `--help` no abre | Registrar el error, versión del SO y controles activos; no ejecutar `--preflight`. |
| Alerta de EDR/Defender | Enviar hash, firma y log al proceso de seguridad; no desactivar ni excluir carpetas. |

## Entrega a seguridad

- [ ] MSI, certificados públicos y scripts proceden del canal aprobado.
- [ ] SHA-256 y huellas llegaron por un canal independiente.
- [ ] Firma, firmante y timestamp fueron verificados.
- [ ] Alcance de red, procesos, carpetas y rollback está documentado.
- [ ] No se solicitaron exclusiones ni desactivación de controles.

## Aceptación en VM enrolada

- [ ] VM autorizada con controles corporativos activos.
- [ ] Enrolamiento limitado a la raíz y hoja públicas del piloto.
- [ ] Instalación registrada con `/L*V` y ruta esperada.
- [ ] `--help` abrió sin cámaras, engines ni inferencia.
- [ ] Reparación, upgrade, desinstalación y rollback tienen responsable asignado.

Para construcción, firma, tablas MSI, toolchain y validación administrativa, usa la
[referencia avanzada del instalador](installer/native/README.md).
