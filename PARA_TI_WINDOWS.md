# Para TI: piloto Windows en cinco pasos

Esta guía permite verificar, instalar y retirar el piloto sin abrir cámaras ni
cargar modelos. La [guía de usuario](INSTALACION_WINDOWS.md) sigue siendo la ruta
para quien necesita abrir el iniciador gráfico y configurar una ejecución.

## Primera pantalla

**Propósito:** preparar una VM Windows enrolada, verificar el paquete aprobado,
instalarlo con registro, elegir cómputo y comprobar el iniciador sin configurar
cámaras ni iniciar inferencia.

**Archivos que recibe TI:**

- MSI `NexoAIVision-<version>-x64-Internal.msi`;
- hash SHA-256 aprobado por un canal independiente;
- certificados públicos raíz y hoja `.cer`, con hashes y huellas aprobados;
- `Install-Pilot.ps1` e `Install-InternalPilotTrust.ps1`;
- autorización de seguridad para enrolar la confianza privada del piloto.

**Ruta de cinco pasos:**

1. Confirma que el paquete es `internal.3` y que la VM está autorizada.
2. Compara SHA-256, huellas y firma antes de modificar confianza.
3. Valida y enrola los certificados públicos con autorización explícita.
4. Instala el MSI con interfaz o en silencio y guarda `/L*V`.
5. Abre **NexoAI Vision**, confirma que aparece el formulario y ciérralo sin
   completar una fuente ni iniciar la inferencia.

**Resultado esperado:** el producto aparece en Aplicaciones instaladas y el
iniciador gráfico abre sin error. Esta aceptación no configura cámaras, no ejecuta
RTSP y no inicia inferencia.

## Versiones y alcance

| Elemento | Estado |
| --- | --- |
| `v0.1.0-internal.3` | Release interna publicada que TI puede evaluar con el paquete aprobado. |
| `0.1.0-internal.4` | Candidato local de ingeniería. No está publicado, aprobado ni autorizado para instalación. |
| Builds desde fuente | Flujo de ingeniería futuro; requiere firma, revisión y sus propios gates. |

El MSI no instala Python, PyTorch, Ultralytics, CVAT, Supervision, `cuajone_qa`, el
binding `.pyd`, fixtures, engines TensorRT ni recibos de paridad. Instala por defecto
el bundle ONNX opcional de EPP y pose; cualquier modelo o engine externo sigue siendo
un artefacto aprobado aparte. Los demás componentes son exclusivamente de desarrollo y QA.

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
  COMPUTE_MODE=auto `
  /L*V "<RUTA_LOG>"
```

`INSTALLFOLDER` cambia los binarios; los datos mutables permanecen bajo
`C:\ProgramData\NexoAI Vision`.

Durante una actualización desde Cuajone PPE Monitor, conserva la carpeta heredada
sin moverla ni eliminarla. NexoAI Vision escribe datos nuevos en su propia carpeta
y el iniciador puede leer modelos de la ubicación heredada. El modo de cómputo se
lee primero desde `HKLM\SOFTWARE\NexoAI Vision\ComputeMode` y, si no existe, desde
la clave heredada; las instalaciones nuevas escriben solo la clave NexoAI Vision.

`COMPUTE_MODE` acepta únicamente `auto`, `cuda` o `cpu`. El valor se persiste en
HKLM. `cuda` falla si el custom action DXGI/CUDA no confirma hardware y driver;
`auto` y `cpu` continúan. El MSI no contiene ni instala drivers NVIDIA.
Para TensorRT 11, el probe exige Driver API CUDA 12.9 (`12090`) y capacidad de
cómputo SM 7.5 o superior. Un driver anterior aparece como `driver_too_old` y no
habilita GPU.

## Logs y aceptación

Conserva el archivo indicado por `/L*V`, el hash del MSI, la salida de firma y la
versión de Windows. La comprobación aceptada es:

1. Confirma el producto en Aplicaciones instaladas.
2. Abre **NexoAI Vision** desde Inicio.
3. Confirma que el formulario permite elegir fuente, análisis, cómputo, modelos
   externos, salida y visualización.
4. Cierra el iniciador sin completar credenciales, abrir la fuente o cargar modelos.

No uses `--preflight` como smoke test: puede seleccionar CUDA y deserializar
engines. **NexoAI Vision - Command Help** y `NexoAIVision.exe` quedan para
diagnóstico o automatización avanzada, no como ruta normal del operador.

## CLI avanzado

El runtime de consola permanece instalado en
`<CARPETA_INSTALACION>\bin\NexoAIVision.exe`. Úsalo solo cuando un procedimiento
de soporte requiera opciones explícitas, `--help`, el probe JSON o automatización.
No entregues al usuario normal una línea de comandos con credenciales RTSP.

El MSI incluye por defecto los modelos ONNX aprobados de EPP y pose con sus
manifests adyacentes. Los modelos fuente o de desarrollo permanecen fuera del MSI;
GPU usa engines TensorRT externos aprobados. Conserva hashes, procedencia y
autorización junto con la evidencia de despliegue, y no copies modelos adicionales
al directorio de binarios.

La salida de producción sigue el
[contrato v1 común](docs/operator-evidence-contract-v1.md):
`Reporte_Eventos_Seguridad.csv` y `Evidencias/`. El MSI no genera XLSX; esa
exportación existe solo en el harness Python de QA local/offline. En una salida
reutilizada, conserva `native_events.csv` y `evidence/` antiguos por separado y no
los concatenes con el contrato nuevo.

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
4. Instálalo y repite la aceptación del iniciador gráfico.

Windows Installer conserva carpetas mutables que todavía contienen datos.

## Solución de problemas

| Síntoma | Acción segura |
| --- | --- |
| Hash o huella distintos | Detener el proceso y solicitar nuevamente el paquete por el canal aprobado. |
| Firma no válida o sin timestamp | No instalar; entregar hash y salida de verificación a seguridad. |
| Acceso denegado al enrolar | Confirmar PowerShell elevado y autorización; no relajar políticas. |
| Error `msiexec` | Conservar el log `/L*V` y buscar `Return value 3` con su contexto. |
| El iniciador no abre | Registrar el error, versión del SO y controles activos; comprobar firma de `NexoAIVisionLauncher.exe` y no ejecutar `--preflight`. |
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
- [ ] El iniciador abrió y cerró sin cámaras, modelos ni inferencia.
- [ ] Reparación, upgrade, desinstalación y rollback tienen responsable asignado.

Para construcción, firma, tablas MSI, toolchain y validación administrativa, usa la
[referencia avanzada del instalador](installer/native/README.md).
