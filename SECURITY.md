# Seguridad de releases

Las releases publicas deben publicarse por HTTPS con hash SHA-256, manifiesto de
archivos y SBOM. Una release de produccion no se considera lista si el ejecutable
propio y el MSI no tienen una firma Authenticode valida, con digest SHA-256 y
sello de tiempo RFC 3161. Las DLL de terceros conservan sus bytes y firmas
upstream; el proyecto no las vuelve a firmar.

## Lista de publicacion

- Publicar instalador, SHA-256, manifiesto/SBOM y fuente correspondiente desde el
  repositorio oficial o un canal HTTPS controlado por el proyecto.
- Mantener estables el publisher, `UpgradeCode` MSI y el esquema de versionado para no
  fragmentar la reputacion de Windows.
- Verificar todas las firmas con `signtool verify /pa /all /v`.
- Probar instalacion, ayuda y desinstalacion en un host limpio con Microsoft
  Defender y SmartScreen habilitados.
- Enviar falsos positivos reproducibles mediante el portal oficial de Microsoft
  Defender, junto con hashes y datos de la firma.
- No usar UPX, ofuscacion ni empaquetadores que oculten el comportamiento del PE.
- No recomendar exclusiones generales del antivirus. Una exclusion puntual solo
  puede decidirse mediante el proceso de seguridad de la organizacion.
- Verificar que el MSI no importe certificados ni modifique almacenes de
  confianza. El enrolamiento piloto siempre ocurre mediante el prerrequisito
  separado y autorizado.

Authenticode y una cadena de publicacion estable mejoran trazabilidad y
reputacion, pero no garantizan cero alertas. La licencia AGPL tampoco evita ni
reduce por si sola las detecciones antivirus.

La CA privada documentada para el piloto interno no satisface los requisitos de
una release pública ni debe enrolarse sin autorización previa de TI/seguridad del
cliente. Solo establece confianza en el publicador para los equipos enrolados: no
garantiza silencio del antivirus/EDR, suprime alertas del SOC, evita SmartScreen o
Smart App Control, ni invalida políticas de AppLocker, WDAC o Defender. Seguridad
debe aprobar la instalación y el comportamiento esperado de procesos y red, y
aplicar allowlisting por certificado hoja o hash del artefacto cuando su
herramienta lo permita. No deben usarse exclusiones amplias ni desactivarse
controles de seguridad.

Consulta [`installer/native/README.md`](installer/native/README.md) para el flujo
de firma, instalacion MSI, reparacion, rollback y aceptacion. Esto es
documentacion de ingenieria, no asesoramiento legal.
