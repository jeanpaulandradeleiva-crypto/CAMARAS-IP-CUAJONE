# Seguridad de releases

Las releases publicas deben publicarse por HTTPS con hash SHA-256, manifiesto de
archivos y SBOM. Una release de produccion no se considera lista si el ejecutable
propio, el instalador y el desinstalador no tienen una firma Authenticode valida,
con digest SHA-256 y sello de tiempo RFC 3161.

## Lista de publicacion

- Publicar instalador, SHA-256, manifiesto/SBOM y fuente correspondiente desde el
  repositorio oficial o un canal HTTPS controlado por el proyecto.
- Mantener estables el publisher, `AppId` y el esquema de versionado para no
  fragmentar la reputacion de Windows.
- Verificar todas las firmas con `signtool verify /pa /all /v`.
- Probar instalacion, ayuda y desinstalacion en un host limpio con Microsoft
  Defender y SmartScreen habilitados.
- Enviar falsos positivos reproducibles mediante el portal oficial de Microsoft
  Defender, junto con hashes y datos de la firma.
- No usar UPX, ofuscacion ni empaquetadores que oculten el comportamiento del PE.
- No recomendar exclusiones generales del antivirus. Una exclusion puntual solo
  puede decidirse mediante el proceso de seguridad de la organizacion.

Authenticode y una cadena de publicacion estable mejoran trazabilidad y
reputacion, pero no garantizan cero alertas. La licencia AGPL tampoco evita ni
reduce por si sola las detecciones antivirus.

Consulta [`installer/native/README.md`](installer/native/README.md) para el flujo
de firma y aceptacion. Esto es documentacion de ingenieria, no asesoramiento legal.
