# Instalar Cuajone PPE Monitor en Windows

Esta guía es para la persona que usará la aplicación. No necesitas conocer
PowerShell ni escribir comandos.

## Antes de empezar

1. Obtén el archivo MSI aprobado por tu organización.
2. Pregunta a TI si el equipo ya está preparado para confiar en el instalador.
3. No continúes si Windows indica que no confía en el instalador. Contacta a TI.

## Instalar

1. Haz doble clic en el archivo MSI.
2. Si Windows pide permiso para hacer cambios, confirma solo si TI aprobó el
   instalador.
3. Deja la carpeta propuesta o selecciona **Browse** para elegir otra.

   La carpeta propuesta es `C:\Program Files\Cuajone PPE Monitor`.

4. Selecciona **Install**.
5. Cuando termine, selecciona **Finish**.

## Comprobar que abre

1. Abre el menú Inicio de Windows.
2. Busca **Cuajone PPE Monitor**.
3. Abre **Cuajone PPE Monitor - Command Help**.
4. Si aparece la ayuda sin un mensaje de error, la comprobación terminó.

Esta comprobación no abre cámaras ni inicia el análisis.

## Configurar cámaras después

La instalación y la configuración de cámaras son pasos separados. El instalador
no agrega una cámara y no deja la aplicación unida de forma permanente a una sola
cámara.

La versión piloto tiene un alcance limitado. La cámara se configura después de
instalar, con ayuda del responsable de la aplicación. La configuración y los datos
se guardan fuera de la carpeta de instalación, bajo
`C:\ProgramData\Cuajone PPE Monitor`.

Se espera que versiones futuras permitan agregar más configuraciones de cámara
sin reinstalar la aplicación en otra carpeta. Esto dependerá del soporte del
producto en cada versión; no significa que la versión actual ya use varias
cámaras.

## Si algo falla

| Problema | Qué hacer |
| --- | --- |
| Windows no confía en el instalador | No cambies la seguridad del equipo. Contacta a TI. |
| La instalación falla | Pide a TI que genere el registro MSI y envíaselo. |
| La aplicación abre, pero no aparece una cámara | La cámara todavía debe configurarse. Es un paso posterior a la instalación. |

## Para TI

TI debe preparar la confianza del certificado cuando corresponda y verificar el
paquete antes de instalarlo. Los pasos técnicos, el despliegue silencioso, los
registros y el mantenimiento están en la
[guía para TI](installer/native/README.md).
