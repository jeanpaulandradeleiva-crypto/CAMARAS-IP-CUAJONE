# SPDX-License-Identifier: AGPL-3.0-only

"""Deprecated prototype. Use ppe_reportev2.py for production monitoring."""

from ultralytics import YOLO
import pandas as pd
from datetime import datetime
from pathlib import Path
import cv2
import os
from dotenv import load_dotenv

print("ADVERTENCIA: ppe_reporte.py está obsoleto; usa ppe_reportev2.py.")
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=False)
RTSP_URL = os.getenv("RTSP_URL")
if not RTSP_URL:
    raise RuntimeError("Falta RTSP_URL en el entorno.")

# Crear carpeta de evidencias
os.makedirs(r"Evidencias", exist_ok=True)

# Modelo
model = YOLO(r"best_ppe.pt")

print("CLASES DEL MODELO:")
print(model.names)

# Cámara
resultados = model.track(
    source=RTSP_URL,
    stream=True,
    show=True,
    persist=True,
    conf=0.25
)

personas_registradas = set()
seguimiento = {}
datos = []

for r in resultados:

    if r.boxes.id is None:
        continue

    ids = r.boxes.id.cpu().numpy().astype(int)
    clases = r.boxes.cls.cpu().numpy().astype(int)

    nombres = [r.names[c] for c in clases]

    print("Detectado:", nombres)

    hay_persona = "Person" in nombres
    hay_casco = "Hard_hat" in nombres
    hay_chaleco = "Vest" in nombres

    for persona_id in ids:

        if persona_id in personas_registradas:
            continue

        if persona_id not in seguimiento:
            seguimiento[persona_id] = {
                "frames": 0,
                "persona": False,
                "casco": False,
                "chaleco": False
            }

        seguimiento[persona_id]["frames"] += 1

        if hay_persona:
            seguimiento[persona_id]["persona"] = True

        if hay_casco:
            seguimiento[persona_id]["casco"] = True

        if hay_chaleco:
            seguimiento[persona_id]["chaleco"] = True

        # Esperar algunos frames
        if seguimiento[persona_id]["frames"] < 20:
            continue

        tiene_persona = seguimiento[persona_id]["persona"]
        tiene_casco = seguimiento[persona_id]["casco"]
        tiene_chaleco = seguimiento[persona_id]["chaleco"]

        if not tiene_persona:
            continue

        if tiene_casco and tiene_chaleco:
            estado = "EPP Completo"
        elif tiene_casco and not tiene_chaleco:
            estado = "Falta Chaleco"
        elif not tiene_casco and tiene_chaleco:
            estado = "Falta Casco"
        else:
            estado = "Sin Casco y Chaleco"

        foto = ""

        # Tomar evidencia si incumple EPP
        if estado != "EPP Completo":

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            foto = (
                rf"Evidencias\ID_{persona_id}_{timestamp}.jpg"
            )

            cv2.imwrite(
                foto,
                r.orig_img
            )

            print(f"Evidencia guardada: {foto}")

        personas_registradas.add(persona_id)

        registro = {
            "ID": persona_id,
            "Fecha": datetime.now().strftime("%Y-%m-%d"),
            "Hora": datetime.now().strftime("%H:%M:%S"),
            "Casco": "SI" if tiene_casco else "NO",
            "Chaleco": "SI" if tiene_chaleco else "NO",
            "Estado": estado,
            "Foto": foto
        }

        datos.append(registro)

        print("==========================")
        print(registro)
        print("==========================")

        df = pd.DataFrame(datos)

        df.to_excel(
            r"Reporte_EPP.xlsx",
            index=False
        )

        print("Reporte actualizado")

    # Detener después de 2 personas (prueba)
    if len(personas_registradas) >= 2:
        break

# Guardado final
df = pd.DataFrame(datos)

df.to_excel(
    r"Reporte_EPP.xlsx",
    index=False
)

print(df)
print("Reporte generado correctamente")
