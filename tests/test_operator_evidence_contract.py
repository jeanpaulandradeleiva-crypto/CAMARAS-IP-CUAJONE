from __future__ import annotations

import ppe_reportev2


EXPECTED_EVENT_HEADER = (
    "Version_Contrato,Evento_ID,Camara,Fecha,Hora,Tipo_Evento,Guantes,Botas_Seguridad,"
    "Chaleco,Respirador,Tapaorejas,Casco,Lentes_Protectores,Faltantes_EPP,Estado_EPP,"
    "Ratio_Guantes,Ratio_Botas_Seguridad,Ratio_Chaleco,Ratio_Respirador,Ratio_Tapaorejas,"
    "Ratio_Casco,Ratio_Lentes_Protectores,Confianza_Guantes,Confianza_Botas_Seguridad,"
    "Confianza_Chaleco,Confianza_Respirador,Confianza_Tapaorejas,Confianza_Casco,"
    "Confianza_Lentes_Protectores,"
    "Confianza_Evento,ID_Seguimiento_Temporal,Estado_Revision,"
    "Identificacion_Humana,Observaciones_Revision,Foto"
)


def test_event_fields_match_operator_evidence_contract_v2() -> None:
    assert ",".join(ppe_reportev2.EVENT_FIELDS) == EXPECTED_EVENT_HEADER
