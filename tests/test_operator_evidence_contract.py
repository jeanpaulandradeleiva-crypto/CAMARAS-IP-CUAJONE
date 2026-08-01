from __future__ import annotations

import ppe_reportev2


EXPECTED_EVENT_HEADER = (
    "Evento_ID,Camara,Fecha,Hora,Tipo_Evento,Casco,Chaleco,Estado_EPP,"
    "Confianza_Evento,ID_Seguimiento_Temporal,Estado_Revision,"
    "Identificacion_Humana,Observaciones_Revision,Foto"
)


def test_event_fields_match_operator_evidence_contract_v1() -> None:
    assert ",".join(ppe_reportev2.EVENT_FIELDS) == EXPECTED_EVENT_HEADER
