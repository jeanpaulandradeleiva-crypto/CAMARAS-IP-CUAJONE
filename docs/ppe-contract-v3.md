# Four-state PPE contract v3

The v3 report evaluates each tracked person against every one of the seven PPE items. Each item has one exact `wear_state`.

| State | Meaning | Alerting |
|---|---|---|
| `PRESENTE_CORRECTAMENTE` | PPE detection is associated with its coarse expected body region. | No |
| `PRESENTE_INCORRECTAMENTE` | A PPE detection is uniquely attributable to a person but spatially incompatible with its expected body region. | Yes |
| `AUSENTE` | No associated PPE detection was observed under the item policy. | Yes |
| `NO_VERIFICABLE` | The item is disabled, has insufficient samples, or facial evidence is insufficient for respirator or eye protection. | No |

`PRESENTE_CORRECTAMENTE` means detector and coarse 2D body-region consistency only. It does not certify PPE fit, fastening, serviceability, or respirator seal.

## Operational switches

The launcher exposes one default-enabled switch for Gloves, Safety_boots, Vest, respirador, tapaorejas, Hard_hat, and lentes_protectores. Person remains decoded and cannot be disabled.

A disabled item is excluded before decode candidate retention and NMS where possible, and is excluded again from association, overlays, compliance, alerts, and evidence. v3 still emits it with `enabled: false`, `wear_state: NO_VERIFICABLE`, and `reason: DISABLED_BY_POLICY`.

Old launcher preferences load every PPE switch as enabled. Saving preferences adds the persisted `ppe_enabled` setting and the launcher passes each explicit `--ppe-enabled name=0|1` argument to the runtime.

## Face policy

Only respirator and eye protection require frontal face evidence to classify lack of detection as absent. They remain `NO_VERIFICABLE` when pose is absent, malformed, rear-facing, asymmetric, or otherwise insufficient. This includes `ppe-only`, which deliberately does not load a pose model.

All other enabled items become `AUSENTE` when their regional association has no detection under current evaluation conditions. Absence never becomes `PRESENTE_INCORRECTAMENTE`.

## Compatibility

`contracts/v1/`, `contracts/v2/`, v1/v2 serializers, and `Reporte_Eventos_Seguridad_v2.csv` are unchanged for consumers. The runtime appends v3 PPE event records to `Reporte_Eventos_Seguridad_v3.jsonl`; each line is a v3 CloudEvent with per-item state, enabled flag, reason, confidence, and detection evidence when present.
