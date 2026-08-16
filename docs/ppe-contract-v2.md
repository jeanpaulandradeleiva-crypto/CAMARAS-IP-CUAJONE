# Fixed seven-item PPE contract v2

> v2 remains frozen for existing consumers. The operational four-state policy and launcher switches are published separately through `contracts/v3/` and `Reporte_Eventos_Seguridad_v3.jsonl`.

Every tracked person is evaluated against the same seven required PPE items. Camera-specific PPE policies are intentionally out of scope.

## Runtime behavior

| Item | Model label | Association region |
|---|---|---|
| Gloves | `Gloves` | Hands and arms |
| Safety boots | `Safety_boots` | Lower legs and feet |
| Vest | `Vest` | Torso |
| Respirator | `respirador` | Head and face |
| Hearing protection | `tapaorejas` | Head and face |
| Hard hat | `Hard_hat` | Head and face |
| Eye protection | `lentes_protectores` | Head and face |

The model label order is fixed:

```text
0 Gloves
1 Person
2 Safety_boots
3 Vest
4 respirador
5 tapaorejas
6 Hard_hat
7 lentes_protectores
```

Startup rejects missing labels, extra labels, duplicate semantics, and a different order. ONNX exports write the same order plus the model SHA-256 into the adjacent manifest; `ppe_reportev2.py` verifies both before startup. TensorRT metadata and native CLI labels are validated against the same fixed registry.

## Temporal decision

Each item has its own confidence history. After `minimum_samples`, an item is present when its observed ratio reaches `present_ratio`; compliance requires all seven items to be present.

`ppe_visibility_sufficient` remains a diagnostic signal. It does not suspend sampling: an item that cannot be associated is recorded as absent, including when the body region is occluded or outside the frame.

## Compatibility

The strict schemas under `contracts/v1/` are frozen. Native `canonicalJson(...)` and the original Python binding methods remain explicit v1 adapters that project only hard-hat and vest state into the existing status values.

Current native and Python monitoring use `contracts/v2/` and publish only `*.v2` event types. A violation creates one internal event candidate; v1 and v2 serializers are alternate views, not two event publications. This prevents duplicate alerts during migration.

## Evidence output

The v2 runtime writes `Reporte_Eventos_Seguridad_v2.csv`; the Python QA harness also writes `Reporte_Eventos_Seguridad_v2.xlsx`. The v1 files are not appended, overwritten, or silently reinterpreted.

Each v2 row contains seven item states, seven temporal ratios, seven item confidences, the complete missing-item list, event metadata, review fields, and the evidence image path.

### Writer queue

`--evidence-writer-queue-capacity 0` is the default and preserves synchronous JPEG,
CSV, and applicable v3 JSONL persistence. A value from `1` to `4096` starts one
bounded FIFO writer worker. Enqueue blocks while full; accepted events are never
dropped or retried silently. The monitor clones the fully annotated BGR frame once
per event-bearing frame and shares that immutable clone across all events from it.

On shutdown, including a signal shutdown, the worker stops accepting jobs and drains
accepted work before the final telemetry report and normal exit. A write failure puts
the worker in a terminal state, rejects later jobs, accounts for accepted queued jobs,
and makes the monitor fail non-zero. `evidence_writer_failures.jsonl` records event,
stage, and error when the output remains writable. JPEG, CSV, and JSONL remain
separate writes: a JPEG may exist without its CSV/JSONL companion after a failure.

## Review path

1. Verify the registry and region strategies in `native/src/analytics/ppe_analytics.cpp`.
2. Verify v1 projection and v2 serialization in `native/src/foundation/contracts.cpp`.
3. Verify v2 schemas in `contracts/v2/`.
4. Verify overlays and evidence in `native/src/app/main.cpp`, `native/src/runtime/evidence.cpp`, and `ppe_reportev2.py`.
