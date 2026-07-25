# Axis Edge and Milestone Integration Architecture

This document separates repository facts from deployment recommendations. It does
not assert that any Cuajone camera supports ACAP, a DLPU, containers, or an
existing metadata producer without an exact camera and VMS inventory.

## Decision Summary

| Horizon | Decision |
|---|---|
| CURRENT | Run one server worker per selected stream with one decoder and a replaceable latest-frame slot. |
| NEAR-TERM | Keep inference on the server and publish deduplicated XProtect Analytics Events. |
| FUTURE | Evaluate native Axis edge inference or Milestone AI Bridge only after hardware, AXIS OS, XProtect, installation, support, and licensing discovery. |

## CURRENT

### Verified Repository Facts

- `ppe_reportev2.py` is the active Python/Ultralytics worker.
- `LatestFrameCapture` owns one decoder and one replaceable latest frame. There is
  no frame queue and no intentional accumulation of stale video.
- `ppe-fall` performs pose tracking followed by PPE inference for each processed
  frame.
- `ppe-only` loads only `best_ppe.pt`, calls PPE tracking with `persist=True`, and
  uses IDs attached to PPE `Person` detections.
- Events and evidence are persisted locally. The repository does not currently
  publish events or metadata to Milestone.
- The current input is a direct camera RTSP URL. There is no proved XProtect or AI
  Bridge video source in this deployment.

### Current Recommendation

Run exactly one video source and one decoder for each analytics worker. Do not
connect the same worker to both direct-camera RTSP and a Milestone/AI Bridge feed.
That would duplicate network traffic and decoding without adding inference value.

Use `ppe-only` where fall analytics is not required. This removes pose work rather
than merely suppressing fall alerts. `TARGET_INFERENCE_FPS` may bound inference
starts while capture continues replacing the latest frame.

No measured throughput or latency claim is made. Validate on the actual server,
camera stream profile, and target GPU.

## NEAR-TERM

### Server Video Path

Recommendation: keep the current host-side model and choose one authoritative
stream path:

1. Direct Axis RTSP when that is the approved operational source today.
2. An XProtect/AI Bridge-provided stream only if that component is installed,
   supported, licensed where required, and selected as the replacement source.

Do not run both paths for the same camera/worker. A single stream and decoder is
the simplest way to control latency, camera fan-out, and GPU workload.

### Milestone Event Path

Recommendation: use XProtect Analytics Event XML as the lowest-complexity first
integration. The
official protocol sample submits an analytics event to the Event Server, allows
alarm definitions to react to it, and exposes events in Smart Client. This is a
smaller change than replacing video ingestion or building a metadata plug-in.

Publish only state transitions and cooldown-approved events already produced by
the worker. Use the application event ID, camera mapping, event type, timestamp,
and confidence as the deduplication/correlation contract. Keep local CSV/JPEG
evidence until XProtect retention and operator workflows are validated.

Operational prerequisites to verify:

- XProtect product, version, Event Server, and Analytics Events configuration.
- Camera-to-XProtect source mapping and clock synchronization.
- Authentication, TLS/network policy, retry policy, and event deduplication.
- Whether port 9090 or a supported API Gateway path is approved in that version.
- Alarm definitions, retention, and operator acknowledgement workflow.

## FUTURE: Axis Edge Compute

### Verified Axis Facts

- ACAP support depends on both Axis hardware and AXIS OS. Axis documents chip
  architecture and software/API compatibility separately; exact device inventory
  is therefore mandatory.
- The ACAP Native SDK officially supports C, C++, and shell. It is the recommended
  current path for native analytics. Python examples exist in the older
  container-oriented Computer Vision examples, but Axis states new products on
  AXIS OS 12.x do not gain container support.
- VDO/Video Capture can provide frames internally to an ACAP application, avoiding
  an external RTSP round trip for on-camera analytics.
- Larod is the C inference API and can use inference hardware made available by
  supported ARTPEC/CV devices. Availability and backend support are device-specific.
- The repository's YOLO11 `.pt` is a PyTorch/Ultralytics artifact, not a directly
  deployable Larod package. Axis examples use SoC-specific conversion paths.

### Required Feasibility Gate

Inventory every target camera before designing an ACAP package:

| Required fact | Why it blocks design |
|---|---|
| Exact model and product number | Establishes ACAP and hardware feature support. |
| AXIS OS release/LTS | Establishes compatible SDK and APIs. |
| SoC: ARTPEC generation or CV25 | Selects architecture and inference conversion path. |
| RAM/storage and application limits | Determines whether the model/runtime fit. |
| DLPU/inference backend availability | Determines hardware acceleration options. |
| Container support | Must not be assumed, especially for new AXIS OS 12 devices. |
| Existing analytics/metadata producers | Prevents duplicate analytics and conflicting overlays. |

If the inventory passes, conversion still requires export, operator compatibility
checking, input preprocessing parity, quantization/compilation where required,
accuracy comparison, memory measurement, thermal/long-run testing, and per-SoC
benchmarking. A converted model that loads is not yet a validated safety model.

Recommendation: treat edge PPE as a separate native ACAP product, likely using
VDO plus Larod and C/C++, rather than attempting to copy this Python worker and
`.pt` file onto a camera.

## FUTURE: Milestone Video and Metadata

### AI Bridge

Recommendation: AI Bridge is the future Python-native analytics source and
integration candidate if it is installed, supported by the deployed XProtect
version, and appropriately licensed. Milestone publishes AI Bridge connectivity
samples and a dedicated sample repository.

Use it to replace, not duplicate, the direct-camera feed for a worker. Validate
camera RTSP/snapshot access, event submission, metadata injection, recorded
playback needs, credentials, and throughput in a staging XProtect system. Do not
claim that AI Bridge removes every internal VMS decode; official interfaces prove
integration capabilities, not that internal implementation detail.

### MIP SDK Metadata and Overlay

Recommendation: reserve MIP SDK metadata/overlay and Smart Client plug-ins for a
later phase requiring live bounding boxes or richer operator visualization. This
path has materially higher complexity than Analytics Events: metadata schemas,
coordinate systems, camera mapping, timestamps, plugin deployment, versioning,
and possibly .NET components all become part of the supported product.

Use Analytics Events first when the user outcome is alarms and evidence
correlation. Add metadata only when an approved operator requirement justifies
the additional lifecycle cost.

## Official Sources

Axis:

- [ACAP developer documentation](https://developer.axis.com/acap/)
- [Supported languages](https://developer.axis.com/acap/reference/supported-languages/)
- [Axis devices and compatibility](https://developer.axis.com/acap/reference/axis-devices-and-compatibility/)
- [Larod API overview](https://developer.axis.com/acap/api/src/api/larod/html/index.html)
- [AxisCommunications GitHub organization](https://github.com/AxisCommunications)
- [ACAP Native SDK](https://github.com/AxisCommunications/acap-native-sdk)
- [Native SDK examples, including VDO/Larod and model conversion](https://github.com/AxisCommunications/acap-native-sdk-examples)
- [Computer Vision SDK examples and container-support notice](https://github.com/AxisCommunications/acap-computer-vision-sdk-examples)

Milestone:

- [milestonesys GitHub organization](https://github.com/milestonesys)
- [MIP SDK protocol samples](https://github.com/milestonesys/mipsdk-samples-protocol)
- [Analytics Event XML sample](https://github.com/milestonesys/mipsdk-samples-protocol/tree/main/TriggerAnalyticsEventXML)
- [MIP AI Bridge samples](https://github.com/milestonesys/MIP-AIBridge-samples)
- [MIP SDK documentation](https://doc.developer.milestonesys.com/)

Sources were reviewed on 2026-07-25. Re-check product/version documentation during
inventory because camera and XProtect capabilities evolve.
