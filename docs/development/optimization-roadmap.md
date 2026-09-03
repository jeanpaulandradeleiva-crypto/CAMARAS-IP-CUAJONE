# Roadmap de optimización: informe → implementación

Este documento traduce las recomendaciones de
[`deep-research-report.md`](deep-research-report.md) y
[`deep-research-report2.md`](deep-research-report2.md) a acciones concretas sobre
el repositorio, y registra su estado. El orden es deliberado y no debe
invertirse: **precisión primero, latencia después, técnicas exóticas al final**.

## Estado de las recomendaciones

| # | Recomendación | Estado | Dónde |
|---|---|---|---|
| 1 | Mantener el checkpoint FP32 como referencia | **Implementado** (continuo) | `best_ppe.pt` / `yolo26s-pose.pt` en la raíz; cada manifest ONNX registra `provenance.source_checkpoint` con SHA-256 del checkpoint fuente |
| 2 | Exportador PPE con YOLO26 end-to-end `[B,300,6]` | **Implementado y por defecto** | `tools/export_runtime_onnx.py`; raw queda como opt-in `--ppe-raw` |
| 3 | Comparar end-to-end contra raw+NMS en el test congelado | **Implementado** (herramienta) | `tools/compare_ppe_models.py` acepta `.onnx`; ver protocolo abajo |
| 4 | Engine TensorRT FP16 en el servidor objetivo | **Implementado y desplegado** (GTX 1650 Ti, gate 2026-09-02) | conversión FP16 del ONNX + `Build-EnginesOnTarget.ps1 -Precision fp16`; ver abajo |
| 5 | Multistream: buffers reutilizables + microbatching | Parcial | Pool de tensores en `LetterboxPreprocessor` (implementado); scheduler multicámara con colas acotadas (pendiente, 32–48 h estimadas) |
| 6 | Pose en GPU y solo cuando una persona/ROI lo requiera | **Implementado** (gate) | `--pose-person-gate` / `POSE_PERSON_GATE`; pose ya va en GPU en la ruta TensorRT |
| 7 | Precisión mixta o QAT si se deterioran guantes/lentes/respirador/caídas | Condicionado al gate del paso 3 | Reentrenamiento en Colab (ver abajo) |
| 8 | Sparsity 2:4 | No empezar antes de INT8/FP16 | — |
| 9 | G-Morph | Solo si PPE+pose siguen dominando la latencia | Referencia: `2024eurosys-gmorph.pdf` |

## Detalle de lo implementado

### Exportador end-to-end por defecto (paso 2)

```powershell
uv run python tools/export_runtime_onnx.py --ppe best_ppe.pt --pose yolo26s-pose.pt --output-dir models
```

- `models/ppe.onnx` es el contrato por defecto del runtime: **end-to-end**
  `[1,300,6]` (x1, y1, x2, y2, score, clase) con decode+NMS fusionados en el
  grafo y `inference_mode: "end2end"`. El runtime nativo lo detecta por forma
  de salida y decodifica sin NMS en CPU (umbral por clase, máscara de clases,
  restore del letterbox y top-k se mantienen idénticos al camino raw).
- `models/ppe-raw.onnx` (opt-in con `--ppe-raw`) es el artefacto raw
  `[1,12,predicciones]` con `inference_mode: "raw-nms"`, para comparaciones
  offline contra el default.

### Comparación end-to-end vs raw+NMS (paso 3)

Protocolo congelado — la misma herramienta, el mismo split y los mismos
umbrales para ambos artefactos; **el split `test` solo se abre una vez
congeladas las decisiones**:

```powershell
uv run python tools/compare_ppe_models.py `
  --baseline models/ppe-raw.onnx `
  --candidate models/ppe.onnx `
  --data <dataset-cuajone.yaml> `
  --split test `
  --device cpu `
  --output artifacts/benchmarks/e2e-vs-raw.json
```

Gate de promoción: end-to-end solo reemplaza raw+NMS si **ninguna clase
crítica** (guantes, lentes, respirador, y las métricas de caída) retrocede
más allá de la tolerancia acordada, aunque el mAP agregado mejore.

### Gate de pose por persona (paso 6)

- CLI nativa: `--pose-person-gate`. Facade QA: `POSE_PERSON_GATE=1` en `.env`.
- Con el gate activo, si PPE no detectó ninguna persona en el frame se omite
  todo el stage de pose (preproceso + inferencia + decode) y `poses` queda
  vacío; la analítica de caídas no cambia para frames con personas.
- El gate es incompatible con los solapamientos PPE/pose (ORT híbrido y
  TensorRT multi-engine) porque exige la decisión de PPE antes de programar
  pose: al activarlo esas rutas pasan a ejecución serie. En escenas vacías el
  ahorro domina; en escenas con personas permanentes evaluar ambos modos con
  el benchmark antes de fijar producción.
- Pose ya se ejecuta en GPU en la ruta TensorRT. En la ruta ONNX Runtime CUDA
  híbrida pose permanece en CPU por la inestabilidad documentada del grafo
  pose en ORT CUDA 1.25 (GTX 1650 Ti); reevaluar sobre TensorRT, no sobre ORT.

### Buffers reutilizables (paso 5, parte 1)

`LetterboxPreprocessor` mantiene un pool pequeño de tensores NCHW packed: un
buffer se reutiliza solo cuando ningún consumidor asíncrono conserva
referencia (`use_count == 1`), eliminando la asignación por frame del hot path
sin carreras con el `PoseExecutor`. Cubierto por
`testPreprocessBufferReuse` y por el guard de compartición ya existente.

### Multistream y microbatching (paso 5, parte 2 — pendiente)

Sigue la recomendación del informe: scheduler con cola acotada por cámara
(sobre el modelo *latest-frame* existente), fairness y micro-batching con
deadline. Requisitos previos: exportar/validar modelos con batch dinámico
(los manifests actuales fijan batch 1), perfiles TensorRT por shape y gate de
paridad. No empezar antes de tener el baseline del paso 4 en el servidor
objetivo.

## Secuencia de precisión (pasos 4, 7, 8)

1. **FP32 de referencia**: los checkpoints fuente son la referencia; toda
   comparación nueva parte de ellos y del test congelado.
2. **FP16 TensorRT** — implementado 2026-09-02:

   TensorRT 11 compila *strongly typed*: `--fp16` ya no existe y la precisión
   proviene del grafo ONNX. Procedimiento real:

   ```powershell
   # 1) Convertir el ONNX end-to-end a FP16 (I/O float32) y correr el gate
   #    contra el test congelado EPP_val_14-08v2.1_seed42 (90 imágenes):
   uv run python tools/compare_precision_gate.py `
     --onnx "<onnx-end2end>" `
     --data "<frozen-split>/data.yaml" `
     --output artifacts/benchmarks/<fecha>/report.json

   # 2) Construir los engines desde los ONNX FP16:
   .\installer
ative\Build-EnginesOnTarget.ps1 -OutputDir <dir> `
     -PpeOnnx <ppe-fp16.onnx> -PoseOnnx <pose-fp16.onnx> -Precision fp16
   ```

   Gate 2026-09-02 (conf 0.30, IoU 0.5): delta máximo −0.0062 recall
   (Safety_boots); 0.0000 en guantes, respirador, lentes, casco, chaleco,
   tapaorejas, persona. Reporte: `artifacts/benchmarks/end2end-fp16-gate-20260902/`.

   Distribución: `internal.33` bundlea ONNX FP16 (`--half`) y el custom action
   `BuildTensorRtEngines -Precision fp16` compila los engines en el target
   durante la instalación. Verificado en instalación limpia: engines FP16
   construidos por el CA en la GTX 1650 Ti y preflight OK. Este variante
   exige GPU NVIDIA en el target; el bundle portable solo-ONNX se construye
   sin `-PpeOnnxPath`/`-PoseOnnxPath`.
3. **QAT / precisión mixta en reentrenamiento** solo si el paso anterior (o
   end-to-end) degrada guantes, lentes, respirador o caídas: ajustar en el
   notebook de Colab y repetir el gate; no promover por FPS.
4. **INT8** únicamente con dataset de calibración (`--data` en
   `export_tensorrt.py`) y gate por clase.
5. **Sparsity 2:4**: solo después de INT8/FP16, si el hardware objetivo lo
   soporta y la latencia aún no cumple el SLO.
6. **G-Morph**: solo si, después de todo lo anterior, PPE+pose siguen
   dominando la latencia (`2024eurosys-gmorph.pdf`).

## Registro de benchmarks

Cada afirmación de mejora debe quedar ligada a hardware, modelos y commit,
con p50/p95/p99 y precisión por clase, bajo
`artifacts/benchmarks/<commit>/` según la convención del informe.
