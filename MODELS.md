# Model matrix

**Last verified against Hugging Face: 2026-11-15.** Licences, sizes and `transformers` versions below were read
from each repo's metadata (`config.json`, model card) on that date. Re-check before production use; the
"How to update" section at the end explains the routine.

Selection rules: open weights, **fine-tunable**, permissive licence (Apache-2.0 / MIT), runs on **CPU** on
ppc64le with the IBM wheel stack (`transformers 4.57.6`, `torch 2.10.0+ppc64le1`) or with llama.cpp/whisper.cpp,
and handles **Arabic** where the task needs language.

## Defaults used by the solutions

| Role | Model | Params | Licence | Runtime on Power | Used in |
|---|---|---:|---|---|---|
| Flagship LLM (reasoning, tools, Arabic tested) | `ibm-granite/granite-4.2-8b` (+ `-GGUF`) | 8.8B | Apache-2.0 | llama-server Q4_0/Q8_0 (MMA) | 01, 02, 04, 06, 08, 09 |
| Small LLM (extraction, labelling) | `ibm-granite/granite-4.2-3b` (+ `-GGUF`) | 3.7B | Apache-2.0 | llama-server Q4_0/Q8_0 | 03, 07 |
| Large LLM (optional) | `ibm-granite/granite-4.2-30b` (+ `-GGUF`) | 29B | Apache-2.0 | llama-server Q4_0, 16+ cores | 09 |
| Multilingual embeddings (Arabic) | `ibm-granite/granite-embedding-278m-multilingual` | 278M | Apache-2.0 | sentence-transformers (torch) | 02 |
| Classifier backbone | same embedding model (XLM-R architecture) | 278M | Apache-2.0 | torch → ONNX Runtime | 07 |
| Document conversion | Docling + `ibm-granite/granite-docling-258M` (optional VLM) | 258M | Apache-2.0 | docling (torch) | 03 |
| OCR | Tesseract 5 with `ara` + `eng` | n/a | Apache-2.0 | RHEL/EPEL package | 03 |
| Speech-to-text (Arabic) | `openai/whisper-large-v3-turbo` (ggml q8_0) | 809M | MIT | whisper.cpp (MMA) | 04 |
| Time-series forecasting | `ibm-granite/granite-timeseries-ttm-r2` | ~1M | Apache-2.0 | granite-tsfm (torch) | 05 |
| Tabular fraud model | XGBoost 3 (trained by you) | n/a | Apache-2.0 | IBM `xgboost` wheel | 06 |
| Translation AR / KU (Sorani, Kurmanji) / EN | `google/madlad400-3b-mt` | 2.9B | Apache-2.0 | transformers (torch, bf16) | 08 |
| Object detection (base for weapons, vehicles, plates) | `PekingU/rtdetr_v2_r18vd` (RT-DETRv2, COCO) | 20M | Apache-2.0 | ONNX Runtime (MLAS, MMA); export via dynamo exporter | 10, 12 |
| Zero-shot weapon detection (interim, labelling teacher) | `google/owlv2-base-patch16-ensemble` (OWLv2) | 155M | Apache-2.0 | transformers (torch), 640 px, ~1.3 s/frame | 10 |
| Plate OCR | CRNN-CTC defined in `solutions/12-traffic-anpr-speed/src/ocr.py`, trained on your crops | ~1M | yours | ONNX Runtime | 12 |
| Event extraction from public news | `ibm-granite/granite-4.2-8b` / `-3b` | 8.8B / 3.7B | Apache-2.0 | llama-server | 11 |

RT-DETRv2 (verified 2026-11-16): `rt_detr_v2` is native in transformers 4.57.6. Export it with
`common/python/vision/export_rtdetr.py`: the classic TorchScript ONNX exporter produced scores ~0.1 off for this
model; the dynamo exporter matches PyTorch to 1.8e-5 (x86) and 5.5e-6 (Power10, 2026-11-19), but only with
onnxscript 0.7.2: onnxscript 0.5.6 exports it with scores 0.19 off. There is no trustworthy permissively licensed weapon or Iraqi
plate model on Hugging Face (checked 2026-11-16: small, unvetted or unlicensed checkpoints only; most plate models
are AGPL YOLO derivatives), so Solutions 10 and 12 train their own on site data. Ultralytics YOLO (AGPL-3.0) is
excluded by the licence rule. Zero-shot alternatives checked 2026-11-17: OmDet-Turbo (Apache-2.0) needs torchvision +
timm and failed with current timm; OWLv2 needs only transformers and was adopted.

Granite 4.2 (released 2026-08-25): dense `GraniteForCausalLM`, 128K native context, built-in
`<think>` reasoning with an `enable_thinking` switch, tool calling, Arabic among the 12 tested languages.
It loads with `transformers>=4.57.1`, so it works with IBM's pinned stack. IBM publishes official GGUF files
(`ibm-granite/granite-4.2-{3b,8b,30b}-GGUF`) including the MMA-friendly `Q4_0` and `Q8_0`.

## Alternatives considered (and why they are not the default)

| Model | Licence | Why not default today | When to use it |
|---|---|---|---|
| `Qwen/Qwen3.5-9B`, `Qwen3.5-35B-A3B`, `Qwen3.8-27B` | Apache-2.0 | `qwen3_5` architecture is **not in transformers 4.57.6** (needs 5.x); GGUF via llama.cpp works | Swap into llama-server as a GGUF if it beats Granite on your Arabic eval |
| `Qwen/Qwen3-ASR-1.7B-hf` | Apache-2.0 | needs `transformers` 5.13 | ASR once you move to the rolling IBM index |
| `mistralai/Voxtral-Mini-4B-Realtime-Arabic` | see card | vLLM-only, newer vLLM than the dated snapshot | Real-time Arabic ASR on the rolling index (`vllm 0.30 +cpu`) |
| `mistralai/LIDstral-Arabic` | Apache-2.0 | fastText (no IBM wheel); dialect ID only | Arabic dialect / Sorani identification front-end |
| `ibm-granite/granite-speech-5.0-2b` | Apache-2.0 | **no Arabic** (EN, FR, DE, ES, PT, JA) | English/European call centers |
| `google/gemma-4-*`, `google/embeddinggemma-2` | Apache-2.0 | need transformers 5.10+/5.18+ | After upgrading to the rolling index |
| `openai/gpt-oss-20b` | Apache-2.0 | 21B MoE in MXFP4; heavier on CPU than Granite 4.2-8B for similar Arabic | Agent workloads needing strong English tool use |
| `humain-ai/ALLaM-7B-Instruct-preview` | Apache-2.0 | 2025 model, no reasoning/tool calling | Arabic-first chat (preset in Solution 01) |
| `BAAI/bge-m3` | MIT | larger (568M), slower on CPU | Highest Arabic retrieval quality if latency allows |
| `ibm-granite/granite-timeseries-tspulse-r1` | Apache-2.0 | anomaly detection on time series, not transactions | Add to Solution 06 for metric streams |
| `ibm-granite/granite-guardian-3.3-8b` | Apache-2.0 | English-centric safety classifier | Guardrails for English traffic in 02/09 |
| NLLB-200 | CC-BY-NC | **non-commercial** | Excluded |

## CPU memory planning

Weights + KV cache + runtime, one or two concurrent users:

| Model | GGUF Q4_0 | GGUF Q8_0 | HF bf16 (for fine-tuning) |
|---|---:|---:|---:|
| Granite 4.2-3B | 3 GB | 4.5 GB | 8 GB (+ LoRA training ≈ 20 GB) |
| Granite 4.2-8B | 6 GB | 10 GB | 18 GB (+ LoRA training ≈ 45 GB) |
| Granite 4.2-30B | 18 GB | 32 GB | 60 GB (training: use 3B/8B instead) |
| MADLAD-400-3B | n/a | n/a | 6 GB (+ LoRA training ≈ 18 GB) |

## How to update this matrix (do it quarterly)

```bash
# newest releases from the publishers we track
for a in ibm-granite Qwen mistralai google openai; do
  curl -s "https://huggingface.co/api/models?author=$a&sort=createdAt&direction=-1&limit=15" | jq -r '.[].id'
done
# licence / transformers version / architecture of a candidate
curl -s https://huggingface.co/api/models/<id> | jq '.tags, .safetensors.total'
curl -s https://huggingface.co/<id>/raw/main/config.json | jq '.transformers_version, .model_type'
```

A candidate qualifies when: licence is Apache-2.0/MIT, `model_type` exists in the transformers version pinned in
the solution (or a GGUF exists for llama.cpp), and it wins on the solution's own evaluation script. Change the
model in the solution's `config.yaml`, run its eval, then record the change in that solution's `AGENTS.md`
change log and regenerate the docx files (`python tools/build_docx.py`).
