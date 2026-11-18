# IBM Power AI Solutions for Iraqi Enterprises

<div dir="rtl">

**مجموعة حلول ذكاء اصطناعي بسيطة وعملية** تعمل على خوادم **IBM Power10 / Power11** باستخدام المعالج المركزي فقط
(وحدات MMA المدمجة)، داخل مركز بيانات المؤسسة في العراق، بنماذج مفتوحة المصدر حديثة وقابلة للتدريب على بياناتكم
(IBM Granite 4.2 وغيرها). كل حل له مجلد خاص يحتوي على: شرح، خطوات تنفيذ على IBM Cloud أو على خوادم محلية، الكود،
مصادر للتعلم، ملف Word، وملف خاص بوكلاء الذكاء الاصطناعي (AGENTS).

</div>

Twelve production-shaped AI solutions that run on **IBM Power10/Power11 (ppc64le)**, on
**IBM Cloud Power Virtual Server** or **on-premises**, built on current **open-weight, fine-tunable** models and
IBM's Power-optimised Python wheels. They are designed for Iraqi enterprises: Arabic (including Iraqi dialect)
and Kurdish where language matters, **CPU-only by default** so no export-controlled GPU is needed, and fully
offline-capable so data can stay in the country.

## The solutions

| # | Solution | Flagship model(s) | Fine-tuning (CPU) | Typical users |
|---|---|---|---|---|
| 01 | [Arabic/English chatbot](solutions/01-arabic-chatbot/) | Granite 4.2-3B/8B | LoRA on your conversations | any organisation (template) |
| 02 | [Enterprise document Q&A (RAG)](solutions/02-enterprise-rag/) | Granite 4.2-8B + Granite Embedding 278M multilingual | embedder on your documents | banks, ministries, oil & gas |
| 03 | [Document intelligence (OCR → JSON)](solutions/03-document-intelligence/) | Tesseract ara+eng, Docling, Granite 4.2-3B | LoRA on reviewed documents | finance, procurement, customs |
| 04 | [Speech analytics for call centers](solutions/04-speech-analytics/) | Whisper large-v3-turbo (whisper.cpp) + Granite 4.2-8B | Whisper on Iraqi recordings | telecoms, banks, utilities |
| 05 | [Load / demand forecasting](solutions/05-time-series-forecasting/) | Granite TimeSeries TTM-R2 | head + exogenous drivers (minutes) | electricity, fuel, water, telecom |
| 06 | [Fraud & anomaly detection](solutions/06-fraud-anomaly-detection/) | XGBoost + Isolation Forest + Granite 4.2 explanations | weekly retraining on analyst labels | banks, wallets, payment switches |
| 07 | [Ticket classification & routing](solutions/07-ticket-classification/) | Granite 4.2 (teacher) → Granite Embedding 278M (student, ONNX) | encoder fine-tuning | contact centers |
| 08 | [Arabic ↔ Kurdish ↔ English translation](solutions/08-translation-ar-ku-en/) | MADLAD-400-3B-MT (+ Granite post-edit) | LoRA on translation memory | government, KRG, NGOs, banks |
| 09 | [Agentic IT-operations assistant](solutions/09-it-ops-agent/) | Granite 4.2-8B/30B tool calling | LoRA on approved agent runs | data-center operations |

### Security and computer vision

| # | Solution | Flagship model(s) | Fine-tuning | Typical users |
|---|---|---|---|---|
| 10 | [Weapon detection across CCTV cameras](solutions/10-cctv-weapon-detection/) | RT-DETRv2 (ONNX Runtime) trained on your footage | detector on site footage + operator verdicts | banks, ministries, universities, oil & gas sites |
| 11 | [National open-source situational awareness](solutions/11-osint-situational-awareness/) | Granite 4.2 event extraction over public feeds, CISA KEV, CT logs | LoRA on analyst corrections | SOC, security, operations, business continuity |
| 12 | [Traffic ANPR and section-speed enforcement](solutions/12-traffic-anpr-speed/) | RT-DETRv2 vehicles + plates, CRNN-CTC plate OCR | detector + OCR on your cameras | traffic directorates, highway operators |

These three run as long-lived services with operator consoles, role-based access, audit logs, Prometheus metrics,
evidence integrity (hashes, signatures), retention jobs, release gates for new models, and test suites. They keep
humans in charge of every consequential decision, and are deliberately limited: no face recognition, no profiling
of individuals, no plate search (see [`COMPLIANCE.md`](COMPLIANCE.md#surveillance-cctv-and-enforcement)).

Every solution folder contains the same set of files:

| File | What it is |
|---|---|
| `README.md` | the problem, architecture, models + licences, compute/compliance, sizing, quick start |
| `IMPLEMENTATION.md` | **step-by-step** on IBM Cloud PowerVS **or** on-prem Power, each step with a check |
| `RESOURCES.md` | where to **learn** to build and run this solution (courses, docs, papers, exercises) |
| `AGENTS.md` | for **AI agents**: brief for coding agents, the solution's **agentic workflow** design, and the **agent change log** |
| `docs/<solution>.docx` | Word version of README + IMPLEMENTATION + RESOURCES |
| `docs/AGENTS.docx` | Word version of AGENTS.md (the "agent docx") |
| `config.yaml`, `requirements.txt`, `src/`, `finetune/`, `eval/`, `scripts/serve.sh` | working code |

## Shared platform

[`common/README.md`](common/README.md) holds the platform steps every solution starts with (C1–C8):
provisioning PowerVS or an on-prem LPAR, OS preparation and the **MMA** check, IBM wheel environments,
llama.cpp/whisper.cpp built for Power10 (`-mcpu=power10`, MMA kernels), Granite 4.2 on `llama-server`,
systemd services, and converting fine-tuned models to GGUF.

```
                        IBM Power10 / Power11 LPAR  (RHEL 9, ppc64le, SMT4, MMA)
 ┌───────────────────────────────────────────────────────────────────────────────────────────┐
 │ llama-server :8080  Granite 4.2-8B Q4_0 (ggml MMA kernels, OpenAI API, --jinja)            │◄─┐
 │ whisper-server :8090  Whisper large-v3-turbo q8_0 (Solution 04)                            │  │ HTTP
 │                                                                                            │  │
 │ Solution APIs (FastAPI, 127.0.0.1)                                                         │  │
 │  01 :7860 chatbot · 02 :8002 RAG · 03 :8003 documents · 04 :8004 calls · 05 :8005 forecast │──┘
 │  06 :8006 fraud · 07 :8007 tickets (ONNX Runtime) · 08 :8008 translation · 09 :8009 agent  │
 │  10 :8010 CCTV weapons · 11 :8011 situational awareness · 12 :8012 traffic ANPR/speed      │
 │                                                                                            │
 │ PyTorch 2.10 / ONNX Runtime 1.23 / XGBoost 3.0 from IBM's ppc64le wheel index (OpenBLAS    │
 │ POWER10 kernels) · common/python: llm_client.py, power_runtime.py, lora_sft.py             │
 └───────────────────────────────────────────────────────────────────────────────────────────┘
          reverse proxy + SSO ──► staff browsers, ERP/CRM, ticketing, monitoring
```

## Key documents

* [`docs/SIZING.pdf`](docs/SIZING.pdf): **sizing guide**: cores, memory, storage, network and servers for every
  solution, from a 300-person organisation to a country-wide programme, best-quality and lean profiles. Built from
  the capacity model in [`tools/sizing_model.py`](tools/sizing_model.py) (`python tools/build_sizing_pdf.py`).
* [`COMPLIANCE.md`](COMPLIANCE.md): why **CPU-only** is the default for Iraq (US EAR 3A090/4A090, Country Group D:1),
  an indicative screening of GPUs, data residency.
* [`MODELS.md`](MODELS.md): the model matrix (licences, sizes, `transformers` compatibility), alternatives considered,
  and the quarterly **update routine** that keeps the repo current with flagship open models.
* [`RESOURCES.md`](RESOURCES.md): learning path from zero to IBM Power AI engineer.
* [`AGENTS.md`](AGENTS.md): repository-wide rules for AI coding agents and the agent change protocol.

## Resources this repo is built on

* **ppc64le/pyeco**: IBM's hub for Power-optimised Python wheels and examples: <https://github.com/ppc64le/pyeco>
* **IBM Power wheel index**: <https://wheels.developerfirst.ibm.com/ppc64le/linux> (dated snapshot
  `linux-v2026.03.31/+simple/` pinned in every `requirements.txt`)
* **RocketCE conda channel**: <https://anaconda.org/rocketce> (older Power conda builds; see `common/README.md` C3)

## Quick start

```bash
# on a RHEL 9 ppc64le LPAR (Power10/Power11)
sudo git clone <this repo> /opt/ibm-power-ai-solutions && cd /opt/ibm-power-ai-solutions
sudo common/scripts/os_prep.sh --smt4 && common/scripts/check_power.sh
common/scripts/build_llama_server.sh
cd solutions/05-time-series-forecasting && ../../common/scripts/make_venv.sh && source venv/bin/activate
python data/make_sample.py && python src/forecast.py
```

Regenerate the Word files after editing any Markdown:

```bash
pip install -r tools/requirements-docs.txt
python tools/build_docx.py            # all solutions + root documents
```

## Validation status (2026-11-15, Solutions 10-12 on 2026-11-16)

Power10/Power11 hardware was **not available** while building this repo. What was verified, and how:

| Item | How | Result |
|---|---|---|
| Package availability | `tools/check_requirements.py`: full pip resolution of every `requirements*.txt` for manylinux ≤ 2.34 ppc64le / CPython 3.12, binary wheels only, markers evaluated for Linux | all 16 files resolve (17–106 packages each, IBM `+ppc64le` builds for torch, numpy, xgboost, onnxruntime, av, docling-parse, ...) |
| Model facts | Hugging Face API: licence, parameters, `transformers_version`, `model_type` for every model in `MODELS.md` | recorded in `MODELS.md` |
| Granite 4.2 chat template | tokenizer on x86, transformers 4.57.6 | `enable_thinking=False` → `<think></think>` prefill; tools rendered |
| Solution 05 | full run on x86 CPU, same library versions | zero-shot MAE 3.60 MW → fine-tuned 2.94 MW; API OK |
| Solution 06 | full run on x86 CPU | PR-AUC 0.749 at 0.6% fraud rate; 28–40 ms per score |
| Solution 07 | full run on x86 CPU | train → ONNX (parity error 1.6e-6) → int8 (278 MB vs 1110 MB) → ONNX Runtime API, ~0.1 s per ticket |
| Solution 09 | agent loop with a scripted LLM stub | runbook tool call, approval denial, injected service name (`rm -rf /`) rejected by the allow-list, audit log written |
| Solutions 03, 04, 08 helpers | unit checks on x86 | invoice validation on gold/bad data, Arabic-Indic digit normalisation, phone/card redaction, Arabic WER, chrF |
| Solution 10 zero-shot | full pipeline on 8 public Wikimedia Commons videos + live service in the browser (2026-11-17) | 0 false alarms on 5.6 min of no-weapon footage; 3 of 5 firearms videos alerted; operator confirm, audit, integrity verified |
| Solution 10 | `pytest` (9) + end-to-end run with the real ONNX detector on a test video | rules, hash chain, retention, escalation, auth pass; 0 false alarms on 30 s video; 108 ms/frame at 8 threads (x86) |
| Shared RT-DETRv2 export | PyTorch vs ONNX Runtime parity | classic TorchScript exporter found **wrong by ~0.1** for RT-DETRv2 → switched to the dynamo exporter: 1.8e-5 |
| Solution 11 | `pytest` (8) + live run against public feeds | UN News, Al Jazeera, CISA KEV collected (1,781 KEV → 87 matched to a sample inventory); stale INA feed and GDELT rate-limit reported |
| Solution 12 | `pytest` (10) + OCR training on 3,000 synthetic plate crops | full read → section speed → signed violation → review → evidence path; OCR 98.7% exact on 300 held-out crops, ONNX parity 1.3e-5; release gate correctly **failed** (99.2% precision on confident reads < 99.5% required) |
| All Python | `py_compile`; all shell scripts `bash -n` | OK |
| Not executed | llama.cpp/whisper.cpp Power builds, PowerVS commands, systemd units, Docling, MADLAD and Whisper inference, live RTSP cameras, trained weapon/plate models (need site data) | run the checks in each IMPLEMENTATION.md on your LPAR |

## License

Apache-2.0 for the code and documents in this repository. Each model keeps its own licence (see `MODELS.md`).
