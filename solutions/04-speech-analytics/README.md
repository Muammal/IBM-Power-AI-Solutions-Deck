# Solution 04 — Arabic speech analytics for call centers (Iraqi dialect)

<div dir="rtl">

**ملخص:** تحويل مكالمات مركز الاتصال (باللهجة العراقية أو الفصحى أو الإنجليزية) إلى نص، ثم تلخيصها وتصنيفها
وقياس رضا العميل وتحديد المكالمات التي تحتاج إلى متابعة، مع إخفاء أرقام الهواتف والبطاقات. يعمل بالكامل على
المعالج المركزي لخوادم IBM Power داخل الشركة، ويمكن تدريب النموذج على لهجة عملائكم.

</div>

## The problem

Telecom operators, banks and ministries in Iraq record thousands of calls a day and listen to almost none of
them. Supervisors cannot see why customers call, which calls ended angry, or which promised call-backs never
happened. Recordings contain personal data and cannot go to a foreign cloud.

## What it does

```
 recording (wav/mp3/m4a/WhatsApp opus/mp4)
        |
 src/audio.py (PyAV) -> 16 kHz mono WAV
        |
 whisper-server (whisper.cpp, large-v3-turbo q8_0, Power10 MMA)  -> Arabic transcript
        |
 redaction: Iraqi mobile numbers, card numbers, IBANs -> [PHONE] [CARD] [IBAN]
        |
 Granite 4.2-8B (llama-server), JSON schema:
   summary_ar, summary_en, topic, customer_sentiment, resolved, action_items, escalate
        |
 data/calls/<call_id>.json  ->  dashboards / CRM / supervisor queue (escalate = true)
```

## Models

| Role | Model | Licence | Notes |
|---|---|---|---|
| Speech-to-text | `openai/whisper-large-v3-turbo` (ggml q8_0, 870 MB) | MIT | multilingual incl. Arabic; dialect quality improves a lot with fine-tuning |
| Fine-tuned ASR | Whisper small or turbo + your Iraqi recordings | yours | `finetune/finetune_whisper.py` |
| Analytics | `ibm-granite/granite-4.2-8b` (GGUF Q4_0) | Apache-2.0 | summaries in Arabic and English |

Not used: IBM Granite Speech 5.0 (excellent, but no Arabic yet); Qwen3-ASR and Mistral Voxtral Realtime
Arabic need newer transformers/vLLM than IBM's dated wheel snapshot (see [`MODELS.md`](../../MODELS.md)).

## Compute and compliance

CPU-only. whisper.cpp uses the same ggml MMA kernels as llama.cpp. Fine-tuning whisper-small on CPU is
practical for 10–50 hours of audio. Recordings never leave the LPAR. See [`COMPLIANCE.md`](../../COMPLIANCE.md).

| Load | LPAR | Rough capacity (measure yours) |
|---|---|---|
| Pilot | 8 cores, 64 GB | batch overnight: a few thousand minutes of audio |
| Production | 16 cores, 128 GB | two whisper-server instances + Granite 8B |

## Quick start (after common steps C1–C6)

```bash
cd /opt/ibm-power-ai-solutions/solutions/04-speech-analytics
../../common/scripts/make_venv.sh && source venv/bin/activate
scripts/run_whisper_server.sh &                      # :8090
python src/pipeline.py /path/to/call.m4a
```

## Files

| Path | Purpose |
|---|---|
| `config.yaml` | ASR/LLM endpoints, language, redaction, retention |
| `src/audio.py` | decode anything to 16 kHz mono with PyAV |
| `src/pipeline.py` | transcribe → redact → analyse → store (CLI, file or folder) |
| `src/app.py` | FastAPI `POST /calls` |
| `src/wer.py` | Arabic-normalised word error rate |
| `eval/eval_asr.py` | WER of the running ASR on a labelled folder |
| `finetune/finetune_whisper.py` | Whisper fine-tuning on Iraqi recordings (CPU) |
| `scripts/run_whisper_server.sh` | start whisper.cpp server with Power settings |
| `scripts/purge_old_calls.sh` | retention enforcement |
