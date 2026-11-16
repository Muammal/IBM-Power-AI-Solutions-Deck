# Solution 04 — Step-by-step implementation

Target: RHEL 9 ppc64le on Power10/Power11, **IBM Cloud PowerVS** or **on-premises**. Each step ends with a **Check**.

## Step 0 — Legal and privacy (before any recording is processed)

* Confirm that callers are told calls are recorded and analysed (IVR message), as your regulator requires.
* Decide the retention period (`storage.retention_days`) with legal/compliance.
* Training data (Step 8) must come from calls whose use for training is allowed, or from staff read-speech.

**Check:** written approval from compliance, retention period agreed.

## Step 1 — Platform

[`common/README.md`](../../common/README.md) C1–C5 (`medium`: 8 cores / 64 GB), then **C6**:

```bash
common/scripts/build_whisper_cpp.sh          # builds whisper-server with MMA, downloads large-v3-turbo q8_0
```

**Check:** `[ OK ] MMA kernels present`; `ls -lh /opt/models/ggml-large-v3-turbo-q8_0.bin` (~870 MB).

## Step 2 — Python environment

```bash
cd /opt/ibm-power-ai-solutions/solutions/04-speech-analytics
../../common/scripts/make_venv.sh && source venv/bin/activate
python -c "import av; print(av.__version__)"
```

**Check:** prints `15.1.0` (IBM `+ppc64le1` build).

## Step 3 — Start the ASR server

```bash
scripts/run_whisper_server.sh &          # 127.0.0.1:8090, Arabic, beam 5
```

**Check:** `curl -s -o /dev/null -w '%{http_code}\n' localhost:8090` → `200`.

## Step 4 — First transcription

Record a 30-second test call on a phone (Iraqi Arabic), copy it to the LPAR:

```bash
python src/pipeline.py test_call.m4a
cat data/calls/*.json | jq '{duration_s, transcript, analytics}'
```

**Check:** transcript is recognisable Arabic; `analytics.summary_ar` and `summary_en` describe the call.
Speed: note `duration_s` vs wall-clock time (the real-time factor). Tune with `BEAM=1` and `THREADS`.

## Step 5 — Redaction check

```bash
python -c "import sys; sys.path.insert(0,'src'); from pipeline import redact; print(redact('رقمي ٠٧٧٠١٢٣٤٥٦٧ والبطاقة 4111 1111 1111 1111'))"
```

**Check:** prints `رقمي [PHONE] والبطاقة [CARD]`. Add patterns for your own identifiers (customer IDs, national ID).

## Step 6 — Baseline accuracy

Create `data/asr/test/` with 50–200 short clips (≤ 30 s) and `metadata.csv` (`file_name,transcription`),
transcribed carefully by a person, in the dialect as spoken.

```bash
python eval/eval_asr.py data/asr/test
```

**Check:** a WER number. For Iraqi dialect call audio, expect the stock model to leave room for improvement.

## Step 7 — Batch processing

```bash
python src/pipeline.py /data/recordings/2026-11-14/
scripts/purge_old_calls.sh
```

Schedule both nightly (cron or systemd timer). Feed `data/calls/*.json` to your dashboard/CRM.

**Check:** one JSON per recording; files older than the retention period are deleted.

## Step 8 — Fine-tune Whisper on Iraqi Arabic (CPU)

Collect `data/asr/train/` (10+ hours, same format as test). Then:

```bash
../../common/scripts/make_venv.sh requirements-finetune.txt venv-ft && source venv-ft/bin/activate
python finetune/finetune_whisper.py --model openai/whisper-small --epochs 3
```

Convert the result for whisper.cpp:

```bash
git clone --depth 1 https://github.com/openai/whisper /opt/openai-whisper     # mel filters + tokenizer assets
python /opt/whisper.cpp/models/convert-h5-to-ggml.py out/whisper-iraqi /opt/openai-whisper out/
cmake --build /opt/whisper.cpp/build --target whisper-quantize   # named "quantize" in older releases
/opt/whisper.cpp/build/bin/whisper-quantize out/ggml-model.bin /opt/models/ggml-whisper-iraqi-q8_0.bin q8_0
scripts/run_whisper_server.sh /opt/models/ggml-whisper-iraqi-q8_0.bin 8091 &
ASR_URL=http://127.0.0.1:8091 python eval/eval_asr.py data/asr/test
```

Switch `asr.url` only if WER is lower than the Step 6 baseline.
GPU option for `whisper-large-v3-turbo`: `--gpu` on a separate compliant machine ([`COMPLIANCE.md`](../../COMPLIANCE.md)).

**Check:** WER(fine-tuned) < WER(baseline) on the same test folder.

## Step 9 — Production

```bash
sudo tee /etc/systemd/system/whisper-server.service >/dev/null <<'EOF'
[Unit]
Description=whisper.cpp server (Arabic ASR)
After=network-online.target
[Service]
User=ai
ExecStart=/opt/ibm-power-ai-solutions/solutions/04-speech-analytics/scripts/run_whisper_server.sh
Restart=on-failure
[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable --now whisper-server ai-app@04-speech-analytics
curl -s localhost:8004/health
```

**Check:** `{"status":"ok","asr":true}` after a reboot.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Transcript in English letters or translated | language auto-detect | keep `-l ar` (`ASR_LANG=ar`) |
| Repeated phrases ("hallucination loop") on silence | long silences | trim silence first, or use `-bs 5`; split calls > 10 min |
| `av` cannot open file | rare codec (AMR from old switches) | convert on the PBX side to WAV/MP3 |
| Slow | beam 5, too many threads | `BEAM=1`, `THREADS=<physical cores>`, SMT4 |
| Fine-tune OOM | batch too big | `--batch 4`, clips ≤ 30 s |
