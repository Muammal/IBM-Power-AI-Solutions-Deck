# Solution 02 — Step-by-step implementation

Target: RHEL 9 ppc64le on IBM Power10/Power11, **IBM Cloud Power Virtual Server** or **on-premises**.
Each step ends with a **Check**. Do not continue until it passes.

## Step 1 — Platform (once per LPAR)

Follow [`common/README.md`](../../common/README.md):

| | IBM Cloud (PowerVS) | On-premises Power |
|---|---|---|
| C1 | `provision_powervs.sh ai-rag medium` (8 cores / 64 GB, `s1122` or `s1022`) | LPAR from your HMC: 8 dedicated cores, 64 GB, RHEL 9.4+, native Power10/11 mode |
| C2 | `os_prep.sh --smt4`, `check_power.sh` | same |
| C4 | build llama.cpp with MMA | same |
| C5 | `granite-4.2-8b-Q4_0.gguf` on llama-server :8080 | same (copy `/opt/models` in if air-gapped) |

**Check:** `curl -s localhost:8080/health` → `{"status":"ok"}`.

## Step 2 — Python environment

```bash
cd /opt/ibm-power-ai-solutions/solutions/02-enterprise-rag
../../common/scripts/make_venv.sh
source venv/bin/activate
```

**Check:** output shows `torch 2.10.0+ppc64le1` and `transformers 4.57.6`.

## Step 3 — Download the embedding model (once, ~1.1 GB)

```bash
hf download ibm-granite/granite-embedding-278m-multilingual
```

Air-gapped: run the same command on a connected machine, then copy `~/.cache/huggingface` to the LPAR
and `export HF_HUB_OFFLINE=1`.

**Check:** `python -c "from transformers import AutoModel; AutoModel.from_pretrained('ibm-granite/granite-embedding-278m-multilingual'); print('ok')"`.

## Step 4 — Put your documents in place

Copy documents into `data/docs/` (sub-folders are fine), or point `corpus.docs_dir` in `config.yaml` at an
NFS/GPFS share. Remove the two sample files when you load real ones.
Scanned PDFs (images) have no text layer: run them through **Solution 03** first, then put the Markdown output here.

**Check:** `find data/docs -type f | wc -l` shows your file count.

## Step 5 — Build the index

```bash
python src/ingest.py
```

Expected: `N chunks from .../data/docs`, `embedded in X s`, `index written to .../index`.
As a rough guide, 8 Power10 cores embed a few hundred chunks per minute in fp32; measure yours.

**Check:** `ls index/` → `chunks.jsonl meta.json vectors.npy`.

## Step 6 — Ask questions from the command line

```bash
python src/rag.py "كم يوم إجازة سنوية يستحق الموظف؟"
python src/rag.py "What is the minimum password length?"
python src/rag.py "What is the canteen menu?"        # not in the docs -> must say it is not found
```

**Check:** the first two answers are correct with citations `[1]`; the third does **not** invent an answer.

## Step 7 — Measure retrieval quality (baseline)

```bash
python eval/eval_retrieval.py --pairs data/eval_questions.jsonl
```

Write your own `eval_questions.jsonl` with 30–100 real staff questions (`question` + `source` file name).
This number is your baseline for Step 8.

**Check:** `Recall@5` printed. Missed questions are listed with `MISS:`.

## Step 8 — Fine-tune the embedder on your documents (CPU, optional but recommended)

```bash
python finetune/make_pairs.py --per-chunk 3            # Granite 4.2 writes questions per chunk
head -5 data/pairs.jsonl                               # review! delete bad lines
python finetune/finetune_embeddings.py --epochs 1 --batch 32
sed -i 's#^  model: ibm-granite/granite-embedding-278m-multilingual#  model: out/embed-finetuned#' config.yaml
python src/ingest.py                                   # re-embed with the new model
python eval/eval_retrieval.py                          # held-out synthetic pairs
python eval/eval_retrieval.py --pairs data/eval_questions.jsonl   # your real questions
```

Keep the fine-tuned model only if **your real questions** score better than the baseline from Step 7.
GPU option: `--gpu` on a separate machine whose GPU passes [`COMPLIANCE.md`](../../COMPLIANCE.md); copy `out/embed-finetuned` back.

**Check:** Recall@5 on `data/eval_questions.jsonl` ≥ baseline.

## Step 9 — Run the API

```bash
python src/app.py &
curl -s localhost:8002/health
curl -s localhost:8002/ask -H 'Content-Type: application/json' \
     -d '{"question":"شلون أبلغ عن رسالة تصيد احتيالي؟"}' | jq
```

**Check:** JSON with `answer` and `sources`.

## Step 10 — Production

```bash
sudo cp common/deploy/ai-app@.service /etc/systemd/system/ && sudo systemctl daemon-reload
sudo systemctl enable --now llama-server@granite-8b ai-app@02-enterprise-rag
```

* Put the API behind your reverse proxy with TLS and staff SSO; it binds to `127.0.0.1` only.
* Re-index nightly: `systemd` timer or cron running `venv/bin/python src/ingest.py`.
* Access control: index each department's documents separately (one `index_dir` per audience) when not
  everyone may see everything.

**Check:** after `sudo reboot`, `/health` answers and `"llm": true`.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Index was built with ... but config uses ...` | model changed after indexing | `python src/ingest.py` |
| Answers in English to Arabic questions | system prompt overridden or 3B model | keep the prompt rule; use 8B |
| Every answer "not found" | `min_score` too high for your corpus | run eval, lower `retrieval.min_score` (e.g. 0.2) |
| Slow answers (> 30 s) | thinking on, or huge `top_k` | `thinking: false`, `top_k: 4`, Q4_0 model, SMT4 |
| `httpx.ConnectError` | llama-server not running | `systemctl status llama-server@granite-8b` |
| PDF text is empty | scanned PDF | OCR it with Solution 03 first |
