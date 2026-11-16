# Solution 02 — Enterprise document Q&A (Arabic/English RAG) on IBM Power

<div dir="rtl">

**ملخص:** مساعد يجيب عن أسئلة الموظفين من وثائق الشركة نفسها (سياسات، إجراءات، عقود) بالعربية الفصحى
واللهجة العراقية والإنجليزية، مع ذكر المصدر لكل إجابة. يعمل بالكامل على المعالج المركزي لخوادم IBM Power10/Power11
داخل مركز بيانات الشركة، دون أي بطاقة رسومية، ودون أن تخرج البيانات من العراق.

</div>

## The problem

Iraqi banks, ministries, oil & gas and telecom operators keep their rules in hundreds of PDF and Word files,
in Arabic and English. Staff ask the same questions again and again, and answers must come from the
**approved** text, not from a chatbot's imagination. The documents are sensitive, so they cannot be sent to a
foreign AI service.

## What it does

* Indexes your `.pdf`, `.docx`, `.md`, `.txt` files (Arabic and English) into a local vector index.
* Answers questions in the user's language with **citations** `[1][2]` to the source passages.
* Says "not found" instead of inventing an answer when the documents do not cover the question.
* Fine-tunes the embedding model **on your own documents** (CPU) to improve retrieval of your terminology.

```
 Documents (.pdf .docx .md)                          Question (AR / EN / Iraqi dialect)
        |                                                       |
  src/ingest.py  chunk (1,200 chars) ---------+                 v
        |                                     |        src/app.py  POST /ask
        v                                     |                 |
 Granite Embedding 278M multilingual   <------+---------  embed question (same model)
 (torch, CLS pooling, MMA via OpenBLAS)                         |
        |                                                       v
 index/vectors.npy + chunks.jsonl  ---------->  cosine top-k (one NumPy matmul, MMA)
                                                                |
                                                                v
                                    Granite 4.2-8B on llama-server (Q4_0, MMA kernels)
                                    "answer only from context, cite [n]"
                                                                |
                                                                v
                                              answer + sources (file, score)
```

## Models

| Role | Model | Licence | Why |
|---|---|---|---|
| Generator | `ibm-granite/granite-4.2-8b` (GGUF Q4_0) | Apache-2.0 | Arabic tested, 128K context, follows "cite or refuse" instructions well |
| Embeddings | `ibm-granite/granite-embedding-278m-multilingual` | Apache-2.0 | Arabic + English in one vector space, small enough for CPU |
| Fine-tuned embeddings | your copy of the above | yours | learns your acronyms, product names, Iraqi phrasing |

## Compute and compliance

**CPU-only, on-premises by default.** Both models run on Power10/Power11 cores using MMA. Fine-tuning the
278M embedder is a CPU job (30–60 min for ~5,000 pairs on 8 cores). No GPU is required. See
[`COMPLIANCE.md`](../../COMPLIANCE.md).

| Size | LPAR | Notes |
|---|---|---|
| Pilot (≤ 5,000 pages, ≤ 20 users) | 8 cores, 64 GB | Granite 4.2-8B Q4_0 + embedder |
| Department (≤ 50,000 pages) | 16 cores, 128 GB | two llama-server slots, or 3B for faster answers |

## Quick start (after common steps C1–C5)

```bash
cd /opt/ibm-power-ai-solutions/solutions/02-enterprise-rag
../../common/scripts/make_venv.sh && source venv/bin/activate
python src/ingest.py                                   # builds index/ from data/docs
python src/rag.py "كم يوم إجازة سنوية يستحق الموظف؟"
python eval/eval_retrieval.py --pairs data/eval_questions.jsonl
python src/app.py                                      # API on 127.0.0.1:8002
```

## Files

| Path | Purpose |
|---|---|
| `config.yaml` | models, chunking, top-k, prompts. The only file you edit |
| `src/embedder.py` | transformers-based embeddings (CLS/mean pooling) |
| `src/ingest.py` | load → chunk → embed → `index/` |
| `src/rag.py` | retrieve + answer with citations (CLI) |
| `src/app.py` | FastAPI: `/ask`, `/search`, `/health` |
| `finetune/make_pairs.py` | Granite 4.2 writes training questions from your chunks |
| `finetune/finetune_embeddings.py` | contrastive fine-tuning on CPU |
| `eval/eval_retrieval.py` | Recall@k and MRR, before/after fine-tuning |
| `data/docs/` | two sample policies (Arabic, English). Replace with yours |
| `IMPLEMENTATION.md` | step-by-step on IBM Cloud PowerVS or on-prem Power |
| `RESOURCES.md` | where to learn each piece |
| `AGENTS.md` | rules for AI coding agents + the agentic workflow design + change log |
| `docs/*.docx` | Word versions of the solution guide and the agent file |
