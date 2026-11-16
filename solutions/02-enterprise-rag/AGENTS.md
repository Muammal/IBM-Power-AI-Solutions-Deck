# AGENTS.md — Solution 02 (Enterprise RAG)

This file has three parts:

1. **Agent brief**: what an AI coding agent (Claude Code, Codex, Copilot, watsonx Code Assistant, ...) must know
   before changing this solution.
2. **Agentic workflow**: how to run this solution as a multi-step AI agent at runtime.
3. **Agent change log**: every change made by an agent is recorded here.

Read the repository-wide rules in [`/AGENTS.md`](../../AGENTS.md) first.

---

## 1. Agent brief

**Purpose:** answer staff questions from company documents, Arabic and English, with citations, on CPU.

**Invariants (never break these):**

* Runs on **ppc64le, CPU-only**. Do not add CUDA-only packages, GPU code paths that are required, or packages
  without a ppc64le wheel. Check with `pip download --platform manylinux_2_34_ppc64le --only-binary=:all:`.
* Answers must come **only** from retrieved context and cite `[n]`. Do not weaken the system prompt rule.
* Model IDs and tunables live in `config.yaml`, not in code.
* Default models stay **Apache-2.0/MIT** licensed (see `/MODELS.md`).
* The index format (`vectors.npy` float32 normalised + `chunks.jsonl` + `meta.json`) is read by `rag.py` and
  by Solution 09's agent tools: change it only together with both readers.

**Map:**

| Change you want | File |
|---|---|
| Model, chunk size, top-k, prompt | `config.yaml` |
| Loading a new file type | `src/ingest.py: read_text()` |
| Retrieval logic (hybrid, rerank, filters) | `src/rag.py: RAG.retrieve()` |
| API endpoints | `src/app.py` |
| Training data generation / training | `finetune/*.py` |

**How to verify a change** (all must pass):

```bash
python -m py_compile src/*.py finetune/*.py eval/*.py
python src/ingest.py
python eval/eval_retrieval.py --pairs data/eval_questions.jsonl     # Recall@5 must not drop
python src/rag.py "What is the canteen menu?"                       # must say "not found", not invent
```

**Do not:** commit `index/`, `out/`, `venv/`, or real company documents; call external AI APIs; log full
document text or questions at INFO level (they can contain personal data).

---

## 2. Agentic workflow (runtime)

Plain RAG answers in one shot. For complex questions ("compare the leave rules for contractors and permanent
staff") an agent loop does better: plan → search several times → check the answer against the sources.

```
 user question
      |
      v
 [Planner]  Granite 4.2-8B, thinking=on (hard questions only)
      |  sub-questions
      v
 [Researcher] -- tool: search_documents(query, top_k) --> POST /search  (no LLM, fast)
      |  passages (deduplicated)
      v
 [Writer]   Granite 4.2-8B, thinking=off, "answer only from passages, cite [n]"
      |
      v
 [Verifier] Granite 4.2-3B, json_schema: {supported: bool, unsupported_sentences: [...]}
      |-- supported ------------> answer + sources
      |-- not supported (1 retry) -> Researcher with the unsupported sentences as new queries
      |-- still not supported ----> "not found" + passages for a human to read
```

**Tools exposed to the model** (OpenAI tool format, served by `llm_client.LLM.chat(tools=...)`):

```json
{"type": "function", "function": {
  "name": "search_documents",
  "description": "Search the company's approved documents. Returns numbered passages with source file names.",
  "parameters": {"type": "object", "properties": {
      "query": {"type": "string", "description": "Search query in Arabic or English"},
      "top_k": {"type": "integer", "minimum": 1, "maximum": 10}},
    "required": ["query"]}}}
```

**Guardrails:** max 4 tool calls per question; read-only tools only; the Verifier must pass before an answer is
shown; every run logs (question hash, tool calls, sources, verdict) for audit.

**Implementation path:** Solution 09 contains a working tool-calling loop (`src/agent.py`). Register
`search_documents` there, pointing at this solution's `/search` endpoint, to get the Researcher role.

**Learning from use (continuous fine-tuning):** store questions where users clicked "not helpful", have a domain
expert write the correct source passage, append to `data/pairs.jsonl`, and re-run Step 8 monthly.

---

## 3. Agent change protocol

Every agent that changes this solution must:

1. Make the change and run **all** verification commands above.
2. If docs changed, regenerate Word files: `python tools/build_docx.py solutions/02-enterprise-rag`.
3. Append one row to the table below (newest last). Never rewrite earlier rows.

### Agent change log

| Date | Agent / author | Change | Verification result |
|---|---|---|---|
| 2026-11-15 | Claude Code (Opus 5.5) | Initial version: ingest, RAG, API, embedding fine-tuning, eval | `py_compile` OK; requirements resolve for ppc64le/cp312 (42 packages, binary only); ingest + retrieval run on x86 (transformers 4.57.6): Recall@1 = 0.75 on `data/eval_questions.jsonl`, misses are Arabic questions about the English policy (cross-lingual, the case Step 8 fine-tuning targets). LLM answering not run in this session; not yet run on Power hardware |
