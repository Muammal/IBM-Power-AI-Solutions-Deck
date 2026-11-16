# Solution 09 — Agentic IT-operations assistant (Granite 4.2 tool calling)

<div dir="rtl">

**ملخص:** مساعد ذكي لفرق تقنية المعلومات يفحص الخوادم والخدمات بنفسه (المساحة، الذاكرة، حالة الخدمات، السجلات،
إعدادات IBM Power)، ويقرأ أدلة التشغيل، ثم يشرح المشكلة ويقترح الحل بالعربية أو الإنجليزية. أي إجراء يغيّر النظام
(إعادة تشغيل خدمة، فتح تذكرة) لا يُنفَّذ إلا بموافقة موظف. يعمل على المعالج المركزي لخوادم IBM Power.

</div>

## The problem

Operations teams in Iraqi banks, ministries and telecoms run many Linux services on IBM Power, often with a
small team covering nights. Diagnosing "the system is slow" means running the same ten commands and reading
runbooks under pressure. Sending logs to a foreign AI service is not allowed.

## What it does

```
 operator: "The RAG API is slow, why?"   (Arabic or English; CLI or API)
        |
        v
 Agent loop (src/agent.py) ── Granite 4.2-8B on llama-server, native tool calling, thinking on
        |   ^
        |   └── tool results (truncated, audited)
        v
 tools (src/tools_ops.py)
   read-only, run freely:      run_diagnostic(disk|memory|load|top_cpu|smt|lpar_info|...)
                               service_status · service_logs · check_http · search_runbooks (→ Solution 02)
   change something, need YES: restart_service · create_ticket
        |
        v
 answer: findings (with numbers) → likely cause → next action / proposed change
 data/audit.jsonl: every task, tool call, approval and answer
        |
        v
 good runs → finetune/collect_traces.py → LoRA on Granite (CPU) → better at your environment
```

This loop is the reference implementation that the `AGENTS.md` of Solutions 02–08 use for their agentic workflows.

## Models

| Role | Model | Licence | Notes |
|---|---|---|---|
| Agent brain | `ibm-granite/granite-4.2-8b` (GGUF Q4_0) | Apache-2.0 | reasoning + tool calling, Arabic tested |
| Harder incidents (optional) | `ibm-granite/granite-4.2-30b` (GGUF Q4_0) | Apache-2.0 | 16+ cores |
| Fine-tuned agent | LoRA on Granite 4.2 with your approved runs | yours | `finetune/collect_traces.py` + `common/python/lora_sft.py` |

## Compute and compliance

CPU-only; the agent itself is a few hundred lines of Python. All diagnostics and logs stay on-prem.
See [`COMPLIANCE.md`](../../COMPLIANCE.md).

## Safety design

* Tools are **fixed commands** (no shell, no model-generated command lines) and **allow-lists** for services and hosts.
* Every tool that changes state requires **human approval** (CLI: y/N prompt; API: queued until `POST /approvals/{id}`).
* Step limit (`max_steps`), output truncation, and an append-only **audit log**.

## Quick start (after common steps C1–C5)

```bash
cd /opt/ibm-power-ai-solutions/solutions/09-it-ops-agent
../../common/scripts/make_venv.sh && source venv/bin/activate
python src/cli.py "How much free disk space do we have, and is SMT set correctly?"
python eval/eval_agent.py
```

## Files

| Path | Purpose |
|---|---|
| `config.yaml` | LLM, step limit, allow-lists, prompt |
| `src/agent.py` | the generic tool-calling loop, approval gate, audit log (`@tool` decorator) |
| `src/tools_ops.py` | operations tools (read-only + approval-gated) |
| `src/cli.py` | interactive terminal assistant |
| `src/app.py` | FastAPI `/tasks`, `/approvals`, `/approvals/{id}` |
| `eval/eval_agent.py`, `eval/scenarios.jsonl` | scenario tests (expected / forbidden tools) |
| `finetune/collect_traces.py` | approved runs → LoRA training data |
| `data/runbooks/*.md` | sample runbooks (fallback when Solution 02 is not running) |
