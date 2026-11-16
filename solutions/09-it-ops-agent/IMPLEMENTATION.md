# Solution 09 — Step-by-step implementation

Target: RHEL 9 ppc64le on Power10/Power11, **IBM Cloud PowerVS** or **on-premises**. Each step ends with a **Check**.

## Step 1 — Platform

[`common/README.md`](../../common/README.md) C1–C5 (`medium`: 8 cores / 64 GB, Granite 4.2-8B on :8080).
llama-server must run with `--jinja` (the C5 script does) so Granite's chat template handles tools.

**Check:** a tool-calling request works:

```bash
curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
 "messages":[{"role":"user","content":"What is the disk usage?"}],
 "tools":[{"type":"function","function":{"name":"run_diagnostic","description":"diagnostic",
   "parameters":{"type":"object","properties":{"check":{"type":"string","enum":["disk","memory"]}},"required":["check"]}}}],
 "chat_template_kwargs":{"enable_thinking":false}}' | jq '.choices[0].message.tool_calls'
```

→ a `run_diagnostic` call with `{"check":"disk"}`.

## Step 2 — Environment

```bash
cd /opt/ibm-power-ai-solutions/solutions/09-it-ops-agent
../../common/scripts/make_venv.sh && source venv/bin/activate
```

**Check:** `python -c "import sys; sys.path.insert(0,'src'); import tools_ops, agent; print(sorted(agent.TOOLS))"`
lists the seven tools.

## Step 3 — Allow-lists and permissions

* Edit `tools.allowed_services` (only services this team owns) and `tools.allowed_http_hosts`.
* For `restart_service`, give the `ai` user a narrow sudo rule, only for the allowed units:

```bash
sudo tee /etc/sudoers.d/ai-agent >/dev/null <<'EOF'
ai ALL=(root) NOPASSWD: /usr/bin/systemctl restart llama-server@granite-8b, /usr/bin/systemctl restart ai-app@02-enterprise-rag
EOF
sudo visudo -cf /etc/sudoers.d/ai-agent
```

**Check:** `visudo` reports `parsed OK`; security team approves the allow-lists.

## Step 4 — First tasks (CLI)

```bash
python src/cli.py
ops> How much free disk space do we have?
ops> هل إعداد SMT مناسب لتشغيل نماذج الذكاء الاصطناعي على هذا الخادم؟
ops> Restart llama-server@granite-8b, it is hung          # you will be asked y/N
```

**Check:** answers quote real numbers from the tools; the restart asks for approval; `data/audit.jsonl` has the events.

## Step 5 — Connect the knowledge base (optional)

Run Solution 02 with your runbooks and KB articles in its `data/docs/`; `search_runbooks` then uses its
`/search` endpoint (`tools.rag_search_url`). Without it, the agent searches `data/runbooks/*.md`.

**Check:** "What does the runbook say about a full filesystem?" cites the runbook source.

## Step 6 — Scenario evaluation

```bash
python eval/eval_agent.py
```

Add one scenario per incident type your team handles (`expect_tools`, `forbid_tools`, `expect_in_answer`).

**Check:** all scenarios pass; never a forbidden tool.

## Step 7 — API with approval queue

```bash
sudo systemctl enable --now ai-app@09-it-ops-agent
curl -s localhost:8009/tasks -H 'Content-Type: application/json' -d '{"task":"Is the RAG service healthy?"}' | jq .answer
curl -s localhost:8009/approvals | jq
```

Connect `/tasks` to your chat tool (Teams/Slack/Mattermost bot) or monitoring alerts; operators approve
through `/approvals/{id}` from an authenticated UI.

**Check:** a task requesting a restart produces a pending approval instead of a restart.

## Step 8 — Fine-tune on your approved runs (CPU)

Operators mark good runs (run IDs) in `data/good_runs.txt`. With 300+:

```bash
python finetune/collect_traces.py
../../common/scripts/make_venv.sh ../../common/requirements-finetune.txt /opt/venv-finetune
source /opt/venv-finetune/bin/activate
python ../../common/python/lora_sft.py --data data/sft_tools.jsonl --model ibm-granite/granite-4.2-8b \
    --out out/lora-ops --epochs 2 --merge
../../common/scripts/hf_to_gguf.sh out/lora-ops/merged /opt/models/granite-4.2-8b-ops Q4_0
```

Serve it on another port, run `LLM_URL=http://127.0.0.1:8083/v1 python eval/eval_agent.py`; switch only if it
passes at least as many scenarios. GPU option: `--gpu` on a compliant machine ([`COMPLIANCE.md`](../../COMPLIANCE.md)).

**Check:** scenario pass count ≥ base model.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Model answers without calling tools | server without `--jinja` | restart llama-server with the C5 script |
| `ERROR: service ... not on the allow-list` | by design | add it to `allowed_services` if appropriate |
| Loops until step limit | vague task / weak model | more specific runbooks; Granite 30B; LoRA (Step 8) |
| `sudo: a password is required` | sudoers rule missing | Step 3 |
| Slow first answer | thinking on | `llm.thinking: false` for simple checks |
