"""app.py: HTTP API for the operations agent. Approval tools are never executed from the API directly:
they are queued, and an operator approves them with POST /approvals/{id}.

    python src/app.py
    curl -s localhost:8009/tasks -H 'Content-Type: application/json' -d '{"task":"Is the RAG service healthy?"}' | jq .answer
    curl -s localhost:8009/approvals | jq
    curl -s -X POST localhost:8009/approvals/<id> -H 'Content-Type: application/json' -d '{"approve": true, "operator": "ahmed"}'
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import tools_ops  # noqa: E402,F401
from agent import ROOT, TOOLS, Agent  # noqa: E402
from tools_ops import CFG  # noqa: E402

import uvicorn  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

app = FastAPI(title="IT-ops agent on IBM Power")
AGENT = Agent(CFG, approver=None)
PENDING = ROOT / CFG["agent"]["pending_file"]


class Task(BaseModel):
    task: str = Field(min_length=3, max_length=4000)


class Decision(BaseModel):
    approve: bool
    operator: str = Field(min_length=2, max_length=60)


def _pending():
    if not PENDING.exists():
        return []
    return [json.loads(line) for line in PENDING.read_text(encoding="utf-8").splitlines() if line.strip()]


@app.get("/health")
def health():
    return {"status": "ok", "llm": AGENT.llm.health(), "tools": sorted(TOOLS)}


@app.post("/tasks")
def run_task(t: Task):
    r = AGENT.run(t.task)
    return {"run_id": r["run_id"], "answer": r["answer"], "steps": r["steps"]}


@app.get("/approvals")
def approvals():
    return _pending()


@app.post("/approvals/{pid}")
def decide(pid: str, d: Decision):
    items = _pending()
    item = next((i for i in items if i["id"] == pid), None)
    if not item:
        raise HTTPException(404, "no such pending approval")
    PENDING.write_text("".join(json.dumps(i) + "\n" for i in items if i["id"] != pid), encoding="utf-8")
    if not d.approve:
        AGENT.log(item["run"], "approval_denied", tool=item["tool"], args=item["args"], operator=d.operator)
        return {"id": pid, "status": "denied"}
    AGENT.log(item["run"], "approval_granted", tool=item["tool"], args=item["args"], operator=d.operator)
    result = TOOLS[item["tool"]]["fn"](**item["args"])
    AGENT.log(item["run"], "tool_result", tool=item["tool"], args=item["args"], result=str(result)[:2000])
    return {"id": pid, "status": "executed", "result": str(result)[:2000]}


if __name__ == "__main__":
    uvicorn.run(app, host=CFG["server"]["host"], port=int(CFG["server"]["port"]))
