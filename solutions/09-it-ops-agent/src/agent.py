"""agent.py: a small, readable tool-calling agent loop with an approval gate and an audit log.

Every other solution's AGENTS.md reuses this loop: register Python functions as tools, mark the ones that change
anything as requires_approval, and run a task. The model (Granite 4.2 via llama-server --jinja) decides which tool
to call; this code runs it, feeds the result back, and stops after max_steps.

    from agent import Agent, tool
    @tool("Disk usage of all mounted filesystems", {})
    def disk_usage(): ...
    Agent(cfg).run("Why is the RAG service slow?")
"""
import datetime as dt
import inspect
import json
import pathlib
import sys
import uuid

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1] / "common" / "python"))
from llm_client import LLM  # noqa: E402

TOOLS = {}


def tool(description, parameters, requires_approval=False):
    """Register a function as a tool. `parameters` is a JSON-schema 'properties' dict; all are required
    unless listed in an `optional` key."""
    def wrap(fn):
        optional = set(parameters.pop("optional", [])) if "optional" in parameters else set()
        TOOLS[fn.__name__] = {
            "fn": fn, "requires_approval": requires_approval,
            "schema": {"type": "function", "function": {
                "name": fn.__name__, "description": description + (" (REQUIRES HUMAN APPROVAL)" if requires_approval else ""),
                "parameters": {"type": "object", "properties": parameters,
                               "required": [p for p in parameters if p not in optional]}}}}
        return fn
    return wrap


class Agent:
    def __init__(self, cfg, approver=None):
        """approver(name, args) -> True/False. None = never execute approval tools; queue them instead."""
        self.cfg, self.approver = cfg, approver
        self.llm = LLM(cfg["llm"]["base_url"])
        self.audit = ROOT / cfg["agent"]["audit_log"]
        self.audit.parent.mkdir(parents=True, exist_ok=True)

    def log(self, run_id, event, **data):
        with open(self.audit, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": dt.datetime.now().isoformat(timespec="seconds"), "run": run_id,
                                "event": event, **data}, ensure_ascii=False) + "\n")

    def call_tool(self, run_id, name, args):
        t = TOOLS.get(name)
        if not t:
            return f"ERROR: unknown tool {name}"
        if t["requires_approval"]:
            if self.approver is None:
                pid = uuid.uuid4().hex[:8]
                with open(ROOT / self.cfg["agent"]["pending_file"], "a", encoding="utf-8") as f:
                    f.write(json.dumps({"id": pid, "run": run_id, "tool": name, "args": args}) + "\n")
                self.log(run_id, "approval_queued", tool=name, args=args, pending_id=pid)
                return f"PENDING: queued for human approval as {pid}. Do not retry; tell the user."
            if not self.approver(name, args):
                self.log(run_id, "approval_denied", tool=name, args=args)
                return "DENIED by the operator. Suggest an alternative or stop."
            self.log(run_id, "approval_granted", tool=name, args=args)
        try:
            accepted = inspect.signature(t["fn"]).parameters
            result = t["fn"](**{k: v for k, v in args.items() if k in accepted})
        except Exception as exc:  # tools must never crash the loop: the model sees the error and adapts
            result = f"ERROR: {type(exc).__name__}: {exc}"
        result = str(result)[:6000]  # keep the context small on CPU
        self.log(run_id, "tool_result", tool=name, args=args, result=result[:2000])
        return result

    def run(self, task, history=None):
        run_id = uuid.uuid4().hex[:12]
        c = self.cfg
        messages = [{"role": "system", "content": c["system_prompt"].strip()}] + (history or []) + \
                   [{"role": "user", "content": task}]
        self.log(run_id, "task", task=task)
        schemas = [t["schema"] for t in TOOLS.values()]
        for step in range(int(c["agent"]["max_steps"])):
            msg = self.llm.chat(messages, tools=schemas, raw=True, thinking=c["llm"]["thinking"],
                                temperature=c["llm"]["temperature"], max_tokens=c["llm"]["max_tokens"])
            calls = msg.get("tool_calls") or []
            messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls} if calls
                            else {"role": "assistant", "content": msg.get("content") or ""})
            if not calls:
                answer = LLM.strip_thinking(msg.get("content") or "")
                self.log(run_id, "answer", answer=answer, steps=step)
                return {"run_id": run_id, "answer": answer, "steps": step, "messages": messages}
            for call in calls:
                fn = call["function"]
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                self.log(run_id, "tool_call", tool=fn["name"], args=args)
                messages.append({"role": "tool", "tool_call_id": call.get("id", fn["name"]),
                                 "content": self.call_tool(run_id, fn["name"], args)})
        messages.append({"role": "user", "content": "Step limit reached. Answer now with what you know."})
        msg = self.llm.chat(messages, raw=True, thinking=False, max_tokens=c["llm"]["max_tokens"])
        answer = LLM.strip_thinking(msg.get("content") or "")
        self.log(run_id, "answer", answer=answer, steps=int(c["agent"]["max_steps"]), limit_reached=True)
        return {"run_id": run_id, "answer": answer, "steps": int(c["agent"]["max_steps"]), "messages": messages}
