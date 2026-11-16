"""eval_agent.py: scenario tests. For each task, did the agent call the expected tools, avoid forbidden ones,
and mention the expected facts? Approval tools are auto-DENIED here, so nothing changes on the host.

    python eval/eval_agent.py                  # eval/scenarios.jsonl
Run it after every prompt, model or tool change. Compare base Granite vs a LoRA-tuned model by LLM_URL.
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import tools_ops  # noqa: E402,F401
from agent import Agent  # noqa: E402
from tools_ops import CFG  # noqa: E402


def main():
    scenarios = [json.loads(line) for line in (ROOT / "eval" / "scenarios.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    agent = Agent(CFG, approver=lambda name, args: False)
    passed = 0
    for s in scenarios:
        r = agent.run(s["task"])
        called = [m["tool_calls"][i]["function"]["name"] for m in r["messages"] if m.get("tool_calls")
                  for i in range(len(m["tool_calls"]))]
        ok_tools = all(t in called for t in s.get("expect_tools", []))
        ok_forbid = not any(t in called for t in s.get("forbid_tools", []))
        ok_text = all(k.lower() in r["answer"].lower() for k in s.get("expect_in_answer", []))
        ok = ok_tools and ok_forbid and ok_text
        passed += ok
        print(f"{'PASS' if ok else 'FAIL'}  {s['task'][:60]:60s} tools={called}")
    print(f"\n{passed}/{len(scenarios)} scenarios passed")


if __name__ == "__main__":
    main()
