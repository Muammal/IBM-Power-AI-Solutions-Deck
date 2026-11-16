"""collect_traces.py: turn good agent runs into LoRA training data, so Granite gets better at YOUR tools.

Source: data/audit.jsonl (written by the agent) + data/good_runs.txt (run IDs an operator marked as good).
Output: data/sft_tools.jsonl in the format common/python/lora_sft.py expects: the conversation with tool results
as context, and the final answer as the target.

    python finetune/collect_traces.py
    python ../../common/python/lora_sft.py --data data/sft_tools.jsonl --model ibm-granite/granite-4.2-8b \
        --out out/lora-ops --epochs 2 --merge
    ../../common/scripts/hf_to_gguf.sh out/lora-ops/merged /opt/models/granite-4.2-8b-ops Q4_0
Collect 300+ good runs first. Re-run eval/eval_agent.py against the new model before switching.
"""
import collections
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1] / "common" / "python"))
from power_runtime import load_yaml  # noqa: E402


def main():
    cfg = load_yaml(ROOT / "config.yaml")
    good = set((ROOT / "data" / "good_runs.txt").read_text().split()) if (ROOT / "data" / "good_runs.txt").exists() else set()
    runs = collections.defaultdict(list)
    for line in (ROOT / cfg["agent"]["audit_log"]).read_text(encoding="utf-8").splitlines():
        e = json.loads(line)
        if e["run"] in good:
            runs[e["run"]].append(e)
    n = 0
    with open(ROOT / "data" / "sft_tools.jsonl", "w", encoding="utf-8") as out:
        for run_id, events in runs.items():
            task = next((e["task"] for e in events if e["event"] == "task"), None)
            answer = next((e["answer"] for e in events if e["event"] == "answer"), None)
            if not (task and answer):
                continue
            evidence = "\n\n".join(f"[{e['tool']}({json.dumps(e['args'], ensure_ascii=False)})]\n{e['result']}"
                                   for e in events if e["event"] == "tool_result")
            msgs = [{"role": "system", "content": cfg["system_prompt"].strip()},
                    {"role": "user", "content": f"{task}\n\nTool results so far:\n{evidence}"},
                    {"role": "assistant", "content": answer}]
            out.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
            n += 1
    print(f"{n} training conversations -> data/sft_tools.jsonl (from {len(good)} runs marked good)")


if __name__ == "__main__":
    main()
