"""cli.py: chat with the operations agent in a terminal. Approval tools ask you y/N before running.

    python src/cli.py                                   # interactive
    python src/cli.py "Why is the RAG API slow?"        # one task
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import tools_ops  # noqa: E402,F401  (registers the tools)
from agent import Agent  # noqa: E402
from tools_ops import CFG  # noqa: E402


def ask(name, args):
    print(f"\n>>> The agent wants to run {name}({json.dumps(args, ensure_ascii=False)})")
    return input("    Approve? [y/N] ").strip().lower() == "y"


def main():
    agent = Agent(CFG, approver=ask)
    if len(sys.argv) > 1:
        print(agent.run(" ".join(sys.argv[1:]))["answer"])
        return
    history = []
    while True:
        try:
            task = input("\nops> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if task in ("exit", "quit"):
            break
        if task:
            r = agent.run(task, history)
            print(f"\n{r['answer']}\n   (run {r['run_id']}, {r['steps']} tool steps)")
            history += [{"role": "user", "content": task}, {"role": "assistant", "content": r["answer"]}]
            history = history[-6:]


if __name__ == "__main__":
    main()
