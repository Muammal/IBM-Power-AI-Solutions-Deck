"""cli.py: chat with the model in the terminal (Step 06).

    python cli.py                      interactive chat   (/reset clears history, /exit quits)
    python cli.py --once "question"    ask one question and exit
    python cli.py --selftest           ask fixed Arabic + English questions, exit code 0 = all passed
"""
import argparse
import sys
import time

import llm

# (question, words of which at least one must appear in the answer, answer must be Arabic)
SELFTEST = [
    ("ما هي عاصمة العراق؟ أجب بجملة واحدة.", ["بغداد"], True),           # Modern Standard Arabic
    ("شنو عاصمة العراق؟ جاوبني بجملة وحدة.", ["بغداد"], True),           # Iraqi dialect
    ("What is the capital of France? Answer in one sentence.", ["Paris"], False),
]


def ask(history):
    """Stream one answer to the terminal. Returns (answer, seconds to first text, total seconds)."""
    start, first, parts = time.time(), None, []
    for piece in llm.generate(history, stream=True):
        if first is None:
            first = time.time() - start
        parts.append(piece)
        print(piece, end="", flush=True)
    print()
    return "".join(parts), first or 0.0, time.time() - start


def selftest():
    all_ok = True
    for question, expected, must_be_arabic in SELFTEST:
        print(f"\n>>> {question}\n", end="")
        answer, ttft, total = ask([{"role": "user", "content": question}])
        ok = any(word.lower() in answer.lower() for word in expected)
        if must_be_arabic:
            ok = ok and llm.detect_language(answer) == "ar"   # replied in the user's language?
        print(f"[{'PASS' if ok else 'FAIL'}] first text after {ttft:.1f}s, total {total:.1f}s")
        all_ok = all_ok and ok
    print("\nSELFTEST", "PASSED" if all_ok else "FAILED")
    return 0 if all_ok else 1


def chat():
    history = []
    print("Type a message (Arabic or English). /reset clears history, /exit quits.")
    while True:
        try:
            user = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user in ("/exit", "/quit"):
            break
        if user == "/reset":
            history.clear()
            print("(history cleared)")
            continue
        history.append({"role": "user", "content": user})
        print("Bot> ", end="", flush=True)
        try:
            answer, _, _ = ask(history)
        except KeyboardInterrupt:          # Ctrl+C stops the current answer, not the program
            print("\n(stopped)")
            history.pop()
            continue
        history.append({"role": "assistant", "content": answer})


def main():
    parser = argparse.ArgumentParser(description="Command-line chatbot")
    parser.add_argument("--config", help="path to config.yaml (default: config.yaml or $CHATBOT_CONFIG)")
    parser.add_argument("--once", metavar="QUESTION", help="ask one question and exit")
    parser.add_argument("--selftest", action="store_true", help="run the Arabic/English smoke test")
    args = parser.parse_args()

    cfg = llm.load_config(args.config)
    if args.selftest:
        cfg["generation"]["temperature"] = 0      # greedy decoding = repeatable test results
    print("Loading model ...", flush=True)
    start = time.time()
    llm.load_model(cfg)
    print(f"Ready in {time.time() - start:.1f}s: {llm.describe()}")

    if args.selftest:
        sys.exit(selftest())
    if args.once:
        ask([{"role": "user", "content": args.once}])
        return
    chat()


if __name__ == "__main__":
    main()
