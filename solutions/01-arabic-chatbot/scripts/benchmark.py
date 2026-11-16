"""scripts/benchmark.py: measure load time, time to first token and decode speed (Step 08).

    python scripts/benchmark.py                                     # bfloat16 and float32, default threads
    python scripts/benchmark.py --dtypes bfloat16 --threads 8 16 32 --tokens 128

Every (dtype, threads) combination loads the model fresh, does one warm-up answer, then
measures --repeats answers per language (English and Arabic). The median is appended to
results/benchmark.csv, so you can compare runs from different days, models and LPARs.
"""
import argparse
import copy
import csv
import os
import platform
import statistics
import sys
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)          # so "import llm" works when run as scripts/benchmark.py
import llm  # noqa: E402

PROMPTS = {
    "en": "Explain in about 150 words how a CPU cache works.",
    "ar": "اشرح في حوالي 150 كلمة كيف تعمل الذاكرة المخبئية (cache) في المعالج.",
}
FIELDS = ["timestamp", "host", "backend", "model", "dtype", "threads", "lang", "load_s",
          "ttft_s", "decode_tok_s", "new_tokens", "total_s", "engine"]


def measure(prompt):
    """One answer. TTFT = time until the first text arrives (prompt processing + first token)."""
    start = time.perf_counter()
    first, parts = None, []
    for piece in llm.generate([{"role": "user", "content": prompt}], stream=True):
        if first is None:
            first = time.perf_counter()
        parts.append(piece)
    end = time.perf_counter()
    first = first or end
    tokens = llm.count_tokens("".join(parts))
    # decode speed excludes the prompt phase: tokens after the first, divided by the time after the first
    decode = (tokens - 1) / (end - first) if tokens > 1 and end > first else 0.0
    return {"ttft_s": first - start, "decode_tok_s": decode, "new_tokens": tokens, "total_s": end - start}


def main():
    p = argparse.ArgumentParser(description="Benchmark the configured model on this machine")
    p.add_argument("--config", help="config file (default: config.yaml)")
    p.add_argument("--dtypes", nargs="+", default=["bfloat16", "float32"], help="transformers only")
    p.add_argument("--threads", nargs="+", type=int, default=[0], help="0 = auto (one per physical core)")
    p.add_argument("--tokens", type=int, default=128, help="max new tokens per answer")
    p.add_argument("--repeats", type=int, default=2, help="measured answers per language")
    p.add_argument("--langs", nargs="+", default=["en", "ar"], choices=list(PROMPTS))
    p.add_argument("--out", default=os.path.join("results", "benchmark.csv"))
    args = p.parse_args()

    out = os.path.abspath(args.out)
    config_path = os.path.abspath(args.config) if args.config else None
    os.chdir(ROOT)                          # config.yaml and .env live in the project root
    base = llm.load_config(config_path)
    dtypes = args.dtypes if base["backend"] == "transformers" else ["gguf"]
    model = base["model"]["id"] if base["backend"] == "transformers" else base["llamacpp"]["gguf_path"]

    os.makedirs(os.path.dirname(out), exist_ok=True)
    new_file = not os.path.exists(out)
    with open(out, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        print(f"{'dtype':9} {'thr':>4} {'lang':4} {'load s':>7} {'TTFT s':>7} {'tok/s':>7} {'tokens':>6}")
        for dtype in dtypes:
            for threads in args.threads:
                cfg = copy.deepcopy(base)
                cfg["model"]["dtype"] = dtype if dtype != "gguf" else cfg["model"].get("dtype")
                cfg.setdefault("runtime", {})["threads"] = threads
                cfg["generation"].update(temperature=0, max_new_tokens=args.tokens)  # repeatable
                start = time.perf_counter()
                llm.load_model(cfg)
                load_s = time.perf_counter() - start
                measure(PROMPTS["en"])        # warm-up: first answer pays one-off costs
                for lang in args.langs:
                    runs = [measure(PROMPTS[lang]) for _ in range(args.repeats)]
                    row = {k: round(statistics.median(r[k] for r in runs), 3) for k in runs[0]}
                    row.update(timestamp=datetime.now().isoformat(timespec="seconds"),
                               host=platform.node(), backend=base["backend"], model=model,
                               dtype=dtype, threads=threads or "auto", lang=lang,
                               load_s=round(load_s, 2), engine=llm.describe())
                    writer.writerow(row)
                    f.flush()
                    print(f"{dtype:9} {str(row['threads']):>4} {lang:4} {load_s:7.1f} {row['ttft_s']:7.2f}"
                          f" {row['decode_tok_s']:7.2f} {row['new_tokens']:6}")
                llm.unload()
    print(f"\nResults appended to {out}")


if __name__ == "__main__":
    main()
