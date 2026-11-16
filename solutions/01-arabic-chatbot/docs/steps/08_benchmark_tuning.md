# Step 08 — Benchmark and tuning

## Goal

Measure, on **your** LPAR, how fast the chatbot is with each setting, and pick the best `dtype` and `threads`
for `config.yaml`. Don't guess: the best value depends on the model, the LPAR size, SMT mode and the library versions.

## Files created in this step

| File | Purpose |
|------|---------|
| `scripts/benchmark.py` | Loads the model for each (dtype, threads) combination and measures it. Appends to `results/benchmark.csv` |
| `docs/steps/08_benchmark_tuning.md` | This document |

## What is measured

| Column | Meaning | Why you care |
|---|---|---|
| `load_s` | Seconds to load the weights into RAM | Service start-up time. The first load after boot reads from disk; later loads come from the page cache |
| `ttft_s` | **Time to first token**: from sending the question until the first text appears. This is the *prompt-processing* (prefill) phase | What users feel as "lag". Grows with prompt and history length |
| `decode_tok_s` | **Tokens per second** after the first one (the *decode* phase) | How fast the answer "types" |
| `new_tokens` | Tokens actually generated (capped by `--tokens`) | Sanity check |
| `lang` | `en` or `ar` prompt | Arabic often needs more tokens per word on English-centric tokenizers, so compare both |

Each combination: load the model → one warm-up answer (discarded) → `--repeats` measured answers per language
→ the **median** is written. Decoding is greedy (`temperature 0`), so runs are comparable.

## Commands to run (copy-paste ready)

```bash
cd /opt/power-chatbot && source venv/bin/activate
export HF_HUB_OFFLINE=1                  # model is already downloaded; avoid network calls

# 1. dtype comparison at the default thread count
python scripts/benchmark.py --dtypes bfloat16 float32

# 2. thread sweep. Example for 8 cores in SMT4 (= 32 logical CPUs, check with: nproc)
python scripts/benchmark.py --dtypes bfloat16 --threads 4 8 16 24 32

# 3. (optional) compare SMT modes: switch, then repeat the sweep
sudo ppc64_cpu --smt=8 && python scripts/benchmark.py --dtypes bfloat16 --threads 8 16 32 64
sudo ppc64_cpu --smt=4

# Look at everything collected so far
column -s, -t < results/benchmark.csv | less -S
```

Tips for clean numbers:
* Stop `app.py` and the systemd service while benchmarking, since they compete for the same cores.
* On a **shared**-processor LPAR, numbers vary with other tenants' load. Repeat runs, or use dedicated cores.
* For float32 on 7B/8B models you need ~32 GB of free RAM for the weights alone.

## Expected output

```
dtype      thr lang  load s  TTFT s   tok/s tokens
bfloat16     8 en       5.9    0.71    9.84    128
bfloat16     8 ar       5.9    0.83    9.61    128
bfloat16    16 en       2.1    0.52   11.20    128
...
Results appended to /opt/power-chatbot/results/benchmark.csv
```

The CSV header is:

```
timestamp,host,backend,model,dtype,threads,lang,load_s,ttft_s,decode_tok_s,new_tokens,total_s,engine
```

> Tested while building this template on an x86 laptop (not Power), with the pinned packages and
> `Qwen/Qwen2.5-0.5B-Instruct`, `--threads 4 8 16 --tokens 64 --repeats 1`. All 12 rows were written. That CPU has
> no native bf16 matrix unit, so **bfloat16 had a 4× worse TTFT than float32** (1.73 s vs 0.44 s at 16 threads),
> while decode speed was similar (~12–13 tok/s). Power10/Power11 run bf16 on the MMA units, so expect a different
> result there. That is exactly why you measure on the target machine.

## How to read the results and tune

1. **dtype.** Use `bfloat16` if its decode speed is ≥ float32's and answers look as good. It also halves RAM.
   Use `float32` if bf16 is slower on your build, or for an accuracy reference.
2. **threads.** Pick the smallest thread count within ~5% of the best `decode_tok_s`. More threads than that
   only add contention. The default (`threads: 0`) is one thread per physical core. Also try **cores × 2**,
   since the Power10 core has several MMA units and SMT threads can keep them busier. Don't trust PyTorch's own default
   on POWER: it uses every hardware thread (see Step 05).
   NUMA: if `check_power.sh` reports more than one NUMA node, benchmark once pinned to a single node and compare.
   Memory traffic across chips is slower than local memory:
   ```bash
   numactl --hardware                                   # nodes, their CPUs and memory
   numactl --cpunodebind=0 --membind=0 python scripts/benchmark.py --dtypes bfloat16
   ```
3. **SMT.** Compare the best row of the SMT4 sweep with the best row of the SMT8 sweep. SMT4 usually wins or ties.
   Keep the winner (see Step 02 for making SMT4 permanent).
4. **TTFT too high?** Shorten `max_history_turns` or the system prompt. Prefill time grows with the prompt length.
5. Write the winners into `config.yaml`:
   ```yaml
   model:
     dtype: bfloat16
   runtime:
     threads: 16
   ```
6. Still too slow, or too little RAM? Try the llama.cpp backend with a Q4_0 or Q8_0 GGUF (Step 09) and benchmark it
   the same way. With `backend: llamacpp` the script ignores `--dtypes`.

## How to verify

```bash
python scripts/benchmark.py --dtypes bfloat16 --tokens 32 --repeats 1 && tail -n 2 results/benchmark.csv
```

**Passes when** the command exits without error and the last two CSV lines show your model with non-zero
`ttft_s` and `decode_tok_s` for `en` and `ar`.

## Common errors and fixes

| Symptom | Cause | Fix |
|---------|-------|-----|
| `Killed` during the float32 run | Not enough RAM for fp32 weights | Benchmark bf16 only, or use a bigger LPAR |
| Numbers jump a lot between runs | Shared cores, or the web service running | Dedicated cores; stop other model processes; raise `--repeats` |
| `decode_tok_s` drops when threads go up | Oversubscription (more threads than useful cores) | Use the lower thread count |
| `ttft_s` near 0 with `backend: llamacpp` | llama.cpp reuses the cached prompt when the same prompt is repeated | Expected for repeats; judge TTFT from the first question of a fresh chat |
| `ttft_s` much higher for `ar` | Tokenizer splits Arabic into more tokens | Expected for English-centric tokenizers; ALLaM/Qwen are more efficient on Arabic |
| `ConnectionError` while loading | Network call to the Hub on load | `export HF_HUB_OFFLINE=1` |
