# Step 06 — Command-line smoke test

## Goal

Talk to the model in the terminal, and run an automatic smoke test proving that the bot answers an
**Arabic** question (Modern Standard and Iraqi dialect) in Arabic and an **English** question in English.

## Files created in this step

| File | Purpose |
|------|---------|
| `cli.py` | Interactive chat, `--once "question"`, and `--selftest` (exit code 0 = pass) |
| `docs/steps/06_cli_test.md` | This document |

## How `cli.py` works

* `ask(history)` calls `llm.generate(history, stream=True)` and prints each piece as it arrives
  (`flush=True` makes it appear immediately). It also measures the time to the first text.
* `chat()` keeps a `history` list. After each answer it appends `{"role": "assistant", ...}`, so the
  model sees the conversation. `/reset` empties it, and Ctrl+C stops only the current answer.
* `selftest()` asks three fixed questions with **temperature 0** (greedy decoding, so results are repeatable) and checks:
  * MSA: `ما هي عاصمة العراق؟` → answer contains **بغداد** and is in Arabic script
  * Iraqi dialect: `شنو عاصمة العراق؟` → same check
  * English: `What is the capital of France?` → answer contains **Paris**

## Commands to run (copy-paste ready)

```bash
cd /opt/power-chatbot && source venv/bin/activate
python cli.py --selftest
```

Try it by hand:

```bash
python cli.py --once "شلونك؟ شنو تكدر تسوي؟"
python cli.py --once "Explain what an LPAR is in two sentences."
python cli.py
```

> **Arabic in a terminal:** most terminals (including the RHEL console and many SSH clients) show Arabic
> letters **left-to-right and sometimes unjoined**. The text itself is correct, as you can see by copying it into
> a browser. The web UI in Step 07 renders it properly right-to-left.

## Expected output

```
Loading model ...
Ready in 11.2s: transformers | meta-llama/Llama-3.2-3B-Instruct | bfloat16 | 32 threads

>>> ما هي عاصمة العراق؟ أجب بجملة واحدة.
عاصمة العراق هي بغداد.
[PASS] first text after 0.9s, total 2.1s

>>> شنو عاصمة العراق؟ جاوبني بجملة وحدة.
عاصمة العراق هي بغداد.
[PASS] first text after 0.8s, total 1.9s

>>> What is the capital of France? Answer in one sentence.
The capital of France is Paris.
[PASS] first text after 0.6s, total 1.4s

SELFTEST PASSED
```

Timings depend on the model, dtype, threads and LPAR size (Step 08 measures them properly).

> Tested while building this template on x86, with the same pinned packages and small stand-in models:
> * `Qwen/Qwen2.5-1.5B-Instruct`: **all three PASS**, exit code 0.
> * `Qwen/Qwen2.5-0.5B-Instruct`: MSA and English pass, **Iraqi dialect FAILS** (it answered "العراق هي عاصمة للبلاد"),
>   exit code 1. This shows the test catches a wrong answer, and that very small models handle dialect poorly.
>   The 3B/7B presets are the intended targets.

## How to verify

```bash
python cli.py --selftest; echo "exit code: $?"
```

**Passes when it prints `SELFTEST PASSED` and `exit code: 0`.** Do not continue to Step 07 until it passes.

## Common errors and fixes

| Symptom | Cause | Fix |
|---------|-------|-----|
| English passes, Arabic `[FAIL]` with an English answer | Model ignores the language instruction | Try a model with better Arabic (Qwen2.5-7B, ALLaM-7B); see Step 11 |
| Iraqi dialect `[FAIL]` only | Small or English-centric model | Use a 7B Arabic-capable model; Llama-3.2-3B is not officially trained for Arabic |
| Arabic shows as `????` or boxes | Terminal not UTF-8 | `export LANG=C.UTF-8` and use a UTF-8 SSH client |
| Answers stop mid-sentence | `max_new_tokens` reached | Raise `generation.max_new_tokens` in `config.yaml` |
| Bot repeats itself | Greedy decoding with a small model | Keep `repetition_penalty: 1.05`–`1.15`; use a larger model |
| Very slow first answer after start | Model weights still being paged in from disk | Normal on the first request; Step 08 measures it |
