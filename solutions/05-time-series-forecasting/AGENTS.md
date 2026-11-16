# AGENTS.md — Solution 05 (Time-series forecasting)

Three parts: **agent brief**, **agentic workflow**, **change log**. Repository rules: [`/AGENTS.md`](../../AGENTS.md).

---

## 1. Agent brief

**Purpose:** hourly demand/load forecasts with Granite TTM-R2, zero-shot and fine-tuned with exogenous drivers.

**Invariants:**

* Pins: `granite-tsfm==0.3.12` requires `torch>=2.10,<2.12` and `transformers>=4.57.6`. Do not bump torch past
  2.11 without bumping granite-tsfm.
* The split is chronological (`get_datasets` with `split_config`). Never shuffle across time.
* `src/forecast.py` and `finetune/finetune_ttm.py` share `make_preprocessor()`: keep them consistent.
* Report MAE in real units, not only scaled MSE, and always compare with zero-shot.

**Map:**

| Change | File |
|---|---|
| Columns, horizon, context | `config.yaml` |
| Evaluation / output | `src/forecast.py` |
| Fine-tuning recipe | `finetune/finetune_ttm.py` |
| API | `src/app.py` |

**Verify:**

```bash
python -m py_compile src/*.py finetune/*.py data/*.py
python data/make_sample.py && python src/forecast.py          # test_mae ~3.6 on the sample
python finetune/finetune_ttm.py && python src/forecast.py --finetuned   # must beat zero-shot
```

---

## 2. Agentic workflow (runtime)

A **forecast analyst agent** turns numbers into a morning briefing and answers "what if" questions.

```
 06:00 job: src/forecast.py (+ fine-tuned model with weather forecast)
        |
        v
 [Analyst agent] Granite 4.2-8B, thinking=on, max 5 tool calls
   tools:
     get_forecast(feeder_id, hours)                -> out/forecast_next.csv rows
     get_actuals(feeder_id, days)                  -> recent actual load
     run_scenario(feeder_id, temp_delta_c)         -> re-run the exogenous model with temperature +/- N °C
     capacity_limits(feeder_id)                    -> transformer / feeder ratings (read-only)
        |
        v
 briefing (Arabic + English): peak hours, feeders over 90% of rating, scenario "+3 °C heat wave"
        |
        v
 dispatcher reviews; any load-shedding plan stays a human decision
```

**Guardrails:** read-only tools; numbers in the briefing must come from tool outputs (the prompt requires quoting
them); the agent never changes schedules or SCADA set-points.

**Implementation path:** Solution 09 `src/agent.py`; `run_scenario` is `forecast.py` with a modified `temp_c` column.

---

## 3. Agent change protocol

1. Change, run every **Verify** command, record `test_mae` zero-shot and fine-tuned.
2. Docs changed → `python tools/build_docx.py solutions/05-time-series-forecasting`.
3. Append a row below.

### Agent change log

| Date | Agent / author | Change | Verification result |
|---|---|---|---|
| 2026-11-15 | Claude Code (Opus 5.5) | Initial version: synthetic data, zero-shot eval + forecast, exogenous fine-tuning, API | Run on x86 CPU (torch 2.10.0+cpu, transformers 4.57.6, granite-tsfm 0.3.12): zero-shot test MAE 3.60 MW; fine-tuned (3 epochs, 2 min) 2.94 MW; API returns 96 points. Not yet run on Power hardware |
