# Solution 05 — Demand and load forecasting with IBM Granite TimeSeries (TTM-R2)

<div dir="rtl">

**ملخص:** التنبؤ بالأحمال الكهربائية أو الطلب على الوقود أو حركة شبكات الاتصالات للأيام القادمة باستخدام نموذج
IBM Granite TimeSeries الصغير جداً (حوالي مليون معامل)، الذي يعمل على المعالج المركزي خلال ثوانٍ، ويمكن تدريبه على
بياناتكم مع درجات الحرارة خلال دقائق، دون أي بطاقة رسومية.

</div>

## The problem

Iraq's grid runs close to its limits every summer, when air-conditioning drives demand with temperatures above
45 °C. Distribution companies, fuel distributors, water utilities and telecom operators all need hour-by-hour
forecasts for the next days to schedule generation, imports, tankers or capacity. Spreadsheets and averages miss
heat waves and holidays.

## What it does

```
 SCADA / AMI / sales history (CSV, hourly)  ──┐
 temperature history + weather forecast ──────┤
                                              v
        Granite TTM-R2 zero-shot  (no training: works on day one)
                                              │   test MAE (sample data): 3.60 MW
                                              v
        fine-tune on your data, backbone frozen, temperature as exogenous driver (CPU, minutes)
                                              │   test MAE (sample data): 2.94 MW
                                              v
        out/forecast_next.csv (next 96 h) · FastAPI /forecast · dashboards / dispatch planning
```

The numbers above are real results of `src/forecast.py` before/after `finetune/finetune_ttm.py` (3 epochs) on
the included synthetic substation data, run on a laptop CPU. Expect a similar pattern, not the same values,
on your data.

## Models

| Role | Model | Licence | Size |
|---|---|---|---|
| Forecaster | `ibm-granite/granite-timeseries-ttm-r2` (TinyTimeMixer) | Apache-2.0 | ~1M parameters per context/horizon variant |
| Optional | `ibm-granite/granite-timeseries-tspulse-r1` (anomaly detection, imputation) | Apache-2.0 | ~1M |
| Optional | `ibm-granite/granite-timeseries-ensemble-r1` (probabilistic ensemble, 2026) | Apache-2.0 | ensemble |

## Compute and compliance

CPU-only and tiny: zero-shot evaluation over a year of hourly data takes seconds; fine-tuning takes minutes.
This is the lightest solution in the repo: a `small` LPAR or even a shared one is enough. See [`COMPLIANCE.md`](../../COMPLIANCE.md).

## Quick start (after common steps C1–C3)

```bash
cd /opt/ibm-power-ai-solutions/solutions/05-time-series-forecasting
../../common/scripts/make_venv.sh && source venv/bin/activate
python data/make_sample.py
python src/forecast.py                 # zero-shot: test MAE + next 96 h
python finetune/finetune_ttm.py        # minutes on CPU
python src/forecast.py --finetuned     # compare test MAE
```

## Files

| Path | Purpose |
|---|---|
| `config.yaml` | data columns, context/horizon, fine-tuning settings |
| `data/make_sample.py` | synthetic hourly load + temperature for an Iraqi substation |
| `src/forecast.py` | evaluation (scaled MSE, MAE in real units) + next-horizon forecast |
| `finetune/finetune_ttm.py` | fine-tuning with exogenous channel mixing |
| `src/app.py` | FastAPI `/forecast` (send history, get forecast) |
