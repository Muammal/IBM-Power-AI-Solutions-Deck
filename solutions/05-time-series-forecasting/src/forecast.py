"""forecast.py: zero-shot (or fine-tuned) forecasting with Granite TimeSeries TTM-R2, plus test-set accuracy.

    python src/forecast.py                  # evaluate zero-shot on the test split + forecast the next 96 h
    python src/forecast.py --finetuned      # same with out/ttm-finetuned
TTM is a ~1M-parameter model: it runs in milliseconds per window on CPU. No GPU, no llama.cpp needed.
"""
import argparse
import json
import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1] / "common" / "python"))
from power_runtime import configure_threads, load_yaml  # noqa: E402

CFG = load_yaml(ROOT / "config.yaml")
configure_threads(CFG["runtime"]["threads"])

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from transformers import Trainer, TrainingArguments  # noqa: E402
from tsfm_public import (TimeSeriesForecastingPipeline, TimeSeriesPreprocessor, TinyTimeMixerForPrediction,  # noqa: E402
                         get_datasets, get_model)


def load_df():
    d = CFG["data"]
    df = pd.read_csv(ROOT / d["csv"], parse_dates=[d["timestamp_column"]])
    return df.sort_values(d["timestamp_column"]).reset_index(drop=True)


def make_preprocessor(with_controls):
    d, m = CFG["data"], CFG["model"]
    return TimeSeriesPreprocessor(
        timestamp_column=d["timestamp_column"], id_columns=d["id_columns"], target_columns=d["target_columns"],
        control_columns=d["control_columns"] if with_controls else [],
        context_length=m["context_length"], prediction_length=m["prediction_length"],
        scaling=True, scaler_type="standard", encode_categorical=False, freq=d["freq"])


def mae_on_test(model, tsp, test_df):
    """Mean absolute error in real units (e.g. MW), averaged over every forecast window and horizon step."""
    pipe = TimeSeriesForecastingPipeline(model, feature_extractor=tsp, device="cpu", batch_size=256)
    fc = pipe(test_df)
    errors = []
    for col in CFG["data"]["target_columns"]:
        pred = np.stack(fc[f"{col}_prediction"].to_numpy())
        true = np.stack(fc[col].to_numpy())
        errors.append(np.nanmean(np.abs(pred - true)))
    return float(np.mean(errors))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--finetuned", action="store_true")
    a = ap.parse_args()
    m, d = CFG["model"], CFG["data"]
    df = load_df()
    path = ROOT / m["finetuned_path"]
    with_controls = a.finetuned and (path / "preprocessor_config.json").exists() and CFG["finetune"]["use_exogenous"]
    tsp = make_preprocessor(with_controls)
    train, valid, test = get_datasets(tsp, df, split_config=d["split"])  # also fits the scaler on train
    model = (TinyTimeMixerForPrediction.from_pretrained(str(path)) if a.finetuned else
             get_model(m["id"], context_length=m["context_length"], prediction_length=m["prediction_length"]))

    with tempfile.TemporaryDirectory() as tmp:
        args = TrainingArguments(output_dir=tmp, per_device_eval_batch_size=256, use_cpu=True, report_to=[])
        loss = Trainer(model=model, args=args).evaluate(test)["eval_loss"]

    # The test windows start where the test split starts, minus one context of history.
    n = len(df)
    test_start = int(n * (1 - d["split"]["test"])) - m["context_length"]
    mae = mae_on_test(model, tsp, df.iloc[test_start:])
    print(json.dumps({"model": str(path) if a.finetuned else m["id"], "test_mse_scaled": round(loss, 4),
                      "test_mae": round(mae, 3), "unit": d["target_columns"]}, indent=2))

    if not with_controls:  # forecast beyond the end of the data (needs only history)
        pipe = TimeSeriesForecastingPipeline(model, feature_extractor=tsp, device="cpu")
        fc = pipe(df.tail(m["context_length"]))
        col = d["target_columns"][0]
        future = pd.date_range(df[d["timestamp_column"]].iloc[-1], periods=m["prediction_length"] + 1, freq=d["freq"])[1:]
        out = pd.DataFrame({d["timestamp_column"]: future, f"{col}_forecast": fc[f"{col}_prediction"].iloc[-1]})
        out_path = ROOT / "out" / "forecast_next.csv"
        out_path.parent.mkdir(exist_ok=True)
        out.to_csv(out_path, index=False)
        print(f"next {m['prediction_length']} steps -> {out_path}\n", out.head(6).to_string(index=False))


if __name__ == "__main__":
    main()
