"""app.py: forecasting API. Send at least context_length past points, get prediction_length future points.

    python src/app.py
    curl -s localhost:8005/forecast -H 'Content-Type: application/json' -d @request.json | jq
request.json: {"timestamps": ["2026-10-01T00:00:00", ...], "values": [61.2, ...]}   (>= 512 hourly points)
Uses the zero-shot model (history only). The exogenous fine-tuned model also needs future drivers
(weather forecast): run it in batch with src/forecast.py --finetuned.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from forecast import CFG  # noqa: E402  (also sets thread counts)

import pandas as pd  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel  # noqa: E402
from tsfm_public import TimeSeriesForecastingPipeline, TimeSeriesPreprocessor, get_model  # noqa: E402

M = CFG["model"]
MODEL = get_model(M["id"], context_length=M["context_length"], prediction_length=M["prediction_length"])
app = FastAPI(title="Time-series forecasting on IBM Power (Granite TTM-R2)")


class Req(BaseModel):
    timestamps: list[str]
    values: list[float]
    freq: str = "h"


@app.get("/health")
def health():
    return {"status": "ok", "model": M["id"], "context_length": M["context_length"],
            "prediction_length": M["prediction_length"]}


@app.post("/forecast")
def forecast(req: Req):
    n = M["context_length"]
    if len(req.values) < n or len(req.values) != len(req.timestamps):
        raise HTTPException(400, f"need >= {n} values and one timestamp per value")
    df = pd.DataFrame({"timestamp": pd.to_datetime(req.timestamps), "y": req.values}).tail(n)
    tsp = TimeSeriesPreprocessor(timestamp_column="timestamp", target_columns=["y"], context_length=n,
                                 prediction_length=M["prediction_length"], scaling=True, freq=req.freq)
    tsp.train(df)  # per-request scaler: the series' own mean/std
    fc = TimeSeriesForecastingPipeline(MODEL, feature_extractor=tsp, device="cpu")(df)
    future = pd.date_range(df["timestamp"].iloc[-1], periods=M["prediction_length"] + 1, freq=req.freq)[1:]
    return {"timestamps": [t.isoformat() for t in future],
            "forecast": [round(float(v), 4) for v in fc["y_prediction"].iloc[-1]]}


if __name__ == "__main__":
    uvicorn.run(app, host=CFG["server"]["host"], port=int(CFG["server"]["port"]))
