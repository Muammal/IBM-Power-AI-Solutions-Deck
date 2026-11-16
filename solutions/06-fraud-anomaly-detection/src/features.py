"""features.py: raw transactions -> model features. Used identically in training and scoring.

Behavioural features compare each transaction with the SAME customer's history before it
(no look-ahead): amount vs the customer's usual amount, how many transactions in the last hour, etc.
"""
import numpy as np
import pandas as pd

CHANNELS = ["pos", "atm", "mobile_wallet", "online", "branch"]
FEATURES = ["log_amount", "hour", "is_night", "is_weekend_fri", "is_foreign", "new_device", "away_from_home",
            "amount_vs_cust_median", "cust_txn_count", "txn_last_1h", "txn_last_24h", "secs_since_prev"] + \
           [f"ch_{c}" for c in CHANNELS]


def build_features(df):
    df = df.sort_values("timestamp").copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    out = pd.DataFrame(index=df.index)
    out["log_amount"] = np.log1p(df["amount_iqd"])
    out["hour"] = df["timestamp"].dt.hour
    out["is_night"] = out["hour"].between(0, 5).astype(int)
    out["is_weekend_fri"] = (df["timestamp"].dt.dayofweek == 4).astype(int)
    out["is_foreign"] = df["is_foreign"].astype(int)
    out["new_device"] = df["new_device"].astype(int)
    out["away_from_home"] = (df["governorate"] != df["home_governorate"]).astype(int)
    for c in CHANNELS:
        out[f"ch_{c}"] = (df["channel"] == c).astype(int)

    g = df.groupby("customer_id", sort=False)
    # expanding median of PREVIOUS amounts (shift(1) = exclude the current transaction)
    prev_median = g["amount_iqd"].transform(lambda s: s.shift(1).expanding().median())
    out["amount_vs_cust_median"] = (df["amount_iqd"] / prev_median).fillna(1.0).clip(0, 1000)
    out["cust_txn_count"] = g.cumcount()
    out["secs_since_prev"] = g["timestamp"].diff().dt.total_seconds().fillna(1e7).clip(0, 1e7)

    # number of the customer's transactions in the previous hour / day (time-based rolling window)
    ts_by_cust = df.set_index("timestamp").groupby("customer_id", sort=False)
    for name, w in (("txn_last_1h", "1h"), ("txn_last_24h", "24h")):
        counts = ts_by_cust["amount_iqd"].transform(lambda s, w=w: s.rolling(w).count() - 1)
        out[name] = counts.values
    return out[FEATURES].astype(float)
