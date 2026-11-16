"""make_transactions.py: synthetic card / mobile-wallet transactions in IQD (training sample only).

Fraud patterns injected (~0.6%): night-time bursts, new device + large amount, foreign online purchases,
far-from-home ATM withdrawals, rapid repeated transfers.
    python data/make_transactions.py           # -> data/transactions.csv (~300k rows)
"""
import pathlib

import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
N_CUST, N = 20000, 300_000
GOVS = ["Baghdad", "Basra", "Nineveh", "Erbil", "Sulaymaniyah", "Najaf", "Karbala", "Kirkuk", "Anbar", "Dhi Qar"]
CHANNELS = ["pos", "atm", "mobile_wallet", "online", "branch"]

cust_home = rng.integers(0, len(GOVS), N_CUST)
cust_scale = rng.lognormal(11.5, 0.8, N_CUST)        # typical spend ~100k IQD, varies by customer
cust = rng.integers(0, N_CUST, N)
ts = pd.Timestamp("2026-01-01") + pd.to_timedelta(np.sort(rng.uniform(0, 300 * 86400, N)), unit="s")
channel = rng.choice(CHANNELS, N, p=[0.38, 0.22, 0.25, 0.10, 0.05])
amount = rng.lognormal(np.log(cust_scale[cust]), 0.7)
gov = np.where(rng.random(N) < 0.92, cust_home[cust], rng.integers(0, len(GOVS), N))
foreign = (channel == "online") & (rng.random(N) < 0.08)
new_device = rng.random(N) < 0.03
fraud = np.zeros(N, bool)

k = int(N * 0.006)
f = rng.choice(N, k, replace=False)
kind = rng.integers(0, 4, k)
amount[f] *= rng.uniform(3, 15, k)
new_device[f[kind == 0]] = True
foreign[f[kind == 1]] = True
channel[f[kind == 1]] = "online"
gov[f[kind == 2]] = (cust_home[cust[f[kind == 2]]] + rng.integers(3, 7, (kind == 2).sum())) % len(GOVS)
channel[f[kind == 2]] = "atm"
night = f[kind == 3]
ts = ts.to_numpy()
ts[night] = (pd.to_datetime(ts[night]).normalize() + pd.to_timedelta(rng.uniform(1, 4.5, len(night)), unit="h")).to_numpy()
fraud[f] = True

df = pd.DataFrame({"txn_id": np.arange(N), "timestamp": ts, "customer_id": cust, "channel": channel,
                   "amount_iqd": amount.round(-2), "governorate": np.array(GOVS)[gov],
                   "home_governorate": np.array(GOVS)[cust_home[cust]], "is_foreign": foreign.astype(int),
                   "new_device": new_device.astype(int), "is_fraud": fraud.astype(int)})
df = df.sort_values("timestamp").reset_index(drop=True)
out = pathlib.Path(__file__).with_name("transactions.csv")
df.to_csv(out, index=False)
print(f"wrote {out}: {len(df)} rows, fraud rate {df.is_fraud.mean():.3%}")
