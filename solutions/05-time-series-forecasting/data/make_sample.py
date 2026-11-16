"""make_sample.py: synthetic hourly load for one distribution substation (training sample only).

Shape follows patterns typical of Iraqi grids: strong summer peak driven by air-conditioning
(temperatures above 45 °C in July/August), evening peak, lower demand on Fridays, plus noise.
    python data/make_sample.py            # -> data/substation_load.csv (2 years, hourly)
Replace it with your SCADA/AMI export: same columns (timestamp, load_mw, temp_c) or edit config.yaml.
"""
import pathlib

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
idx = pd.date_range("2024-11-01", periods=24 * 730, freq="h")
doy, hour = idx.dayofyear.to_numpy(), idx.hour.to_numpy()

# Temperature: Baghdad-like seasonal + daily cycle
season = 24 + 14 * np.sin(2 * np.pi * (doy - 110) / 365.25)
daily = 7 * np.sin(2 * np.pi * (hour - 9) / 24)
temp = season + daily + rng.normal(0, 1.5, len(idx))

# Load: base + cooling demand above 26 °C + evening peak + Friday dip + noise
cooling = 1.9 * np.clip(temp - 26, 0, None)
evening = 9 * np.exp(-((hour - 21) ** 2) / 8)
friday = np.where(idx.dayofweek.to_numpy() == 4, -6, 0)
load = 55 + cooling + evening + friday + rng.normal(0, 2.5, len(idx))

out = pathlib.Path(__file__).with_name("substation_load.csv")
pd.DataFrame({"timestamp": idx, "load_mw": load.round(2), "temp_c": temp.round(1)}).to_csv(out, index=False)
print(f"wrote {out} ({len(idx)} rows)")
