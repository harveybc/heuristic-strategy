"""Illustration for the owner: 'search the data where the Hilbert phase = x and look at what followed'. Outcome = 12 h
forward log return (bps) on TRAIN 2020-2022. Raw conditional mean by phase quadrant vs the same comparison INSIDE strata
of volatility (px.rv_24 terciles) x hour-of-day block (6 blocks): association vs adjusted association."""
import numpy as np, pandas as pd
df = pd.read_parquet("/var/tmp/features-candidates/eurusd_h1_full_pool_s_a1/full_pool_eurusd_h1.parquet",
                     columns=["timestamp", "CLOSE", "hb.eur.phase_sin", "hb.eur.phase_cos", "px.rv_24"])
t = pd.to_datetime(df["timestamp"], utc=True)
df["fwd"] = 1e4 * (np.log(df["CLOSE"].shift(-12)) - np.log(df["CLOSE"]))
df = df[(t >= "2020-01-01") & (t < "2023-01-01")].dropna()
t = pd.to_datetime(df["timestamp"], utc=True)
ph = np.degrees(np.arctan2(df["hb.eur.phase_sin"], df["hb.eur.phase_cos"])) % 360
df["quad"] = pd.cut(ph, [0, 90, 180, 270, 360], labels=["0-90", "90-180", "180-270", "270-360"], include_lowest=True)
df["vol"] = pd.qcut(df["px.rv_24"], 3, labels=["low", "mid", "high"])
df["hblk"] = (t.dt.hour // 4).astype(int)
raw = df.groupby("quad", observed=True)["fwd"].agg(["mean", "count", "std"])
raw["t"] = raw["mean"] / (raw["std"] / np.sqrt(raw["count"]))
# adjusted: mean of within-stratum deviations from the stratum mean, then by quadrant (stratification)
df["dev"] = df["fwd"] - df.groupby(["vol", "hblk"], observed=True)["fwd"].transform("mean")
adj = df.groupby("quad", observed=True)["dev"].agg(["mean", "count", "std"]); adj["t"] = adj["mean"] / (adj["std"] / np.sqrt(adj["count"]))
mix = pd.crosstab(df["quad"], df["vol"], normalize="index").round(3)
print("RAW (bps, 12 h ahead):\n", raw.round(2).to_string()); print("\nADJUSTED within vol x hour strata:\n", adj.round(2).to_string())
print("\nshare of each volatility tercile inside each phase quadrant:\n", mix.to_string())
# overlap: share of 2020 vs 2021 vs 2022 sign agreement of adjusted quadrant effect
for y in (2020, 2021, 2022):
    m = t.dt.year == y
    print(y, df[m.values].groupby("quad", observed=True)["dev"].mean().round(2).to_dict())
