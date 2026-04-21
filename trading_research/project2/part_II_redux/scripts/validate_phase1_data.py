#!/usr/bin/env python3
"""Stage II-7.1.e: Validation battery for newly acquired data.

Produces:
  - data/validation/II-7_data_validation.md
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_BINANCE_DIR = os.path.join(BASE_DIR, "data", "raw", "binance")
RAW_CM_DIR = os.path.join(BASE_DIR, "data", "raw", "coinmetrics")
RAW_BC_DIR = os.path.join(BASE_DIR, "data", "raw", "blockchain_com")
VALIDATION_DIR = os.path.join(BASE_DIR, "data", "validation")


def acf_lag1(series: pd.Series) -> float:
    if len(series) < 3:
        return 0.0
    mean = float(series.mean())
    x0 = series.iloc[:-1].to_numpy() - mean
    x1 = series.iloc[1:].to_numpy() - mean
    denom = float(np.sum((series.to_numpy() - mean) ** 2))
    if denom == 0:
        return 0.0
    return float(np.sum(x0 * x1) / denom)


def ljung_box_pvalue(series: pd.Series, lags: int = 10) -> float:
    n = len(series)
    if n <= lags + 2:
        return 1.0

    centered = series - float(series.mean())
    c0 = float(np.sum(centered * centered) / n)
    if c0 <= 0:
        return 1.0

    q_stat = 0.0
    arr = centered.to_numpy()
    for k in range(1, lags + 1):
        ck = float(np.sum(arr[:-k] * arr[k:]) / n)
        rho = ck / c0
        q_stat += (rho * rho) / (n - k)
    q_stat *= n * (n + 2)
    return float(1 - stats.chi2.cdf(q_stat, lags))


def runs_test_pvalue(returns: pd.Series) -> float:
    signs = np.sign(returns.to_numpy())
    signs = signs[signs != 0]
    n = len(signs)
    if n < 20:
        return 1.0

    n_pos = float(np.sum(signs > 0))
    n_neg = float(np.sum(signs < 0))
    if n_pos == 0 or n_neg == 0:
        return 1.0

    runs = 1
    for i in range(1, n):
        if signs[i] != signs[i - 1]:
            runs += 1

    expected = (2 * n_pos * n_neg) / (n_pos + n_neg) + 1
    var = (2 * n_pos * n_neg * (2 * n_pos * n_neg - n_pos - n_neg)) / (
        (n_pos + n_neg) ** 2 * (n_pos + n_neg - 1)
    )
    if var <= 0:
        return 1.0
    z = (runs - expected) / math.sqrt(var)
    return float(2 * (1 - stats.norm.cdf(abs(z))))


@dataclass
class OhlcvValidationResult:
    file_name: str
    rows: int
    start: str
    end: str
    checks: Dict[str, bool]
    metrics: Dict[str, float]


def expected_rows(start: pd.Timestamp, end: pd.Timestamp, tf: str) -> int:
    mins = {"5m": 5, "15m": 15, "1h": 60, "4h": 240}[tf]
    span_minutes = (end - start).total_seconds() / 60.0
    return int(span_minutes // mins)


def validate_ohlcv_file(path: str) -> OhlcvValidationResult:
    df = pd.read_parquet(path)
    df["DateTime"] = pd.to_datetime(df["DateTime"], utc=True)
    df = df.sort_values("DateTime").drop_duplicates(subset=["DateTime"])

    start = df["DateTime"].iloc[0]
    end = df["DateTime"].iloc[-1]

    tf = os.path.basename(path).split("_")[1]
    exp = expected_rows(start, end, tf)
    rows = len(df)

    close = pd.to_numeric(df["Close"], errors="coerce")
    returns = np.log(close / close.shift(1)).dropna()

    kurt = float(stats.kurtosis(returns, fisher=False, nan_policy="omit")) if len(returns) > 20 else 0.0
    acf_r2 = acf_lag1((returns ** 2).dropna())
    acf_r = acf_lag1(returns)

    lb_p = ljung_box_pvalue((returns ** 2).dropna())
    jb_stat, jb_p = stats.jarque_bera(returns) if len(returns) > 20 else (0.0, 1.0)
    runs_p = runs_test_pvalue(returns)
    gbm_rejects = int(lb_p < 0.01) + int(jb_p < 0.01) + int(runs_p < 0.05)

    checks = {
        "bar_count_realistic": (exp * 0.85) <= rows <= (exp * 1.15),
        "monotonic_no_dupes": bool(df["DateTime"].is_monotonic_increasing),
        "fat_tails": kurt > 3.5,
        "volatility_clustering": acf_r2 > 0.03,
        "tiny_nonzero_autocorr": abs(acf_r) > 0.002,
        "no_gbm_fingerprint": gbm_rejects >= 2,
    }

    metrics = {
        "expected_rows": float(exp),
        "kurtosis": kurt,
        "acf1_r2": acf_r2,
        "acf1_r": acf_r,
        "ljung_box_p": float(lb_p),
        "jarque_bera_p": float(jb_p),
        "runs_p": float(runs_p),
        "gbm_rejects_3": float(gbm_rejects),
    }

    return OhlcvValidationResult(
        file_name=os.path.basename(path),
        rows=rows,
        start=str(start),
        end=str(end),
        checks=checks,
        metrics=metrics,
    )


def validate_tabular_csv(path: str, time_col: str = "time") -> dict:
    df = pd.read_csv(path)
    if time_col in df.columns:
        ts = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    elif "funding_time" in df.columns:
        ts = pd.to_datetime(df["funding_time"], utc=True, errors="coerce")
    else:
        ts = pd.Series([pd.NaT] * len(df))

    numeric_cols = [c for c in df.columns if c not in {time_col, "funding_time", "symbol", "asset"}]
    missing_ratio = float(df[numeric_cols].isna().mean().mean()) if numeric_cols else 0.0

    return {
        "file": os.path.basename(path),
        "rows": int(len(df)),
        "start": str(ts.min()) if ts.notna().any() else "N/A",
        "end": str(ts.max()) if ts.notna().any() else "N/A",
        "numeric_columns": int(len(numeric_cols)),
        "mean_missing_ratio": missing_ratio,
        "pass_nonempty": len(df) > 0,
        "pass_missing_ratio": missing_ratio < 0.30,
    }


def main() -> None:
    os.makedirs(VALIDATION_DIR, exist_ok=True)

    parquet_files = sorted(
        [os.path.join(RAW_BINANCE_DIR, f) for f in os.listdir(RAW_BINANCE_DIR) if f.endswith(".parquet")]
    )
    funding_files = sorted(
        [os.path.join(RAW_BINANCE_DIR, f) for f in os.listdir(RAW_BINANCE_DIR) if f.startswith("funding_") and f.endswith(".csv")]
    )
    cm_files = sorted([os.path.join(RAW_CM_DIR, f) for f in os.listdir(RAW_CM_DIR) if f.endswith(".csv")])
    bc_files = sorted([os.path.join(RAW_BC_DIR, f) for f in os.listdir(RAW_BC_DIR) if f.endswith(".csv")])

    ohlcv_results: List[OhlcvValidationResult] = [validate_ohlcv_file(p) for p in parquet_files]
    funding_results = [validate_tabular_csv(p, time_col="funding_time") for p in funding_files]
    cm_results = [validate_tabular_csv(p, time_col="time") for p in cm_files]
    bc_results = [validate_tabular_csv(p, time_col="time") for p in bc_files]

    all_ohlcv_pass = all(all(r.checks.values()) for r in ohlcv_results)
    all_aux_pass = all(
        r["pass_nonempty"] and r["pass_missing_ratio"] for r in (funding_results + cm_results + bc_results)
    )
    overall_pass = all_ohlcv_pass and all_aux_pass

    report_path = os.path.join(VALIDATION_DIR, "II-7_data_validation.md")
    json_path = os.path.join(VALIDATION_DIR, "II-7_data_validation.json")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Stage II-7.1 Data Validation\n\n")
        f.write(f"Overall gate: {'PASS' if overall_pass else 'FAIL'}\n\n")

        f.write("## OHLCV (6-test battery)\n\n")
        for r in ohlcv_results:
            f.write(f"### {r.file_name}\n")
            f.write(f"- Rows: {r.rows}\n")
            f.write(f"- Range: {r.start} -> {r.end}\n")
            for k, v in r.checks.items():
                f.write(f"- {k}: {'PASS' if v else 'FAIL'}\n")
            f.write(
                "- Metrics: "
                + ", ".join([f"{mk}={mv:.6f}" for mk, mv in r.metrics.items()])
                + "\n\n"
            )

        f.write("## Funding / On-chain / Supplementary Checks\n\n")
        for section, results in [
            ("Funding", funding_results),
            ("CoinMetrics", cm_results),
            ("Blockchain.com", bc_results),
        ]:
            f.write(f"### {section}\n")
            for r in results:
                f.write(
                    f"- {r['file']}: rows={r['rows']} range={r['start']} -> {r['end']} "
                    f"missing={r['mean_missing_ratio']:.3f} "
                    f"nonempty={'PASS' if r['pass_nonempty'] else 'FAIL'} "
                    f"missing_check={'PASS' if r['pass_missing_ratio'] else 'FAIL'}\n"
                )
            f.write("\n")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "overall_pass": overall_pass,
                "ohlcv": [
                    {
                        "file": r.file_name,
                        "rows": r.rows,
                        "start": r.start,
                        "end": r.end,
                        "checks": r.checks,
                        "metrics": r.metrics,
                    }
                    for r in ohlcv_results
                ],
                "funding": funding_results,
                "coinmetrics": cm_results,
                "blockchain_com": bc_results,
            },
            f,
            indent=2,
        )

    print(f"Validation report: {report_path}")
    print(f"Validation json:   {json_path}")
    print(f"Overall gate: {'PASS' if overall_pass else 'FAIL'}")


if __name__ == "__main__":
    main()
