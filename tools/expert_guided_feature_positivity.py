"""How predictable is each recipe feature from the other 120 on TRAIN rows (R^2 of a ridge fit)? A treatment that the
other covariates almost determine has a degenerate propensity, so IPTW / do(feature) weights explode."""
import json, numpy as np, pandas as pd
rec = json.load(open("docs/short_horizon/clean_round/evidence/a1/RECIPE_v2_a1.json"))
cols = [c["store_column"] for c in rec["tensor"]["columns"]]
df = pd.read_parquet("__import__("sys").argv[1]", columns=cols + ["timestamp"])
idx = pd.to_datetime(df.pop("timestamp"), utc=True); idx = idx.dt.tz_localize(None) if hasattr(idx, "dt") else idx
X = df[(idx >= "2020-01-01") & (idx < "2023-01-01")].dropna()
X = X.loc[:, X.std() > 0]
Z = ((X - X.mean()) / X.std()).to_numpy()
n, k = Z.shape
r2 = []
for j in range(k):
    y = Z[:, j]; A = np.delete(Z, j, axis=1)
    w = np.linalg.solve(A.T @ A + 1e-3 * n * np.eye(k - 1), A.T @ y)
    r2.append(1 - np.mean((y - A @ w) ** 2) / np.var(y))
r2 = np.array(r2)
# ESS of stabilized Gaussian generalized-propensity weights for the median feature
j = int(np.argsort(r2)[len(r2) // 2]); y = Z[:, j]; A = np.delete(Z, j, axis=1)
w = np.linalg.solve(A.T @ A + 1e-3 * n * np.eye(k - 1), A.T @ y); res = y - A @ w; s = res.std()
dens = np.exp(-0.5 * (res / s) ** 2) / s; marg = np.exp(-0.5 * y ** 2)
wt = marg / dens; ess = wt.sum() ** 2 / (wt ** 2).sum()
out = {"rows": n, "features": k, "r2_quantiles": {q: float(np.quantile(r2, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
       "share_r2_ge_0.9": float((r2 >= 0.9).mean()), "share_r2_ge_0.99": float((r2 >= 0.99).mean()),
       "median_feature": X.columns[j], "median_feature_ipw_ess": float(ess), "median_feature_ipw_ess_share": float(ess / n),
       "max_weight_over_mean": float(wt.max() / wt.mean())}
print(json.dumps(out, indent=1))
