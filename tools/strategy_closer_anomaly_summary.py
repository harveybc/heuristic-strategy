import json, sys, glob, numpy as np
rows = [json.loads(l) for f in sys.argv[1:] for l in open(f) if l.startswith("{")]
by = {}
for r in rows: by.setdefault(r["q"], []).append(r)
for q in sorted(by, reverse=True):
    R = by[q]
    g = lambda k: np.nanmean([float(x[k]) if x[k] is not None else np.nan for x in R])
    early = np.nanmean([x["mix"][s]["early"] for x in R for s in ("BUY", "SELL")])
    print(f"q{q:.2f} n={len(R)} sharpe {g('sharpe'):.2f} net {g('net_usd'):,.0f} trades {g('trades'):.0f} hold {g('mean_hold_bars'):.1f} net/trade {g('mean_net_per_trade'):,.0f} win {g('win_rate'):.3f} early {early:.3f} "
          f"maxDD% {np.nanmean([x['m'].get('max_drawdown_pct', np.nan) for x in R]):.2f} E4 {np.nanmean([e for x in R for e in x['E4']]):.3f}")
c = {r["cell"]: {} for r in rows}
for r in rows: c[r["cell"]][r["q"]] = r
for qa in (0.85, 0.7):
    up = [k for k, v in c.items() if qa in v and 1.0 in v and v[qa]["sharpe"] > v[1.0]["sharpe"]]
    upnet_down = [k for k in up if c[k][qa]["net_usd"] < c[k][1.0]["net_usd"]]
    netup = [k for k, v in c.items() if qa in v and 1.0 in v and v[qa]["net_usd"] > v[1.0]["net_usd"]]
    rel = [ (c[k][qa]["sharpe"]/c[k][1.0]["sharpe"], c[k][qa]["net_usd"]/c[k][1.0]["net_usd"]) for k in c if qa in c[k] and 1.0 in c[k]]
    print(f"q{qa}: Sharpe above q1 in {len(up)}/{len(c)} cells; of these net below q1 in {len(upnet_down)}; net above q1 in {len(netup)}; median Sharpe ratio {np.median([a for a,b in rel]):.3f} median net ratio {np.median([b for a,b in rel]):.3f}")
