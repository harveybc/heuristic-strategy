# Controlled modular specialists deployment

The owner-approved finite study is running, not merely queued. First JOINT16
cell closed; JOINT_MATCHED began automatically. There are 15 cells in total.
Runtime f6ddd53c18c34a12f809498d124874ba82ee908f; spec digest
fb808aa85830deac27cc838fad9cc326ac3d4bb57790b51e782f583250b946a6.

43 focused tests passed locally and on the worker. Actual synthetic GPU training
and exact reload passed. The first scientific BC round ran 216 epochs (best206)
with the declared mean TRAIN/2023H1 monitor and patience10. This is an operational
proof of execution, not a financial conclusion. First terminal: 8 governed
reports delivered/read back, zero pending or mismatched; independent physical
copy authenticated. No unrelated process was stopped or modified.

Worker and publication user timers are enabled on the worker, and the collector
timer is enabled on the independent storage role. Linger is enabled. The worker
uses one GPU process group, strict 5090 identity, 7 GiB host memory limit,
bounded fresh-root retries and per-cell timeout. All scheduled trials proceed
even when performance is negative. Integrity faults are explicit rather than
silently accepting inconsistent results. Reboots retain the queue and receipts.

## Read-only status

From the pinned runtime, with the private root set by the operator:

```bash
python -m tools.modular_campaign status --root "$CAMPAIGN_ROOT"
```

Also inspect PUBLICATION_STATUS.json and the collector's COLLECTION_STATUS.json.
The first trial took about87 seconds. A whole-campaign ETA from that one JOINT
trial is only a rough extrapolation: FULL has four models and more controls.
Use measured costs from each architecture before treating a total ETA as reliable.

Artifacts: CONFIG.json, per-round *.pt and learning curves, FROZEN.json,
trade/equity ledgers, RESULT/CLOSE, source RESULTS.csv/RESULTS.json, independent
ANALYSIS.json and READOUT.md. The collector separates original and new seeds.
The initial no-fit spec is retained as a prelaunch predecessor; resealing was
permitted only while the queue contained no cell and no runs directory.

Scientific completion, warehouse completion and independent-copy completion
have separate sentinels. Their timers become no-ops after the corresponding
completion marker; no agent needs to monitor them or authorize the next cell.
