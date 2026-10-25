# Clean expert-guided round A1 — Stage A LAUNCHED (2026-10-24 21:24Z, lane L3)

- Authorization: owner, "entonces orquesta y ejecuta" (2026-10-24); gate LAUNCH_EXPERT_GUIDED.json in this root (byte-identical mirror on the coordinator, because the emitter checks the gate path locally).
- Spec clean-expert-guided-a1-20261024 (sha dd37a9c6…), recipe recipe_5a8f2155318ba7c7 (sha 93ac66dc…), store cd0eb8d2…; PINS.sha256 12/12 OK; launcher check: valid, probe reproduced exactly (tensor f087d1a7…); package code byte-identical to the hs worktree tip ccafab2 for the 8 consumed modules.
- Queue: queue_A.sh (emitted on the coordinator from the spec; 7 arms: zz_hold, zz_hold_shorter, opp_hold, opp_optimum, both_hold, both_hold_shorter, opp_hold_shorter; 3 seeds each inside E3-M), sequential, one unit at a time, each under crispdm-run -m 6G -t 43200s, GPU index 0 (the only GPU), recipe digest + gate re-checked before every unit; not-admitted units retry every 10 min up to 72 times.
- Layout fix disclosed: the emitted command reads the recipe at <root>/data/recipe/CONSUMER_RECIPE.json while the package keeps it at <root>/recipe/; data/recipe/CONSUMER_RECIPE.json is a HARD LINK to the pinned file (same bytes, sha 93ac66dc…). No code changed.
- State at launch: GPU idle (14 MiB, no compute apps), RAM 10 GiB available of 30. First unit A_zz_hold admitted and training (process on GPU, 874 MiB, 53 % util; log shows recipe mode: differenced [], verify true, agent_state none).
- Expected duration: T3 v5 unit C took 3 160 s wall on the 5090 host; on this 4090 ~1 h per arm → Stage A ≈ 7–9 h if every unit is admitted at first try.
- Where: logs out/A_<arm>.log, results out/A_<arm>/E3M.json (+ SCALER.json, probe, 12 checkpoints), completion marks STATE/A_<arm>, queue log LOG, driver log nohup_A.log.
- Resume: re-run "GPU=0 setsid nohup bash queue_A.sh > nohup_A.log 2>&1 < /dev/null & disown" from this root; finished arms are skipped by their STATE mark.
- Decision after all 7: on the coordinator, tools/expert_guided_clean_round.py decide --stage A --results <root>/out (frozen rule; INNER_A only). Stage B needs a new owner word (new gate stage).

# Update 2026-10-25 01:30Z (lane L4)

- Stage A finished 23:50Z: 4 arms DONE (zz_hold, both_hold, opp_hold, opp_optimum); 3 arms STOPPED by the admission monitor
  (zz_hold_shorter, both_hold_shorter, opp_hold_shorter: PRESSURE_STOP_SUSTAINED_ABOVE_RESPOND, observed tree peak
  5.99-6.02 GB against the 6G cap; incident records in the admission store's incidents/). They died in round 1 of seed 1
  (SCALER.json + TRIAL_LEDGER.jsonl kept, no E3M.json).
- Interim decision (frozen rule, INNER_A only, 4 arms): both_hold 1.48 +- 1.16 > zz_hold 1.13 +- 0.97 > opp_hold 0.14 >
  opp_optimum 0.08 -> DECISION_A_INTERIM.{json,md} in this root. INTERIM until the three stopped arms are run elsewhere.
- Rerun attempt: 9G was QUEUED on HOST_HEADROOM (host_free_for_new 8.57 GB: 3 GiB desktop reserve + the resident VM);
  the waiter was stopped; a 7G retry was REFUSED CAP_LOWERED_AFTER_REFUSAL by the admission store. Verdict:
  NOT_RUNNABLE_ON_THIS_HOST under the admission rules; they need a host that admits >= 9G (or the VM memory released).
  Nothing was renamed or re-labelled to dodge the refusal. queue_A_rerun.sh / run_after_A.sh are inert artefacts.
- Stage A_prime (spec v3 CLEAN_ROUND_SPEC_v3.json; the pinned CLEAN_ROUND_SPEC.json is the v2 that ran stage A):
  queue_Aprime.sh launched 01:26Z (GPU=0 setsid nohup ... nohup_Aprime.log), 3 arms x 3 seeds, 7G per unit, sequential:
  multires_default -> multires_deap (recipe/CHOSEN_THRESHOLDS.json, sha a84c5f57..., pinned) -> multires_deap_weighted
  (--sample-weights). First unit ADMITTED at 7G and training on the GPU. Expected ~35 min per arm as in stage A.
- Package code synced from hs tip 8df54ab for tools/expert_guided_clean_round.py, tools/expert_guided_e3m.py,
  app/short_horizon/expert_guided/learn.py, app/short_horizon/expert_guided/multires.py (all pinned in PINS.sha256,
  18 lines, sha256sum -c OK). Gate LAUNCH_EXPERT_GUIDED.json now lists stage ["A","A_prime"] (mirror on the coordinator).
- Decision after A_prime: on the coordinator, decide over A and A_prime arms together (same recipe/support), with the
  three stopped arms declared; results out/A_prime_<arm>/E3M.json, marks STATE/A_prime_<arm>, queue log LOG.
- Disclosed error: the new spec was first copied over the pinned CLEAN_ROUND_SPEC.json; restored byte-identical from
  git (sha dd37a9c6...) before any unit ran; PINS re-verified.

## 2026-10-25 re-queue of the three shorter arms at 10G
queue_A_shorter_10G.sh (= queue_A.sh with -m 10G, job names clean-A10-*) launched detached; it skips arms with STATE marks
and retries every 10 min for up to 12 h while the admission store answers QUEUED HOST_HEADROOM. It admits as soon as the
host frees ~10 GiB (e.g. after the owner stops the resident VM or restarts the host). After a host restart, relaunch with:
  cd /var/tmp/expert-guided-clean-a1-20261024 && GPU=0 setsid nohup bash queue_A_shorter_10G.sh > nohup_A_shorter_10G.log 2>&1 < /dev/null & disown
(queue_Aprime.sh resumes the same way for the unfinished A_prime unit.)
