# INVALIDATION NOTE

**Date:** 2025-04-19

**Reason:** Results in this directory were invalidated because the data pipeline operated on synthetic GBM (Geometric Brownian Motion) data via a silent fallback mechanism in the data acquisition scripts. The synthetic data passed superficially but lacked real market microstructure (fat tails, volatility clustering, weekend gaps).

**Consequence:** All experimental results (Path A experiments A1-A5, CI-2 causal analysis, static baselines) are unreliable and must not be cited.

**Replacement:** See `../part_II_redux/` for the re-execution with verified real market data.

**Reusable artifacts:** `infrastructure/rolling_orchestrator.py` was copied to Part II-Redux as it contains logic (not data-dependent).
