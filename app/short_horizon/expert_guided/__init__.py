"""Expert-guided (hindsight-demonstration) learning ladder E0-E2 for EURUSD H1.

DEVELOPMENT only: TRAIN 2020-2022 fits, 2023 is joint development (INNER_A 2023H1 monitors/calibrates,
INNER_B 2023H2 is scored after the freeze), 2024 is previously observed development read once after the
freeze. TEST 2025 is never read (the loaders refuse it). No order is ever sent anywhere.

The expert is the MetaTrader ZigZag hindsight expert (HINDSIGHT_NON_CAUSAL_EXPERT): its labels may use the
future, the student's inputs never do. Terminology: learning from (hindsight-expert) demonstrations /
interactive imitation; not human-preference RLHF.
"""
