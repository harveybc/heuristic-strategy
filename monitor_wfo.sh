#!/bin/bash
# Monitor all 3 WFO machines
echo "=== WFO CLUSTER STATUS ($(date +%H:%M:%S)) ==="
echo ""
echo "--- OMEGA (Ryzen 7, folds 2011-2013) ---"
tail -3 /home/harveybc/Documents/GitHub/heuristic-strategy/wfo_omega_2011_2013.log 2>/dev/null || echo "  No log file"
echo ""
echo "--- DRAGON (i9, folds 2014-2016) ---"
ssh -p 62024 -o ConnectTimeout=3 harveybc@192.168.1.235 "tail -3 /home/harveybc/Documents/GitHub/heuristic-strategy/wfo_dragon_2014_2016.log" 2>/dev/null || echo "  Unreachable"
echo ""
echo "--- GAMMA (Ryzen 9, folds 2017-2019) ---"
ssh -p 62024 -o ConnectTimeout=3 harveybc@192.168.0.106 "tail -3 /home/harveybc/Documents/GitHub/heuristic-strategy/wfo_gamma_2017_2019.log" 2>/dev/null || echo "  Unreachable"
echo ""
