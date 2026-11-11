#!/usr/bin/env bash
# one-off: drop the 32B thinking-mode readout (too slow on one GPU) so run_sft.sh can start
until grep -q "qwen3-32b-fp8 readout start" ../results/logs/probes.log; do sleep 15; done
sleep 5; pkill -f "probes.py --model Qwen/Qwen3-32B --out qwen3-32b-fp8 --mode readout"; echo "32B readout skipped $(date +%H:%M)" >> ../results/logs/probes.log
