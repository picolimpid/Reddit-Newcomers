#!/bin/bash
# 2026-11-06 21:10: the /mnt/disk1 Ministral-3 copies were removed; re-download the 8B into the /data HF cache.
export HF_HOME=/data/hanqim2/.cache/huggingface
echo "$(date +%H:%M) start"
/data/hanqim2/.conda/envs/vllm311/bin/hf download mistralai/Ministral-3-8B-Instruct-2512-BF16 --max-workers 2 && touch .dl_Ministral-3-8B
echo "$(date +%H:%M) end rc=$?"
