#!/bin/bash
# 2026-11-08: Qwen2.5-72B fp8 (online, TP=2) gives readout mass 0 on vLLM 0.19.1; use the official AWQ checkpoint.
export HF_HOME=/data/hanqim2/.cache/huggingface
echo "$(date +%H:%M) start"
/data/hanqim2/.conda/envs/vllm311/bin/hf download Qwen/Qwen2.5-72B-Instruct-AWQ --max-workers 2 && touch .dl_Qwen2.5-72B-AWQ
echo "$(date +%H:%M) end rc=$?"
