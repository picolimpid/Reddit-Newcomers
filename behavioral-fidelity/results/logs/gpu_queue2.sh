#!/bin/bash
# GPU 2 queue v2 (2026-11-04): GPU idle while Arctic Shift collection runs -> 32B probes first,
# then Futurology 8B and 32B once inputs exist
cd /data/hanqim2/behavioral-fidelity/src
export CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 HF_HOME=/data/hanqim2/.cache/huggingface NCCL_P2P_DISABLE=1
PY=/data/hanqim2/.conda/envs/vllm311/bin/python; M32=/data/hanqim2/models/Qwen3-32B; L=../results/logs
$PY probes.py --model $M32 --out qwen3-32b-fp8 --quant fp8 --gpu-util 0.90 > $L/probes_qwen3-32b-fp8_main.log 2>&1
while [ ! -s ../data/futurology2020/inputs.parquet ]; do sleep 30; done
$PY futurology_agents.py --model Qwen/Qwen3-8B --out futurology-qwen3-8b > $L/futurology_qwen3-8b.log 2>&1
$PY futurology_agents.py --model $M32 --quant fp8 --gpu-util 0.90 --out futurology-qwen3-32b > $L/futurology_qwen3-32b.log 2>&1
