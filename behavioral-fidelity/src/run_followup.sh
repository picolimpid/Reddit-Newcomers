#!/usr/bin/env bash
# Does Qwen3-32B really use the trajectory, or only the explicit window summary?
# Runs after the 72B job has released the GPUs:
#   GPUs 1,2  32B with --persona timeline (no window summary), all reply conditions
#   GPUs 5,7  32B with --history reverse (mirrored ages), observed condition only
cd "$(dirname "$0")"
PY=/data/hanqim2/.conda/envs/vllm311/bin/python
LOG=../results/logs
export HF_HOME=/data/hanqim2/.cache/huggingface HF_HUB_OFFLINE=1

while pgrep -f "agent_sim.py --model Qwen/Qwen2.5-72B-Instruct" > /dev/null; do sleep 30; done

CUDA_VISIBLE_DEVICES=1,2 $PY agent_sim.py --model Qwen/Qwen3-32B --out qwen3-32b-timeline --persona timeline \
    --tp 2 --gpu-util 0.88 > $LOG/qwen3-32b-timeline.log 2>&1 && python3 evaluate.py qwen3-32b-timeline \
    > $LOG/qwen3-32b-timeline_eval.log 2>&1 &
CUDA_VISIBLE_DEVICES=5,7 $PY agent_sim.py --model Qwen/Qwen3-32B --out qwen3-32b-reverse --history reverse \
    --conditions observed --tp 2 --gpu-util 0.88 > $LOG/qwen3-32b-reverse.log 2>&1 && python3 evaluate.py \
    qwen3-32b-reverse > $LOG/qwen3-32b-reverse_eval.log 2>&1 &
wait
python3 reversal.py qwen3-32b qwen3-32b-reverse
python3 compare_runs.py qwen3-1.7b qwen3-8b qwen3-14b qwen3-32b qwen2.5-72b qwen3-8b-timeline qwen3-32b-timeline \
    qwen3-8b-reverse qwen3-32b-reverse
