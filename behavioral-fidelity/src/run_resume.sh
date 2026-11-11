#!/usr/bin/env bash
# Resume after the session that ran run_scale.sh ended mid-72B (2026-10-28):
#   1. Qwen3-32B mechanism checks (timeline persona on GPUs 1,2; reversed history on GPUs 5,7)
#   2. Qwen2.5-72B-Instruct on GPUs 1,2,5,7 (TP4)
# Launch detached so it survives the Claude Code session:
#   setsid nohup ./run_resume.sh > ../results/logs/resume.log 2>&1 &
cd "$(dirname "$0")"
PY=/data/hanqim2/.conda/envs/vllm311/bin/python
LOG=../results/logs
export HF_HOME=/data/hanqim2/.cache/huggingface HF_HUB_OFFLINE=1

echo "follow-ups start $(date +%H:%M)"
./run_followup.sh
echo "72B start $(date +%H:%M)"
CUDA_VISIBLE_DEVICES=1,2,5,7 $PY agent_sim.py --model Qwen/Qwen2.5-72B-Instruct --out qwen2.5-72b --tp 4 \
    --gpu-util 0.88 > $LOG/qwen2.5-72b.log 2>&1 && python3 evaluate.py qwen2.5-72b > $LOG/qwen2.5-72b_eval.log 2>&1
echo "72B exit $? $(date +%H:%M)"
python3 compare_runs.py qwen3-1.7b qwen3-8b qwen3-14b qwen3-32b qwen2.5-72b qwen3-8b-timeline qwen3-32b-timeline \
    qwen3-8b-reverse qwen3-32b-reverse
echo "all done $(date +%H:%M)"
