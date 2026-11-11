#!/usr/bin/env bash
# Observational SFT experiment (sft.py); waits until run_probes.sh has released the GPU.
#   setsid nohup ./run_sft.sh 9 > ../results/logs/sft.log 2>&1 < /dev/null &
cd "$(dirname "$0")"
PY=/data/hanqim2/.conda/envs/vllm311/bin/python
LOG=../results/logs
export HF_HOME=/data/hanqim2/.cache/huggingface HF_HUB_OFFLINE=1
G=${1:-9}
while pgrep -f run_probes.sh > /dev/null; do sleep 30; done
for mode in train eval; do
    echo "sft $mode start $(date +%H:%M)"
    CUDA_VISIBLE_DEVICES=$G $PY sft.py $mode --out sft-qwen3-8b > $LOG/sft_$mode.log 2>&1
    echo "sft $mode exit $? $(date +%H:%M)"
done
