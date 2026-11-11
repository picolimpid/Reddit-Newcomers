#!/usr/bin/env bash
# Probe experiments (probes.py) on one GPU: Qwen3-8B (bf16, all agents), then Qwen3-32B with fp8
# weights on a 1,200-agent subsample (GPU 0 on this box fails CUDA init; the free GPUs vary).
#   setsid nohup ./run_probes.sh 9 > ../results/logs/probes.log 2>&1 < /dev/null &
cd "$(dirname "$0")"
PY=/data/hanqim2/.conda/envs/vllm311/bin/python
LOG=../results/logs
export HF_HOME=/data/hanqim2/.cache/huggingface HF_HUB_OFFLINE=1
G=${1:-9}
for mode in main readout; do
    echo "qwen3-8b $mode start $(date +%H:%M)"
    CUDA_VISIBLE_DEVICES=$G $PY probes.py --model Qwen/Qwen3-8B --out qwen3-8b --mode $mode > $LOG/probes_qwen3-8b_$mode.log 2>&1
    echo "qwen3-8b $mode exit $? $(date +%H:%M)"
done
for mode in main readout; do
    echo "qwen3-32b-fp8 $mode start $(date +%H:%M)"
    CUDA_VISIBLE_DEVICES=$G $PY probes.py --model Qwen/Qwen3-32B --out qwen3-32b-fp8 --mode $mode --quant fp8 --limit 1200 \
        --gpu-util 0.86 > $LOG/probes_qwen3-32b-fp8_$mode.log 2>&1
    echo "qwen3-32b-fp8 $mode exit $? $(date +%H:%M)"
done
