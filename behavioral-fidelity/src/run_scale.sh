#!/usr/bin/env bash
# Scale runs: each model starts as soon as its download has finished.
#   phase 1  Qwen3-14B on GPUs 1,2 (TP2) | Qwen3-32B on GPUs 5,7 (TP2)
#   phase 2  Qwen2.5-72B-Instruct on GPUs 1,2,5,7 (TP4)
cd "$(dirname "$0")"
PY=/data/hanqim2/.conda/envs/vllm311/bin/python
LOG=../results/logs
export HF_HOME=/data/hanqim2/.cache/huggingface

wait_download() {  # until no `hf download <model>` process is left
    while pgrep -f "hf download $1\$" > /dev/null; do sleep 30; done
}

run() {  # gpus model out tp
    wait_download "$2"
    CUDA_VISIBLE_DEVICES=$1 HF_HUB_OFFLINE=1 $PY agent_sim.py --model "$2" --out "$3" --tp "$4" --gpu-util 0.88 \
        > $LOG/$3.log 2>&1 && python3 evaluate.py "$3" > $LOG/$3_eval.log 2>&1
    echo "$3 exit $?"
}

run 1,2 Qwen/Qwen3-14B qwen3-14b 2 &
run 5,7 Qwen/Qwen3-32B qwen3-32b 2 &
wait
run 1,2,5,7 Qwen/Qwen2.5-72B-Instruct qwen2.5-72b 4
python3 compare_runs.py qwen3-1.7b qwen3-8b qwen3-14b qwen3-32b qwen2.5-72b
