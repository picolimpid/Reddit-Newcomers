#!/usr/bin/env bash
# Other open-weight families (ungated on HF), same protocol as the Qwen runs.
#   lane A, GPUs 1,2: Mistral-Small-24B-Instruct-2501 (TP2) -> GLM-4-32B-0414 (TP2)
#   lane B, GPUs 5,7: Olmo-3.1-32B-Instruct (TP2)          -> phi-4 (GPU 5)
# Launch detached:  setsid nohup ./run_families.sh > ../results/logs/families.log 2>&1 < /dev/null &
cd "$(dirname "$0")"
PY=/data/hanqim2/.conda/envs/vllm311/bin/python
HF=/data/hanqim2/.conda/envs/vllm311/bin/hf
LOG=../results/logs
export HF_HOME=/data/hanqim2/.cache/huggingface

MODELS="mistralai/Mistral-Small-24B-Instruct-2501 zai-org/GLM-4-32B-0414 allenai/Olmo-3.1-32B-Instruct microsoft/phi-4"
for m in $MODELS; do
    ( $HF download $m > $LOG/dl_$(basename $m).log 2>&1; touch $LOG/.dl_$(basename $m).done ) &
done

run() {  # gpus model out tp
    while [ ! -f $LOG/.dl_$(basename $2).done ]; do sleep 20; done
    echo "$3 start $(date +%H:%M)"
    CUDA_VISIBLE_DEVICES=$1 HF_HUB_OFFLINE=1 $PY agent_sim.py --model "$2" --out "$3" --tp "$4" --gpu-util 0.88 \
        > $LOG/$3.log 2>&1 && python3 evaluate.py "$3" > $LOG/$3_eval.log 2>&1
    echo "$3 exit $? $(date +%H:%M)"
}

( run 1,2 mistralai/Mistral-Small-24B-Instruct-2501 mistral-small-24b 2
  run 1,2 zai-org/GLM-4-32B-0414 glm4-32b 2 ) &
( run 5,7 allenai/Olmo-3.1-32B-Instruct olmo3.1-32b 2
  run 5 microsoft/phi-4 phi-4 1 ) &
wait
ALL="qwen3-8b qwen3-14b qwen3-32b qwen2.5-72b mistral-small-24b glm4-32b olmo3.1-32b phi-4"
python3 compare_runs.py $ALL
python3 robust_ols.py $ALL
echo "all done $(date +%H:%M)"
