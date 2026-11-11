#!/usr/bin/env bash
# Sequential model downloads for the panel (2026-11-06); each finished model touches results/queue/.dl_<name>
HF=/data/hanqim2/.conda/envs/vllm311/bin/hf
export HF_HOME=/data/hanqim2/.cache/huggingface
Q=/data/hanqim2/behavioral-fidelity/results/queue
for m in unsloth/Llama-3.1-8B-Instruct Qwen/Qwen3-4B Qwen/Qwen3-14B microsoft/phi-4 mistralai/Mistral-Small-24B-Instruct-2501 \
         unsloth/gemma-3-27b-it allenai/Olmo-3.1-32B-Instruct Qwen/Qwen2.5-72B-Instruct zai-org/GLM-4-32B-0414; do
    n=$(basename $m)
    echo "$(date +%H:%M) start $m"
    [ -f $Q/.dl_$n ] && continue
    $HF download $m --exclude "original/*" --max-workers 2 > $Q/dl_$n.log 2>&1 && touch $Q/.dl_$n
    echo "$(date +%H:%M) end $m rc=$?"
done
