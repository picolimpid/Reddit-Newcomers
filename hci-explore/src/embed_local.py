"""Sentence embeddings from a small open-weight encoder, run locally (CPU is enough for a few thousand texts).

  $PY src/embed_local.py IN.jsonl OUT.npz [--model /data/hanqim2/models/bge-small-en-v1.5] [--max-len 512]

IN rows: {"id", "text"}.  OUT .npz: ids, emb (L2-normalised [CLS] vectors, the pooling BGE models are trained with).
Weights are loaded from safetensors only; no remote code.  $PY = /data/hanqim2/.conda/envs/vllm311/bin/python
"""
import argparse
import json
import time

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--model", default="/data/hanqim2/models/bge-small-en-v1.5")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.inp, encoding="utf-8")]
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModel.from_pretrained(a.model, use_safetensors=True).eval().to(a.device)
    t0, embs, n_trunc = time.time(), [], 0
    with torch.inference_mode():
        for i in range(0, len(rows), a.batch):
            texts = [r["text"] for r in rows[i:i + a.batch]]
            n_trunc += sum(len(x) > a.max_len for x in tok(texts)["input_ids"])
            enc = tok(texts, padding=True, truncation=True, max_length=a.max_len, return_tensors="pt").to(a.device)
            v = model(**enc).last_hidden_state[:, 0]
            embs.append(torch.nn.functional.normalize(v, dim=-1).float().cpu().numpy())
    emb = np.concatenate(embs).astype(np.float32)
    np.savez(a.out, ids=np.array([r["id"] for r in rows]), emb=emb)
    print(f"{len(rows)} texts -> {emb.shape} in {time.time() - t0:.0f}s; truncated at {a.max_len} tokens: {n_trunc}")


if __name__ == "__main__":
    main()
