"""bge-small embeddings of r/science 2016 newcomers' first comments and history items (CPU; 2026-11-06), for who_returns.py.
    CUDA_VISIBLE_DEVICES="" /data/hanqim2/.conda/envs/vllm311/bin/python src/embed_bge.py   (run from the repo root)"""
import json, numpy as np, pandas as pd, torch, os
from transformers import AutoTokenizer, AutoModel
torch.set_num_threads(16)
OUT = "data/rscience2016/emb_bge"
R = pd.read_parquet("data/rscience2016/retro.parquet")
tok = AutoTokenizer.from_pretrained("/data/hanqim2/models/bge-small-en-v1.5"); m = AutoModel.from_pretrained("/data/hanqim2/models/bge-small-en-v1.5").eval()
def emb(texts, bs=128):
    out = []
    for i in range(0, len(texts), bs):
        b = tok(texts[i:i+bs], padding=True, truncation=True, max_length=256, return_tensors="pt")
        with torch.no_grad():
            h = m(**b).last_hidden_state[:, 0]
        out.append(torch.nn.functional.normalize(h, dim=-1).numpy())
    return np.vstack(out)
first = emb([str(x)[:1500] for x in R.first_body])
items, owner = [], []
for k, h in enumerate(R.history):
    xs = [str(x["text"])[:600] for x in json.loads(h) if len(str(x["text"]).strip()) >= 5 and str(x["text"]).strip() not in ("[deleted]", "[removed]")][:20]
    items += xs; owner += [k] * len(xs)
E = emb(items)
hist = np.zeros((len(R), E.shape[1]))
owner = np.array(owner)
for k in range(len(R)):
    if (owner == k).any(): hist[k] = E[owner == k].mean(0)
np.save(f"{OUT}/first.npy", first); np.save(f"{OUT}/hist.npy", hist)
print(first.shape, hist.shape, len(items))
