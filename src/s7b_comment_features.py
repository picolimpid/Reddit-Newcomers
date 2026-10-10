"""Stage 7b: comment-content and community-integration measures for each
subject's (newcomer or existing member) first comment in the target thread.

B. content:     question, stance toward parent (LLM), sentiment, toxicity/insult, style
C. integration: comment <-> prior-discussion similarity, tone of replies received,
                back-and-forth continuation
"""
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from api import DATA

BAD = {"[deleted]", "[removed]", "", None}
df = pd.read_parquet(os.path.join(DATA, "analysis_table.parquet"))
posts = pd.read_parquet(os.path.join(DATA, "target_posts.parquet")).set_index("post_id")
tc = pd.read_parquet(os.path.join(DATA, "thread_comments.parquet"))
tc = tc.drop_duplicates("id").set_index("id")
subj = df[((df.newcomer_12m == 1) | (df.existing == 1)) & df.hist_idx.notna()].copy()
print(len(subj), "subjects", flush=True)

# ---------------------------------------------------------------- thread structure
children = defaultdict(list)
for cid, par in zip(tc.index, tc.parent_id):
    children[par].append(cid)


def subtree(cid):
    out, stack = [], list(children.get("t1_" + cid, []))
    while stack:
        c = stack.pop()
        out.append(c)
        stack.extend(children.get("t1_" + c, []))
    return out


rows_reply = []  # (subject index, reply id) for direct replies received
for i, r in subj.iterrows():
    first = r.first_comment_id
    direct = [c for c in children.get("t1_" + first, [])
              if tc.at[c, "author"] not in (r.author, "AutoModerator")]
    rows_reply += [(i, c) for c in direct]
    sub = subtree(first)
    own = [c for c in sub if tc.at[c, "author"] == r.author]
    # newcomer answered someone who replied inside their exchange
    subj.at[i, "answered_replier"] = int(any(
        tc.at[c, "parent_id"].startswith("t1_") and tc.at[c, "parent_id"][3:] in tc.index
        and tc.at[tc.at[c, "parent_id"][3:], "author"] != r.author for c in own))
    subj.at[i, "exchange_turns"] = len(own)

# ---------------------------------------------------------------- B: question + style
WORD = re.compile(r"[A-Za-z']+")
QSTART = re.compile(r"^\s*(who|what|when|where|why|how|is|are|was|were|do|does|did|can|could|would|"
                    r"should|will|has|have|anyone|any)\b", re.I)
HEDGE = {"maybe", "perhaps", "probably", "possibly", "might", "seems", "guess", "think", "suppose",
         "apparently", "kinda", "sorta"}
LAUGH = re.compile(r"\b(lol|lmao|lmfao|rofl|haha\w*|hehe\w*)\b|[\U0001F600-\U0001F64F\U0001F923]", re.I)


def style(t):
    t = t or ""
    w = [x.lower() for x in WORD.findall(t)]
    n = max(len(w), 1)
    return {
        "question": int("?" in t or any(QSTART.match(s) for s in re.split(r"[.!\n]+", t) if s.strip())),
        "n_words": len(w),
        "rate_i": sum(x in ("i", "i'm", "i've", "i'd", "i'll", "me", "my", "mine") for x in w) / n,
        "rate_you": sum(x in ("you", "you're", "your", "yours", "you've", "you'd", "u", "ur") for x in w) / n,
        "rate_we": sum(x in ("we", "we're", "our", "ours", "us", "we've") for x in w) / n,
        "rate_hedge": sum(x in HEDGE for x in w) / n,
        "exclaim": int("!" in t),
        "laugh": int(bool(LAUGH.search(t))),
    }


subj = subj.join(pd.DataFrame([style(t) for t in subj.first_body], index=subj.index))

# ---------------------------------------------------------------- sentiment + toxicity
reply_ids = [c for _, c in rows_reply]
texts = list(subj.first_body.fillna("")) + [tc.at[c, "body"] or "" for c in reply_ids]
sent = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=0, top_k=None, truncation=True, max_length=256)
tox = pipeline("text-classification", model="unitary/unbiased-toxic-roberta", device=0, top_k=None,
               function_to_apply="sigmoid", truncation=True, max_length=256)
S = [{d["label"]: d["score"] for d in r} for r in sent(texts, batch_size=64)]
T = [{d["label"]: d["score"] for d in r} for r in tox(texts, batch_size=64)]
sentiment = np.array([s["positive"] - s["negative"] for s in S])
insult = np.array([t["insult"] for t in T])
toxicity = np.array([t["toxicity"] for t in T])
n = len(subj)
subj["sentiment"], subj["insult"], subj["toxicity"] = sentiment[:n], insult[:n], toxicity[:n]
print("sentiment/toxicity done", flush=True)

# ---------------------------------------------------------------- stance (LLM, replies only)
LLM = "Qwen/Qwen2.5-3B-Instruct"
tok = AutoTokenizer.from_pretrained(LLM)
tok.padding_side = "left"
del sent, tox
torch.cuda.empty_cache()
llm = AutoModelForCausalLM.from_pretrained(LLM, dtype=torch.bfloat16).to("cuda").eval()
LABELS = {"agree": tok.encode("agree")[0], "disagree": tok.encode("disagree")[0],
          "neutral": tok.encode("neutral")[0]}


def prompt(parent, reply):
    msg = ("You are annotating Reddit conversations. Classify the stance of the Reply toward the "
           "Parent comment.\n"
           "- agree: supports, confirms, praises, or builds on the parent (e.g. adds a similar example).\n"
           "- disagree: contradicts, corrects, argues against, or mocks the parent's point.\n"
           "- neutral: a question, an unrelated joke or tangent, or no position toward the parent.\n\n"
           "Examples:\n"
           "Parent: \"Same thing happened at my school.\" Reply: \"Happened at mine too, every year.\" -> agree\n"
           "Parent: \"Best movie of the decade.\" Reply: \"It really wasn't, the plot made no sense.\" -> disagree\n"
           "Parent: \"My cat does this every morning.\" Reply: \"What breed is it?\" -> neutral\n\n"
           f"Parent comment:\n\"\"\"{(parent or '')[:600]}\"\"\"\n\n"
           f"Reply:\n\"\"\"{(reply or '')[:600]}\"\"\"\n\n"
           "Answer with exactly one word: agree, disagree, or neutral.")
    return tok.apply_chat_template([{"role": "user", "content": msg}], tokenize=False,
                                   add_generation_prompt=True)


@torch.no_grad()
def stance(pairs, bs=16):
    out = []
    for k in range(0, len(pairs), bs):
        enc = tok([prompt(p, r) for p, r in pairs[k:k + bs]], return_tensors="pt", padding=True).to("cuda")
        logits = llm(**enc).logits[:, -1, :].float()
        pr = torch.softmax(logits[:, list(LABELS.values())], dim=-1).cpu().numpy()
        out.extend(pr)
    return np.array(out)  # columns: agree, disagree, neutral


# (1) subject's first comment toward its parent comment (replies only)
is_reply = subj.index[subj.top_level == 0]
pairs, keep = [], []
for i in is_reply:
    par = tc.at[subj.at[i, "first_comment_id"], "parent_id"][3:]
    if par in tc.index and tc.at[par, "body"] not in BAD and subj.at[i, "first_body"] not in BAD:
        pairs.append((tc.at[par, "body"], subj.at[i, "first_body"]))
        keep.append(i)
P1 = stance(pairs)
subj.loc[keep, "stance_disagree"] = (P1.argmax(1) == 1).astype(int)
subj.loc[keep, "stance_agree"] = (P1.argmax(1) == 0).astype(int)
subj.loc[keep, "p_disagree"] = P1[:, 1]
# (2) replies received, toward the subject's first comment
P2 = stance([(subj.at[i, "first_body"], tc.at[c, "body"]) for i, c in rows_reply])
print("stance done:", len(P1), "first-comment pairs,", len(P2), "reply pairs", flush=True)
del llm
torch.cuda.empty_cache()

rep = pd.DataFrame({"i": [i for i, _ in rows_reply], "sent": sentiment[n:], "insult": insult[n:],
                    "disagree": (P2.argmax(1) == 1).astype(int) if len(P2) else []})
agg = rep.groupby("i").agg(reply_tone_sentiment=("sent", "mean"), reply_tone_insult=("insult", "max"),
                           reply_tone_disagree=("disagree", "max"))
subj = subj.join(agg)

# ---------------------------------------------------------------- C: comment <-> discussion similarity
emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")
body = tc.body.fillna("")
valid = ~body.isin(["[deleted]", "[removed]", ""])
E = emb.encode(list(body.str[:1000]), batch_size=256, normalize_embeddings=True,
               show_progress_bar=False).astype(np.float32)
E[~valid.values] = 0
pos = {cid: k for k, cid in enumerate(tc.index)}
title_E = emb.encode(list((posts.title + "\n" + posts.selftext.fillna("").str[:1000]).str.strip()),
                     normalize_embeddings=True)
title_pos = {p: k for k, p in enumerate(posts.index)}

# prefix sums of embeddings per thread (time-ordered) -> centroid of discussion before entry
thread_prefix = {}
for pid, g in tc.reset_index().sort_values("created_utc").groupby("post_id"):
    idx = [pos[c] for c in g.id]
    cs = np.cumsum(E[idx], axis=0)
    cnt = np.cumsum(valid.values[idx])
    thread_prefix[pid] = (g.id.tolist(), cs, cnt)
cent = np.load(os.path.join(DATA, "user_centroids_mini.npy"))
cent_idx = {k: j for j, k in enumerate(np.load(os.path.join(DATA, "user_centroids_mini_idx.npy")))}


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else np.nan


for i, r in subj.iterrows():
    e = E[pos[r.first_comment_id]]
    ids, cs, cnt = thread_prefix[r.post_id]
    k = ids.index(r.first_comment_id)
    subj.at[i, "sim_thread"] = cos(e, cs[k - 1]) if k > 0 and cnt[k - 1] > 0 else np.nan
    subj.at[i, "sim_post"] = cos(e, title_E[title_pos[r.post_id]])
    par = tc.at[r.first_comment_id, "parent_id"]
    subj.at[i, "sim_parent"] = cos(e, E[pos[par[3:]]]) if par.startswith("t1_") and par[3:] in pos else np.nan
    subj.at[i, "sim_own_history"] = cos(e, cent[cent_idx[i]]) if i in cent_idx else np.nan

cols = ["answered_replier", "exchange_turns", "question", "n_words", "rate_i", "rate_you", "rate_we",
        "rate_hedge", "exclaim", "laugh", "sentiment", "insult", "toxicity", "stance_disagree",
        "stance_agree", "p_disagree", "reply_tone_sentiment", "reply_tone_insult", "reply_tone_disagree",
        "sim_thread", "sim_post", "sim_parent", "sim_own_history"]
subj[cols].to_parquet(os.path.join(DATA, "comment_features.parquet"))
print("saved", subj[cols].shape)
print(subj.groupby("newcomer_12m")[cols].mean().T.round(3))
