"""E4h: confirm D4's post hoc fairness finding on new data with a new judge (PLAN.md D9, pre-registered).

POPQUORN offensiveness (Pei & Jurgens 2023): 1,500 Reddit comments, 262 annotators recruited to US census quotas,
~50 comments each, with demographics.  y = offensiveness >= 3 (1-5 scale).  Judge = local Qwen3-32B (bf16, vLLM),
one question "Is `comment` offensive?" (P(Yes) from the first token).  E4c's pipeline: 5 comment folds x 5 annotator
folds, the shared parts (calibration of the judge, prior over people) fitted jointly on the training annotators'
labels, new annotators answer k random anchors among the pool comments they labelled.

  H1 (primary): the largest gap between race groups (>= 10 annotators) in excess log loss shrinks from k = 0 to
      k = 20; annotator-bootstrap 95% CI of the reduction excludes 0.
  H2: split-half reliability (Spearman-Brown) of the per-person intercept >= 0.7 with lower bound >= 0.6.
  H3: within-comment AUROC at k = 20 has its lower 95% bound above 0.5.

  python3 -I src/e4h_popquorn.py
"""
import json
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.special import expit, logit  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
import jev  # noqa: E402
import e1f_onboarding as e1f  # noqa: E402
import e4b_construct as e4b  # noqa: E402
import e4c_single_label as e4c  # noqa: E402

OUT = os.path.join(jev.ROOT, "results", "e4h")
os.makedirs(OUT, exist_ok=True)
CSV = os.path.join(jev.ROOT, "data", "ext", "popquorn", "raw_data.csv")
KS = e4c.KS
B = 2000


def build():
    d = pd.read_csv(CSV)
    n = d.groupby("user_id").size()
    d = d[d.user_id.isin(n.index[n >= 40])]
    items = np.sort(d.instance_id.unique())
    raters = np.sort(d.user_id.unique())
    ipos = {i: k for k, i in enumerate(items)}
    rpos = {r: k for k, r in enumerate(raters)}
    Y = np.full((len(items), len(raters)), np.nan)
    Y[d.instance_id.map(ipos).values, d.user_id.map(rpos).values] = (d.offensiveness >= 3).astype(float).values
    P = {}
    for line in open(os.path.join(OUT, "llm_qwen3-32b-local.jsonl")):
        r = json.loads(line)
        pr = r["probs"]
        P[int(r["id"])] = pr["Yes"] / max(pr["Yes"] + pr["No"], 1e-12)
    Z = logit(np.clip(np.array([P[i] for i in items]), 1e-6, 1 - 1e-6))[:, None]
    demo = d.drop_duplicates("user_id").set_index("user_id").loc[raters]
    return Y, Z, demo, items, raters


def main():
    pd.set_option("display.width", 250)
    Y, Z, demo, items, raters = build()
    I, P = Y.shape
    lab = ~np.isnan(Y)
    print(f"POPQUORN: {P} annotators (>= 40 labels) x {I} comments, {int(lab.sum())} labels, offensive rate "
          f"{np.nanmean(Y):.3f}; judge AUROC vs majority label "
          f"{__import__('sklearn.metrics', fromlist=['roc_auc_score']).roc_auc_score((np.nanmean(Y, 1) >= .5).astype(int), Z[:, 0]):.3f}")
    cfold = np.empty(I, int)
    cfold[np.random.default_rng(0).permutation(I)] = np.arange(I) % 5
    rfold = np.empty(P, int)
    rfold[np.random.default_rng(2).permutation(P)] = np.arange(P) % 5
    e4c._G.update(Y=Y, rated=lab, Z=Z, cfold=cfold, rfold=rfold)
    jobs = [(f, g, "all labels") for f in range(5) for g in range(5)]
    with ProcessPoolExecutor(25, mp_context=mp.get_context("fork"), initializer=e1f._init_worker) as ex:
        out = list(ex.map(e4c.cell_job, jobs))
    M = {}
    for f, g, cond, test, teR, res, info in out:
        for rep in range(e1f.R):
            for ki, k in enumerate(KS):
                M.setdefault((k, rep), np.full((I, P), np.nan))[np.ix_(test, teR)] = res[rep, ki]
    print("prior SD of people (mean over cells): %.2f" % np.sqrt(np.mean([o[6]["tau2"] for o in out])))
    # within-comment AUROC and pooled metrics (two-way bootstrap, as E4c)
    Mz = {key: np.where(lab, v, 0.) for key, v in M.items()}
    brng = np.random.default_rng(1)
    C = np.stack([np.bincount(brng.integers(0, I, I), minlength=I) for _ in range(e1f.B)]).astype(float)
    W = np.stack([np.bincount(brng.integers(0, P, P), minlength=P) for _ in range(e1f.B)]).astype(float)
    rows = []
    for k in KS:
        reps = [e1f.boot_stats(Mz[k, rep], Y, C, W, rfold) for rep in range(e1f.R)]
        pt, lb, wb = (np.mean([x[j] for x in reps], 0) for j in range(3))
        rows.append(dict(k=k, within_auroc=f"{pt[3]:.3f} [{np.quantile(wb, .025):.3f}, {np.quantile(wb, .975):.3f}]",
                         pooled_auroc=round(pt[2], 3), logloss=round(pt[0], 4)))
    t = pd.DataFrame(rows)
    print("\n== learning curve (random anchors)\n" + t.to_string(index=False))
    lo_k20 = float(t[t.k == 20].within_auroc.str.extract(r"\[([\d.]+),")[0].iloc[0])
    print(f"H3 (within-comment AUROC at k = 20, lower bound > 0.5): {'MET' if lo_k20 > .5 else 'NOT met'}")

    # H2: split-half reliability of the intercept, judge margins (k = 0) as offset
    M0 = np.where(lab, M[0, 0], np.nan)
    rng = np.random.default_rng(0)
    halves = [e4b.split_half(M0, Y, rng) for _ in range(e4b.SPLITS)]
    r = float(np.mean([np.corrcoef(a, b)[0, 1] for a, b in halves]))
    bo = []
    for _ in range(B):
        idx = brng.integers(0, P, P)
        a, b = halves[brng.integers(len(halves))]
        bo.append(e4b.sb(np.corrcoef(a[idx], b[idx])[0, 1]))
    lo, hi = np.quantile(bo, [.025, .975])
    print(f"\nH2 split-half reliability: r = {r:.3f}, Spearman-Brown {e4b.sb(r):.3f} [{lo:.3f}, {hi:.3f}] -> "
          f"{'MET' if e4b.sb(r) >= .7 and lo >= .6 else 'NOT met'} (labels per annotator median {int(np.median(lab.sum(0)))})")

    # H1: excess log loss by race group
    def ll_cols(Mm):
        y = np.nan_to_num(Y)
        Mm = np.nan_to_num(Mm)
        return np.nanmean(np.where(lab, np.logaddexp(0, Mm) - y * Mm, np.nan), 0)
    u_best = np.array([e4b.u_hat(M0[lab[:, q], q], Y[lab[:, q], q], t2=100.) for q in range(P)])
    Lb = ll_cols(np.where(lab, M0 + u_best[None, :], np.nan))
    ex0 = ll_cols(np.where(lab, M[0, 0], np.nan)) - Lb
    ex20 = np.mean([ll_cols(np.where(lab, M[20, rep], np.nan)) for rep in range(e1f.R)], 0) - Lb
    per = pd.DataFrame(dict(rater=raters, excess_k0=ex0, excess_k20=ex20, intercept=u_best,
                            race=demo.race.values, gender=demo.gender.values, age=demo.age.values,
                            education=demo.education.values))
    per.to_csv(os.path.join(OUT, "per_rater.csv"), index=False)
    print(f"\nexcess log loss: judge alone {ex0.mean():.4f}, with k = 20 anchors {ex20.mean():.4f}")
    res = []
    for att in ("race", "gender", "age", "education"):
        g = per.groupby(att).agg(n=("rater", "size"), excess_k0=("excess_k0", "mean"), excess_k20=("excess_k20", "mean"))
        g = g[g.n >= 10]
        if len(g) < 2:
            continue
        groups = list(g.index)
        sub = per[per[att].isin(groups)]
        bs = []
        for _ in range(B):
            s = sub.sample(frac=1, replace=True, random_state=int(brng.integers(1e9)))
            gg = s.groupby(att)[["excess_k0", "excess_k20"]].mean()
            bs.append((gg.excess_k0.max() - gg.excess_k0.min()) - (gg.excess_k20.max() - gg.excess_k20.min()))
        gap0, gap20 = g.excess_k0.max() - g.excess_k0.min(), g.excess_k20.max() - g.excess_k20.min()
        lo, hi = np.quantile(bs, [.025, .975])
        res.append(dict(attribute=att, groups=", ".join(f"{k} (n={int(row.n)}) {row.excess_k0:.3f}" for k, row in g.iterrows()),
                        gap_k0=round(gap0, 4), gap_k20=round(gap20, 4),
                        reduction=f"{gap0 - gap20:+.4f} [{lo:+.4f}, {hi:+.4f}]",
                        verdict=("H1 MET" if lo > 0 else "H1 NOT met") if att == "race" else ""))
    print(pd.DataFrame(res).to_string(index=False))
    pd.DataFrame(res).to_csv(os.path.join(OUT, "gaps.csv"), index=False)
    t.to_csv(os.path.join(OUT, "curve.csv"), index=False)


if __name__ == "__main__":
    main()
