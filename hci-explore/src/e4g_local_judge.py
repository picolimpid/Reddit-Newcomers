"""E4g: the same per-person calibration with a local open-weight judge (PLAN.md D8).  DICES-350 (104 raters, E1f's
two-way folds), E4c's estimator and random anchors; the judge's six answers are either JEV's or local Qwen3-32B's
(bf16 via vLLM, from E1e; Yes/No probabilities on the same question wording).  Training logs: all labels or one
label per conversation (m = 1).

Primary (pre-registered): within-conversation AUROC at k = 20, Qwen minus JEV (all labels); non-inferior if the lower
95% bound (two-way bootstrap) is above -0.02.

  python3 -I src/e4g_local_judge.py
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

sys.path.insert(0, os.path.dirname(__file__))
import jev  # noqa: E402
import e1e_dices as e1e  # noqa: E402
import e1f_onboarding as e1f  # noqa: E402
import e4c_single_label as e4c  # noqa: E402

OUT = os.path.join(jev.ROOT, "results", "e4g")
os.makedirs(OUT, exist_ok=True)
KS = e4c.KS
CONDS = ["all labels", "m=1"]
JUDGES = ["JEV", "Qwen3-32B (local)"]
Z_BY = {}


def qwen_Z(items):
    P = {}
    for line in open(os.path.join(jev.ROOT, "results", "e1e", "llm_qwen3-32b-local.jsonl")):
        r = json.loads(line)
        i, q = r["id"].split("|")
        pr = r["probs"]
        P[(i, q)] = pr["Yes"] / max(pr["Yes"] + pr["No"], 1e-12)
    qs = list(e1e.QUESTIONS)
    Zq = np.array([[P[(str(i), q)] for q in qs] for i in items])
    return e1e.logit(np.clip(Zq, 1e-6, 1 - 1e-6))


def job(args):
    judge, f, g, cond = args
    e4c._G["Z"] = Z_BY[judge]
    out = e4c.cell_job((f, g, cond))
    return (judge,) + out


def main():
    pd.set_option("display.width", 250)
    G = e1e.build("104")
    I, P = G["I"], G["P"]
    Z_BY["JEV"] = G["Z"]
    Z_BY["Qwen3-32B (local)"] = qwen_Z(G["items"])
    print(f"DICES-350: {P} raters x {I} conversations; judge inputs {Z_BY['JEV'].shape} and {Z_BY['Qwen3-32B (local)'].shape}")
    cfold = np.empty(I, int)
    cfold[np.random.default_rng(0).permutation(I)] = np.arange(I) % 5
    rfold = np.empty(P, int)
    rfold[np.random.default_rng(2).permutation(P)] = np.arange(P) % 5
    rated = np.ones(G["Y"].shape, bool)
    e4c._G.update(Y=G["Y"], rated=rated, Z=None, cfold=cfold, rfold=rfold)
    jobs = [(j, f, g, c) for j in JUDGES for c in CONDS for f in range(5) for g in range(5)]
    with ProcessPoolExecutor(25, mp_context=mp.get_context("fork"), initializer=e1f._init_worker) as ex:
        out = list(ex.map(job, jobs))
    M = {}
    for judge, f, g, cond, test, teR, res, info in out:
        for rep in range(e1f.R):
            for ki, k in enumerate(KS):
                M.setdefault((judge, cond, k, rep), np.full((I, P), np.nan))[np.ix_(test, teR)] = res[rep, ki]
    lab = ~np.isnan(G["Y"])
    for key in M:
        M[key] = np.where(lab, M[key], 0.)
    brng = np.random.default_rng(1)
    C = np.stack([np.bincount(brng.integers(0, I, I), minlength=I) for _ in range(e1f.B)]).astype(float)
    W = np.stack([np.bincount(brng.integers(0, P, P), minlength=P) for _ in range(e1f.B)]).astype(float)
    st = {}
    for judge in JUDGES:
        for cond in CONDS:
            for k in KS:
                reps = [e1f.boot_stats(M[judge, cond, k, rep], G["Y"], C, W, rfold) for rep in range(e1f.R)]
                st[judge, cond, k] = tuple(np.mean([x[j] for x in reps], 0) for j in range(3))
    tab = pd.DataFrame([dict(judge=j, training=c, k=k, within_auroc=pt[3], pooled_auroc=pt[2], logloss=pt[0])
                        for (j, c, k), (pt, lb, wb) in st.items()])
    tab.to_csv(os.path.join(OUT, "curve.csv"), index=False)
    print(tab.round(4).to_string(index=False))
    rows = []
    for tag, a, b in [("primary", ("Qwen3-32B (local)", "all labels", 20), ("JEV", "all labels", 20)),
                      ("", ("Qwen3-32B (local)", "m=1", 20), ("JEV", "m=1", 20)),
                      ("", ("Qwen3-32B (local)", "all labels", 5), ("JEV", "all labels", 5)),
                      ("", ("Qwen3-32B (local)", "all labels", 0), ("JEV", "all labels", 0)),
                      ("", ("Qwen3-32B (local)", "m=1", 20), ("Qwen3-32B (local)", "all labels", 20))]:
        pa, la, wa = st[a]
        pb, lb, wb = st[b]
        lo = np.quantile(wa - wb, .025)
        rows.append(dict(test=tag, pair=f"{a[0]}, {a[1]}, k={a[2]}  -  {b[0]}, {b[1]}, k={b[2]}",
                         within=f"{pa[3] - pb[3]:+.4f} [{lo:+.4f}, {np.quantile(wa - wb, .975):+.4f}]",
                         pooled_auroc=f"{pa[2] - pb[2]:+.4f}",
                         logloss=f"{pa[0] - pb[0]:+.4f} [{np.quantile(la - lb, .025):+.4f}, {np.quantile(la - lb, .975):+.4f}]",
                         noninferior=(lo > -.02) if tag == "primary" else ""))
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(OUT, "tests.csv"), index=False)
    print("\n== paired differences (two-way bootstrap)\n" + d.to_string(index=False))


if __name__ == "__main__":
    main()
