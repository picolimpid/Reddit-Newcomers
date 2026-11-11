"""E4a: a fixed judge calibrated per community (the Eshwar-lens test of "judge fixed, calibrate the people").

Data: Jigsaw "Agile Community Rules" training set (Kaggle 2025, mirrored on Hugging Face by phucthaiv02): 2,029 Reddit
comments from 100 subreddits, each paired with one of two rules ("No legal advice", "No advertising") and a binary
label of whether the comment breaks that rule.  The same rule is enforced very differently across communities
(e.g. the advertising rule in r/soccerstreams vs r/churning), which is the community-specific-norms problem that
Crossmod, SLM-Mod and VASTU study.

Judge (JEV, two requests per comment, nothing else is ever shown to it):
  generic    state {comment, rule}:             "Does `comment` break `rule`?"
  community  state {comment, rule, community}:  "...Would that community's moderators remove `comment` for breaking `rule`?"
A "community" here is a (subreddit, rule) cell.  Evaluation cells: n >= 20 after removing duplicate comments within a
cell.  For each cell and each of R random draws the cell is split in half: anchors (a moderator's k decisions) come
from the first half, everything is scored on the second half.  Shared parts (calibration of the judge, the prior over
community offsets, the cross-community text classifier) are fitted on all OTHER cells (leave-one-cell-out).

Pre-registered in PLAN.md (D1) before any JEV output was looked at:
  H1  pooled AUROC, generic + cell intercept (k = 10)  >  generic (k = 0)           95% CI over cells excludes 0
  H2  pooled AUROC, generic + cell intercept (k = 10)  >  community-name prompt      (labels beat telling the judge)
  H3  pooled AUROC, community prompt + cell intercept (k = 10)  >  community prompt  (still needed with the name)

  python3 -I src/e4a_community_calib.py prep --dry
  python3 -I src/e4a_community_calib.py prep
  python3 -I src/e4a_community_calib.py analyze
"""
import json
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import brentq, minimize  # noqa: E402
from scipy.special import expit, logit  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
import jev  # noqa: E402

OUT = os.path.join(jev.ROOT, "results", "e4a")
os.makedirs(OUT, exist_ok=True)
CSV = os.path.join(jev.ROOT, "data", "ext", "jigsaw_acrc", "train.csv")
Q_GENERIC = ("Does `comment` break `rule`? `comment` is a Reddit comment and `rule` is a rule of the community where "
             "it was posted.")
Q_COMMUNITY = ("`comment` was posted in the Reddit community `community`, which has the rule `rule`. Would that "
               "community's moderators remove `comment` for breaking `rule`?")
KS = (0, 2, 5, 10)
R = 40
B = 2000
MIN_CELL = 20


def load():
    d = pd.read_csv(CSV)
    d["rule_short"] = np.where(d.rule.str.startswith("No legal"), "legal", "advert")
    d["cell"] = d.subreddit + "|" + d.rule_short
    return d


def requests_for(d):
    gen, com = [], []
    for r in d.itertuples():
        gen.append({"state": {"comment": r.body, "rule": r.rule},
                    "questions": {"violates": {"type": "noul", "instructions": Q_GENERIC}}, "tag": "e4a-generic"})
        com.append({"state": {"comment": r.body, "rule": r.rule, "community": "r/" + r.subreddit},
                    "questions": {"removes": {"type": "noul", "instructions": Q_COMMUNITY}}, "tag": "e4a-community"})
    return gen, com


def eval_rows(d):
    """Evaluation cells: (subreddit, rule) cells with >= MIN_CELL distinct comments.  Only these go to JEV (budget:
    a 20-request trial measured 375 input tokens per request, so all 2,029 rows x 2 prompts would cost ~$0.06)."""
    d = d.drop_duplicates(["cell", "body"])
    size = d.cell.value_counts()
    return d[d.cell.isin(size[size >= MIN_CELL].index)]


def prep(dry):
    d = eval_rows(load())
    gen, com = requests_for(d)
    uniq = {json.dumps({k: v for k, v in q.items() if k != "tag"}, sort_keys=True) for q in gen + com}
    print(f"{len(d)} rows in {d.cell.nunique()} evaluation cells, {len(uniq)} distinct requests, expected "
          f"${len(uniq) * 375 * jev.USD_PER_TOKEN:.4f} (375 tokens/request); JEV spent so far ${jev.spent_usd():.4f}, "
          f"cap ${jev.BUDGET:.2f}")
    if dry:
        return
    outs = jev.ask_many(gen + com, workers=8)
    n = len(d)
    d = d.copy()
    d["p_generic"] = [jev.p_yes(o, "violates") for o in outs[:n]]
    d["p_community"] = [jev.p_yes(o, "removes") for o in outs[n:]]
    d.drop(columns=["positive_example_1", "positive_example_2", "negative_example_1", "negative_example_2"]).to_parquet(
        os.path.join(OUT, "jev.parquet"))
    print(f"done; JEV spent so far ${jev.spent_usd():.4f}")


# ------------------------------------------------------------------ analysis helpers
def lr_fit(X, y, l2=1e-4):
    """Plain logistic regression (intercept included in X)."""
    def f(w):
        z = X @ w
        return np.logaddexp(0, z).sum() - y @ z + .5 * l2 * w @ w, X.T @ (expit(z) - y) + l2 * w
    return minimize(f, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B").x


def post_u(m, y, t2):
    """Laplace posterior of a cell offset u ~ N(0, t2) given base margins m and labels y (bracketed root)."""
    if len(m) == 0:
        return 0.0, t2
    g = lambda u: (expit(m + u) - y).sum() + u / t2
    w = t2 * len(m) + 10
    u = brentq(g, -w, w, xtol=1e-10)
    p = expit(m + u)
    return u, 1 / ((p * (1 - p)).sum() + 1 / t2)


def em_tau2(groups, iters=60):
    """Variance of cell offsets by EM with Laplace E-steps over the training cells (mean fixed at 0)."""
    t2 = .5
    for _ in range(iters):
        s = [post_u(m, y, t2) for m, y in groups]
        t2 = max(np.mean([u * u + v for u, v in s]), 1e-4)
    return t2


def auc(y, s):
    o = np.argsort(s, kind="mergesort")
    r = np.empty(len(s))
    r[o] = np.arange(1, len(s) + 1)
    # average ranks for ties
    ss = s[o]
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        if j > i:
            r[o[i:j + 1]] = (i + j + 2) / 2
        i = j + 1
    n1 = y.sum()
    n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def tfidf_margins(d, train_idx, test_idx):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    Xtr = vec.fit_transform(d.body.values[train_idx] + " RULE_" + d.rule_short.values[train_idx])
    clf = LogisticRegression(C=4.0, max_iter=2000).fit(Xtr, d.y.values[train_idx])
    Xte = vec.transform(d.body.values[test_idx] + " RULE_" + d.rule_short.values[test_idx])
    return clf.decision_function(Xte), clf.decision_function(Xtr)


def analyze():
    allrows = load().drop_duplicates(["cell", "body"]).reset_index(drop=True)
    allrows["y"] = allrows.rule_violation.astype(float)
    d = pd.read_parquet(os.path.join(OUT, "jev.parquet")).reset_index(drop=True)
    d["y"] = d.rule_violation.astype(float)
    eps = 1e-4
    d["z_gen"] = logit(d.p_generic.clip(eps, 1 - eps))
    d["z_com"] = logit(d.p_community.clip(eps, 1 - eps))
    d["is_legal"] = (d.rule_short == "legal").astype(float)
    cells = sorted(d.cell.unique())
    print(f"{len(allrows)} distinct (cell, comment) rows in {allrows.cell.nunique()} cells; {len(cells)} evaluation "
          f"cells (n >= {MIN_CELL}) with {len(d)} rows (JEV-scored)")

    # raw zero-shot decisions (what LLM-Mod / SLM-Mod report): p >= 0.5, no calibration at all
    for name, col in (("generic", "p_generic"), ("community", "p_community")):
        e = d[d.cell.isin(cells)]
        pr = (e[col] >= .5).astype(float)
        tp = ((pr == 1) & (e.y == 1)).sum()
        print(f"raw {name:9s} p>=0.5 on evaluation cells: accuracy {np.mean(pr == e.y):.3f}, precision "
              f"{tp / max(pr.sum(), 1):.3f}, recall {tp / e.y.sum():.3f}, AUROC {auc(e.y.values, e[col].values):.3f}")

    # cross-fitted (5 folds by cell) text-model margins, used only to estimate that model's prior over cell offsets
    # (its margins on the cells it was trained on are in-sample and would understate how far communities sit from it)
    cf = np.full(len(allrows), np.nan)
    fold = {c: i % 5 for i, c in enumerate(np.random.default_rng(5).permutation(allrows.cell.unique()))}
    af = allrows.cell.map(fold).values
    for g in range(5):
        m_te, _ = tfidf_margins(allrows, np.flatnonzero(af != g), np.flatnonzero(af == g))
        cf[af == g] = m_te
    tf_cf = pd.Series(cf, index=allrows.cell + "\x00" + allrows.body).reindex(d.cell + "\x00" + d.body).values

    rng = np.random.default_rng(0)
    methods = ["rule base rate", "generic", "community", "TF-IDF cross-community"]
    store = {}  # (method, k) -> list over draws of list over cells of (y, p)
    tau = {}
    for c in cells:
        tr = np.flatnonzero(d.cell.values != c)
        te_all = np.flatnonzero(d.cell.values == c)
        # leakage guard for the text model: drop training rows whose text appears in this cell
        y = d.y.values
        base = {}
        X0 = np.c_[np.ones(len(d)), d.is_legal.values]
        w = lr_fit(X0[tr], y[tr]); base["rule base rate"] = X0 @ w
        for name, col in (("generic", "z_gen"), ("community", "z_com")):
            X = np.c_[np.ones(len(d)), d[col].values, d.is_legal.values]
            w = lr_fit(X[tr], y[tr]); base[name] = X @ w
        # cross-community text classifier (Crossmod-style): trained on every other cell's rows, including cells too
        # small to evaluate; rows whose text also appears in this cell are dropped (leakage guard)
        a_tr = np.flatnonzero((allrows.cell.values != c) & ~np.isin(allrows.body.values, d.body.values[te_all]))
        mt, _ = tfidf_margins(allrows, a_tr, np.arange(len(allrows)))
        mt = pd.Series(mt, index=allrows.cell + "\x00" + allrows.body)
        base["TF-IDF cross-community"] = mt.reindex(d.cell + "\x00" + d.body).values
        # prior variance of cell offsets, from the other cells (base margins on training rows are in-sample; the
        # text model's in-sample margins are overconfident, so its prior uses the same EM on those margins)
        for mth in methods:
            groups = []
            src = tf_cf if mth == "TF-IDF cross-community" else base[mth]
            for c2 in d.cell[tr].unique():
                r2 = np.flatnonzero(d.cell.values == c2)
                groups.append((src[r2], y[r2]))
            tau[(c, mth)] = em_tau2(groups)
        for r in range(R):
            perm = rng.permutation(te_all)
            half = (len(perm) + 1) // 2
            pool, ev = perm[:half], perm[half:]
            for mth in methods:
                t2 = tau[(c, mth)]
                for k in KS:
                    a = pool[:k]
                    u, v = post_u(base[mth][a], y[a], t2)
                    p = expit((base[mth][ev] + u) / np.sqrt(1 + np.pi * v / 8))
                    store.setdefault((mth, k), [[] for _ in range(R)])[r].append((c, y[ev], p))
    print("prior SD of cell offsets (median over folds): " + ", ".join(
        f"{m} {np.sqrt(np.median([tau[(c, m)] for c in cells])):.2f}" for m in methods))

    def metrics(entries, cell_w):
        """entries: list over cells of (cell, y, p); cell_w: dict cell -> multiplicity (bootstrap)."""
        ys, ps, accs = [], [], []
        within = []
        for c, y, p in entries:
            for _ in range(cell_w.get(c, 0)):
                ys.append(y); ps.append(p)
                if 0 < y.sum() < len(y):
                    within.append(auc(y, p))
        y = np.concatenate(ys); p = np.concatenate(ps)
        pc = np.clip(p, 1e-6, 1 - 1e-6)
        pr = (p >= .5)
        tp = (pr & (y == 1)).sum()
        return dict(auroc=auc(y, p), logloss=-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc)),
                    accuracy=np.mean(pr == y), recall=tp / max(y.sum(), 1), precision=tp / max(pr.sum(), 1),
                    within_cell_auroc=np.mean(within))

    names = [(m, k) for m in methods for k in KS]
    w1 = {c: 1 for c in cells}
    point = {nk: pd.DataFrame([metrics(store[nk][r], w1) for r in range(R)]).mean() for nk in names}
    brng = np.random.default_rng(1)
    boots = {nk: [] for nk in names}
    for b in range(B):
        cw = pd.Series(brng.choice(cells, len(cells))).value_counts().to_dict()
        rr = brng.integers(R)  # one random draw per replicate keeps the cost down
        for nk in names:
            boots[nk].append(metrics(store[nk][rr], cw))
    boots = {nk: pd.DataFrame(v) for nk, v in boots.items()}
    cols = ["auroc", "logloss", "accuracy", "recall", "precision", "within_cell_auroc"]
    rows = []
    for nk in names:
        lo, hi = boots[nk][cols].quantile(.025), boots[nk][cols].quantile(.975)
        rows.append({"method": f"{nk[0]} | k={nk[1]}", **{c: f"{point[nk][c]:.3f} [{lo[c]:.3f}, {hi[c]:.3f}]"
                                                          for c in cols}})
    tab = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print("\n== metrics on the held-out half of each evaluation cell (mean over draws; 95% CI, cell bootstrap)\n"
          + tab.to_string(index=False))

    def diff(a, b_):
        pa, pb = point[a], point[b_]
        dd = boots[a][cols] - boots[b_][cols]
        lo, hi = dd.quantile(.025), dd.quantile(.975)
        return {"pair": f"{a[0]} k={a[1]}  -  {b_[0]} k={b_[1]}",
                **{c: f"{pa[c] - pb[c]:+.3f} [{lo[c]:+.3f}, {hi[c]:+.3f}]" for c in cols[:5]}}
    pairs = [(("generic", 10), ("generic", 0)),           # H1
             (("generic", 10), ("community", 0)),         # H2
             (("community", 10), ("community", 0)),       # H3
             (("community", 0), ("generic", 0)),
             (("generic", 5), ("generic", 0)), (("generic", 2), ("generic", 0)),
             (("generic", 10), ("rule base rate", 10)),
             (("generic", 10), ("TF-IDF cross-community", 10)),
             (("TF-IDF cross-community", 10), ("TF-IDF cross-community", 0))]
    dt = pd.DataFrame([diff(a, b_) for a, b_ in pairs])
    print("\n== paired differences (H1-H3 first)\n" + dt.to_string(index=False))
    tab.to_csv(os.path.join(OUT, "summary.csv"), index=False)
    dt.to_csv(os.path.join(OUT, "diffs.csv"), index=False)
    # per-cell view: how far each community sits from the judge, and how much its offset moves things
    pc = []
    for c in cells:
        e = d[d.cell == c]
        pc.append({"cell": c, "n": len(e), "violation_rate": e.y.mean(), "mean_p_generic": e.p_generic.mean(),
                   "mean_p_community": e.p_community.mean(),
                   "within_auroc_generic": auc(e.y.values, e.z_gen.values) if 0 < e.y.sum() < len(e) else np.nan})
    pc = pd.DataFrame(pc).sort_values("violation_rate")
    print("\n== per cell\n" + pc.round(3).to_string(index=False))
    pc.to_csv(os.path.join(OUT, "per_cell.csv"), index=False)


if __name__ == "__main__":
    if sys.argv[1] == "prep":
        prep("--dry" in sys.argv)
    else:
        analyze()
