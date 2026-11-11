"""E1c: can a learned per-person offset on top of JEV's probability predict *individual* coders' labels?

Measuring Hate Speech (MHS) keeps every annotator's id and demographics.  70 reference comments were labelled by
>= 100 annotators each (they link the IRT scale), and every one of the 7,912 annotators labelled ~6 of them.  So:
  anchors  = an annotator's labels on the reference comments (what a coder would label once, up front)
  eval     = the same annotators' labels on the E1 sample comments (new items, 392 after removing 8 references)
JEV gives one content probability per comment and code (constant across people).  The question is whether an
external model that learns each person's offset (linear, or a small network on person x item features) adds to it.

  python3 src/e1c_personal.py prep     # JEV on the reference comments (~$0.0012), text file for embeddings
  CUDA_VISIBLE_DEVICES="" $PY src/embed_local.py results/e1c/emb_in.jsonl results/e1c/emb_bge-small.npz
  python3 src/e1c_personal.py analyze
"""
import json
import os
import sys
import time

# one BLAS thread everywhere: worker processes are forked, and forking after a multi-threaded BLAS call can
# deadlock the children on an inherited lock (it did, 2026-11-04)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
import jev  # noqa: E402
import e1_perspectivist as e1  # noqa: E402

OUT = os.path.join(jev.ROOT, "results", "e1c")
os.makedirs(OUT, exist_ok=True)
MHS = os.path.join(jev.ROOT, "data", "ext", "mhs", "mhs.parquet")
REF_MIN = 50  # a comment with at least this many annotators is a reference (anchor) comment
CODES = list(e1.CODES)


def sets():
    m = pd.read_parquet(MHS, columns=["comment_id", "annotator_id", "text"])
    n = m.groupby("comment_id").size()
    ref = set(n[n >= REF_MIN].index)
    s = pd.read_parquet(os.path.join(e1.OUT, "sample.parquet"))
    ev = set(s.comment_id) - ref
    return m, ref, ev


def prep():
    m, ref, ev = sets()
    txt = m.drop_duplicates("comment_id").set_index("comment_id").text
    qs = {c: {"type": "noul", "instructions": q} for c, (_, q) in e1.CODES.items()}  # identical to E1 -> cache hits
    ids = sorted(ref)
    t0, before = time.time(), jev.spent_usd()
    outs = jev.ask_many([{"state": {"comment": txt[i]}, "questions": qs, "tag": "e1c-ref"} for i in ids])
    pd.DataFrame([{"comment_id": i, **{"j_" + c: jev.p_yes(o, c) for c in CODES}, "cached": o["_cached"]}
                  for i, o in zip(ids, outs)]).to_parquet(os.path.join(OUT, "jev_ref.parquet"))
    with open(os.path.join(OUT, "emb_in.jsonl"), "w") as f:
        for i in sorted(ref | ev):
            f.write(json.dumps({"id": str(i), "text": txt[i]}) + "\n")
    print(f"{len(ids)} reference comments ({sum(o['_cached'] for o in outs)} cached) in {time.time() - t0:.0f}s, "
          f"spent now ${jev.spent_usd() - before:.4f}; JEV total ${jev.spent_usd():.4f}; "
          f"{len(ref | ev)} texts for embeddings")


# ----------------------------------------------------------------------------------------------------------- analysis
import multiprocessing as mp  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

from scipy.optimize import minimize  # noqa: E402
from scipy.special import expit  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from offset_mlp import OffsetMLP  # noqa: E402

LAMS_U = (0.5, 1., 2., 4., 8.)  # prior precision of a person's offset (sd 1.4 ... 0.35 logits)
LAMS_D = (3., 10., 30., 100.)   # prior precision of demographic weights
LAMS_S = (0.5, 1., 2., 4., 8., 32., 128., 1e4)  # shared severity; 1e4 = effectively none
NFOLD, B = 5, 1000
MLP_CFG = dict(hidden=(64,), wd=1e-4, lr=1e-3, epochs=30, batch=512, dropout=0.1, patience=3)
MLP_SEEDS = 3
_G = {}


def logit(p):
    p = np.clip(np.asarray(p, float), .005, .995)
    return np.log(p / (1 - p))


def demographics(m):
    a = m.drop_duplicates("annotator_id").set_index("annotator_id")
    cols = [c for c in a.columns if c.startswith("annotator_") and a[c].dtype == bool]
    D = a[cols].astype(float)
    age = a.annotator_age
    D["age_z"] = ((age - age.mean()) / age.std()).fillna(0)
    D["age_missing"] = age.isna().astype(float)
    return D


def tfidf_prior(m, ids, exclude):
    """E1's supervised baseline, retrained without the reference AND evaluation comments (cached)."""
    path = os.path.join(OUT, "tfidf_prior.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    tr = m[~m.comment_id.isin(exclude)]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=200000, sublinear_tf=True)
    X = vec.fit_transform(tr.text)
    txt = m.drop_duplicates("comment_id").set_index("comment_id").text
    Xs = vec.transform(txt[ids])
    out = pd.DataFrame({"comment_id": ids})
    for c, (rule, _) in e1.CODES.items():
        out[c] = LogisticRegression(C=4.0, max_iter=2000).fit(X, rule(tr).astype(int)).predict_proba(Xs)[:, 1]
    print(f"  TF-IDF prior trained on {len(tr)} labels of {tr.comment_id.nunique()} other comments")
    out.to_parquet(path)
    return out


def fit_lin_person(X, y, pid, P, w0, lam_w, lower, lam_u):
    """logit = X w + b + u[pid]; MAP with w ~ N(w0, 1/lam_w), b ~ N(0, 1e3), u ~ N(0, 1/lam_u); lam_u None: no u."""
    k, use_u = X.shape[1], lam_u is not None

    def f(th):
        w, b = th[:k], th[k]
        mg = X @ w + b
        if use_u:
            u = th[k + 1:]
            mg = mg + u[pid]
        r = expit(mg) - y
        loss = np.logaddexp(0, mg).sum() - y @ mg + (lam_w * (w - w0) ** 2).sum() / 2 + 5e-4 * b * b
        g = [X.T @ r + lam_w * (w - w0), [r.sum() + 1e-3 * b]]
        if use_u:
            loss += lam_u / 2 * u @ u
            g.append(np.bincount(pid, weights=r, minlength=P) + lam_u * u)
        return loss, np.concatenate(g)

    nu = P if use_u else 0
    bounds = [(lo if np.isfinite(lo) else None, None) for lo in lower] + [(None, None)] * (1 + nu)
    th = minimize(f, np.r_[w0, 0., np.zeros(nu)], jac=True, method="L-BFGS-B", bounds=bounds,
                  options={"maxiter": 3000}).x
    return th[:k], th[k], (th[k + 1:] if use_u else np.zeros(P))


def cv_fit(X, y, w0, lower, grid):
    """grid: list of (lam_w, lam_u).  Picks the setting with the best held-out log loss over reference-comment folds
    (so a person's offset is always estimated from anchors other than the item it is scored on), refits on all
    anchors.  Returns margins for the evaluation rows, cross-fitted margins for the anchor rows, offsets."""
    pid, P, isref, fold = _G["pid"], _G["P"], _G["isref"], _G["fold_r"]
    R, S = np.flatnonzero(isref), np.flatnonzero(~isref)
    best = None
    for lam_w, lam_u in grid:
        ll, cf, uf = 0.0, np.full(len(y), np.nan), []
        for f in range(NFOLD):
            trn, tst = R[fold[R] != f], R[fold[R] == f]
            w, b, u = fit_lin_person(X[trn], y[trn], pid[trn], P, w0, lam_w, lower, lam_u)
            cf[tst] = X[tst] @ w + b + u[pid[tst]]
            ll += (np.logaddexp(0, cf[tst]) - y[tst] * cf[tst]).sum()
            uf.append(u)
        if best is None or ll < best[0]:
            best = (ll, lam_w, lam_u, cf, uf)
    ll, lam_w, lam_u, cf, uf = best
    w, b, u = fit_lin_person(X[R], y[R], pid[R], P, w0, lam_w, lower, lam_u)
    cf[S] = X[S] @ w + b + u[pid[S]]
    return dict(m=cf, u_final=u, u_fold=uf, lam_u=lam_u, lam_w=np.atleast_1d(lam_w)[-1], cv_ll=ll / len(R), w=w, b=b)


def fit_code(c):
    y, z, zt, D = _G["Y"][c], _G["Zj"][c], _G["Zt"][c], _G["Drow"]
    inf, d = np.inf, D.shape[1]
    one, X = np.array([1.]), z[:, None]
    XD = np.c_[z, D]
    w0D, loD = np.r_[1., np.zeros(d)], [0.] + [-inf] * d
    out = {"JEV + Platt": cv_fit(X, y, one, [0.], [(1., None)]),
           "JEV + person offset": cv_fit(X, y, one, [0.], [(1., lu) for lu in LAMS_U]),
           "JEV + demographics": cv_fit(XD, y, w0D, loD, [(np.r_[1., np.full(d, ld)], None) for ld in LAMS_D])}
    ld = out["JEV + demographics"]["lam_w"]
    out["JEV + demographics + person offset"] = cv_fit(XD, y, w0D, loD, [(np.r_[1., np.full(d, ld)], lu) for lu in LAMS_U])
    out["TF-IDF + Platt"] = cv_fit(zt[:, None], y, one, [0.], [(1., None)])
    out["TF-IDF + person offset"] = cv_fit(zt[:, None], y, one, [0.], [(1., lu) for lu in LAMS_U])
    return c, out


def fit_shared(lam_s, Z, Y, pid, P):
    """One severity per person shared by all six codes: logit_c = a_c z_c + b_c + s[pid].  (A first version also had
    per-code loadings beta_c * s; loadings and severities traded off -- beta 8-36, sd(s) 0.006 -- so the scale was not
    identified.  With the loading fixed at 1 the prior on s sets the scale.)"""
    C = Z.shape[1]

    def f(th):
        a, b, s = th[:C], th[C:2 * C], th[2 * C:]
        mg = Z * a + b + s[pid][:, None]
        r = expit(mg) - Y
        loss = (np.logaddexp(0, mg) - Y * mg).sum() + ((a - 1) ** 2).sum() / 2 + 5e-4 * b @ b + lam_s / 2 * s @ s
        return loss, np.r_[(r * Z).sum(0) + (a - 1), r.sum(0) + 1e-3 * b,
                           np.bincount(pid, weights=r.sum(1), minlength=P) + lam_s * s]

    th = minimize(f, np.r_[np.ones(C), np.zeros(C), np.zeros(P)], jac=True, method="L-BFGS-B",
                  options={"maxiter": 3000}).x
    return th[:C], th[C:2 * C], np.ones(C), th[2 * C:]


def _shared_data():
    return (np.c_[tuple(_G["Zj"][c] for c in CODES)], np.c_[tuple(_G["Y"][c] for c in CODES)], _G["pid"], _G["P"],
            np.flatnonzero(_G["isref"]), np.flatnonzero(~_G["isref"]), _G["fold_r"])


def shared_cv_job(job):
    """Held-out log loss of the shared-severity model for one (lam, reference fold)."""
    lam, f = job
    Z, Y, pid, P, R, S, fold = _shared_data()
    trn, tst = R[fold[R] != f], R[fold[R] == f]
    a, b, be, s = fit_shared(lam, Z[trn], Y[trn], pid[trn], P)
    mg = Z[tst] * a + b + np.outer(s[pid[tst]], be)
    return job, float((np.logaddexp(0, mg) - Y[tst] * mg).sum())


def shared_final(lam):
    Z, Y, pid, P, R, S, fold = _shared_data()
    a, b, be, s = fit_shared(lam, Z[R], Y[R], pid[R], P)
    print(f"  shared severity (lam {lam:g}): slopes {np.round(a, 2)}, intercepts {np.round(b, 2)}, loadings "
          f"{np.round(be, 2)} (fixed), sd of severity {s[np.unique(pid[S])].std():.3f}", flush=True)
    return {c: Z[S][:, k] * a[k] + b[k] + s[pid[S]] * be[k] for k, c in enumerate(CODES)}, s


def _init_worker():
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)


def mlp_job(job):
    variant, f_out, seed = job
    X, base, y, comment_fold, is_val = _G["mlp_X"][variant], _G["mlp_base"][variant], _G["mlp_y"], _G["mlp_fold"], _G["mlp_val"]
    tst = comment_fold == f_out
    val = (~tst) & is_val[f_out]
    trn = ~tst & ~val
    mdl = OffsetMLP(**MLP_CFG, seed=seed)
    hist = mdl.fit(X[trn], base[trn], y[trn], val=(X[val], base[val], y[val]))
    return job, mdl.decision(X[tst], base[tst]), len(hist)


def pair_stats(y, p, comment):
    """Per comment: concordant pairs (+ half ties) and pairs of (annotator who said yes, annotator who said no)."""
    out = {}
    for j in np.unique(comment):
        k = comment == j
        pos, neg = p[k][y[k] == 1], p[k][y[k] == 0]
        if len(pos) and len(neg):
            d = pos[:, None] - neg[None, :]
            out[j] = ((d > 0).sum() + .5 * (d == 0).sum(), d.size)
    return out


def evaluate(y, p, comment, picks):
    """Metrics on individual labels: log loss, Brier, pooled AUROC, within-comment AUROC (which of the annotators of
    the same comment say yes; anything constant across annotators scores exactly 0.5)."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    ll = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    br = (p - y) ** 2
    ps = pair_stats(y, p, comment)
    cids = np.unique(comment)
    members = {j: np.flatnonzero(comment == j) for j in cids}

    def one(sel):
        idx = np.concatenate([members[j] for j in sel])
        c = np.array([ps[j] for j in sel if j in ps])
        return [ll[idx].mean(), br[idx].mean(), roc_auc_score(y[idx], p[idx]), c[:, 0].sum() / c[:, 1].sum()]

    full = one(cids)
    boot = np.array([one(cids[pk]) for pk in picks])
    return np.array(full), boot


EVAL_METRICS = ["logloss", "brier", "auroc", "within_comment_auroc"]


def analyze():
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 70)
    t0 = time.time()
    m = pd.read_parquet(MHS)
    n = m.groupby("comment_id").size()
    ref = set(n[n >= REF_MIN].index)
    ev = set(pd.read_parquet(os.path.join(e1.OUT, "sample.parquet")).comment_id) - ref
    for c, (rule, _) in e1.CODES.items():
        m["y_" + c] = rule(m).astype(float)
    D = demographics(m)
    rows = m[m.comment_id.isin(ref | ev)][["comment_id", "annotator_id"] + ["y_" + c for c in CODES]].reset_index(drop=True)
    rows["is_ref"] = rows.comment_id.isin(ref)
    jp = pd.concat([pd.read_parquet(os.path.join(OUT, "jev_ref.parquet")),
                    pd.read_parquet(os.path.join(e1.OUT, "jev.parquet"))]).drop_duplicates("comment_id").set_index("comment_id")
    tf = tfidf_prior(m, sorted(ref | ev), ref | ev).set_index("comment_id")
    persons, pid = np.unique(rows.annotator_id, return_inverse=True)
    rng = np.random.default_rng(0)
    ref_ids = rng.permutation(sorted(ref))
    fold_of = {j: k % NFOLD for k, j in enumerate(ref_ids)}
    ev_ids = rng.permutation(sorted(ev))
    fold_ev = {j: k % NFOLD for k, j in enumerate(ev_ids)}
    _G.update(pid=pid, P=len(persons), isref=rows.is_ref.values,
              fold_r=rows.comment_id.map(fold_of).fillna(-1).astype(int).values,
              Y={c: rows["y_" + c].values for c in CODES},
              Zj={c: logit(rows.comment_id.map(jp["j_" + c]).values) for c in CODES},
              Zt={c: logit(rows.comment_id.map(tf[c]).values) for c in CODES},
              Drow=D.loc[rows.annotator_id].values)
    R, S = np.flatnonzero(rows.is_ref), np.flatnonzero(~rows.is_ref)
    na = rows.iloc[R].groupby("annotator_id").size()
    print(f"anchor rows {len(R)} ({len(ref)} reference comments, {rows.iloc[R].annotator_id.nunique()} annotators); "
          f"evaluation rows {len(S)} ({len(ev)} comments, {rows.iloc[S].annotator_id.nunique()} annotators, whose anchors: "
          f"median {na.loc[rows.iloc[S].annotator_id.unique()].median():.0f}); {D.shape[1]} demographic features")

    # ---- linear models, one process per code
    sjobs = [(lam, f) for lam in LAMS_S for f in range(NFOLD)]
    with ProcessPoolExecutor(len(CODES) + len(sjobs), mp_context=mp.get_context("fork"), initializer=_init_worker) as ex:
        fut_codes = [ex.submit(fit_code, c) for c in CODES]
        fut_shared = [ex.submit(shared_cv_job, j) for j in sjobs]
        sll = {}
        for fu in fut_shared:
            (lam, f), ll = fu.result()
            sll[lam] = sll.get(lam, 0.0) + ll
        lam_s = min(sll, key=sll.get)
        nR = int(_G['isref'].sum())
        print('  shared severity, held-out anchor log loss per label (6 codes summed) by lam: ' +
              str({k: round(v / nR, 4) for k, v in sll.items()}), flush=True)
        fut_final = ex.submit(shared_final, lam_s)
        fits = dict(fu.result() for fu in fut_codes)
        shared, sev = fut_final.result()
    print(f"  linear fits done in {time.time() - t0:.0f}s; chosen lam_u per code (JEV + person offset): "
          f"{ {c: fits[c]['JEV + person offset']['lam_u'] for c in CODES} }; lam_d: "
          f"{ {c: fits[c]['JEV + demographics']['lam_w'] for c in CODES} }; shared-severity lam {lam_s}")
    cvt = pd.DataFrame({k: {c: fits[c][k]["cv_ll"] for c in CODES} for k in fits[CODES[0]]})
    print("  held-out anchor log loss (reference-comment folds):\n" + cvt.round(4).to_string())

    # ---- external network on top: one MLP shared by the six codes (code one-hot as input)
    emb = np.load(os.path.join(OUT, "emb_bge-small.npz"), allow_pickle=False)
    E = emb["emb"].astype(float)
    E = E - E.mean(0)
    U_, s_, Vt = np.linalg.svd(E, full_matrices=False)
    pcs = E @ Vt[:32].T
    pcs = pcs / pcs.std(0)
    pc_of = dict(zip(emb["ids"], pcs))
    Pc = np.array([pc_of[str(j)] for j in rows.comment_id])
    Zall = np.c_[tuple(_G["Zj"][c] for c in CODES)]
    Dz = _G["Drow"]
    fold_r = _G["fold_r"]

    def offsets_feature(name):  # each person's six offsets, from anchors other than the row's comment
        F = np.zeros((len(rows), len(CODES)))
        for k, c in enumerate(CODES):
            fr = fits[c][name]
            F[S, k] = fr["u_final"][pid[S]]
            for f in range(NFOLD):
                rr = R[fold_r[R] == f]
                F[rr, k] = fr["u_fold"][f][pid[rr]]
        return F

    Uf = offsets_feature("JEV + person offset")
    blocks, ys, bases, folds, codes_col, comment_col = [], [], {"item": [], "itemxperson": [], "cold": []}, [], [], []
    feats = {"item": [], "itemxperson": [], "cold": []}
    for k, c in enumerate(CODES):
        oh = np.zeros((len(rows), len(CODES)))
        oh[:, k] = 1
        it = np.c_[oh, Zall, Pc]
        feats["item"].append(it)
        feats["itemxperson"].append(np.c_[it, Dz, Uf])
        feats["cold"].append(np.c_[it, Dz])
        bases["item"].append(fits[c]["JEV + person offset"]["m"])
        bases["itemxperson"].append(fits[c]["JEV + person offset"]["m"])
        bases["cold"].append(fits[c]["JEV + Platt"]["m"])
        ys.append(_G["Y"][c])
        folds.append(np.where(rows.is_ref, -1, rows.comment_id.map(fold_ev).fillna(-1)).astype(int))
        codes_col.append(np.full(len(rows), k))
        comment_col.append(rows.comment_id.values)
    mlp_X = {v: np.vstack(feats[v]) for v in feats}
    mlp_base = {v: np.concatenate(bases[v]) for v in bases}
    mlp_y, mlp_fold = np.concatenate(ys), np.concatenate(folds)
    code_col, cm_col = np.concatenate(codes_col), np.concatenate(comment_col)
    # validation (early stopping) = 20% of the evaluation comments outside the test fold; anchors always train
    vrng = np.random.default_rng(1)
    mlp_val = {}
    for f in range(NFOLD):
        others = [j for j in ev_ids if fold_ev[j] != f]
        vset = set(vrng.choice(others, size=len(others) // 5, replace=False))
        mlp_val[f] = np.isin(cm_col, list(vset))
    _G.update(mlp_X=mlp_X, mlp_base=mlp_base, mlp_y=mlp_y, mlp_fold=mlp_fold, mlp_val=mlp_val)
    jobs = [(v, f, s) for v in mlp_X for f in range(NFOLD) for s in range(MLP_SEEDS)]
    t1 = time.time()
    with ProcessPoolExecutor(min(len(jobs), 48), mp_context=mp.get_context("fork"), initializer=_init_worker) as ex:
        mres = list(ex.map(mlp_job, jobs))
    mlp_out = {v: np.zeros(len(mlp_y)) for v in mlp_X}
    epochs = {v: [] for v in mlp_X}
    for (v, f, s), dec, ne in mres:
        mlp_out[v][mlp_fold == f] += dec / MLP_SEEDS
        epochs[v].append(ne)
    print(f"  {len(jobs)} MLP fits ({mlp_X['itemxperson'].shape[0]} rows x {mlp_X['itemxperson'].shape[1]} inputs "
          f"for the person-aware one) in {time.time() - t1:.0f}s; epochs run (incl. patience) median "
          f"{ {v: float(np.median(e)) for v, e in epochs.items()} }")

    # ---- evaluation on the new comments
    picks = [np.random.default_rng(100 + b).integers(0, len(ev), len(ev)) for b in range(B)]
    names = ["JEV zero-shot", "JEV + Platt", "JEV + demographics", "JEV + person offset",
             "JEV + demographics + person offset", "JEV + shared person severity",
             "JEV + person offset + MLP (item features)", "JEV + person offset + MLP (item x person features)",
             "JEV + Platt + MLP (item x demographics, no anchors)", "TF-IDF + Platt", "TF-IDF + person offset"]
    mlp_name = {"item": names[6], "itemxperson": names[7], "cold": names[8]}
    full, boot = {}, {}
    yS_all = {}
    for k, c in enumerate(CODES):
        yS = _G["Y"][c][S]
        cm = rows.comment_id.values[S]
        sel = (code_col == k) & (mlp_fold >= 0)
        assert (cm_col[sel] == cm).all()
        preds = {"JEV zero-shot": expit(_G["Zj"][c][S]), "JEV + shared person severity": expit(shared[c]),
                 **{nm: expit(fits[c][nm]["m"][S]) for nm in fits[c]},
                 **{mlp_name[v]: expit(mlp_out[v][sel]) for v in mlp_out}}
        for nm in names:
            full[(c, nm)], boot[(c, nm)] = evaluate(yS, preds[nm], cm, picks)
        yS_all[c] = yS.mean()
    rows_out, drows = [], []
    for nm in names:
        for c in CODES + ["mean"]:
            fv = np.mean([full[(cc, nm)] for cc in CODES], 0) if c == "mean" else full[(c, nm)]
            bv = np.mean([boot[(cc, nm)] for cc in CODES], 0) if c == "mean" else boot[(c, nm)]
            for j, k in enumerate(EVAL_METRICS):
                rows_out.append(dict(method=nm, code=c, metric=k, mean=fv[j], lo=np.quantile(bv[:, j], .025),
                                     hi=np.quantile(bv[:, j], .975)))
    pairs = [("JEV + person offset", "JEV + Platt"), ("JEV + demographics", "JEV + Platt"),
             ("JEV + demographics + person offset", "JEV + person offset"),
             ("JEV + shared person severity", "JEV + person offset"),
             ("JEV + person offset + MLP (item x person features)", "JEV + person offset"),
             ("JEV + person offset + MLP (item x person features)", "JEV + person offset + MLP (item features)"),
             ("JEV + person offset + MLP (item features)", "JEV + person offset"),
             ("JEV + Platt + MLP (item x demographics, no anchors)", "JEV + Platt"),
             ("TF-IDF + person offset", "TF-IDF + Platt"), ("JEV + person offset", "TF-IDF + person offset"),
             ("JEV + Platt", "TF-IDF + Platt")]
    for a_, b_ in pairs:
        for c in CODES + ["mean"]:
            cc = CODES if c == "mean" else [c]
            fv = np.mean([full[(x, a_)] - full[(x, b_)] for x in cc], 0)
            bv = np.mean([boot[(x, a_)] - boot[(x, b_)] for x in cc], 0)
            for j, k in enumerate(EVAL_METRICS):
                drows.append(dict(pair=f"{a_} - {b_}", code=c, metric=k, mean=fv[j], lo=np.quantile(bv[:, j], .025),
                                  hi=np.quantile(bv[:, j], .975)))
    tab, dif = pd.DataFrame(rows_out), pd.DataFrame(drows)
    tab.to_csv(os.path.join(OUT, "summary.csv"), index=False)
    dif.to_csv(os.path.join(OUT, "diffs.csv"), index=False)
    cell = lambda d: d.assign(cell=[f"{a:.3f} [{lo:.3f}, {hi:.3f}]" for a, lo, hi in zip(d["mean"], d.lo, d.hi)])
    print(f"\n== individual labels on {len(ev)} new comments ({len(S)} labels x 6 codes); mean over codes, comment "
          f"bootstrap 95% CI.  Share of 'yes' labels per code: { {c: round(v, 3) for c, v in yS_all.items()} }")
    print(cell(tab[tab.code == "mean"]).pivot(index="method", columns="metric", values="cell")
          .reindex(index=names, columns=EVAL_METRICS).to_string())
    print("\n== paired differences (mean over codes)")
    print(cell(dif[dif.code == "mean"]).pivot(index="pair", columns="metric", values="cell")
          .reindex(index=[f"{a} - {b}" for a, b in pairs], columns=EVAL_METRICS).to_string())
    print("\n== within-comment AUROC per code")
    print(tab[tab.metric == "within_comment_auroc"].pivot(index="method", columns="code", values="mean")
          .reindex(index=names, columns=CODES + ["mean"]).round(3).to_string())
    print("\n== log loss per code")
    print(tab[tab.metric == "logloss"].pivot(index="method", columns="code", values="mean")
          .reindex(index=names, columns=CODES + ["mean"]).round(4).to_string())
    # how big are the offsets, and are they stable?  correlation of the anchor-based offset with the person's mean
    # residual on the new comments (under JEV + Platt)
    st = []
    for c in CODES:
        u = fits[c]["JEV + person offset"]["u_final"]
        res = _G["Y"][c][S] - expit(fits[c]["JEV + Platt"]["m"][S])
        r_by = pd.Series(res).groupby(pid[S]).mean()
        st.append(dict(code=c, sd_offset=u[np.unique(pid[S])].std(), lam_u=fits[c]["JEV + person offset"]["lam_u"],
                       corr_offset_vs_new_residual=np.corrcoef(u[r_by.index], r_by.values)[0, 1]))
    st = pd.DataFrame(st)
    st.to_csv(os.path.join(OUT, "offset_stability.csv"), index=False)
    print("\n== person offsets: spread and whether they carry over to new comments\n" + st.round(3).to_string(index=False))
    print(f"\nshared severity vs MHS's own IRT severity (computed from ALL labels, so only a sanity check): r = "
          f"{np.corrcoef(sev, m.drop_duplicates('annotator_id').set_index('annotator_id').annotator_severity.loc[persons])[0, 1]:.3f}")
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    {"prep": prep, "analyze": analyze}[sys.argv[1]]()
