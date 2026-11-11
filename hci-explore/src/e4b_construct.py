"""E4b: is the per-person intercept a real, stable trait, and whom does the fixed judge fail?  (Koustuv-lens checks,
pre-registered in PLAN.md as D3 and D4.)  No new JEV calls: everything reuses E1i's DICES-990 out-of-fold predictions
(141 quality-filtered raters with >= 100 conversations) and E2f's RealHumanEval out-of-fold gate margins.

D3  reliability and generality of the intercept
    (1) split-half: each rater's conversations split at random in two; an intercept is estimated on each half with the
        judge's out-of-fold prediction as offset; correlation across raters, Spearman-Brown corrected; 100 splits.
    (2) across harm categories: intercepts estimated separately from the harmful-content, unfair-bias and
        misinformation labels (each with JEV's own category answer as offset); correlations, disattenuated by the
        categories' split-half reliabilities.
    (3) over time (RealHumanEval): each programmer's offset from the first vs second half of their suggestions.
D4  whom the judge fails: per-rater log loss with the judge alone (k = 0) vs judge + own intercept (k = 20 random
    anchors), by demographic group and by how extreme the person's standard is (|intercept| from all their labels).

  python3 -I src/e4b_construct.py
"""
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import brentq, minimize  # noqa: E402
from scipy.special import expit, logit  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
import jev  # noqa: E402
import e1i_dices990 as e1i  # noqa: E402

OUT = os.path.join(jev.ROOT, "results", "e4b")
os.makedirs(OUT, exist_ok=True)
SPLITS, B = 100, 2000
T2 = 4.0  # weak prior for per-half estimates (the reliability question is about the data, not the prior)


def u_hat(m, y, t2=T2):
    if len(m) == 0:
        return np.nan
    g = lambda u: (expit(m + u) - y).sum() + u / t2
    w = t2 * len(m) + 10
    return brentq(g, -w, w, xtol=1e-9)


def sb(r):
    return 2 * r / (1 + r)


def split_half(M, Y, rng):
    """M, Y: (items x raters) offsets and labels (nan = missing).  Returns (r, Spearman-Brown) for one random split."""
    a, b = [], []
    for j in range(Y.shape[1]):
        lab = np.flatnonzero(~np.isnan(Y[:, j]))
        perm = rng.permutation(lab)
        h = len(perm) // 2
        a.append(u_hat(M[perm[:h], j], Y[perm[:h], j]))
        b.append(u_hat(M[perm[h:], j], Y[perm[h:], j]))
    a, b = np.array(a), np.array(b)
    return a, b


def dices():
    G = e1i.build("filtered")
    z = np.load(os.path.join(jev.ROOT, "results", "e1i", "oof_filtered.npz"))
    Y = z["Y"]
    lab = ~np.isnan(Y)
    # the saved out-of-fold predictions are margins (logits), not probabilities (v1 of this script treated them as
    # probabilities; its log is kept as results/e4b/log_v1_margin_bug.txt and is void)
    M0 = np.where(lab, z["0|0|0"], np.nan)
    P0, P20 = expit(z["0|0|0"]), expit(z["0|20|0"])
    I, P = Y.shape
    print(f"DICES-990: {P} raters x {I} conversations, {int(lab.sum())} labels")

    # ---------------- D3 (1) split-half reliability of the overall intercept
    rng = np.random.default_rng(0)
    rs, halves = [], []
    for s in range(SPLITS):
        a, b = split_half(M0, Y, rng)
        halves.append((a, b))
        rs.append(np.corrcoef(a, b)[0, 1])
    r = float(np.mean(rs))
    brng = np.random.default_rng(1)
    boot = []
    for _ in range(B):
        idx = brng.integers(0, P, P)
        a, b = halves[brng.integers(SPLITS)]
        boot.append(sb(np.corrcoef(a[idx], b[idx])[0, 1]))
    lo, hi = np.quantile(boot, [.025, .975])
    print(f"\n== D3(1) split-half reliability of the per-person intercept (overall verdict)\n"
          f"mean split-half r = {r:.3f}; Spearman-Brown = {sb(r):.3f} [{lo:.3f}, {hi:.3f}] (rater bootstrap); "
          f"pre-registered criterion: >= 0.7 with lower bound >= 0.6 -> "
          f"{'MET' if sb(r) >= .7 and lo >= .6 else 'NOT met'}")
    n_lab = lab.sum(0)
    print(f"(labels per rater: median {int(np.median(n_lab))}, range {n_lab.min()}-{n_lab.max()})")

    # ---------------- D3 (2) across harm categories
    df, items = e1i.load()
    df = df[df.rater_id.isin(set(df.rater_id) - e1i.REMOVED)]
    n = df.groupby("rater_id").size()
    df = df[df.rater_id.isin(n.index[n >= e1i.MIN_RATED])]
    ipos = {i: k for k, i in enumerate(items.index)}
    rids = sorted(df.rater_id.unique())
    assert len(rids) == P
    rpos = {q: k for k, q in enumerate(rids)}
    ii, rr = df.item_id.map(ipos).values, df.rater_id.map(rpos).values
    j = pd.read_parquet(os.path.join(jev.ROOT, "results", "e1i", "jev.parquet")).set_index("item_id").loc[items.index]
    cats = {"harmful": "Q2_harmful_content_overall", "bias": "Q3_unfair_bias_overall",
            "misinfo": "Q4_misinformation_overall", "overall": "Q_overall"}
    U, REL = {}, {}
    for c, col in cats.items():
        Yc = np.full((I, P), np.nan)
        Yc[ii, rr] = df[col].map({"Yes": 1., "No": 0.}).values
        zc = logit(np.clip(j["j_" + c].values, 1e-4, 1 - 1e-4))
        # global two-parameter calibration of JEV's category answer (in-sample; two parameters)
        mk = ~np.isnan(Yc)
        X = np.repeat(zc[:, None], P, 1)[mk]
        yv = Yc[mk]
        def f(w):
            zz = w[0] + w[1] * X
            return np.logaddexp(0, zz).sum() - yv @ zz, np.array([(expit(zz) - yv).sum(), ((expit(zz) - yv) * X).sum()])
        w = minimize(f, np.zeros(2), jac=True, method="L-BFGS-B").x
        Mc = np.where(mk, (w[0] + w[1] * zc)[:, None] + 0 * Yc, np.nan)
        U[c] = np.array([u_hat(Mc[~np.isnan(Yc[:, q]), q], Yc[~np.isnan(Yc[:, q]), q]) for q in range(P)])
        rr_ = [np.corrcoef(*split_half(Mc, Yc, np.random.default_rng([2, s])))[0, 1] for s in range(30)]
        REL[c] = sb(float(np.mean(rr_)))
    print("\n== D3(2) intercepts estimated from each harm category separately")
    print("split-half reliability (Spearman-Brown): " + ", ".join(f"{c} {REL[c]:.3f}" for c in cats))
    rows = []
    names = list(cats)
    for a_ in range(len(names)):
        for b_ in range(a_ + 1, len(names)):
            x, y = U[names[a_]], U[names[b_]]
            ok = np.isfinite(x) & np.isfinite(y)
            rr2 = np.corrcoef(x[ok], y[ok])[0, 1]
            rows.append(dict(pair=f"{names[a_]} vs {names[b_]}", r=round(rr2, 3),
                             disattenuated=round(rr2 / np.sqrt(REL[names[a_]] * REL[names[b_]]), 3)))
    print(pd.DataFrame(rows).to_string(index=False))

    # ---------------- D4 whom does the judge fail?
    def ll(Pm):
        pc = np.clip(Pm, 1e-6, 1 - 1e-6)
        L = -(np.nan_to_num(Y) * np.log(pc) + (1 - np.nan_to_num(Y)) * np.log(1 - pc))
        return np.where(lab, L, np.nan)
    L0, L20 = np.nanmean(ll(P0), 0), np.nanmean(ll(P20), 0)
    u_all = np.array([u_hat(M0[lab[:, q], q], Y[lab[:, q], q]) for q in range(P)])
    ext = np.abs(u_all)
    demo = df.drop_duplicates("rater_id").set_index("rater_id").loc[rids]
    per = pd.DataFrame(dict(rater=rids, logloss_k0=L0, logloss_k20=L20, intercept=u_all, extremity=ext,
                            yes_rate=np.nanmean(Y, 0), n=n_lab, gender=demo.rater_gender.values,
                            locale=demo.rater_locale.values, race=demo.rater_race.values, age=demo.rater_age.values,
                            education=demo.rater_education.values))
    per.to_csv(os.path.join(OUT, "per_rater_dices990.csv"), index=False)
    rho0 = spearmanr(L0, ext).correlation
    rho20 = spearmanr(L20, ext).correlation
    brng = np.random.default_rng(3)
    bo = []
    for _ in range(B):
        idx = brng.integers(0, P, P)
        bo.append((spearmanr(L0[idx], ext[idx]).correlation, spearmanr(L20[idx], ext[idx]).correlation,
                   L0[idx].std() - L20[idx].std(), L0[idx].mean() - L20[idx].mean()))
    bo = np.array(bo)
    q = lambda col: np.quantile(bo[:, col], [.025, .975])
    print(f"\n== D4 whom does the fixed judge fail? (per-rater log loss, DICES-990)")
    print(f"mean per-rater log loss: judge alone {L0.mean():.4f}, + own intercept (k = 20) {L20.mean():.4f}; "
          f"difference {L0.mean() - L20.mean():+.4f} [{q(3)[0]:+.4f}, {q(3)[1]:+.4f}]")
    print(f"(1) Spearman(log loss, |intercept|): judge alone {rho0:+.3f} [{q(0)[0]:+.3f}, {q(0)[1]:+.3f}]; "
          f"with intercept {rho20:+.3f} [{q(1)[0]:+.3f}, {q(1)[1]:+.3f}]")
    print(f"(2) SD across raters of log loss: judge alone {L0.std():.4f}, with intercept {L20.std():.4f}; "
          f"reduction {L0.std() - L20.std():+.4f} [{q(2)[0]:+.4f}, {q(2)[1]:+.4f}]")
    top = ext >= np.quantile(ext, .75)
    rows = [dict(attribute="extremity", group="most extreme quarter", n=int(top.sum()), logloss_k0=L0[top].mean(),
                 logloss_k20=L20[top].mean()),
            dict(attribute="extremity", group="other three quarters", n=int((~top).sum()), logloss_k0=L0[~top].mean(),
                 logloss_k20=L20[~top].mean())]
    gaps = [dict(attribute="extremity", gap_k0=L0[top].mean() - L0[~top].mean(),
                 gap_k20=L20[top].mean() - L20[~top].mean())]
    for att in ("gender", "locale", "race", "age", "education"):
        g = per.groupby(att).agg(n=("rater", "size"), logloss_k0=("logloss_k0", "mean"),
                                 logloss_k20=("logloss_k20", "mean"))
        g = g[g.n >= 10]
        for k_, row in g.iterrows():
            rows.append(dict(attribute=att, group=k_, n=int(row.n), logloss_k0=row.logloss_k0,
                             logloss_k20=row.logloss_k20))
        gaps.append(dict(attribute=att, gap_k0=g.logloss_k0.max() - g.logloss_k0.min(),
                         gap_k20=g.logloss_k20.max() - g.logloss_k20.min()))
    tab = pd.DataFrame(rows)
    print("\nper-group mean of per-rater log loss (groups with >= 10 raters)\n" + tab.round(4).to_string(index=False))
    gp = pd.DataFrame(gaps)
    print("\nlargest gap between groups (max - min of group means)\n" + gp.round(4).to_string(index=False))
    tab.to_csv(os.path.join(OUT, "groups_dices990.csv"), index=False)
    gp.to_csv(os.path.join(OUT, "gaps_dices990.csv"), index=False)
    # who are the extreme raters?
    ex = per[top]
    print("\nmost extreme quarter: share saying 'unsafe' to most conversations "
          f"{np.mean(ex.intercept > 0):.2f}; their yes-rate median {ex.yes_rate.median():.2f} vs others "
          f"{per[~top].yes_rate.median():.2f}")
    for att in ("gender", "locale", "race", "age"):
        print(f"  {att}: extreme quarter " + ", ".join(f"{k}={v:.2f}" for k, v in
                                                       ex[att].value_counts(normalize=True).items())
              + " | all " + ", ".join(f"{k}={v:.2f}" for k, v in per[att].value_counts(normalize=True).items()))


def rhe():
    o = pd.read_parquet(os.path.join(jev.ROOT, "results", "e2f", "oof.parquet"))
    col = "CDHF features + JEV, LR"
    o = o.sort_values(["programmer_id", "idx"])
    n = o.groupby("programmer_id").size()
    keep = n.index[n >= 40]
    a, b, ra, rb = [], [], [], []
    rng = np.random.default_rng(4)
    for pid in keep:
        g = o[o.programmer_id == pid]
        m, y = g[col].values, g.accepted.values.astype(float)
        h = len(g) // 2
        a.append(u_hat(m[:h], y[:h])); b.append(u_hat(m[h:], y[h:]))
        perm = rng.permutation(len(g))
        ra.append(u_hat(m[perm[:h]], y[perm[:h]])); rb.append(u_hat(m[perm[h:]], y[perm[h:]]))
    a, b, ra, rb = map(np.array, (a, b, ra, rb))
    brng = np.random.default_rng(5)
    bt = []
    for _ in range(B):
        i = brng.integers(0, len(a), len(a))
        bt.append((np.corrcoef(a[i], b[i])[0, 1], np.corrcoef(ra[i], rb[i])[0, 1]))
    bt = np.array(bt)
    print(f"\n== D3(3) RealHumanEval: {len(keep)} programmers with >= 40 suggestions; offset on the out-of-fold "
          f"gate '{col}'")
    print(f"first half vs second half (time): r = {np.corrcoef(a, b)[0, 1]:.3f} "
          f"[{np.quantile(bt[:, 0], .025):.3f}, {np.quantile(bt[:, 0], .975):.3f}]")
    print(f"random halves (same people):       r = {np.corrcoef(ra, rb)[0, 1]:.3f} "
          f"[{np.quantile(bt[:, 1], .025):.3f}, {np.quantile(bt[:, 1], .975):.3f}]")
    print(f"mean offset first half {a.mean():+.3f}, second half {b.mean():+.3f}")


def _main():
    pd.set_option("display.width", 250)
    if "--posthoc" in sys.argv:
        posthoc()
    elif "--replicate350" in sys.argv:
        replicate350()
    else:
        dices()
        rhe()


def posthoc():
    """POST HOC (written after seeing D4's pre-registered result, which mixed two things: per-rater log loss is high
    both when the judge is miscalibrated for a person and when the person's labels are simply hard to predict).
    Excess log loss = log loss of a prediction minus log loss of the same judge with this person's own best intercept
    (fitted on all their labels, in-sample; a reference point, not a deployable model).  It isolates the cost of the
    judge being off for this person; groups are compared on it."""
    z = np.load(os.path.join(jev.ROOT, "results", "e1i", "oof_filtered.npz"))
    Y = z["Y"]
    lab = ~np.isnan(Y)
    M0, M20 = np.where(lab, z["0|0|0"], np.nan), np.where(lab, z["0|20|0"], np.nan)
    P = Y.shape[1]

    def ll_cols(M):
        y = np.nan_to_num(Y)
        L = np.where(lab, np.logaddexp(0, np.nan_to_num(M)) - y * np.nan_to_num(M), np.nan)
        return np.nanmean(L, 0)
    u_best = np.array([u_hat(M0[lab[:, q], q], Y[lab[:, q], q], t2=100.) for q in range(P)])
    Lbest = ll_cols(M0 + u_best[None, :])
    ex0, ex20 = ll_cols(M0) - Lbest, ll_cols(M20) - Lbest
    per = pd.read_csv(os.path.join(OUT, "per_rater_dices990.csv"))
    per["excess_k0"], per["excess_k20"] = ex0, ex20
    ext = np.abs(u_best)
    top = ext >= np.quantile(ext, .75)
    print("\n== POST HOC: excess log loss over the person's own best intercept (miscalibration cost)")
    print(f"all raters: judge alone {ex0.mean():.4f}, with k = 20 anchors {ex20.mean():.4f}")
    print(f"Spearman(excess, |best intercept|): judge alone {spearmanr(ex0, ext).correlation:+.3f}, "
          f"with anchors {spearmanr(ex20, ext).correlation:+.3f}")
    print(f"most extreme quarter: {ex0[top].mean():.4f} -> {ex20[top].mean():.4f}; others {ex0[~top].mean():.4f} -> "
          f"{ex20[~top].mean():.4f}")
    rows = []
    brng = np.random.default_rng(7)
    for att in ("gender", "locale", "race", "age", "education"):
        g = per.groupby(att).agg(n=("rater", "size"), excess_k0=("excess_k0", "mean"), excess_k20=("excess_k20", "mean"))
        g = g[g.n >= 10]
        gap0, gap20 = g.excess_k0.max() - g.excess_k0.min(), g.excess_k20.max() - g.excess_k20.min()
        bs = []
        groups = list(g.index)
        for _ in range(B):
            s = per[per[att].isin(groups)].sample(frac=1, replace=True, random_state=int(brng.integers(1e9)))
            gg = s.groupby(att)[["excess_k0", "excess_k20"]].mean()
            bs.append((gg.excess_k0.max() - gg.excess_k0.min()) - (gg.excess_k20.max() - gg.excess_k20.min()))
        lo, hi = np.quantile(bs, [.025, .975])
        rows.append(dict(attribute=att, groups=", ".join(f"{k} {v:.3f}" for k, v in g.excess_k0.items()),
                         gap_k0=round(gap0, 4), gap_k20=round(gap20, 4), gap_reduction=f"{gap0 - gap20:+.4f} "
                         f"[{lo:+.4f}, {hi:+.4f}]"))
    rows.append(dict(attribute="extremity (top quarter vs rest)", groups="", gap_k0=round(ex0[top].mean() - ex0[~top].mean(), 4),
                     gap_k20=round(ex20[top].mean() - ex20[~top].mean(), 4), gap_reduction=""))
    print(pd.DataFrame(rows).to_string(index=False))
    per.to_csv(os.path.join(OUT, "per_rater_dices990.csv"), index=False)



def replicate350():
    """Replication on DICES-350 (104 raters, E1f's out-of-fold margins, variant 0 = random anchors, JEV difficulties):
    split-half reliability (D3-1) and the post hoc excess-log-loss gaps (D4 post hoc)."""
    import e1e_dices as e1e
    G = e1e.build("104")
    z = np.load(os.path.join(jev.ROOT, "results", "e1f", "oof_104.npz"))
    Y = z["Y"]
    lab = ~np.isnan(Y)
    M0, M20 = np.where(lab, z["0|0|0"], np.nan), np.where(lab, z["0|20|0"], np.nan)
    P = Y.shape[1]
    rs, halves = [], []
    rng = np.random.default_rng(0)
    for s in range(SPLITS):
        a, b = split_half(M0, Y, rng)
        halves.append((a, b))
        rs.append(np.corrcoef(a, b)[0, 1])
    brng = np.random.default_rng(1)
    boot = []
    for _ in range(B):
        idx = brng.integers(0, P, P)
        a, b = halves[brng.integers(SPLITS)]
        boot.append(sb(np.corrcoef(a[idx], b[idx])[0, 1]))
    r = float(np.mean(rs))
    print(f"== DICES-350 replication ({P} raters x {Y.shape[0]} conversations)\nsplit-half reliability: r = {r:.3f}, "
          f"Spearman-Brown {sb(r):.3f} [{np.quantile(boot, .025):.3f}, {np.quantile(boot, .975):.3f}]")

    def ll_cols(M):
        y = np.nan_to_num(Y)
        L = np.where(lab, np.logaddexp(0, np.nan_to_num(M)) - y * np.nan_to_num(M), np.nan)
        return np.nanmean(L, 0)
    u_best = np.array([u_hat(M0[lab[:, q], q], Y[lab[:, q], q], t2=100.) for q in range(P)])
    Lb = ll_cols(M0 + u_best[None, :])
    ex0, ex20 = ll_cols(M0) - Lb, ll_cols(M20) - Lb
    df, _ = e1e.load()
    demo = df.drop_duplicates("rater_id").set_index("rater_id").loc[G["rids"]]
    per = pd.DataFrame(dict(rater=G["rids"], excess_k0=ex0, excess_k20=ex20, gender=demo.rater_gender.values,
                            race=demo.rater_race.values, age=demo.rater_age.values,
                            education=demo.rater_education.values))
    print(f"excess log loss: judge alone {ex0.mean():.4f}, with k = 20 anchors {ex20.mean():.4f}")
    rows = []
    brng = np.random.default_rng(7)
    for att in ("gender", "race", "age", "education"):
        g = per.groupby(att).agg(n=("rater", "size"), excess_k0=("excess_k0", "mean"), excess_k20=("excess_k20", "mean"))
        g = g[g.n >= 10]
        if len(g) < 2:
            continue
        groups = list(g.index)
        bs = []
        for _ in range(B):
            s = per[per[att].isin(groups)].sample(frac=1, replace=True, random_state=int(brng.integers(1e9)))
            gg = s.groupby(att)[["excess_k0", "excess_k20"]].mean()
            bs.append((gg.excess_k0.max() - gg.excess_k0.min()) - (gg.excess_k20.max() - gg.excess_k20.min()))
        gap0, gap20 = g.excess_k0.max() - g.excess_k0.min(), g.excess_k20.max() - g.excess_k20.min()
        rows.append(dict(attribute=att, groups=", ".join(f"{k} {v:.3f}" for k, v in g.excess_k0.items()),
                         gap_k0=round(gap0, 4), gap_k20=round(gap20, 4),
                         gap_reduction=f"{gap0 - gap20:+.4f} [{np.quantile(bs, .025):+.4f}, {np.quantile(bs, .975):+.4f}]"))
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    _main()
