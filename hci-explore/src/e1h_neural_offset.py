"""E1h: a meta-learned, amortised offset network -- the strongest form of the user's idea "an external deep network
that learns a person's offset on top of JEV's probability".  Earlier deep offsets (E1e, E2d, E2e) were fitted per
person from that person's own labels and needed 100+ labels; this one is trained across many raters to map "a few
anchors of a new person" -> "that person's predictions" (a neural process), so it can learn how people differ.

Same design and anchors as E1f (DICES-350, 5 x 5 two-way CV, new raters, new conversations, JEV-predicted
difficulties only).  Per cell the network is trained episodically on the training raters' pool labels (k anchors,
k log-uniform in 0..50, the rest as targets; 20% of training raters held out for early stopping).
  NP alone             logit = JEV content margin + net(item, set-encoding of the anchors)
  NP on the intercept  logit = JEV content margin + Bayesian intercept + net(item, set-encoding, intercept, its variance)
                       (last layer starts at zero, so it starts as the E1f baseline and can only learn a correction)
Baseline: E1f's "random | jev difficulties" (Bayesian intercept), recomputed here on the same anchors.
Pre-registered in PLAN.md (A2): within-conversation AUROC at k = 10 or 20 above the baseline, 95% CI excluding 0.

  python3 -I src/e1h_neural_offset.py [--all-raters]
"""
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.special import expit  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
import e1e_dices as e1e  # noqa: E402
import e1f_onboarding as e1f  # noqa: E402

OUT = os.path.join(e1e.jev.ROOT, "results", "e1h")
os.makedirs(OUT, exist_ok=True)
NF, NIN, B, R, KS = e1f.NF, e1f.NIN, e1f.B, e1f.R, e1f.KS
NPCA, STEPS, BATCH, NTGT, EVAL_EVERY = 16, 3000, 32, 64, 100
NAMES = ["Bayesian intercept (E1f baseline)", "NP alone", "NP on the intercept"]
_G = {}


def batch_post_u(b, y, m, t2, iters=40):
    """Vectorised posterior mode / variance of the intercept for many anchor sets (rows), mask m.  Newton steps are
    clipped to +-1, which cannot diverge on this convex 1-D problem."""
    u = np.zeros(b.shape[0])
    for _ in range(iters):
        p = expit(b + u[:, None])
        g = ((p - y) * m).sum(1) + u / t2
        h = (p * (1 - p) * m).sum(1) + 1 / t2
        u -= np.clip(g / h, -1, 1)
    p = expit(b + u[:, None])
    return u, 1 / ((p * (1 - p) * m).sum(1) + 1 / t2)


def build_net(dx, residual, torch, nn):
    class NP(nn.Module):
        def __init__(self):
            super().__init__()
            h, dz = 64, 16
            self.enc = nn.Sequential(nn.Linear(dx + 2, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
            self.rho = nn.Sequential(nn.Linear(h + 1, h), nn.ReLU(), nn.Linear(h, dz))
            self.dec = nn.Sequential(nn.Linear(dx + dz + (2 if residual else 0), h), nn.ReLU(), nn.Dropout(.1),
                                     nn.Linear(h, 1))
            nn.init.zeros_(self.dec[-1].weight)
            nn.init.zeros_(self.dec[-1].bias)

        def forward(self, xc, yc, rc, mc, xt, bt, uh, vh):
            e = self.enc(torch.cat([xc, yc[..., None], rc[..., None]], -1)) * mc[..., None]
            n = mc.sum(1, keepdim=True)
            z = self.rho(torch.cat([e.sum(1) / n.clamp(min=1), torch.log1p(n)], -1))
            parts = [xt, z[:, None, :].expand(-1, xt.shape[1], -1)]
            if residual:
                parts += [uh[:, None, None].expand(-1, xt.shape[1], 1), vh[:, None, None].expand(-1, xt.shape[1], 1)]
                bt = bt + uh[:, None]
            return bt + self.dec(torch.cat(parts, -1)).squeeze(-1)
    return NP()


def episodes(rng, raters, Ylab, X, bpool, t2, n):
    """n training episodes: (context tensors, target tensors).  Ylab: pool x raters labels (NaN = none)."""
    K = max(KS)
    xc, yc, mc, xt, yt, bt, ic = (np.zeros((n, K, X.shape[1])), np.zeros((n, K)), np.zeros((n, K)),
                                  np.zeros((n, NTGT, X.shape[1])), np.zeros((n, NTGT)), np.zeros((n, NTGT)),
                                  np.zeros((n, K), int))
    for e in range(n):
        r = raters[rng.integers(len(raters))]
        lab = np.flatnonzero(~np.isnan(Ylab[:, r]))
        perm = rng.permutation(lab)
        k = min(int(np.floor(np.exp(rng.uniform(0, np.log(K + 2))))) - 1, K)  # log-uniform over 0..K
        k = max(0, min(k, len(perm) - NTGT))
        c, t = perm[:k], perm[k:k + NTGT]
        xc[e, :k], yc[e, :k], mc[e, :k], ic[e, :k] = X[c], Ylab[c, r], 1., c
        xt[e], yt[e], bt[e] = X[t], Ylab[t, r], bpool[t]
    bc = bpool[ic]
    uh, vh = batch_post_u(bc, yc, mc, t2)
    rc = (yc - expit(bc)) * mc
    return xc, yc, rc, mc, xt, bt, uh, vh, yt


def cell_job(args):
    import torch
    import torch.nn as nn
    torch.set_num_threads(1)
    f, g, residual = args
    Y, Z, E, cfold, rfold = (_G[x] for x in ("Y", "Z", "E", "cfold", "rfold"))
    pool, test = np.flatnonzero(cfold != f), np.flatnonzero(cfold == f)
    trR, teR = np.flatnonzero(rfold != g), np.flatnonzero(rfold == g)
    Yp = Y[np.ix_(pool, trR)]
    b_crowd, u_tr = e1f.fit_rasch(Yp)
    pc = expit(b_crowd[:, None] + u_tr[None, :])
    tau2 = max(u_tr.var() - np.mean(1 / (np.where(~np.isnan(Yp), pc * (1 - pc), 0.).sum(0) + e1f.LAM_U)), .05)
    mz, sz = Z[pool].mean(0), Z[pool].std(0) + 1e-8
    Zs = (Z - mz) / sz
    w, c = e1f.fit_content(Zs[pool], Yp, u_tr)
    b_test = Zs[test] @ w + c
    inner = np.random.default_rng([f, g, 3]).permutation(len(pool)) % NIN  # E1f's cross-fitting
    b_jev = np.empty(len(pool))
    for h in range(NIN):
        wh, ch = e1f.fit_content(Zs[pool][inner != h], Yp[inner != h], u_tr)
        b_jev[inner == h] = Zs[pool][inner == h] @ wh + ch
    s2 = float(np.var(b_crowd - b_jev))
    Ec = E - E[pool].mean(0)
    _, _, Vt = np.linalg.svd(Ec[pool], full_matrices=False)
    Ep = Ec @ Vt[:NPCA].T
    Ep /= Ep[pool].std(0) + 1e-8
    bfull = np.zeros(len(Z))
    bfull[pool], bfull[test] = b_jev, b_test
    X = np.c_[Zs, Ep, bfull].astype(np.float32)  # item features: JEV six answers, embedding PCA, JEV margin
    Xp, Xt = X[pool], X[test]
    rng = np.random.default_rng([f, g, 21])
    tr_r = rng.permutation(len(trR))
    va_r, fit_r = tr_r[:max(4, len(trR) // 5)], tr_r[max(4, len(trR) // 5):]
    T = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)
    val = [T(a) for a in episodes(np.random.default_rng([f, g, 22]), va_r, Yp, Xp, b_jev, tau2, 400)]
    torch.manual_seed(f * 10 + g)
    net = build_net(X.shape[1], residual, torch, nn)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()

    def val_loss():
        net.eval()
        with torch.no_grad():
            out = net(*val[:8])
            l = bce(out, val[8]).item()
        net.train()
        return l

    best, best_state, hist = val_loss(), {k: v.clone() for k, v in net.state_dict().items()}, []
    for step in range(1, STEPS + 1):
        ep = [T(a) for a in episodes(rng, fit_r, Yp, Xp, b_jev, tau2, BATCH)]
        loss = bce(net(*ep[:8]), ep[8])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % EVAL_EVERY == 0:
            l = val_loss()
            hist.append(l)
            if l < best:
                best, best_state = l, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    net.eval()
    res = {(m, rep): np.full((len(KS), len(test), len(teR)), np.nan) for m in range(2) for rep in range(R)}
    with torch.no_grad():
        for ri, r in enumerate(teR):
            yr = Y[pool, r]
            for rep in range(R):
                order = np.random.default_rng([f, g, int(r), rep, 11]).permutation(len(pool))  # E1f's anchors
                for ki, k in enumerate(KS):
                    a = order[:k]
                    lab = a[~np.isnan(yr[a])]
                    u, v = e1f.post_u(b_jev[lab], yr[lab], 0., tau2) if len(lab) else (0., tau2)
                    res[0, rep][ki, :, ri] = (b_test + u) / np.sqrt(1 + np.pi * (s2 + v) / 8)
                    K = max(KS)
                    xc, yc, mc = np.zeros((1, K, X.shape[1])), np.zeros((1, K)), np.zeros((1, K))
                    xc[0, :len(lab)], yc[0, :len(lab)], mc[0, :len(lab)] = Xp[lab], yr[lab], 1.
                    rc = (yc - expit(np.r_[b_jev[lab], np.zeros(K - len(lab))][None, :])) * mc
                    out = net(T(xc), T(yc), T(rc), T(mc), T(Xt[None]), T(b_test[None]), T([u]), T([v]))
                    res[1, rep][ki, :, ri] = out.numpy()[0]
    return f, g, residual, test, teR, res, dict(best_val=best, first_val=hist[0] if hist else np.nan,
                                                 best_step=EVAL_EVERY * (int(np.argmin(hist)) + 1) if hist else 0)


def main(raters):
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 80)
    G = e1e.build(raters)
    I, P = G["I"], G["P"]
    cfold = np.empty(I, int)
    cfold[np.random.default_rng(0).permutation(I)] = np.arange(I) % NF
    rfold = np.empty(P, int)
    rfold[np.random.default_rng(2).permutation(P)] = np.arange(P) % NF
    _G.update(Y=G["Y"], Z=G["Z"], E=G["E"], cfold=cfold, rfold=rfold)
    jobs = [(f, g, res) for res in (False, True) for f in range(NF) for g in range(NF)]
    t0 = time.time()
    with ProcessPoolExecutor(len(jobs), mp_context=mp.get_context("fork")) as ex:
        out = list(ex.map(cell_job, jobs))
    print(f"DICES-350, raters = {raters}: {len(jobs)} (cell, variant) trainings in {time.time() - t0:.0f}s")
    M = {}
    infos = []
    for f, g, residual, test, teR, res, info in out:
        name = NAMES[2] if residual else NAMES[1]
        for (m, rep), arr in res.items():
            nm = NAMES[0] if m == 0 else name
            for ki, k in enumerate(KS):
                M.setdefault((nm, k, rep), np.full((I, P), np.nan))[np.ix_(test, teR)] = arr[ki]
        infos.append(dict(variant=name, **info))
    print(pd.DataFrame(infos).groupby("variant").agg(["mean", "min", "max"]).round(4).to_string())
    brng = np.random.default_rng(1)
    C = np.stack([np.bincount(brng.integers(0, I, I), minlength=I) for _ in range(B)]).astype(float)
    W = np.stack([np.bincount(brng.integers(0, P, P), minlength=P) for _ in range(B)]).astype(float)
    st = {}
    for nm in NAMES:
        for k in KS:
            reps = [e1f.boot_stats(M[nm, k, rep], G["Y"], C, W, rfold) for rep in range(R)]
            st[nm, k] = tuple(np.mean([x[j] for x in reps], 0) for j in range(3))
    tab = pd.DataFrame([dict(model=nm, k=k, within_auroc=pt[3], logloss=pt[0], auroc=pt[2])
                        for (nm, k), (pt, lb, wb) in st.items()])
    tab.to_csv(os.path.join(OUT, f"curve_{raters}.csv"), index=False)
    piv = lambda col: tab.pivot(index="model", columns="k", values=col).reindex(NAMES)
    print("\n== within-conversation AUROC (pairs within a rater fold)\n" + piv("within_auroc").round(3).to_string())
    print("\n== log loss\n" + piv("logloss").round(4).to_string())
    rows = []
    for a in NAMES[1:]:
        for k in KS:
            pa, la, wa = st[a, k]
            pb, lb, wb = st[NAMES[0], k]
            rows.append(dict(pair=f"{a}  -  {NAMES[0]}", k=k,
                             within=f"{pa[3] - pb[3]:+.3f} [{np.quantile(wa - wb, .025):+.3f}, {np.quantile(wa - wb, .975):+.3f}]",
                             logloss=f"{pa[0] - pb[0]:+.4f} [{np.quantile(la - lb, .025):+.4f}, {np.quantile(la - lb, .975):+.4f}]"))
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(OUT, f"diffs_{raters}.csv"), index=False)
    print("\n== paired differences (two-way bootstrap, 95% CI)\n" + d.to_string(index=False))


if __name__ == "__main__":
    main("all" if "--all-raters" in sys.argv else "104")
