"""A small numpy MLP that learns a *correction on top of a fixed offset* (e.g. JEV's calibrated logit):

    logit = offset + MLP(x)

The output layer starts at zero, so an untrained model returns the offset exactly and training only moves it where the
data say so.  Adam, L2 weight decay, optional dropout on hidden units, optional early stopping on a validation set,
sample weights.  numpy only (the system python has no torch), which is plenty for a few hundred thousand rows.
"""
import numpy as np
from scipy.special import expit


def _logloss(m, y, w=None):
    v = np.logaddexp(0, m) - y * m
    return float(v.mean() if w is None else (w * v).sum() / w.sum())


class OffsetMLP:
    def __init__(self, hidden=(64,), wd=1e-3, lr=1e-3, epochs=50, batch=256, dropout=0.0, seed=0, patience=5):
        self.hidden, self.wd, self.lr, self.epochs, self.batch = tuple(hidden), wd, lr, epochs, batch
        self.dropout, self.seed, self.patience = dropout, seed, patience

    def _init(self, d):
        rng = np.random.default_rng(self.seed)
        self.W, self.b = [], []
        for h in self.hidden:
            self.W.append(rng.normal(0, np.sqrt(2 / d), (d, h)))
            self.b.append(np.zeros(h))
            d = h
        self.wo, self.bo = np.zeros(d), 0.0  # zero output layer: start exactly at the offset

    def _params(self):
        return self.W + self.b + [self.wo, np.array([self.bo])]

    def _set(self, ps):
        k = len(self.hidden)
        self.W, self.b, self.wo, self.bo = ps[:k], ps[k:2 * k], ps[2 * k], float(ps[2 * k + 1][0])

    def residual(self, X):
        h = X
        for W, b in zip(self.W, self.b):
            h = np.maximum(h @ W + b, 0)
        return h @ self.wo + self.bo

    def decision(self, X, o):
        return o + self.residual(X)

    def predict_proba(self, X, o):
        return expit(self.decision(X, o))

    def fit(self, X, o, y, w=None, val=None, rng=None):
        """val: optional (Xv, ov, yv[, wv]) for early stopping (best epoch kept). Returns per-epoch validation losses."""
        X, o, y = np.asarray(X, float), np.asarray(o, float), np.asarray(y, float)
        w = np.ones(len(y)) if w is None else np.asarray(w, float)
        self._init(X.shape[1])
        rng = rng or np.random.default_rng(self.seed + 1)
        ps = self._params()
        m1, m2 = [np.zeros_like(p) for p in ps], [np.zeros_like(p) for p in ps]
        b1, b2, eps, step = .9, .999, 1e-8, 0
        hist, best, best_ps, bad = [], np.inf, None, 0
        k = len(self.hidden)
        for ep in range(self.epochs):
            order = rng.permutation(len(y))
            for s in range(0, len(y), self.batch):
                idx = order[s:s + self.batch]
                xb, ob, yb, wb = X[idx], o[idx], y[idx], w[idx]
                hs, masks, h = [xb], [], xb
                for W, b in zip(self.W, self.b):
                    a = h @ W + b
                    msk = (a > 0).astype(float)
                    if self.dropout:
                        msk *= (rng.random(a.shape) >= self.dropout) / (1 - self.dropout)
                    h = a * msk
                    hs.append(h)
                    masks.append(msk)
                m = ob + h @ self.wo + self.bo
                dm = wb * (expit(m) - yb) / wb.sum()
                g_wo, g_bo = hs[-1].T @ dm + self.wd * self.wo, dm.sum()
                dh = np.outer(dm, self.wo)
                gW, gb = [None] * k, [None] * k
                for l in range(k - 1, -1, -1):
                    da = dh * masks[l]
                    gW[l] = hs[l].T @ da + self.wd * self.W[l]
                    gb[l] = da.sum(0)
                    dh = da @ self.W[l].T
                grads = gW + gb + [g_wo, np.array([g_bo])]
                step += 1
                ps = self._params()
                for i, (p, g) in enumerate(zip(ps, grads)):
                    m1[i] = b1 * m1[i] + (1 - b1) * g
                    m2[i] = b2 * m2[i] + (1 - b2) * g * g
                    p -= self.lr * (m1[i] / (1 - b1 ** step)) / (np.sqrt(m2[i] / (1 - b2 ** step)) + eps)
                self._set(ps)
            if val is not None:
                Xv, ov, yv = val[:3]
                lv = _logloss(self.decision(Xv, ov), yv, val[3] if len(val) > 3 else None)
                hist.append(lv)
                if lv < best - 1e-5:
                    best, best_ps, bad = lv, [p.copy() for p in self._params()], 0
                else:
                    bad += 1
                    if self.patience and bad >= self.patience:
                        break
        if best_ps is not None:
            self._set(best_ps)
        return hist


def ensemble_decision(models, X, o):
    return np.mean([mdl.decision(X, o) for mdl in models], 0)
