"""Forest-RI/RC (Breiman 2001) - implementation shared within the group.

Sonar labels are 0 (R) and 1 (M). RI uses sklearn's CART forest;
RC implements random linear combinations (L=3) at every node.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier


def fit_forest_ri(X, y, F, n_trees=100, seed=0):
    """Bootstrap + random F features at every CART node; fully grown trees."""
    return RandomForestClassifier(
        n_estimators=n_trees, max_features=F, bootstrap=True,
        criterion="gini", max_depth=None, min_samples_leaf=1,
        random_state=seed, n_jobs=1,
    ).fit(X, y)


class RCTree:
    """Forest-RC tree: for each node try F random projections of L inputs."""
    def __init__(self, F=2, L=3, rng=None):
        self.F, self.L = F, L
        self.rng = rng if rng is not None else np.random.default_rng()

    def fit(self, X, y):
        self.M = X.shape[1]
        self.vars, self.coefs, self.thr = [], [], []
        self.left, self.right, self.leaf_value = [], [], []
        self._grow(X, y)
        self.vars, self.coefs = np.array(self.vars), np.array(self.coefs)
        self.thr, self.leaf_value = np.array(self.thr), np.array(self.leaf_value)
        self.left, self.right = np.array(self.left), np.array(self.right)
        return self

    def _new_node(self):
        self.vars.append(np.zeros(self.L, dtype=int))
        self.coefs.append(np.zeros(self.L))
        self.thr.append(0.0)
        self.left.append(-1)
        self.right.append(-1)
        self.leaf_value.append(-1)
        return len(self.thr) - 1

    def _grow(self, X, y):
        node = self._new_node()
        n1 = y.sum()
        if n1 == 0 or n1 == len(y):
            self.leaf_value[node] = int(y[0])
            return node
        split = self._best_split(X, y)
        if split is None:
            self.leaf_value[node] = int(n1 * 2 > len(y))
            return node
        v, c, t = split
        go_left = X[:, v] @ c <= t
        self.vars[node], self.coefs[node], self.thr[node] = v, c, t
        self.left[node] = self._grow(X[go_left], y[go_left])
        self.right[node] = self._grow(X[~go_left], y[~go_left])
        return node

    def _best_split(self, X, y, max_tries=10):
        n, total1 = len(y), y.sum()
        for _ in range(max_tries):
            best = None
            for _ in range(self.F):
                v = self.rng.choice(self.M, size=self.L, replace=False)
                c = self.rng.uniform(-1, 1, size=self.L)
                z = X[:, v] @ c
                order = np.argsort(z)
                zs, ys = z[order], y[order]
                n_left = np.arange(1, n)
                left1 = np.cumsum(ys)[:-1]
                right1 = total1 - left1
                n_right = n - n_left
                p_l, p_r = left1 / n_left, right1 / n_right
                gini = (n_left * 2 * p_l * (1 - p_l) + n_right * 2 * p_r * (1 - p_r)) / n
                valid = zs[:-1] < zs[1:]
                if not valid.any():
                    continue
                gini[~valid] = np.inf
                i = np.argmin(gini)
                if best is None or gini[i] < best[0]:
                    t = (zs[i] + zs[i + 1]) / 2
                    if not (zs[i] <= t < zs[i + 1]):
                        t = zs[i]
                    best = (gini[i], v, c, t)
            if best is not None:
                return best[1], best[2], best[3]
        return None

    def predict(self, X):
        node = np.zeros(len(X), dtype=int)
        active = self.leaf_value[node] < 0
        while active.any():
            idx = np.where(active)[0]
            nd = node[idx]
            z = (X[idx[:, None], self.vars[nd]] * self.coefs[nd]).sum(axis=1)
            node[idx] = np.where(z <= self.thr[nd], self.left[nd], self.right[nd])
            active = self.leaf_value[node] < 0
        return self.leaf_value[node]


def fit_forest_rc(X, y, F, L=3, n_trees=100, seed=0):
    """Forest-RC following the group's shared implementation (one RNG)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    trees, mask = [], np.ones((n_trees, n), dtype=bool)
    for k in range(n_trees):
        idx = rng.integers(0, n, size=n)
        mask[k, idx] = False
        trees.append(RCTree(F, L, rng).fit(X[idx], y[idx]))
    return trees, mask


def fit_forest(mode, X, y, F, seed, n_trees=100):
    if mode == "RI":
        return fit_forest_ri(X, y, F, n_trees, seed)
    if mode == "RC":
        return fit_forest_rc(X, y, F, 3, n_trees, seed)
    raise ValueError(mode)


def votes_and_mask(mode, model, X, n_train=None):
    """Return votes (n_trees,n_samples), and optionally in-bag-derived OOB mask."""
    if mode == "RI":
        votes = np.array([t.predict(X) for t in model.estimators_], dtype=np.int8)
        if n_train is None:
            return votes, None
        mask = np.ones((len(model.estimators_), n_train), dtype=bool)
        for k, inbag in enumerate(model.estimators_samples_):
            mask[k, inbag] = False
        return votes, mask
    trees, mask = model
    votes = np.array([t.predict(X) for t in trees], dtype=np.int8)
    return votes, mask if n_train is not None else None


def majority(votes):
    """One tree = one vote; ties -> class 0, as in the shared code."""
    return (np.mean(votes, axis=0) > .5).astype(np.int8)


def test_error(mode, model, X, y):
    votes, _ = votes_and_mask(mode, model, X)
    return float(np.mean(majority(votes) != y) * 100)


def oob_stats(votes, mask, y):
    """OOB test error, mean individual tree error, strength, correlation.

    Uses the mathematically correct minus sign in the variance formula.
    """
    n_oob = mask.sum(axis=0)
    ok = n_oob > 0
    p_vote1 = np.sum(votes * mask, axis=0)[ok] / n_oob[ok]
    pred_oob = (p_vote1 > 0.5).astype(np.int8)
    oob_err = float(np.mean(pred_oob != y[ok]) * 100)
    q_true = np.where(y[ok] == 1, p_vote1, 1 - p_vote1)
    margins = 2 * q_true - 1
    strength = float(np.mean(margins))
    sds, errs = [], []
    for k in range(len(votes)):
        o = mask[k]
        if not o.any():
            continue
        good = np.mean(votes[k, o] == y[o])
        sds.append(np.sqrt(max(0., 1 - (2 * good - 1)**2)))
        errs.append(100 * (1 - good))
    denom = float(np.mean(sds)**2)
    rho = float(np.var(margins) / denom) if denom > 1e-14 else np.nan
    return dict(oob_error=oob_err, one_tree=float(np.mean(errs)), strength=strength,
                correlation=rho, rho_over_s2=rho / strength**2 if abs(strength) > 1e-14 else np.nan)


def measure(mode, F, Xtr, ytr, Xte, yte, seed, trees):
    model = fit_forest(mode, Xtr, ytr, F, seed, trees)
    vtr, mask = votes_and_mask(mode, model, Xtr, len(ytr))
    info = oob_stats(vtr, mask, ytr)
    info["test_error"] = test_error(mode, model, Xte, yte)
    return info


def perm_importance(X, y, seed, n_trees=1000, F=1):
    """Permutation in EACH tree's OOB rows; report error change by feature."""
    rf = fit_forest_ri(X, y, F, n_trees=n_trees, seed=seed)
    vtr, mask = votes_and_mask("RI", rf, X, len(y))
    base = oob_stats(vtr, mask, y)["oob_error"]
    n_oob = mask.sum(axis=0)
    ok = n_oob > 0
    rng = np.random.default_rng(seed + 123456)
    out = []
    for j in range(X.shape[1]):
        count_one = np.zeros(len(y), dtype=int)
        for k, tree in enumerate(rf.estimators_):
            idx = np.flatnonzero(mask[k])
            if len(idx) == 0:
                continue
            Xo = X[idx].copy()
            Xo[:, j] = rng.permutation(Xo[:, j])
            count_one[idx] += tree.predict(Xo).astype(int)
        err = np.mean((count_one[ok] / n_oob[ok] > 0.5).astype(int) != y[ok]) * 100
        out.append(dict(feature=j+1, oob_original_pct=base, oob_permuted_pct=err,
                        increase_relative_pct=100 * (err - base) / base if base > 0 else np.nan))
    return out
