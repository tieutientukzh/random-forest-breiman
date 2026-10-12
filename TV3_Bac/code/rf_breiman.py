"""
rf_breiman.py - Code cua TV3 (Sonar): tai hien Breiman (2001) "Random Forests".

Gom:
  1. make_rc_matrix        : sinh dac trung to hop tuyen tinh ngau nhien (Forest-RC)
  2. RCTree / BreimanForest: rung tu cai dat (mode "RI" hoac "RC"; RC sinh to hop tai tung nut), giu chi so OOB
  3. oob_report            : OOB error, One Tree, Strength, Correlation (Phu luc II)
  4. variable_importance   : Variable Importance (Muc 10)
  5. run_table / sweep_F / noise_experiment : Bang 2, 3, 4 va Hinh 1

Chay thu:  python rf_breiman.py duong_dan/sonar.all-data
"""
import os
import sys
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


# ---------------------------------------------------------------- 1. Forest-RC
def make_rc_matrix(rng, M, P=50, L=3):
    """Ma tran W (M x P). Moi cot la 1 to hop tuyen tinh cua L bien chon ngau nhien,
    he so ~ U[-1, 1]. Dac trung moi = X @ W."""
    W = np.zeros((M, P))
    for j in range(P):
        chosen = rng.choice(M, size=L, replace=False)
        W[chosen, j] = rng.uniform(-1, 1, size=L)
    return W


# ---------------------------------------------------------------- 2. Rung
class RCTree:
    """Cay Forest-RC (Muc 5 cua paper): TAI MOI NUT sinh F dac trung moi, moi dac trung la to hop tuyen tinh
    cua L bien goc voi he so ~ U[-1, 1]; chon phep chia Gini tot nhat. Cay moc toi da, khong cat tia.
    Tim phep chia bang cumsum (vector hoa) nen nhanh hon vong lap tung mau rat nhieu."""

    def __init__(self, F=2, L=3, seed=42, n_classes=2, max_tries=10):
        self.F, self.L, self.n_classes_, self.max_tries = F, L, n_classes, max_tries
        self.rng = np.random.default_rng(seed)

    def fit(self, X, y):
        self.root_ = self._build(np.asarray(X, float), np.asarray(y, int))
        return self

    def _best_split(self, X, y):
        n, M, C = len(y), X.shape[1], self.n_classes_
        Y = np.eye(C)[y]                                   # one-hot (n x C)
        total = Y.sum(0)
        nl = np.arange(1, n)[:, None]
        nr = n - nl
        for _ in range(self.max_tries):                    # neu ca F dac trung deu khong chia duoc thi sinh lai
            best = None
            for _ in range(self.F):
                chosen = self.rng.choice(M, size=min(self.L, M), replace=False)
                coef = self.rng.uniform(-1, 1, size=len(chosen))
                z = X[:, chosen] @ coef
                order = np.argsort(z, kind="stable")
                zs = z[order]
                left = np.cumsum(Y[order], axis=0)[:-1]    # so mau moi lop o ben trai
                right = total - left
                gl = 1 - ((left / nl) ** 2).sum(1)
                gr = 1 - ((right / nr) ** 2).sum(1)
                w = (nl[:, 0] * gl + nr[:, 0] * gr) / n    # Gini co trong so sau khi chia
                valid = zs[:-1] < zs[1:]                   # chi chia giua hai gia tri khac nhau
                if not valid.any():
                    continue
                w = np.where(valid, w, np.inf)
                j = int(np.argmin(w))
                if best is None or w[j] < best[0]:
                    t = (zs[j] + zs[j + 1]) / 2
                    if not (zs[j] <= t < zs[j + 1]):       # hai gia tri qua sat nhau
                        t = zs[j]
                    best = (w[j], chosen, coef, t)
            if best is not None:
                return best[1], best[2], best[3]
        return None

    def _build(self, X, y):
        counts = np.bincount(y, minlength=self.n_classes_)
        value = int(np.argmax(counts))
        leaf = {"leaf": True, "value": value}
        if (counts > 0).sum() <= 1:                        # nut thuan nhat -> la
            return leaf
        split = self._best_split(X, y)                     # chon phep chia tot nhat KE CA khi gain = 0
        if split is None:                                  # (nhu sklearn) de cay moc toi da
            return leaf
        chosen, coef, thr = split
        go_left = X[:, chosen] @ coef <= thr
        if go_left.all() or not go_left.any():
            return leaf
        return {"leaf": False, "chosen": chosen, "coef": coef, "threshold": thr, "value": value,
                "left": self._build(X[go_left], y[go_left]),
                "right": self._build(X[~go_left], y[~go_left])}

    def _predict_into(self, node, X, idx, out):
        if len(idx) == 0:
            return
        if node["leaf"]:
            out[idx] = node["value"]
            return
        go_left = X[idx][:, node["chosen"]] @ node["coef"] <= node["threshold"]
        self._predict_into(node["left"], X, idx[go_left], out)
        self._predict_into(node["right"], X, idx[~go_left], out)

    def predict(self, X):
        X = np.asarray(X, float)
        out = np.empty(len(X), dtype=int)
        self._predict_into(self.root_, X, np.arange(len(X)), out)
        return out


class BreimanForest:
    """mode="RI": cay CART cua sklearn, moi nut chon ngau nhien F bien goc.
    mode="RC": RCTree o tren (to hop tuyen tinh sinh tai tung nut, dung bai bao). Du doan bang bieu quyet cung."""

    def __init__(self, mode="RI", n_trees=100, F=1, L=3, P=50, seed=42, scale=False):
        self.mode, self.n_trees, self.F, self.L, self.P, self.seed = mode, n_trees, F, L, P, seed   # P khong con dung
        # RC: scale=True -> chuan hoa Z-score truoc khi tao to hop; False -> dung gia tri goc.
        # Sonar: 60 bien deu la nang luong chuan hoa trong [0, 1] (cung thang do) nen bai bao khong chuan hoa
        # (Z-score lam Test Error cua Forest-RC tang tu ~14.6% len ~17.1%).
        self.scale = scale

    def fit(self, X, y):
        X = np.asarray(X, float)
        self.classes_, yi = np.unique(np.asarray(y), return_inverse=True)
        N, M = X.shape
        rng = np.random.default_rng(self.seed)
        if self.mode == "RC":
            if self.scale:
                self.mu_, self.sd_ = X.mean(axis=0), X.std(axis=0)
                self.sd_[self.sd_ == 0] = 1.0
            else:
                self.mu_, self.sd_ = np.zeros(M), np.ones(M)
            Z = (X - self.mu_) / self.sd_                  # Z-score theo tap huan luyen (muc 3.2.4)
        else:
            Z = X
        self.X_, self.y_ = X, yi
        self.trees_, self.oob_, self.W_ = [], [], []
        for _ in range(self.n_trees):
            idx = rng.integers(0, N, N)                    # bootstrap co hoan lai
            self.oob_.append(np.setdiff1d(np.arange(N), np.unique(idx)))
            tree_seed = int(rng.integers(2**31 - 1))
            if self.mode == "RC":
                tree = RCTree(self.F, self.L, tree_seed, n_classes=len(self.classes_)).fit(Z[idx], yi[idx])
            else:
                tree = DecisionTreeClassifier(max_features=self.F, random_state=tree_seed).fit(Z[idx], yi[idx])
            self.trees_.append(tree)
            self.W_.append(None)
        return self

    def _votes_tree(self, k, X):
        X = np.asarray(X, float)
        Z = (X - self.mu_) / self.sd_ if self.mode == "RC" else X
        return self.trees_[k].predict(Z)

    def predict(self, X):
        X = np.asarray(X, float)
        votes = np.zeros((len(X), len(self.classes_)), dtype=int)
        for k in range(self.n_trees):
            votes[np.arange(len(X)), self._votes_tree(k, X)] += 1
        return self.classes_[votes.argmax(axis=1)]


# ---------------------------------------------------------------- 3. OOB, Strength, Correlation
def oob_report(forest, sd_sign=-1):
    """Tinh theo Phu luc II. Tra ve dict: oob_error, one_tree_error, strength, correlation, c_s2.

    sd_sign: sd(Theta_k) = sqrt(p1 + p2 + sd_sign*(p1-p2)^2).
      Bai bao in dau CONG (A2), nhung rmg nhan gia tri {1,0,-1} nen
      Var(rmg) = p1 + p2 - (p1-p2)^2  -> dau TRU. Mac dinh dung dau tru; dat sd_sign=+1 de theo dung chu in."""
    K, y = forest.n_trees, forest.y_
    N, C = len(y), len(forest.classes_)
    votes, preds = np.zeros((N, C)), []
    for k in range(K):
        o = forest.oob_[k]
        p = forest._votes_tree(k, forest.X_[o])
        votes[o, p] += 1
        preds.append(p)
    n_oob = votes.sum(1)
    ok = n_oob > 0                                             # mau nam ngoai bag it nhat 1 cay
    Q = np.zeros((N, C))
    Q[ok] = votes[ok] / n_oob[ok, None]                        # Q(x, j)
    rows = np.arange(N)
    masked = Q.copy()
    masked[rows, y] = -np.inf
    jhat = masked.argmax(1)                                    # lop sai pho bien nhat
    margin = Q[rows, y] - masked.max(1)                        # Q(x,y) - max_{j!=y} Q(x,j)

    s = margin[ok].mean()
    var_mr = (margin[ok] ** 2).mean() - s ** 2
    sds, tree_err = [], []
    for k in range(K):
        o, p = forest.oob_[k], preds[k]
        if len(o) == 0:
            continue
        p1, p2 = np.mean(p == y[o]), np.mean(p == jhat[o])
        sds.append(np.sqrt(max(p1 + p2 + sd_sign * (p1 - p2) ** 2, 0.0)))
        tree_err.append(np.mean(p != y[o]))
    rho = var_mr / np.mean(sds) ** 2
    return dict(oob_error=np.mean(votes[ok].argmax(1) != y[ok]) * 100,
                one_tree_error=np.mean(tree_err) * 100,
                strength=s, correlation=rho, c_s2=rho / s ** 2)


# ---------------------------------------------------------------- 4. Variable Importance
def variable_importance(forest, seed=0):
    """Muc 10: voi moi bien m, hoan vi ngau nhien gia tri bien m trong cac mau OOB cua TUNG cay,
    cho qua cay, lay da so phieu OOB, so voi ty le loi OOB goc.
    Tra ve % tang ty le phan loai sai cua tung bien (mang do dai M). Dung cho forest mode "RI"."""
    rng = np.random.default_rng(seed)
    y, X = forest.y_, forest.X_
    N, M = X.shape
    C = len(forest.classes_)

    def oob_error(perm_col=None):
        votes = np.zeros((N, C))
        for k in range(forest.n_trees):
            o = forest.oob_[k]
            Xo = X[o].copy()
            if perm_col is not None:
                Xo[:, perm_col] = rng.permutation(Xo[:, perm_col])
            votes[o, forest._votes_tree(k, Xo)] += 1
        ok = votes.sum(1) > 0
        return np.mean(votes[ok].argmax(1) != y[ok])

    base = oob_error()
    err_m = np.array([oob_error(m) for m in range(M)])
    return 100 * (err_m - base) / max(base, 1e-12)


# ---------------------------------------------------------------- 5. Thi nghiem
def one_repeat(X, y, i, mode, Fs, n_trees=100):
    """1 lan lap: cung 1 phep chia train/test cho moi F (seed = 42 + i)."""
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.1, random_state=42 + i)
    res = {}
    for F in Fs:
        f = BreimanForest(mode, n_trees, F, seed=42 + i).fit(Xtr, ytr)
        r = oob_report(f)
        r["test_error"] = np.mean(f.predict(Xte) != yte) * 100
        res[F] = r
    best = min(Fs, key=lambda F: res[F]["oob_error"])          # chon F theo OOB thap nhat
    return dict(selection=res[best]["test_error"],             # cot "Selection"
                single=res[Fs[0]]["test_error"],               # cot "Single Input" / "Two Features"
                one_tree=res[best]["one_tree_error"],          # cot "One Tree"
                best_F=best, strength=res[best]["strength"],
                correlation=res[best]["correlation"], c_s2=res[best]["c_s2"])


def run_table(X, y, mode, Fs, n_rep=100, n_trees=100, n_jobs=-1):
    """Bang 2 (mode="RI", Fs=[1, int(log2 M + 1)]) va Bang 3 (mode="RC", Fs=[2, 8]). Tra ve trung binh."""
    rows = Parallel(n_jobs=n_jobs)(delayed(one_repeat)(X, y, i, mode, Fs, n_trees) for i in range(n_rep))
    return pd.DataFrame(rows).mean().round(3)


def sweep_F(X, y, F_list, n_rep=50, n_trees=100, n_jobs=-1):
    """Hinh 1: Forest-RI voi F thay doi. Tra ve DataFrame (test_error, oob_error, strength, correlation)."""
    def job(i):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.1, random_state=42 + i)
        out = []
        for F in F_list:
            f = BreimanForest("RI", n_trees, F, seed=42 + i).fit(Xtr, ytr)
            r = oob_report(f)
            out.append(dict(F=F, test_error=np.mean(f.predict(Xte) != yte) * 100, **r))
        return out
    rows = [r for rep in Parallel(n_jobs=n_jobs)(delayed(job)(i) for i in range(n_rep)) for r in rep]
    return pd.DataFrame(rows).groupby("F").mean()[["test_error", "oob_error", "strength", "correlation"]]


def plot_figure1(df, path="hinh1_sonar.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(6, 8), sharex=True)
    ax[0].plot(df.index, df["correlation"], label="correlation")
    ax[0].plot(df.index, df["strength"], label="strength")
    ax[0].set_ylabel("giá trị"); ax[0].legend()
    ax[1].plot(df.index, df["oob_error"], label="OOB error")
    ax[1].plot(df.index, df["test_error"], label="test set error")
    ax[1].set_xlabel("số biến đầu vào F"); ax[1].set_ylabel("%"); ax[1].legend()
    fig.suptitle("Ảnh hưởng của F lên dữ liệu Sonar (tái hiện Hình 1)")
    fig.tight_layout(); fig.savefig(path, dpi=150)


def add_label_noise(y, rate, rng):
    """Doi ngau nhien `rate` ty le nhan sang 1 nhan khac chon deu trong cac nhan con lai."""
    y = np.asarray(y).copy()
    classes = np.unique(y)
    flip = rng.choice(len(y), size=int(round(rate * len(y))), replace=False)
    for i in flip:
        y[i] = rng.choice(classes[classes != y[i]])
    return y


def noise_experiment(X, y, mode, F, n_rep=50, n_trees=100, rate=0.05):
    """Bang 4: % tang test error khi huan luyen tren nhan nhieu 5% (test van sach)."""
    clean, noisy = [], []
    for i in range(n_rep):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.1, random_state=42 + i)
        ytr_n = add_label_noise(ytr, rate, np.random.default_rng(1000 + i))
        for ylab, store in ((ytr, clean), (ytr_n, noisy)):
            f = BreimanForest(mode, n_trees, F, seed=42 + i).fit(Xtr, ylab)
            store.append(np.mean(f.predict(Xte) != yte) * 100)
    return 100 * (np.mean(noisy) - np.mean(clean)) / np.mean(clean)


def load_sonar(path):
    """File UCI 'sonar.all-data': 60 cot so + cot nhan R/M, khong co header."""
    df = pd.read_csv(path, header=None)
    return df.iloc[:, :60].to_numpy(float), df.iloc[:, 60].to_numpy()


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE_DIR, "data", "sonar.all-data")

    RESULTS_DIR = os.path.join(BASE_DIR, "results")
    FIGURES_DIR = os.path.join(BASE_DIR, "figures")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    X, y = load_sonar(path)
    M = X.shape[1]
    F2 = int(np.log2(M) + 1)

    print("Bang 2 (Forest-RI):\n", run_table(X, y, "RI", [1, F2], n_rep=100))
    print("Bang 3 (Forest-RC):\n", run_table(X, y, "RC", [2, 8], n_rep=100))

    f = BreimanForest("RI", n_trees=1000, F=1, seed=42).fit(X, y)
    vi = variable_importance(f)
    print("Top 10 bien quan trong (chi so cot, % tang loi):")
    for m in np.argsort(vi)[::-1][:10]:
        print(f"  bien {m + 1:2d}: {vi[m]:.1f}%")
    pd.DataFrame({"bien": np.arange(1, M + 1), "importance": vi}).to_csv(
        os.path.join(RESULTS_DIR, "variable_importance.csv"), index=False)

    df = sweep_F(X, y, F_list=list(range(1, 51)), n_rep=80)
    df.to_csv(os.path.join(RESULTS_DIR, "figure1_sonar_ri.csv"))
    plot_figure1(df, os.path.join(FIGURES_DIR, "hinh1_sonar.png"))
    print(df.round(3))

    print("Bang 4 - tang loi do 5% nhieu (RI, RC):",
          round(noise_experiment(X, y, "RI", F=1), 1),
          round(noise_experiment(X, y, "RC", F=2), 1))
