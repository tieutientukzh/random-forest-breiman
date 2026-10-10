"""Forest-RC (Mục 5 của paper): tại mỗi nút, sinh F đặc trưng mới, mỗi đặc trưng là
tổ hợp tuyến tính ngẫu nhiên của L biến gốc với hệ số lấy từ U[-1, 1];
tìm phép chia tốt nhất (Gini) trong F đặc trưng đó. Cây mọc tối đa, không cắt tỉa."""
import numpy as np


class RCTree:
    def __init__(self, F=2, L=3, rng=None):
        self.F, self.L = F, L
        self.rng = rng if rng is not None else np.random.default_rng()

    # ---------- huấn luyện ----------
    def fit(self, X, y):
        self.M = X.shape[1]
        self.vars, self.coefs, self.thr = [], [], []   # thông tin phép chia của từng nút
        self.left, self.right, self.leaf_value = [], [], []
        self._grow(X, y)
        # đổi sang mảng numpy để dự đoán nhanh
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
        self.leaf_value.append(-1)                       # -1 nghĩa là nút trong
        return len(self.thr) - 1

    def _grow(self, X, y):
        node = self._new_node()
        n1 = y.sum()
        if n1 == 0 or n1 == len(y):                     # nút thuần nhất -> lá
            self.leaf_value[node] = int(y[0])
            return node
        split = self._best_split(X, y)
        if split is None:                               # không chia được (các mẫu trùng nhau) -> lá
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
        for _ in range(max_tries):                      # nếu cả F đặc trưng đều không chia được thì sinh lại
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
                valid = zs[:-1] < zs[1:]                # chỉ chia giữa hai giá trị khác nhau
                if not valid.any():
                    continue
                gini[~valid] = np.inf
                i = np.argmin(gini)
                if best is None or gini[i] < best[0]:
                    t = (zs[i] + zs[i + 1]) / 2
                    if not (zs[i] <= t < zs[i + 1]):     # hai giá trị quá sát nhau, trung điểm bị làm tròn
                        t = zs[i]
                    best = (gini[i], v, c, t)
            if best is not None:
                return best[1], best[2], best[3]
        return None

    # ---------- dự đoán ----------
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


def fit_forest_rc(X_tr, y_tr, F, L=3, n_trees=100, seed=0):
    """Trả về danh sách cây và ma trận OOB (số cây x n_train)."""
    rng = np.random.default_rng(seed)
    n = len(y_tr)
    trees, mask = [], np.ones((n_trees, n), dtype=bool)
    for k in range(n_trees):
        idx = rng.integers(0, n, size=n)              # mẫu bootstrap (rút có hoàn lại)
        mask[k, idx] = False                          # mẫu đã được rút -> không phải OOB
        trees.append(RCTree(F, L, rng).fit(X_tr[idx], y_tr[idx]))
    return trees, mask


def tree_votes_rc(trees, X):
    return np.array([t.predict(X) for t in trees]).astype(int)