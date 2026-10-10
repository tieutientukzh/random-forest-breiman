"""Tái hiện Bảng 4 của paper (Mục 8): ảnh hưởng của 5% nhãn nhiễu.
Mỗi lần: giữ 10% làm test; huấn luyện 2 lần trên cùng tập train - một lần nhãn sạch,
một lần đổi ngẫu nhiên 5% nhãn train sang lớp còn lại. Tập test giữ nhãn sạch. Lặp 50 lần.
Mỗi phương pháp dùng cấu hình có test error thấp nhất ở các bước trước:
Forest-RI F=1, Forest-RC F=2, Adaboost cây độ sâu 5."""
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from data_loader import load_breast_cancer_original
from protocol import split_and_impute
from forest_utils import fit_forest_ri, tree_votes, majority_vote
from forest_rc import fit_forest_rc, tree_votes_rc

N_REPEATS = 50
NOISE_RATE = 0.05
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def add_label_noise(y, rate, seed):
    """Đổi ngẫu nhiên `rate` phần nhãn sang lớp còn lại (bài toán 2 lớp)."""
    rng = np.random.default_rng(seed)
    y_noisy = y.copy()
    idx = rng.choice(len(y), size=int(round(rate * len(y))), replace=False)
    y_noisy[idx] = 1 - y_noisy[idx]
    return y_noisy


def test_errors(X_tr, y_tr, X_te, y_te, seed):
    """Test error của 3 phương pháp khi huấn luyện trên (X_tr, y_tr)."""
    rf = fit_forest_ri(X_tr, y_tr, F=1, n_trees=100, seed=seed)
    trees, _ = fit_forest_rc(X_tr, y_tr, F=2, L=3, n_trees=100, seed=seed)
    ada = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=5, random_state=seed),
                             n_estimators=50, random_state=seed).fit(X_tr, y_tr)
    return {
        "Adaboost": np.mean(ada.predict(X_te) != y_te),
        "Forest-RI": np.mean(majority_vote(tree_votes(rf, X_te)) != y_te),
        "Forest-RC": np.mean(majority_vote(tree_votes_rc(trees, X_te)) != y_te),
    }


X, y, names = load_breast_cancer_original()

rows = []
start = time.time()
for seed in range(N_REPEATS):
    X_tr, X_te, y_tr, y_te = split_and_impute(X, y, seed)
    y_tr_noisy = add_label_noise(y_tr, NOISE_RATE, seed + 1000)
    clean = test_errors(X_tr, y_tr, X_te, y_te, seed)
    noisy = test_errors(X_tr, y_tr_noisy, X_te, y_te, seed)
    for method in clean:
        rows.append({"seed": seed, "method": method, "clean": clean[method], "noisy": noisy[method]})
    if (seed + 1) % 10 == 0:
        print(f"  xong {seed + 1}/{N_REPEATS} lan ({time.time() - start:.0f}s)")

df = pd.DataFrame(rows)
df.to_csv(RESULTS_DIR / "table4_noise_runs.csv", index=False)

paper = {"Adaboost": 43.2, "Forest-RI": 1.8, "Forest-RC": 11.1}
print("\nBANG 4 - Anh huong cua 5% nhan nhieu tren Breast Cancer (trung binh 50 lan)")
print(f"{'Phuong phap':12s} {'Sach %':>7s} {'Nhieu %':>8s} {'Tang (diem %)':>14s} {'SE':>5s} "
      f"{'Tang tuong doi':>15s} {'Paper':>7s}")
for method in paper:
    d = df[df.method == method]
    diff = d.noisy - d.clean
    rel = 100 * (d.noisy.mean() - d.clean.mean()) / d.clean.mean()
    print(f"{method:12s} {100 * d.clean.mean():7.2f} {100 * d.noisy.mean():8.2f} "
          f"{100 * diff.mean():14.2f} {100 * diff.std() / np.sqrt(len(d)):5.2f} "
          f"{rel:14.1f}% {paper[method]:6.1f}%")