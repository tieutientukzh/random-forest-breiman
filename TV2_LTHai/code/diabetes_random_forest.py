"""TV2 - Pima Indians Diabetes, sử dụng module chung của nhóm.
Chạy từ thư mục bất kỳ: python TV2_LTHai/code/diabetes_random_forest.py
QUICK=True để kiểm tra; QUICK=False cho thí nghiệm 100 seed, 100 cây.
Kết quả là thực nghiệm trên Diabetes, không phải số liệu gốc của paper.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
TV2_ROOT = HERE.parent
REPO = TV2_ROOT.parent
SHARED = REPO / 'TV_NPHieu' / 'code'
if not (SHARED / 'forest_utils.py').is_file() or not (SHARED / 'forest_rc.py').is_file():
    raise FileNotFoundError(f'Thiếu forest_utils.py hoặc forest_rc.py trong {SHARED}')
# Giữ thư mục TV2 ưu tiên cho data_loader.py và protocol.py riêng.
sys.path.insert(1, str(SHARED))
from forest_utils import (fit_forest_ri, tree_votes, majority_vote, oob_mask,
                          oob_error, one_tree_error, strength_correlation)
from forest_rc import fit_forest_rc, tree_votes_rc
from protocol import split_and_impute
from data_loader import load_diabetes

QUICK = False  # Đổi False để chạy 100 lần; Forest-RC có thể mất nhiều thời gian
N_REPEATS = 2 if QUICK else 100
N_TREES = 5 if QUICK else 100
SEEDS = range(N_REPEATS)  # 0..99 khi chạy chính thức
ZERO_AS_MISSING = True  # Cần nhóm xác nhận quy ước này cho Diabetes
ZERO_COLS = ('Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI')
RESULTS = TV2_ROOT / 'results'
RESULTS.mkdir(exist_ok=True, parents=True)

X, y, feature_names = load_diabetes()
M = X.shape[1]
ZERO_INDICES = tuple(feature_names.index(c) for c in ZERO_COLS) if ZERO_AS_MISSING else ()
F_RI = sorted(set((1, int(np.log2(M) + 1))))
F_RC = (2, 8)


def split(seed):
    return split_and_impute(X, y, seed=seed, test_size=0.1,
                            zero_missing_indices=ZERO_INDICES)


def err(y_true, pred):
    return 1.0 - accuracy_score(y_true, pred)


def fit_ri_error(X_tr, y_tr, X_te, y_te, F, seed):
    rf = fit_forest_ri(X_tr, y_tr, F=F, n_trees=N_TREES, seed=seed)
    return err(y_te, majority_vote(tree_votes(rf, X_te))), rf


def best_ri(X_tr, y_tr, X_te, y_te, seed):
    """Chọn F bằng OOB trên train; không dùng test để chọn tham số."""
    candidates = []
    for F in F_RI:
        rf = fit_forest_ri(X_tr, y_tr, F=F, n_trees=N_TREES, seed=seed)
        mask = oob_mask(rf, len(y_tr))
        votes_tr = tree_votes(rf, X_tr)
        candidates.append((oob_error(votes_tr, mask, y_tr), F, rf, votes_tr, mask))
    _, F, rf, votes_tr, mask = min(candidates, key=lambda t: (t[0], t[1]))
    test_error = err(y_te, majority_vote(tree_votes(rf, X_te)))
    return test_error, F, rf, votes_tr, mask


def boost_error(X_tr, y_tr, X_te, y_te, depth, seed):
    # API sklearn >= 1.2, tương thích phiên bản người dùng đang sử dụng.
    model = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=depth),
                               n_estimators=5 if QUICK else 50, random_state=seed)
    model.fit(X_tr, y_tr)
    return err(y_te, model.predict(X_te))


# Cell 1-4: Table 2; 100 holdout 90/10, Forest-RI, One Tree, AdaBoost
rows2, rows3, rows4, rows_diag = [], [], [], []
for seed in SEEDS:
    X_tr, X_te, y_tr, y_te = split(seed)
    selection_error, chosen_F, rf, votes_tr, mask = best_ri(X_tr, y_tr, X_te, y_te, seed)
    rows2.append(dict(seed=seed, method='Selection', F=chosen_F, error_rate=selection_error))
    single_error, _ = fit_ri_error(X_tr, y_tr, X_te, y_te, F=1, seed=seed)
    rows2.append(dict(seed=seed, method='Single Input', F=1, error_rate=single_error))
    single_tree = DecisionTreeClassifier(random_state=seed).fit(X_tr, y_tr)
    rows2.append(dict(seed=seed, method='One Tree', F=np.nan,
                      error_rate=err(y_te, single_tree.predict(X_te))))
    for depth in (1, 3, 5):
        rows2.append(dict(seed=seed, method=f'AdaBoost depth={depth}', F=np.nan,
                          error_rate=boost_error(X_tr, y_tr, X_te, y_te, depth, seed)))

    strength, correlation = strength_correlation(votes_tr, mask, y_tr)
    rows_diag.append(dict(seed=seed, F=chosen_F,
                          oob_error=oob_error(votes_tr, mask, y_tr),
                          one_tree_oob_error=one_tree_error(votes_tr, mask, y_tr),
                          strength=strength, correlation=correlation,
                          correlation_over_strength_sq=(correlation / strength**2
                              if abs(strength) > 1e-12 else np.nan)))

    # Cell 5: Forest-RC, scale train only, F=2 và 8, L=3
    scaler = StandardScaler().fit(X_tr)
    X_rc_tr, X_rc_te = scaler.transform(X_tr), scaler.transform(X_te)
    for F in F_RC:
        trees, rc_mask = fit_forest_rc(X_rc_tr, y_tr, F=F, L=3,
                                       n_trees=N_TREES, seed=seed)
        pred = majority_vote(tree_votes_rc(trees, X_rc_te))
        rows3.append(dict(seed=seed, method='Forest-RC', F=F, L=3,
                          error_rate=err(y_te, pred)))

    # Cell 6: Label noise, train only, test labels stay unchanged
    for noise in (0.0, 0.05):
        y_noisy = y_tr.copy()
        if noise:
            rng = np.random.default_rng(100000 + seed)
            flip = rng.choice(len(y_noisy), size=round(noise * len(y_noisy)), replace=False)
            y_noisy[flip] = 1 - y_noisy[flip]
        for F in F_RI:
            e, _ = fit_ri_error(X_tr, y_noisy, X_te, y_te, F, seed)
            rows4.append(dict(seed=seed, noise_fraction=noise, method='Forest-RI',
                              F=F, test_error_rate=e))
    print(f'Hoàn thành seed {seed + 1}/{N_REPEATS}', flush=True)

pd.DataFrame(rows2).to_csv(RESULTS / 'results_diabetes_table2.csv', index=False)
pd.DataFrame(rows3).to_csv(RESULTS / 'results_diabetes_table3.csv', index=False)
pd.DataFrame(rows4).to_csv(RESULTS / 'results_diabetes_table4.csv', index=False)
pd.DataFrame(rows_diag).to_csv(RESULTS / 'diabetes_oob_diagnostics.csv', index=False)

# Cell 7-8: Variable importance và feature subsets trên seed=0
X_tr, X_te, y_tr, y_te = split(0)
_, F, reference, _, _ = best_ri(X_tr, y_tr, X_te, y_te, 0)
perm = permutation_importance(reference, X_te, y_te, n_repeats=3 if QUICK else 15,
                              random_state=0, scoring='accuracy')
imp = pd.Series(perm.importances_mean, index=feature_names).sort_values()
fig, ax = plt.subplots(figsize=(9, 5))
imp.plot.barh(ax=ax)
ax.set(title='Diabetes - Permutation importance', xlabel='Mean decrease in accuracy')
fig.tight_layout()
fig.savefig(RESULTS / 'variable_importance.png', dpi=160)
plt.close(fig)

subset_rows = []
for k in (2, 4, 6, 8):
    selector = SelectKBest(f_classif, k=k).fit(X_tr, y_tr)
    a, b = selector.transform(X_tr), selector.transform(X_te)
    f_sub = min(F, k)
    e, _ = fit_ri_error(a, y_tr, b, y_te, f_sub, 0)
    subset_rows.append(dict(n_features=k,
                            features=','.join(np.array(feature_names)[selector.get_support()]),
                            accuracy=1-e))
pd.DataFrame(subset_rows).to_csv(RESULTS / 'feature_subsets.csv', index=False)

print('\nTABLE 2:\n', pd.DataFrame(rows2).groupby('method')['error_rate'].agg(['mean','std']))
print('\nTABLE 3:\n', pd.DataFrame(rows3).groupby('F')['error_rate'].agg(['mean','std']))
print('\nĐã xuất kết quả tại:', RESULTS)
