"""TV2 - Pima Indians Diabetes, bản đã sửa để bám sát plan/paper hơn.

QUICK=True  : chạy thử nhanh.
QUICK=False : thí nghiệm chính 100 seed, 100 cây; variable importance 1.000 cây.

"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
TV2_ROOT = HERE.parent
REPO = TV2_ROOT.parent
SHARED = REPO / 'TV_NPHieu' / 'code'

if not (SHARED / 'forest_utils.py').is_file() or not (SHARED / 'forest_rc.py').is_file():
    raise FileNotFoundError(f'Thiếu forest_utils.py hoặc forest_rc.py trong {SHARED}')

# Thư mục code của TV2 vẫn đứng trước SHARED trên sys.path khi chạy file trực tiếp.
sys.path.insert(1, str(SHARED))

from forest_utils import (
    fit_forest_ri,
    tree_votes,
    majority_vote,
    oob_mask,
    oob_error,
    one_tree_error,
    strength_correlation,
)
from forest_rc import fit_forest_rc, tree_votes_rc
from protocol import split_and_impute
from data_loader import load_diabetes


QUICK = False
N_REPEATS = 2 if QUICK else 100
N_TREES = 5 if QUICK else 100
N_IMPORTANCE_TREES = 30 if QUICK else 1000
SEEDS = range(N_REPEATS)

ZERO_AS_MISSING = True
ZERO_COLS = ('Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI')

RESULTS = TV2_ROOT / 'results'
RESULTS.mkdir(exist_ok=True, parents=True)

X, y, feature_names = load_diabetes()
M = X.shape[1]
ZERO_INDICES = tuple(feature_names.index(c) for c in ZERO_COLS) if ZERO_AS_MISSING else ()
F_RI = sorted(set((1, int(np.log2(M) + 1))))  # Diabetes: [1, 4]
F_RC = (2, 8)

# Theo thứ tự biến trong paper: 2=Glucose, 6=BMI, 8=Age.
PAPER_SUBSETS = {
    '2': (1,),
    '2+8': (1, 7),
    '2+6': (1, 5),
    'All 8': tuple(range(M)),
}


def split(seed):
    return split_and_impute(
        X,
        y,
        seed=seed,
        test_size=0.1,
        zero_missing_indices=ZERO_INDICES,
    )


def err(y_true, pred):
    return 1.0 - accuracy_score(y_true, pred)


def fit_ri_error(X_tr, y_tr, X_te, y_te, F, seed, n_trees=None):
    if n_trees is None:
        n_trees = N_TREES
    rf = fit_forest_ri(X_tr, y_tr, F=F, n_trees=n_trees, seed=seed)
    test_error = err(y_te, majority_vote(tree_votes(rf, X_te)))
    return test_error, rf


def best_ri(X_tr, y_tr, X_te, y_te, seed):
    """Chọn F của Forest-RI bằng OOB trên train, không dùng test để chọn."""
    candidates = []
    for F in F_RI:
        rf = fit_forest_ri(X_tr, y_tr, F=F, n_trees=N_TREES, seed=seed)
        mask = oob_mask(rf, len(y_tr))
        votes_tr = tree_votes(rf, X_tr)
        oob = oob_error(votes_tr, mask, y_tr)
        candidates.append((oob, F, rf, votes_tr, mask))

    _, F, rf, votes_tr, mask = min(candidates, key=lambda t: (t[0], t[1]))
    test_error = err(y_te, majority_vote(tree_votes(rf, X_te)))
    return test_error, F, rf, votes_tr, mask


def fit_rc_candidate(X_tr, y_tr, X_te, y_te, F, seed, n_trees=None):
    if n_trees is None:
        n_trees = N_TREES
    trees, mask = fit_forest_rc(
        X_tr,
        y_tr,
        F=F,
        L=3,
        n_trees=n_trees,
        seed=seed,
    )
    votes_tr = tree_votes_rc(trees, X_tr)
    votes_te = tree_votes_rc(trees, X_te)
    return {
        'F': F,
        'trees': trees,
        'mask': mask,
        'votes_tr': votes_tr,
        'test_error': err(y_te, majority_vote(votes_te)),
        'oob_error': oob_error(votes_tr, mask, y_tr),
        'one_tree_error': one_tree_error(votes_tr, mask, y_tr),
    }


def best_rc(X_tr, y_tr, X_te, y_te, seed):
    """Chọn F của Forest-RC bằng OOB trên train."""
    candidates = [
        fit_rc_candidate(X_tr, y_tr, X_te, y_te, F=F, seed=seed)
        for F in F_RC
    ]
    best = min(candidates, key=lambda d: (d['oob_error'], d['F']))
    return best, candidates


def boost_error(X_tr, y_tr, X_te, y_te, depth, seed):
    model = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=depth, random_state=seed),
        n_estimators=5 if QUICK else 50,
        random_state=seed,
    )
    model.fit(X_tr, y_tr)
    return err(y_te, model.predict(X_te))


def make_noisy_labels(y_tr, seed, fraction=0.05):
    y_noisy = y_tr.copy()
    if fraction <= 0:
        return y_noisy
    rng = np.random.default_rng(100000 + seed)
    n_flip = round(fraction * len(y_noisy))
    flip = rng.choice(len(y_noisy), size=n_flip, replace=False)
    y_noisy[flip] = 1 - y_noisy[flip]
    return y_noisy


def save_progress(rows2, rows3, rows4, rows_diag, rows_subsets):
    pd.DataFrame(rows2).to_csv(RESULTS / 'results_diabetes_table2.csv', index=False)
    pd.DataFrame(rows3).to_csv(RESULTS / 'results_diabetes_table3.csv', index=False)
    pd.DataFrame(rows4).to_csv(RESULTS / 'results_diabetes_table4.csv', index=False)
    pd.DataFrame(rows_diag).to_csv(RESULTS / 'diabetes_oob_diagnostics.csv', index=False)
    pd.DataFrame(rows_subsets).to_csv(RESULTS / 'paper_feature_subsets.csv', index=False)


def oob_permutation_importance_sklearn(rf, X_tr, y_tr, mask, seed):
    """Permutation importance kiểu paper: hoán vị từng biến chỉ trên OOB của từng cây."""
    base_votes = tree_votes(rf, X_tr)
    baseline = oob_error(base_votes, mask, y_tr)
    rows = []

    for j, name in enumerate(feature_names):
        rng = np.random.default_rng(seed + 1000 + j)
        votes_for_1 = np.zeros(len(y_tr), dtype=float)
        counts = np.zeros(len(y_tr), dtype=int)

        for k, tree in enumerate(rf.estimators_):
            idx = np.flatnonzero(mask[k])
            if len(idx) == 0:
                continue
            X_oob = X_tr[idx].copy()
            X_oob[:, j] = X_oob[rng.permutation(len(idx)), j]
            pred = tree.predict(X_oob).astype(int)
            votes_for_1[idx] += pred
            counts[idx] += 1

        ok = counts > 0
        pred = (votes_for_1[ok] / counts[ok] > 0.5).astype(int)
        perm_error = np.mean(pred != y_tr[ok])
        pct = 100.0 * (perm_error - baseline) / baseline if baseline > 0 else np.nan
        rows.append({
            'variable_number': j + 1,
            'feature': name,
            'baseline_oob_error': baseline,
            'permuted_oob_error': perm_error,
            'percent_increase_error': pct,
        })
    return pd.DataFrame(rows)


def oob_permutation_importance_rc(trees, X_tr, y_tr, mask, seed):
    base_votes = tree_votes_rc(trees, X_tr)
    baseline = oob_error(base_votes, mask, y_tr)
    rows = []

    for j, name in enumerate(feature_names):
        rng = np.random.default_rng(seed + 2000 + j)
        votes_for_1 = np.zeros(len(y_tr), dtype=float)
        counts = np.zeros(len(y_tr), dtype=int)

        for k, tree in enumerate(trees):
            idx = np.flatnonzero(mask[k])
            if len(idx) == 0:
                continue
            X_oob = X_tr[idx].copy()
            X_oob[:, j] = X_oob[rng.permutation(len(idx)), j]
            pred = tree.predict(X_oob).astype(int)
            votes_for_1[idx] += pred
            counts[idx] += 1

        ok = counts > 0
        pred = (votes_for_1[ok] / counts[ok] > 0.5).astype(int)
        perm_error = np.mean(pred != y_tr[ok])
        pct = 100.0 * (perm_error - baseline) / baseline if baseline > 0 else np.nan
        rows.append({
            'variable_number': j + 1,
            'feature': name,
            'baseline_oob_error': baseline,
            'permuted_oob_error': perm_error,
            'percent_increase_error': pct,
        })
    return pd.DataFrame(rows)


def plot_importance(df, title, out_name):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df['variable_number'], df['percent_increase_error'], marker='D', linestyle='None')
    ax.axhline(0, linewidth=0.8)
    ax.set_xticks(range(1, M + 1))
    ax.set_xlabel('Variable')
    ax.set_ylabel('Percent increase in OOB error')
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(RESULTS / out_name, dpi=180)
    plt.close(fig)


def make_summary(df, value_col):
    return df.groupby('method')[value_col].agg(['mean', 'std'])


rows2, rows3, rows4, rows_diag, rows_subsets = [], [], [], [], []

for seed in SEEDS:
    X_tr, X_te, y_tr, y_te = split(seed)

    # ---------------- Table 2: Forest-RI ----------------
    selection_error, chosen_F, rf, votes_tr, mask = best_ri(
        X_tr, y_tr, X_te, y_te, seed
    )
    rows2.append({
        'seed': seed,
        'method': 'Selection',
        'F': chosen_F,
        'error_rate': selection_error,
    })

    single_error, _ = fit_ri_error(X_tr, y_tr, X_te, y_te, F=1, seed=seed)
    rows2.append({
        'seed': seed,
        'method': 'Single Input',
        'F': 1,
        'error_rate': single_error,
    })

    # One Tree đúng theo paper: lỗi OOB từng cây của cấu hình RI được chọn.
    rows2.append({
        'seed': seed,
        'method': 'One Tree',
        'F': chosen_F,
        'error_rate': one_tree_error(votes_tr, mask, y_tr),
    })

    # Giữ CART độc lập làm baseline bổ sung, không dùng để so cột One Tree của paper.
    cart = DecisionTreeClassifier(random_state=seed).fit(X_tr, y_tr)
    rows2.append({
        'seed': seed,
        'method': 'Single CART baseline',
        'F': np.nan,
        'error_rate': err(y_te, cart.predict(X_te)),
    })

    for depth in (1, 3, 5):
        rows2.append({
            'seed': seed,
            'method': f'AdaBoost depth={depth}',
            'F': np.nan,
            'error_rate': boost_error(X_tr, y_tr, X_te, y_te, depth, seed),
        })

    strength, correlation = strength_correlation(votes_tr, mask, y_tr)
    rows_diag.append({
        'seed': seed,
        'F': chosen_F,
        'oob_error': oob_error(votes_tr, mask, y_tr),
        'one_tree_oob_error': one_tree_error(votes_tr, mask, y_tr),
        'strength': strength,
        'correlation': correlation,
        'correlation_over_strength_sq': (
            correlation / strength**2 if abs(strength) > 1e-12 else np.nan
        ),
    })

    # ---------------- Table 3: Forest-RC ----------------
    scaler = StandardScaler().fit(X_tr)
    X_rc_tr = scaler.transform(X_tr)
    X_rc_te = scaler.transform(X_te)

    best_rc_result, rc_candidates = best_rc(
        X_rc_tr, y_tr, X_rc_te, y_te, seed
    )
    rc_by_f = {d['F']: d for d in rc_candidates}

    rows3.append({
        'seed': seed,
        'method': 'Selection',
        'F': best_rc_result['F'],
        'L': 3,
        'error_rate': best_rc_result['test_error'],
        'oob_error': best_rc_result['oob_error'],
    })
    rows3.append({
        'seed': seed,
        'method': 'Two Features',
        'F': 2,
        'L': 3,
        'error_rate': rc_by_f[2]['test_error'],
        'oob_error': rc_by_f[2]['oob_error'],
    })
    rows3.append({
        'seed': seed,
        'method': 'Eight Features',
        'F': 8,
        'L': 3,
        'error_rate': rc_by_f[8]['test_error'],
        'oob_error': rc_by_f[8]['oob_error'],
    })
    rows3.append({
        'seed': seed,
        'method': 'One Tree',
        'F': best_rc_result['F'],
        'L': 3,
        'error_rate': best_rc_result['one_tree_error'],
        'oob_error': best_rc_result['oob_error'],
    })

    # ---------------- Table 4: 5% label noise ----------------
    # Giữ F đã chọn từ dữ liệu sạch để việc so clean/noise dùng cùng cấu hình.
    rows4.append({
        'seed': seed,
        'noise_fraction': 0.0,
        'method': 'Forest-RI',
        'F': chosen_F,
        'test_error_rate': selection_error,
    })
    rows4.append({
        'seed': seed,
        'noise_fraction': 0.0,
        'method': 'Forest-RC',
        'F': best_rc_result['F'],
        'test_error_rate': best_rc_result['test_error'],
    })

    y_noisy = make_noisy_labels(y_tr, seed, 0.05)

    noisy_ri_error, _ = fit_ri_error(
        X_tr, y_noisy, X_te, y_te, F=chosen_F, seed=seed
    )
    rows4.append({
        'seed': seed,
        'noise_fraction': 0.05,
        'method': 'Forest-RI',
        'F': chosen_F,
        'test_error_rate': noisy_ri_error,
    })

    noisy_rc = fit_rc_candidate(
        X_rc_tr,
        y_noisy,
        X_rc_te,
        y_te,
        F=best_rc_result['F'],
        seed=seed,
    )
    rows4.append({
        'seed': seed,
        'noise_fraction': 0.05,
        'method': 'Forest-RC',
        'F': best_rc_result['F'],
        'test_error_rate': noisy_rc['test_error'],
    })

    # ---------------- Paper feature subsets ----------------
    for subset_name, indices in PAPER_SUBSETS.items():
        idx = np.asarray(indices, dtype=int)
        sub_error, _ = fit_ri_error(
            X_tr[:, idx],
            y_tr,
            X_te[:, idx],
            y_te,
            F=1,
            seed=seed,
        )
        rows_subsets.append({
            'seed': seed,
            'subset': subset_name,
            'features': ','.join(feature_names[i] for i in indices),
            'F': 1,
            'error_rate': sub_error,
        })

    if (seed + 1) % 10 == 0 or seed == N_REPEATS - 1:
        save_progress(rows2, rows3, rows4, rows_diag, rows_subsets)

    print(f'Hoàn thành seed {seed + 1}/{N_REPEATS}', flush=True)


# ---------------- Lưu bảng chính ----------------
save_progress(rows2, rows3, rows4, rows_diag, rows_subsets)

df2 = pd.DataFrame(rows2)
df3 = pd.DataFrame(rows3)
df4 = pd.DataFrame(rows4)
df_diag = pd.DataFrame(rows_diag)
df_sub = pd.DataFrame(rows_subsets)

# Tóm tắt Table 4: % tăng lỗi trung bình do noise.
noise_pivot = (
    df4.groupby(['method', 'noise_fraction'])['test_error_rate']
    .mean()
    .unstack('noise_fraction')
)
noise_summary = noise_pivot.rename(columns={0.0: 'clean_error', 0.05: 'noise_error'})
noise_summary['relative_increase_percent'] = (
    (noise_summary['noise_error'] - noise_summary['clean_error'])
    / noise_summary['clean_error']
    * 100.0
)
noise_summary.reset_index().to_csv(
    RESULTS / 'results_diabetes_table4_summary.csv', index=False
)

# Tóm tắt đúng các tổ hợp biến của paper.
subset_summary = (
    df_sub.groupby(['subset', 'features'])['error_rate']
    .agg(['mean', 'std'])
    .reset_index()
)
subset_summary.to_csv(RESULTS / 'paper_feature_subsets_summary.csv', index=False)


# ---------------- Hình 4/5: OOB variable importance, 1.000 cây ----------------
# Paper dùng một lần chạy rừng lớn; ở đây dùng seed=0 để tái lập được.
X_tr0, X_te0, y_tr0, y_te0 = split(0)

# Hình 4: Forest-RI, F=1, 1.000 cây.
rf_imp = fit_forest_ri(
    X_tr0,
    y_tr0,
    F=1,
    n_trees=N_IMPORTANCE_TREES,
    seed=0,
)
ri_mask = oob_mask(rf_imp, len(y_tr0))
ri_imp = oob_permutation_importance_sklearn(rf_imp, X_tr0, y_tr0, ri_mask, seed=0)
ri_imp.to_csv(RESULTS / 'variable_importance_figure4_ri.csv', index=False)
plot_importance(
    ri_imp,
    f'Diabetes - OOB Variable Importance (Forest-RI, F=1, {N_IMPORTANCE_TREES} trees)',
    'figure4_variable_importance_ri.png',
)

# Hình 5: Forest-RC, L=3, F=2, 1.000 cây.
scaler0 = StandardScaler().fit(X_tr0)
X_rc_tr0 = scaler0.transform(X_tr0)
rc_trees_imp, rc_mask_imp = fit_forest_rc(
    X_rc_tr0,
    y_tr0,
    F=2,
    L=3,
    n_trees=N_IMPORTANCE_TREES,
    seed=0,
)
rc_imp = oob_permutation_importance_rc(
    rc_trees_imp,
    X_rc_tr0,
    y_tr0,
    rc_mask_imp,
    seed=0,
)
rc_imp.to_csv(RESULTS / 'variable_importance_figure5_rc.csv', index=False)
plot_importance(
    rc_imp,
    f'Diabetes - OOB Variable Importance (Forest-RC, F=2, L=3, {N_IMPORTANCE_TREES} trees)',
    'figure5_variable_importance_rc.png',
)


# ---------------- In kết quả tóm tắt ----------------
print('\nTABLE 2:')
print(make_summary(df2, 'error_rate'))

print('\nTABLE 3:')
print(make_summary(df3, 'error_rate'))

print('\nTABLE 4 - relative increase (%):')
print(noise_summary)

print('\nOOB DIAGNOSTICS (mean):')
print(df_diag.select_dtypes(include=[np.number]).mean())

print('\nPAPER FEATURE SUBSETS:')
print(subset_summary)

print('\nFIGURE 4 - RI variable importance:')
print(ri_imp[['variable_number', 'feature', 'percent_increase_error']])

print('\nFIGURE 5 - RC variable importance:')
print(rc_imp[['variable_number', 'feature', 'percent_increase_error']])

print('\nĐã xuất kết quả tại:', RESULTS)
