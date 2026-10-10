# Tên file: TV2_LTHai/diabetes_random_forest.py
"""Thực nghiệm Random Forest trên Pima Indians Diabetes.
Bản triển khai tham khảo theo cấu trúc 10 Cell trong hướng dẫn TV2.
Không khẳng định tái hiện chính xác các bảng trong bài báo Breiman.
"""

# %% Cell 1: Cấu hình và đọc dữ liệu
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance

QUICK = False
SEED = 42
N_TREES = 30 if QUICK else 200
N_REPEATS = 2 if QUICK else 10
ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
RESULTS.mkdir(parents=True, exist_ok=True)
DATA_CANDIDATES = [ROOT / 'data' / 'diabetes.csv']
if Path('/kaggle/input').exists():
    DATA_CANDIDATES += list(Path('/kaggle/input').rglob('diabetes.csv'))
DATA_FILE = next((p for p in DATA_CANDIDATES if p.is_file()), None)
if DATA_FILE is None:
    raise FileNotFoundError('Không tìm thấy TV2_LTHai/data/diabetes.csv')
df = pd.read_csv(DATA_FILE)
required = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']
if not set(required).issubset(df.columns):
    raise ValueError(f'Thiếu cột: {sorted(set(required)-set(df.columns))}')
df = df[required].copy()
if df.isna().any().any():
    raise ValueError('Dữ liệu có giá trị NaN; cần kiểm tra trước khi chạy')
X_raw = df.drop(columns='Outcome').astype(float)
y = df['Outcome'].astype(int).to_numpy()
if not set(np.unique(y)).issubset({0,1}):
    raise ValueError('Outcome phải là nhãn 0 hoặc 1')
print('Python:', sys.version.split()[0], 'scikit-learn:', sklearn.__version__)
print('Dữ liệu:', DATA_FILE, 'Kích thước:', df.shape, 'QUICK:', QUICK)

# %% Cell 2: Tiền xử lý, Forest-RI, OOB, Strength, Correlation
ZERO_MISSING = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']

def impute_train_test(x_train, x_test):
    """Tính median CHỈ trên tập train để tránh rò rỉ dữ liệu."""
    a, b = x_train.copy(), x_test.copy()
    for col in ZERO_MISSING:
        med = a[col].replace(0, np.nan).median()
        if pd.isna(med):
            raise ValueError(f'Cột {col} không có giá trị hợp lệ trong train')
        a[col] = a[col].replace(0, med)
        b[col] = b[col].replace(0, med)
    return a, b

class ForestRI:
    """Forest-RI tự cài phần bagging và lấy ngẫu nhiên biến tại mỗi split qua cây sklearn.
    Đây là hiện thực thuật toán ở mức ensemble, không tự viết thuật toán CART.
    """
    def __init__(self, n_estimators=100, max_features='sqrt', random_state=42):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.random_state = random_state

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError('Ví dụ này chỉ hỗ trợ phân loại nhị phân')
        rng = np.random.default_rng(self.random_state)
        self.trees_ = []
        self.inbag_ = []
        for _ in range(self.n_estimators):
            idx = rng.integers(0, len(y), size=len(y))
            tree = DecisionTreeClassifier(max_features=self.max_features,
                                          random_state=int(rng.integers(0, 2**31-1)))
            tree.fit(X[idx], y[idx])
            self.trees_.append(tree)
            self.inbag_.append(np.unique(idx))
        return self

    def predict_proba(self, X):
        X = np.asarray(X)
        votes = np.stack([t.predict(X) for t in self.trees_])
        p1 = (votes == self.classes_[1]).mean(axis=0)
        return np.column_stack([1-p1, p1])

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def oob_diagnostics(self, X, y):
        """OOB error, strength (mean correct-class margin), và tương quan margin của cây.
        Đây là phép ước lượng thực nghiệm; không đồng nhất với mọi định nghĩa trong bài báo.
        """
        X, y = np.asarray(X), np.asarray(y)
        n = len(y)
        vote_sum = np.zeros(n)
        vote_count = np.zeros(n, dtype=int)
        margins = np.full((len(self.trees_), n), np.nan)
        for k, (tree, inbag) in enumerate(zip(self.trees_, self.inbag_)):
            mask = np.ones(n, dtype=bool)
            mask[inbag] = False
            if not mask.any():
                continue
            pred = tree.predict(X[mask])
            correct = (pred == y[mask]).astype(float)
            vote_sum[mask] += correct
            vote_count[mask] += 1
            margins[k, mask] = 2*correct - 1
        valid = vote_count > 0
        if not valid.any():
            return {'oob_error': np.nan, 'strength': np.nan, 'correlation': np.nan, 'c_over_s2': np.nan}
        vote_fraction = vote_sum[valid] / vote_count[valid]
        oob_error = float(np.mean(vote_fraction <= 0.5))
        strength = float(np.mean(2*vote_fraction-1))
        corrs = []
        for i in range(len(self.trees_)):
            for j in range(i+1, len(self.trees_)):
                shared = np.isfinite(margins[i]) & np.isfinite(margins[j])
                if shared.sum() >= 3 and np.std(margins[i, shared]) > 0 and np.std(margins[j, shared]) > 0:
                    corrs.append(np.corrcoef(margins[i, shared], margins[j, shared])[0,1])
        corr = float(np.mean(corrs)) if corrs else np.nan
        return {'oob_error': oob_error, 'strength': strength,
                'correlation': corr, 'c_over_s2': corr/(strength**2) if abs(strength)>1e-12 else np.nan}


def make_model(kind, seed, n_trees=N_TREES):
    if kind == 'Selection':
        return RandomForestClassifier(n_estimators=n_trees, max_features='sqrt', random_state=seed, n_jobs=-1)
    if kind == 'Single Input':
        return RandomForestClassifier(n_estimators=n_trees, max_features=1, random_state=seed, n_jobs=-1)
    if kind == 'One Tree':
        return DecisionTreeClassifier(random_state=seed)
    if kind == 'Adaboost':
        return AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), n_estimators=n_trees, random_state=seed)
    raise ValueError(kind)

# %% Cell 3: Kiểm tra nhanh so với sklearn
xtr, xte, ytr, yte = train_test_split(X_raw, y, test_size=0.25, stratify=y, random_state=SEED)
xtr, xte = impute_train_test(xtr, xte)
custom = ForestRI(n_estimators=N_TREES, random_state=SEED).fit(xtr, ytr)
reference = RandomForestClassifier(n_estimators=N_TREES, max_features='sqrt', random_state=SEED, n_jobs=-1).fit(xtr, ytr)
print('Sanity check custom RF:', round(accuracy_score(yte, custom.predict(xte)),4),
      '| sklearn RF:', round(accuracy_score(yte, reference.predict(xte)),4))
assert custom.predict_proba(xte).shape == (len(xte), 2)
assert np.allclose(custom.predict_proba(xte).sum(axis=1), 1)

# %% Cell 4: Bảng 2 (bảng so sánh tham khảo, không phải tái hiện chính xác bài báo)
rows = []
for repeat in range(N_REPEATS):
    folds = StratifiedKFold(n_splits=3 if QUICK else 5, shuffle=True, random_state=SEED+repeat)
    for fold, (tr, te) in enumerate(folds.split(X_raw,y), 1):
        a,b = impute_train_test(X_raw.iloc[tr], X_raw.iloc[te])
        for name in ['Selection','Single Input','One Tree','Adaboost']:
            model = make_model(name, SEED+repeat*10+fold)
            model.fit(a,y[tr])
            rows.append({'repeat':repeat+1,'fold':fold,'method':name,
                         'error_rate':1-accuracy_score(y[te],model.predict(b))})
table2 = pd.DataFrame(rows)
table2.to_csv(RESULTS/'results_diabetes_table2.csv',index=False)
print('\nBảng 2 - tỷ lệ lỗi trung bình (tham khảo):')
print(table2.groupby('method')['error_rate'].agg(['mean','std']).round(4))

# %% Cell 5: Bảng 4 - thử nghiệm nhiễu nhãn 5%
rows = []
for repeat in range(N_REPEATS):
    tr, te = train_test_split(np.arange(len(y)), test_size=0.25, stratify=y, random_state=SEED+repeat)
    a,b = impute_train_test(X_raw.iloc[tr],X_raw.iloc[te])
    for noise in [0.0,0.05]:
        yy = y[tr].copy()
        rng = np.random.default_rng(SEED+repeat+int(noise*1000))
        if noise:
            flip = rng.choice(len(yy),size=round(noise*len(yy)),replace=False)
            yy[flip] = 1-yy[flip]
        for name in ['Selection','Single Input','One Tree','Adaboost']:
            model = make_model(name,SEED+repeat)
            model.fit(a,yy)
            rows.append({'repeat':repeat+1,'noise_fraction':noise,'method':name,
                         'test_error_rate':1-accuracy_score(y[te],model.predict(b))})
table4 = pd.DataFrame(rows)
table4.to_csv(RESULTS/'results_diabetes_table4.csv',index=False)
print('\nBảng 4 - nhiễu nhãn (tham khảo):')
print(table4.groupby(['noise_fraction','method'])['test_error_rate'].mean().round(4))

# %% Cell 6: Strength, Correlation, c/s² theo số biến F
for f in [1,2,3,round(np.sqrt(X_raw.shape[1])),X_raw.shape[1]]:
    forest = ForestRI(n_estimators=N_TREES,max_features=f,random_state=SEED).fit(xtr,ytr)
    d = forest.oob_diagnostics(xtr,ytr)
    print('F=',f, {k:round(v,4) for k,v in d.items()})

# %% Cell 7: Variable Importance (permutation importance trên hold-out)
perm = permutation_importance(reference,xte,yte,n_repeats=3 if QUICK else 15,
                              random_state=SEED,n_jobs=-1,scoring='accuracy')
importance = pd.Series(perm.importances_mean,index=X_raw.columns).sort_values()
fig,ax = plt.subplots(figsize=(9,5))
importance.plot.barh(ax=ax)
ax.set_title('Diabetes - Variable Importance (hold-out permutation)')
ax.set_xlabel('Mean decrease in accuracy')
fig.tight_layout()
fig.savefig(RESULTS/'variable_importance.png',dpi=160)
plt.close(fig)

# %% Cell 8: Chạy lại với tập con biến (selection chỉ trên train)
rows=[]
for k in [2,4,6,8]:
    selector = SelectKBest(score_func=f_classif,k=k).fit(xtr,ytr)
    a = selector.transform(xtr)
    b = selector.transform(xte)
    model = RandomForestClassifier(n_estimators=N_TREES,random_state=SEED,n_jobs=-1)
    model.fit(a,ytr)
    rows.append({'n_features':k,'features':','.join(xtr.columns[selector.get_support()]),
                 'accuracy':accuracy_score(yte,model.predict(b))})
subset_results = pd.DataFrame(rows)
subset_results.to_csv(RESULTS/'feature_subsets.csv',index=False)
print('\nTập con biến:')
print(subset_results.to_string(index=False))

# %% Cell 9: Forest-RC (chờ module Thành viên 3)
print('\nCell 9: Forest-RC chưa triển khai, chờ module của TV3.')

# %% Cell 10: Tổng kết file kết quả
print('\nCác file kết quả:')
for path in sorted(RESULTS.iterdir()):
    print('-',path.name)
print('\n5 dòng đầu Bảng 2:')
print(pd.read_csv(RESULTS/'results_diabetes_table2.csv').head().to_string(index=False))
