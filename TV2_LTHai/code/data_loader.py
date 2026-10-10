"""Đọc bộ dữ liệu Pima Indians Diabetes cho TV2.

Không xử lý giá trị 0 hoặc điền dữ liệu thiếu ở bước đọc; các thao tác
đó phải được fit trên tập train sau khi chia dữ liệu để tránh leakage.
"""
from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age',
]
TARGET_COLUMN = 'Outcome'


def load_diabetes(csv_path=None):
    """Trả về (X, y, feature_names), với X/y là numpy arrays."""
    if csv_path is None:
        csv_path = Path(__file__).resolve().parent.parent / 'data' / 'diabetes.csv'
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f'Không tìm thấy dữ liệu: {csv_path}')

    df = pd.read_csv(csv_path)
    required = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f'Dataset thiếu cột: {missing}')

    X = df[FEATURE_COLUMNS].apply(pd.to_numeric, errors='raise').to_numpy(dtype=float)
    if not np.isfinite(X[~np.isnan(X)]).all():
        raise ValueError('Dữ liệu chứa giá trị vô cực')
    y = pd.to_numeric(df[TARGET_COLUMN], errors='raise').to_numpy()
    if pd.isna(y).any() or not set(np.unique(y)).issubset({0, 1}):
        raise ValueError('Outcome chỉ được chứa nhãn 0 hoặc 1')
    return X, y.astype(int), FEATURE_COLUMNS.copy()


if __name__ == '__main__':
    X, y, names = load_diabetes()
    print('X:', X.shape, '| y:', y.shape)
    print('Features:', names)
