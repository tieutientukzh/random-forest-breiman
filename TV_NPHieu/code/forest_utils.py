import numpy as np
from sklearn.ensemble import RandomForestClassifier


def fit_forest_ri(X_tr, y_tr, F, n_trees=100, seed=0):
    """Forest-RI: bagging + chọn ngẫu nhiên F biến tại mỗi nút.
    Cây mọc tối đa (max_depth=None, min_samples_leaf=1), không cắt tỉa, tiêu chí Gini."""
    rf = RandomForestClassifier(
        n_estimators=n_trees,
        max_features=F,        # F biến được xét tại mỗi nút
        bootstrap=True,        # mỗi cây học trên một mẫu bootstrap
        criterion="gini",
        max_depth=None,
        min_samples_leaf=1,
        random_state=seed,
    )
    rf.fit(X_tr, y_tr)
    return rf


def tree_votes(rf, X):
    """Phiếu của từng cây: ma trận (số cây x số mẫu), mỗi ô là lớp 0 hoặc 1."""
    return np.array([tree.predict(X) for tree in rf.estimators_]).astype(int)


def majority_vote(votes):
    """Bỏ phiếu đa số (mỗi cây một phiếu). Hòa phiếu thì chọn lớp 0."""
    return (votes.mean(axis=0) > 0.5).astype(int)


def oob_mask(rf, n_train):
    """Ma trận (số cây x n_train): True nếu mẫu i KHÔNG có trong mẫu bootstrap của cây k."""
    mask = np.ones((len(rf.estimators_), n_train), dtype=bool)
    for k, in_bag in enumerate(rf.estimators_samples_):
        mask[k, in_bag] = False
    return mask


def oob_error(votes_tr, mask, y_tr):
    """OOB error: với mỗi mẫu train, chỉ các cây chưa học mẫu đó được bỏ phiếu."""
    n_oob_trees = mask.sum(axis=0)              # mỗi mẫu là OOB ở bao nhiêu cây
    votes_for_1 = (votes_tr * mask).sum(axis=0)  # số phiếu OOB cho lớp 1
    ok = n_oob_trees > 0                         # bỏ qua mẫu không có cây OOB nào (hầu như không xảy ra)
    pred = (votes_for_1[ok] / n_oob_trees[ok] > 0.5).astype(int)
    return np.mean(pred != y_tr[ok])


def one_tree_error(votes_tr, mask, y_tr):
    """Cột 'One Tree': lỗi của từng cây trên chính các mẫu OOB của nó, lấy trung bình qua các cây."""
    errs = [np.mean(votes_tr[k, mask[k]] != y_tr[mask[k]]) for k in range(len(votes_tr))]
    return np.mean(errs)
def strength_correlation(votes_tr, mask, y_tr):
    """Ước lượng strength s và correlation rho bằng OOB, theo Phụ lục II của paper
    (bài toán 2 lớp, nhãn 0/1)."""
    n_oob = mask.sum(axis=0)
    ok = n_oob > 0
    v, m, y = votes_tr[:, ok], mask[:, ok], y_tr[ok]

    # Q(x, j): tỷ lệ phiếu OOB cho lớp j tại x
    q1 = (v * m).sum(axis=0) / m.sum(axis=0)
    q_true = np.where(y == 1, q1, 1 - q1)          # Q(x, y): phiếu cho lớp đúng
    mr = q_true - (1 - q_true)                     # margin: lớp đúng - lớp sai còn lại

    s = mr.mean()                                  # strength
    var_mr = (mr ** 2).mean() - s ** 2             # phương sai của margin (A1)

    # sd(Theta_k) cho từng cây, chỉ tính trên các mẫu OOB của cây đó (A2)
    sds = []
    for k in range(len(v)):
        p1 = np.mean(v[k, m[k]] == y[m[k]])        # cây k đoán đúng
        p2 = 1 - p1                                # cây k đoán lớp sai (2 lớp)
        sds.append(np.sqrt(p1 + p2 - (p1 - p2) ** 2))   # dấu TRỪ (bản PDF in nhầm dấu +)
    rho = var_mr / np.mean(sds) ** 2               # correlation trung bình

    return s, rho