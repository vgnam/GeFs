"""Debug MI and Chow-Liu on synthetic data."""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from gefs import RandomForest
from gefs.dgef import get_tree_leaf_ids, compute_mi_matrix

# ---- Create synthetic mixed data ----
np.random.seed(0)
n_samples = 500
n_features = 5
X_cont = np.random.randn(n_samples, 3)
X_cat = np.random.randint(0, 3, size=(n_samples, 2))
X = np.hstack([X_cont, X_cat])
y = (X[:, 0] + X[:, 1] > 0).astype(np.int64)
data = np.hstack([X, y[:, None]]).astype(np.float64)
ncat = np.array([1, 1, 1, 3, 3, 2], dtype=np.int64)
split = int(0.8 * n_samples)
data_train = data[:split]
X_train = data_train[:, :-1]
y_train = data_train[:, -1].astype(np.int64)

rf = RandomForest(ncat=ncat, n_estimators=5, max_features=3, random_state=42)
rf.fit(X_train, y_train)

leaf_ids = np.zeros((data_train.shape[0], rf.n_estimators), dtype=np.int64)
for t in range(rf.n_estimators):
    leaf_ids[:, t] = get_tree_leaf_ids(rf.estimators[t], X_train)

mi = compute_mi_matrix(leaf_ids)
print("MI matrix:\n", mi)

n = mi.shape[0]
cost = -mi
np.fill_diagonal(cost, 0.0)
csr = csr_matrix(cost)
mst = minimum_spanning_tree(csr)
print("MST data:\n", mst.data)
print("MST row:\n", mst.row)
print("MST col:\n", mst.col)
print("MST toarray():\n", mst.toarray())

rows, cols = mst.nonzero()
print("nonzero rows:", rows)
print("nonzero cols:", cols)

edges = []
for idx in range(len(rows)):
    i, j = int(rows[idx]), int(cols[idx])
    if i < j:
        w = mi[i, j]
        edges.append((i, j, w))
print("edges:", edges)
