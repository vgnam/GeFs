"""Quick smoke-test for Deep GeF (D-GeF) using synthetic data."""
import numpy as np
from gefs import RandomForest
from gefs.nodes import eval_root

# ---- Create synthetic mixed data ----
np.random.seed(0)
n_samples = 500
n_features = 5

# 3 continuous, 2 categorical (3 cats each)
X_cont = np.random.randn(n_samples, 3)
X_cat = np.random.randint(0, 3, size=(n_samples, 2))
X = np.hstack([X_cont, X_cat])

# Simple class label based on first two features
y = (X[:, 0] + X[:, 1] > 0).astype(np.int64)

# Combine into GeF-format data (class last)
data = np.hstack([X, y[:, None]]).astype(np.float64)
ncat = np.array([1, 1, 1, 3, 3, 2], dtype=np.int64)

# Train/test split
split = int(0.8 * n_samples)
data_train = data[:split]
data_test = data[split:]
X_train = data_train[:, :-1]
y_train = data_train[:, -1].astype(np.int64)
X_test = data_test[:, :-1]
y_test = data_test[:, -1].astype(np.int64)

# Train RF
print("Training RF...")
rf = RandomForest(ncat=ncat, n_estimators=5, max_features=3, random_state=42)
rf.fit(X_train, y_train)

# Standard GeF
print("\nBuilding standard GeF...")
gef = rf.topc()
ll_gef_train = gef.log_likelihood(data_train).mean()
ll_gef_test = gef.log_likelihood(data_test).mean()
print(f"Standard GeF LL (train): {ll_gef_train:.4f}")
print(f"Standard GeF LL (test) : {ll_gef_test:.4f}")

# Deep GeF
print("\nBuilding Deep GeF...")
dgef = rf.topc(deep=True, data=data_train, n_groups=3, n_em_iter=5)
ll_dgef_train = dgef.log_likelihood(data_train).mean()
ll_dgef_test = dgef.log_likelihood(data_test).mean()
print(f"Deep GeF LL (train): {ll_dgef_train:.4f}")
print(f"Deep GeF LL (test) : {ll_dgef_test:.4f}")

# Classification
y_pred_gef = gef.classify_avg(X_test)
y_pred_dgef = dgef.classify_avg(X_test)
acc_gef = np.mean(y_pred_gef == y_test)
acc_dgef = np.mean(y_pred_dgef == y_test)
print(f"\nAccuracy standard GeF: {acc_gef:.4f}")
print(f"Accuracy Deep GeF    : {acc_dgef:.4f}")

# Check that D-GeF is a valid PC (can evaluate root)
print("\nEvaluating D-GeF root directly...")
val = eval_root(dgef.root, data_test[:5])
print("Direct eval shape:", val.shape, "values:", val)

print("\n=== Smoke test passed! ===")
