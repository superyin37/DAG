"""
Minimal smoke test for NOTEARS via gcastle.

Install:
    pip install gcastle

Run:
    python experiments/test_notears.py
"""

import time
import numpy as np

print("=== NOTEARS smoke test ===")

# --- 1. import ---
print("[1/3] importing castle.algorithms.Notears ...", flush=True)
try:
    from castle.algorithms import Notears
    print("      OK")
except Exception as e:
    print(f"      FAILED: {e}")
    raise SystemExit(1)

# --- 2. generate tiny synthetic data (n=200, d=5) ---
print("[2/3] generating synthetic data n=200, d=5 ...", flush=True)
rng = np.random.default_rng(42)
d = 5
# simple chain DAG: 0->1->2->3->4
W_true = np.zeros((d, d))
for i in range(d - 1):
    W_true[i, i + 1] = rng.uniform(0.5, 1.5)
X = np.zeros((200, d))
for i in range(d):
    noise = rng.standard_normal(200)
    X[:, i] = X @ W_true[:, i] + noise
print(f"      X.shape={X.shape}")

# --- 3. run ---
print("[3/3] running Notears(lambda1=0.1, loss_type='l2') ...", flush=True)
t0 = time.perf_counter()
model = Notears(lambda1=0.1, loss_type="l2", w_threshold=0.3)
model.learn(X)
elapsed = time.perf_counter() - t0

print(f"      done in {elapsed:.2f}s")
print(f"      causal_matrix shape: {model.causal_matrix.shape}")
print(f"      estimated edges:\n{model.causal_matrix}")
print("\nNOTEARS smoke test PASSED")
