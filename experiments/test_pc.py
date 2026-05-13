"""
Minimal smoke test for PC via py-tetrad/Tetrad.

Requirements:
    - JDK 21+ installed, JAVA_HOME set or discoverable by fges_compat.py
    - pip install JPype1
    - pip install git+https://github.com/cmu-phil/py-tetrad

Run:
    python experiments/test_pc.py
"""

import os
import sys
import time

import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from fges_compat import TetradSearch

print("=== PC smoke test ===")

print("[1/3] importing Tetrad compatibility wrapper ...", flush=True)
try:
    ts = TetradSearch
    print("      OK")
except Exception as e:
    print(f"      FAILED: {e}")
    raise SystemExit(1)

print("[2/3] generating synthetic data n=500, d=5 ...", flush=True)
rng = np.random.default_rng(42)
d = 5
W_true = np.zeros((d, d))
for i in range(d - 1):
    W_true[i, i + 1] = rng.uniform(0.5, 1.5)

X = np.zeros((500, d))
for i in range(d):
    noise = rng.standard_normal(500)
    X[:, i] = X @ W_true[:, i] + noise

df = pd.DataFrame(X, columns=[f"x{i}" for i in range(d)])
df = df.astype({col: "float64" for col in df.columns})
print(f"      DataFrame shape={df.shape}")

print("[3/3] running PC with Fisher Z test ...", flush=True)
t0 = time.perf_counter()
search = ts(df)
search.set_verbose(False)
search.run_pc(alpha=0.01, stable=True)
elapsed = time.perf_counter() - t0

result = search.get_graph_to_matrix()
print(f"      done in {elapsed:.2f}s")
print(f"      result type: {type(result)}")
print(f"      estimated graph:\n{result}")
print("\nPC smoke test PASSED")
