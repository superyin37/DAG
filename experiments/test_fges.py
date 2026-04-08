"""
Minimal smoke test for FGES via py-tetrad.

Requirements:
    - JDK 21+ installed, JAVA_HOME set
    - pip install JPype1
    - pip install git+https://github.com/cmu-phil/py-tetrad

Run:
    python experiments/test_fges.py
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

print("=== FGES smoke test ===")

if not os.environ.get("JAVA_HOME"):
    jdk_candidate = r"C:\Program Files\Microsoft\jdk-21.0.10.7-hotspot"
    if os.path.isdir(jdk_candidate):
        os.environ["JAVA_HOME"] = jdk_candidate

# --- 1. import ---
print("[1/3] importing FGES compatibility wrapper ...", flush=True)
try:
    ts = TetradSearch
    print("      OK")
except Exception as e:
    print(f"      FAILED: {e}")
    raise SystemExit(1)

# --- 2. generate tiny synthetic data (n=200, d=5) ---
print("[2/3] generating synthetic data n=200, d=5 ...", flush=True)
rng = np.random.default_rng(42)
d = 5
W_true = np.zeros((d, d))
for i in range(d - 1):
    W_true[i, i + 1] = rng.uniform(0.5, 1.5)
X = np.zeros((200, d))
for i in range(d):
    noise = rng.standard_normal(200)
    X[:, i] = X @ W_true[:, i] + noise

df = pd.DataFrame(X, columns=[f"x{i}" for i in range(d)])
df = df.astype({col: "float64" for col in df.columns})
print(f"      DataFrame shape={df.shape}")

# --- 3. run ---
print("[3/3] running FGES with SEM BIC score ...", flush=True)
t0 = time.perf_counter()
search = ts(df)
search.set_verbose(False)
search.use_sem_bic()
search.run_fges()
elapsed = time.perf_counter() - t0

result = search.get_graph_to_matrix()
print(f"      done in {elapsed:.2f}s")
print(f"      result type: {type(result)}")
print(f"      estimated graph:\n{result}")
print("\nFGES smoke test PASSED")
