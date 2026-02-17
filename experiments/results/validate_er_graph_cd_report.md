# ER-Graph Validation Report

- Generated at: 2026-02-17 20:21:19
- Trials: 10, n=10000, d=5, degree=1.0
- Noise: gaussian_ev, threshold=0.05, lambda_l0=0.1

## Summary (mean metrics)

| algorithm   |   mec_match_mean |   exact_match_mean |   cpdag_shd_mean |   cpdag_shd_std |   runtime_sec_mean |   runtime_sec_std |
|:------------|-----------------:|-------------------:|-----------------:|----------------:|-------------------:|------------------:|
| cd_A        |              0.6 |                0.6 |              1.1 |         1.72884 |           0.978797 |         0.0652761 |
| cd_B        |              0.6 |                0.6 |              0.9 |         1.37032 |           0.737641 |         0.0349016 |
| cd_BOmega   |              0.6 |                0.6 |              0.9 |         1.37032 |           0.81426  |         0.037709  |


## Failure stats

- Failure count: 12
- Failure CSV: c:\Users\super\DAG\experiments\results\validate_er_graph_cd_failures.csv
- Failure JSON: c:\Users\super\DAG\experiments\results\validate_er_graph_cd_failures.json