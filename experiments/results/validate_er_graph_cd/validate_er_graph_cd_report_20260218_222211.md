# ER-Graph Validation Report

- Generated at: 2026-02-18 22:22:11
- Run ID: 20260218_222211
- Trials: 100, n=10000, d=5, degree=2.0
- Noise: gaussian_nv, threshold=0.05, lambda_l0_list=[0.0, 0.1, 0.2]

## Summary (mean metrics)

| algorithm        |   mec_match_mean |   exact_match_mean |   cpdag_shd_mean |   cpdag_shd_std |   runtime_sec_mean |
|:-----------------|-----------------:|-------------------:|-----------------:|----------------:|-------------------:|
| cd_A_l0_0.0      |             0.18 |               0.17 |             2.89 |         2.45318 |          3.85604   |
| cd_A_l0_0.1      |             0.25 |               0.24 |             3.17 |         2.64787 |          3.69989   |
| cd_A_l0_0.2      |             0.22 |               0.22 |             3.28 |         2.6365  |          3.63121   |
| cd_BOmega_l0_0.0 |             0.2  |               0.17 |             3.25 |         2.58346 |          2.71339   |
| cd_BOmega_l0_0.1 |             0.17 |               0.17 |             3.67 |         2.51884 |          1.51562   |
| cd_BOmega_l0_0.2 |             0.16 |               0.16 |             3.71 |         2.40494 |          1.4273    |
| cd_B_l0_0.0      |             0.18 |               0.14 |             3.72 |         2.72319 |          2.7363    |
| cd_B_l0_0.1      |             0.15 |               0.15 |             3.9  |         2.43916 |          1.36263   |
| cd_B_l0_0.2      |             0.15 |               0.15 |             3.93 |         2.49547 |          1.3587    |
| ges              |             0.36 |               0.07 |             0.77 |         1.85241 |          0.136974  |
| golem_ev         |             0.26 |               0.26 |             2.74 |         2.63856 |         32.9557    |
| golem_nv         |             0.04 |               0    |             5.96 |         2.5183  |         27.5181    |
| sp               |             0.25 |               0.1  |             3.83 |         2.85351 |          0.0234658 |


## Failure stats

- Failure count: 1100
- Failure CSV: c:\Users\super\DAG\experiments\results\validate_er_graph_cd_failures.csv
- Failure JSON: c:\Users\super\DAG\experiments\results\validate_er_graph_cd_failures.json