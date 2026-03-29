# ER-Graph Validation Report

- Generated at: 2026-02-18 18:54:10
- Run ID: 20260218_185410
- Trials: 100, n=10000, d=5, degree=1.0
- Noise: gaussian_nv, threshold=0.05, lambda_l0_list=[0.0, 0.1, 0.2]

## Summary (mean metrics)

| algorithm        |   mec_match_mean |   exact_match_mean |   cpdag_shd_mean |   cpdag_shd_std |   runtime_sec_mean |
|:-----------------|-----------------:|-------------------:|-----------------:|----------------:|-------------------:|
| cd_A_l0_0.0      |             0.53 |               0.51 |             1.08 |         1.55492 |          5.59759   |
| cd_A_l0_0.1      |             0.61 |               0.59 |             0.98 |         1.52408 |          2.15481   |
| cd_A_l0_0.2      |             0.56 |               0.56 |             1.13 |         1.60589 |          2.13801   |
| cd_BOmega_l0_0.0 |             0.54 |               0.53 |             1.39 |         2.02457 |          1.4091    |
| cd_BOmega_l0_0.1 |             0.52 |               0.52 |             1.21 |         1.60363 |          1.27831   |
| cd_BOmega_l0_0.2 |             0.51 |               0.51 |             1.25 |         1.62291 |          1.25455   |
| cd_B_l0_0.0      |             0.43 |               0.39 |             1.67 |         2.14172 |         13.807     |
| cd_B_l0_0.1      |             0.49 |               0.49 |             1.27 |         1.60085 |          1.18437   |
| cd_B_l0_0.2      |             0.49 |               0.49 |             1.27 |         1.58181 |          1.15146   |
| ges              |             0.55 |               0.17 |             0    |         0       |          0.0700904 |
| golem_ev         |             0.6  |               0.59 |             0.69 |         1.22016 |         27.2553    |
| golem_nv         |             0.27 |               0.05 |             2.44 |         2.33255 |         53.7988    |
| sp               |             0.7  |               0.35 |             0.95 |         1.61667 |          0.0210971 |


## Failure stats

- Failure count: 725
- Failure CSV: c:\Users\super\DAG\experiments\results\validate_er_graph_cd_failures.csv
- Failure JSON: c:\Users\super\DAG\experiments\results\validate_er_graph_cd_failures.json