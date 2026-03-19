# THRESHOLD CALIBRATION RESULTS -- Pareto Rule (80/20)

Comparing PerSyst thresholds to Pareto rule (80/20) thresholds.

| Metric                     | Current |    Pareto |    Change | Threshold |
| -------------------------- | ------: | --------: | --------: | --------: |
| check_branch_mis_rate      |    0.01 |     1.837 | +18266.8% |       80% |
| check_branch_mis_ratio     |    0.05 |     9.097 | +18094.4% |       80% |
| check_l3_bandwidth         |     0.4 |     0.208 |    -48.0% |       80% |
| check_l3_miss_rate         |  0.0003 |     0.635 |  +211566% |       80% |
| check_l3_miss_ratio        |   0.741 |     0.763 |     +3.0% |       80% |
| check_cpi                  |     1.6 |     10.35 |   +546.8% |       80% |
| check_flop_rate_for_cpi    |    0.15 |   0.09158 |    -38.9% |       80% |
| check_total_flop_rate_prec |    0.15 | 0.0003963 |    -99.7% |       20% |
| check_sp_to_dp_ratio       |   0.035 |         0 |   -100.0% |       20% |
| check_vec_to_scalar        |    0.01 |         0 |   -100.0% |       20% |
| check_avx_sse_ratio        |   0.013 |   0.04212 |   +224.0% |       20% |
| check_stall_rate           |     0.2 |     84.55 | +42176.2% |       80% |
| check_loads_to_stores      |       4 |     14.73 |   +268.2% |       80% |
