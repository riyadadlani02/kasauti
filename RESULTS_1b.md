# Results — ibm-granite/granite-3.0-1b-a400m-instruct

BF16 baseline: perplexity 27.65, calibration 0.500, format 0.875, long_horizon 0.759, recall 0.650, recovery 0.100

Excluded from the agentic mean, at or near their floor at BF16: recovery (0.10). They are still reported per probe below.

## Configs

| config | bits | method | ppl ratio | recall (control) | agentic mean | flip rate | recon err |
|---|---|---|---|---|---|---|---|
| bf16 | 16 | none | 1.000 | 1.000 | 1.000 | 0.0000 | 0.0000 |
| w8-ea | 8 | rtn | 1.003 | 1.000 | 1.000 | 0.0255 | 0.0203 |
| w6-ea | 6 | rtn | 0.984 | 1.154 | 1.009 | 0.0580 | 0.0552 |
| w4-ea | 4 | rtn | 0.955 | 0.962 | 0.889 | 0.1814 | 0.2301 |
| w4-ea-awq | 4 | awq | 1.375 | 0.885 | 0.834 | 0.2050 | 0.2877 |
| w3-ea | 3 | rtn | 1.929 | 0.385 | 0.839 | 0.3546 | 0.3316 |
| w4-experts | 4 | rtn | 1.113 | 1.000 | 0.967 | 0.1225 | 0.1312 |
| w4-attention | 4 | rtn | 0.851 | 0.923 | 0.813 | 0.1506 | 0.1625 |
| w4-all | 4 | rtn | 0.966 | 0.962 | 0.813 | 0.2297 | 0.2055 |

## Per-probe, relative to BF16

| config | long_horizon | format | calibration | recovery |
|---|---|---|---|---|
| bf16 | 1.000 | 1.000 | 1.000 | 1.000 |
| w8-ea | 1.000 | 1.000 | 1.000 | 0.000 |
| w6-ea | 0.927 | 1.000 | 1.100 | 0.000 |
| w4-ea | 0.805 | 0.762 | 1.100 | 1.000 |
| w4-ea-awq | 0.732 | 0.571 | 1.200 | 0.000 |
| w3-ea | 0.951 | 0.667 | 0.900 | 0.000 |
| w4-experts | 0.854 | 1.048 | 1.000 | 0.000 |
| w4-attention | 0.585 | 0.952 | 0.900 | 0.000 |
| w4-all | 0.683 | 0.857 | 0.900 | 2.000 |

## H3 — predictors of agentic degradation (n=8 configs)

| predictor | pearson r | spearman rho |
|---|---|---|
| router expert-set change rate | -0.821 | -0.667 |
| router top-1 flip rate | -0.773 | -0.667 |
| router top-k jaccard | -0.748 | -0.667 |
| expert load KL | -0.432 | -0.667 |
| perplexity ratio | -0.257 | 0.262 |
| layer reconstruction error | -0.794 | -0.571 |

Strongest predictor: **router expert-set change rate** (|r| = 0.821).

## Mechanism — divergence by base router margin quartile

Each margin is paired with the event it governs: the top-1 margin with a change of argmax, the k-th/(k+1)-th margin with a change of the selected set.

| config | top-1 flips, narrow to wide | set changes, narrow to wide |
|---|---|---|
| w8-ea | 0.100 0.002 0.000 0.000 | 0.289 0.051 0.008 0.000 |
| w6-ea | 0.211 0.019 0.002 0.000 | 0.481 0.213 0.059 0.006 |
| w4-ea | 0.443 0.205 0.066 0.011 | 0.717 0.614 0.462 0.181 |
| w4-ea-awq | 0.470 0.240 0.092 0.018 | 0.746 0.657 0.498 0.216 |
| w3-ea | 0.580 0.437 0.295 0.106 | 0.885 0.850 0.778 0.565 |
| w4-experts | 0.358 0.108 0.023 0.001 | 0.605 0.462 0.262 0.064 |
| w4-attention | 0.396 0.158 0.043 0.005 | 0.675 0.554 0.365 0.116 |
| w4-all | 0.497 0.294 0.111 0.017 | 0.760 0.684 0.520 0.235 |
