# Results — ibm-granite/granite-3.0-1b-a400m-instruct

BF16 baseline: perplexity 27.65, calibration 0.500, format 0.875, long_horizon 0.759, recall 0.650, recovery 0.100

Excluded from the agentic mean, at or near their floor at BF16: recovery (0.10). They are still reported per probe below.

## Configs

| config | bits | method | ppl ratio | recall (control) | agentic | core | flip rate | recon err |
|---|---|---|---|---|---|---|---|---|
| bf16 | 16 | none | 1.000 | 1.000 | 1.000 | 1.000 | 0.0000 | 0.0000 |
| w8-ea | 8 | rtn | 1.003 | 1.000 | 1.000 | 1.000 | 0.0255 | 0.0203 |
| w6-ea | 6 | rtn | 0.984 | 1.154 | 1.009 | 0.963 | 0.0580 | 0.0552 |
| w4-ea | 4 | rtn | 0.955 | 0.962 | 0.889 | 0.783 | 0.1814 | 0.2301 |
| w4-ea-awq | 4 | awq | 1.375 | 0.885 | 0.834 | 0.652 | 0.2050 | 0.2877 |
| w3-ea | 3 | rtn | 1.929 | 0.385 | 0.839 | 0.809 | 0.3546 | 0.3316 |
| w4-experts | 4 | rtn | 1.113 | 1.000 | 0.967 | 0.951 | 0.1225 | 0.1312 |
| w4-attention | 4 | rtn | 0.851 | 0.923 | 0.813 | 0.769 | 0.1506 | 0.1625 |
| w4-all | 4 | rtn | 0.966 | 0.962 | 0.813 | 0.770 | 0.2297 | 0.2055 |
| x8-ea | 8 | rtn | 1.003 | nan | 0.985 | 0.977 | 0.0255 | 0.0203 |
| x7-ea | 7 | rtn | 0.995 | nan | 1.010 | 1.016 | 0.0383 | 0.0277 |
| x5-ea | 5 | rtn | 1.049 | nan | 0.899 | 0.848 | 0.1043 | 0.1142 |
| x6-experts | 6 | rtn | 0.957 | nan | 0.981 | 0.921 | 0.0390 | 0.0423 |
| x3-experts | 3 | rtn | 1.247 | nan | 1.092 | 0.988 | 0.2330 | 0.2347 |
| x6-attention | 6 | rtn | 0.996 | nan | 0.994 | 0.941 | 0.0533 | 0.0511 |
| x3-attention | 3 | rtn | 1.148 | nan | 1.124 | 1.085 | 0.2702 | 0.2016 |
| x6-gate | 6 | rtn | 1.003 | nan | 0.985 | 0.977 | 0.0468 | 0.0243 |
| x4-gate | 4 | rtn | 0.971 | nan | 1.041 | 1.012 | 0.1502 | 0.0546 |
| x3-gate | 3 | rtn | 1.006 | nan | 0.999 | 0.899 | 0.2623 | 0.1107 |
| x2-gate | 2 | rtn | 1.714 | nan | 1.206 | 1.009 | 0.4808 | 0.3230 |
| x3-all | 3 | rtn | 1.828 | nan | 0.824 | 0.836 | 0.4121 | 0.2909 |
| x4-ea-g32 | 4 | rtn | 1.032 | nan | 0.947 | 0.820 | 0.1534 | 0.1300 |
| x4-ea-g512 | 4 | rtn | 1.179 | nan | 0.967 | 0.850 | 0.2217 | 0.2109 |
| x4-ea-perchan | 4 | rtn | 1.203 | nan | 0.724 | 0.635 | 0.2431 | 0.2244 |
| x3-ea-g32 | 3 | rtn | 1.130 | nan | 0.733 | 0.600 | 0.2720 | 0.2431 |
| x4-experts-awq | 4 | awq | 1.148 | nan | 0.966 | 0.799 | 0.1441 | 0.1029 |
| x3-ea-awq | 3 | awq | 4.543 | nan | 0.285 | 0.277 | 0.4002 | 0.3464 |
| x8-gate-only-ea4 | 4 | awq | 1.419 | nan | 0.960 | 0.790 | 0.2816 | 0.2021 |

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
| x8-ea | 1.000 | 0.955 | 1.000 | nan |
| x7-ea | 1.122 | 0.909 | 1.000 | nan |
| x5-ea | 0.878 | 0.818 | 1.000 | nan |
| x6-experts | 1.024 | 0.818 | 1.100 | nan |
| x3-experts | 0.976 | 1.000 | 1.300 | nan |
| x6-attention | 0.927 | 0.955 | 1.100 | nan |
| x3-attention | 1.171 | 1.000 | 1.200 | nan |
| x6-gate | 1.000 | 0.955 | 1.000 | nan |
| x4-gate | 1.024 | 1.000 | 1.100 | nan |
| x3-gate | 1.024 | 0.773 | 1.200 | nan |
| x2-gate | 0.927 | 1.091 | 1.600 | nan |
| x3-all | 0.854 | 0.818 | 0.800 | nan |
| x4-ea-g32 | 0.732 | 0.909 | 1.200 | nan |
| x4-ea-g512 | 0.610 | 1.091 | 1.200 | nan |
| x4-ea-perchan | 0.634 | 0.636 | 0.900 | nan |
| x3-ea-g32 | 0.927 | 0.273 | 1.000 | nan |
| x4-experts-awq | 0.780 | 0.818 | 1.300 | nan |
| x3-ea-awq | 0.463 | 0.091 | 0.300 | nan |
| x8-gate-only-ea4 | 0.854 | 0.727 | 1.300 | nan |

## H3 — predictors of agentic degradation (n=27 configs)

`all probes` is the mean over every probe with headroom at BF16; `core` drops calibration, which rewards abstention and can rise under damage.

| predictor | r (all probes) | rho | r (core) | rho |
|---|---|---|---|---|
| routing change x compute damage | -0.603 | -0.580 | -0.584 | -0.560 |
| quantized-weight error (expert+attn) | -0.553 | -0.549 | -0.575 | -0.540 |
| router expert-set change rate | -0.341 | -0.337 | -0.477 | -0.403 |
| router top-1 flip rate | -0.295 | -0.324 | -0.417 | -0.402 |
| router top-k jaccard | -0.348 | -0.342 | -0.455 | -0.410 |
| expert load KL | -0.145 | -0.337 | -0.194 | -0.393 |
| perplexity ratio | -0.703 | -0.192 | -0.664 | -0.217 |
| layer reconstruction error | -0.444 | -0.489 | -0.577 | -0.542 |

Strongest predictor of the core target: **perplexity ratio** (|r| = 0.664).

## The deployable regime (n=21 configs with perplexity within 25%)

The whole argument is about configs that *look fine*, so this restricts to the ones a practitioner would actually ship and asks which predictor still finds the damage. Chosen after seeing that one catastrophic config was carrying the full-set Pearson.

| predictor | r (core) | rho |
|---|---|---|
| routing change x compute damage | -0.290 | -0.409 |
| quantized-weight error (expert+attn) | -0.331 | -0.408 |
| router expert-set change rate | -0.509 | -0.455 |
| router top-1 flip rate | -0.467 | -0.438 |
| router top-k jaccard | -0.487 | -0.452 |
| expert load KL | -0.279 | -0.445 |
| perplexity ratio | -0.118 | -0.012 |
| layer reconstruction error | -0.590 | -0.561 |

In the regime that matters: **layer reconstruction error** (|r| = 0.590).

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
| x8-ea | 0.100 0.002 0.000 0.000 | 0.289 0.051 0.008 0.000 |
| x7-ea | 0.147 0.005 0.000 0.000 | 0.371 0.102 0.017 0.001 |
| x5-ea | 0.328 0.076 0.012 0.001 | 0.594 0.406 0.203 0.034 |
| x6-experts | 0.149 0.007 0.000 0.000 | 0.378 0.119 0.017 0.001 |
| x3-experts | 0.484 0.287 0.138 0.024 | 0.757 0.671 0.547 0.275 |
| x6-attention | 0.200 0.013 0.001 0.000 | 0.444 0.178 0.048 0.004 |
| x3-attention | 0.517 0.332 0.179 0.053 | 0.823 0.773 0.668 0.418 |
| x6-gate | 0.182 0.005 0.000 0.000 | 0.392 0.094 0.014 0.001 |
| x4-gate | 0.414 0.160 0.026 0.001 | 0.608 0.449 0.225 0.035 |
| x3-gate | 0.521 0.344 0.158 0.026 | 0.760 0.680 0.532 0.246 |
| x2-gate | 0.663 0.584 0.464 0.211 | 0.930 0.914 0.875 0.737 |
| x3-all | 0.630 0.497 0.370 0.151 | 0.918 0.894 0.837 0.659 |
| x4-ea-g32 | 0.402 0.165 0.042 0.005 | 0.685 0.541 0.366 0.111 |
| x4-ea-g512 | 0.487 0.268 0.109 0.023 | 0.770 0.685 0.550 0.274 |
| x4-ea-perchan | 0.495 0.299 0.139 0.039 | 0.793 0.709 0.595 0.308 |
| x3-ea-g32 | 0.527 0.331 0.181 0.050 | 0.809 0.755 0.655 0.395 |
| x4-experts-awq | 0.383 0.143 0.047 0.004 | 0.644 0.500 0.321 0.093 |
| x3-ea-awq | 0.630 0.491 0.338 0.142 | 0.912 0.882 0.830 0.652 |
| x8-gate-only-ea4 | 0.530 0.367 0.188 0.041 | 0.810 0.730 0.629 0.343 |
