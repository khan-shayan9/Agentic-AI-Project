# Ablations: each component switched off in turn (variant - reference)

Missing runs: b0-test-s1, b0-test-s2, b0-test-s3, b1-test-s1, b1-test-s2, b1-test-s3, b2-test-s1, b2-test-s2, b2-test-s3, sysA-test-s2, sysA-test-s3, sysB-test-s1, b4-test-s1, b4-test-s2, b4-test-s3, b5-test-s1, none-test-s1, noblock-test-s1, debate-test-s1

| component | reference | n_seeds | f1_reference | f1_variant | delta_f1 | delta_f1_sd | delta_f1_seed1 | delta_f1_ci_low | delta_f1_ci_high | delta_f1_p | delta_acc | delta_acc_ci_low | delta_acc_ci_high | delta_coverage | calls_reference | calls_variant | seconds_reference | seconds_variant | missing |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Second round off | System A | 1 | 0.694 | 0.709 | 0.015 | - | 0.015 | 0.002 | 0.027 | 0.0200 | 0.011 | -0.002 | 0.025 | -0.087 | 3.475 | 3.000 | 13.864 | 11.795 | - |
| No evidence | System A | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | runs missing |
| No source blocking | System A | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | runs missing |
| One-family panel (B4) | System A round 1 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | runs missing |
| Debate instead of the independent vote | System A round 1 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | runs missing |
| Drop llama (2 of 2 must agree) | System A round 1 | 1 | 0.709 | 0.734 | 0.025 | - | 0.025 | -0.006 | 0.056 | 0.1026 | 0.050 | 0.022 | 0.080 | -0.263 | 3.000 | 2.000 | 11.795 | 7.576 | - |
| Drop mistral-local (2 of 2 must agree) | System A round 1 | 1 | 0.709 | 0.788 | 0.079 | - | 0.079 | 0.052 | 0.107 | 0.0000 | 0.066 | 0.034 | 0.098 | -0.292 | 3.000 | 2.000 | 11.795 | 6.736 | - |
| Drop phi (2 of 2 must agree) | System A round 1 | 1 | 0.709 | 0.722 | 0.013 | - | 0.013 | -0.019 | 0.042 | 0.4190 | -0.007 | -0.039 | 0.026 | -0.328 | 3.000 | 2.000 | 11.795 | 9.278 | - |
| Forced answer (no abstention) | System A | 1 | 0.694 | 0.668 | -0.027 | - | -0.027 | -0.039 | -0.014 | 0.0000 | -0.031 | -0.044 | -0.018 | 0.073 | 3.475 | 3.475 | 13.864 | 13.864 | - |
| Vote 1 of 3 | System A (2 of 3) | 1 | 0.694 | 0.686 | -0.009 | - | -0.009 | -0.016 | -0.002 | 0.0088 | -0.009 | -0.017 | -0.002 | 0.023 | 3.475 | 3.475 | 13.864 | 13.864 | - |
| Vote 2 of 3 | System A (2 of 3) | 1 | 0.694 | 0.694 | 0.000 | - | 0.000 | 0.000 | 0.000 | 1.0000 | 0.000 | 0.000 | 0.000 | 0.000 | 3.475 | 3.475 | 13.864 | 13.864 | - |
| Vote 3 of 3 | System A (2 of 3) | 1 | 0.694 | 0.781 | 0.086 | - | 0.086 | 0.046 | 0.126 | 0.0000 | 0.080 | 0.036 | 0.124 | -0.517 | 3.475 | 3.475 | 13.864 | 13.864 | - |

Deltas are means over matching seeds; intervals and p values are the seed-1 paired bootstrap. Offline views (round 1, dropped verifier, forced answer, vote thresholds) come from the System A runs themselves.
