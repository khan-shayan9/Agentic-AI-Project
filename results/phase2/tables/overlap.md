# Error overlap between verifiers (round 1)

Missing runs: b0-test-s1, b0-test-s2, b0-test-s3, b1-test-s1, b1-test-s2, b1-test-s3, b2-test-s1, b2-test-s2, b2-test-s3, sysA-test-s2, sysA-test-s3, sysB-test-s1, b4-test-s1, b4-test-s2, b4-test-s3, b5-test-s1, none-test-s1, noblock-test-s1, debate-test-s1

| level | pair_type | pair | run | seed | n_seeds | n | both_wrong | both_wrong_sd | expected_if_independent | overlap_ratio | overlap_ratio_sd | phi | phi_sd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pair | cross-family | llama vs mistral-local | sysA-test-s1 | 1 | - | 411 | 0.263 | - | 0.152 | 1.734 | - | 0.468 | - |
| pair | cross-family | llama vs phi | sysA-test-s1 | 1 | - | 493 | 0.187 | - | 0.123 | 1.518 | - | 0.281 | - |
| pair | cross-family | mistral-local vs phi | sysA-test-s1 | 1 | - | 470 | 0.217 | - | 0.119 | 1.821 | - | 0.437 | - |
| pair (mean over seeds) | cross-family | llama vs mistral-local | - | - | 1 | - | 0.263 | - | 0.152 | 1.734 | - | 0.468 | - |
| pair (mean over seeds) | cross-family | llama vs phi | - | - | 1 | - | 0.187 | - | 0.123 | 1.518 | - | 0.281 | - |
| pair (mean over seeds) | cross-family | mistral-local vs phi | - | - | 1 | - | 0.217 | - | 0.119 | 1.821 | - | 0.437 | - |
| pair type (mean over seeds) | cross-family | all cross-family pairs | - | - | 1 | - | 0.222 | - | 0.131 | 1.691 | - | 0.395 | - |

Ratio = both wrong / expected if independent; 1.0 means independent errors.
