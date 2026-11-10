# Paired significance tests (A - B, same items)

Missing runs: b0-test-s1, b0-test-s2, b0-test-s3, b1-test-s1, b1-test-s2, b1-test-s3, b2-test-s1, b2-test-s2, b2-test-s3, sysA-test-s2, sysA-test-s3, sysB-test-s1, b4-test-s1, b4-test-s2, b4-test-s3, b5-test-s1, none-test-s1, noblock-test-s1, debate-test-s1

| comparison | n | f1_diff | f1_ci_low | f1_ci_high | f1_p | acc_diff | acc_ci_low | acc_ci_high | acc_p | mcnemar_b | mcnemar_c | mcnemar_p | mean_diff_f1 | sd_diff_f1 | missing |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| System A vs B0 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | runs missing |
| System A vs B1 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | runs missing |
| System B vs B1 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | runs missing |
| System A round-1 vote vs B4 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | runs missing |
| System A round-1 vote vs best single verifier (llama, chosen on val) | 600 | 0.012 | -0.019 | 0.042 | 0.4596 | 0.048 | 0.012 | 0.084 | 0.0090 | 42 | 27 | 0.0912 | - | - | - |
| System A vs System A round 1 | 600 | -0.015 | -0.027 | -0.002 | 0.0200 | -0.011 | -0.025 | 0.002 | 0.1096 | 0 | 0 | 1.0000 | - | - | - |
| System A vs B5 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | runs missing |
| System A vs no evidence | - | - | - | - | - | - | - | - | - | - | - | - | - | - | runs missing |
| No blocking vs System A | - | - | - | - | - | - | - | - | - | - | - | - | - | - | runs missing |
| Debate vs System A round 1 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | runs missing |

Seed 1 vs seed 1: paired bootstrap over items (10,000 resamples, seed 0; each side's F1 and accuracy on its own answered items) with 95% interval and two-sided p; McNemar exact test on items both sides answered (b = A right and B wrong, c = A wrong and B right). Mean difference over matching seeds when both sides have 3 seeds.
