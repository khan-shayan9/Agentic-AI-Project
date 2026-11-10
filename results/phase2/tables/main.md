# Main results (mean ± SD over seeds)

Missing runs: b0-test-s1, b0-test-s2, b0-test-s3, b1-test-s1, b1-test-s2, b1-test-s3, b2-test-s1, b2-test-s2, b2-test-s3, sysA-test-s2, sysA-test-s3, sysB-test-s1, b4-test-s1, b4-test-s2, b4-test-s3, b5-test-s1, none-test-s1, noblock-test-s1, debate-test-s1

| configuration | source | seeds | F1 | precision | recall | coverage | acc. answered | easy F1 | medium F1 | hard F1 | calls/item | model s/item |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| System A (PubMed, source blocked) | ours (rerun) | 1 | 0.694 | 0.622 | 0.787 | 0.927 | 0.644 | 0.771 | 0.708 | 0.608 | 3.475 | 13.864 |
| MedHallu reported: no knowledge | paper | - | 0.522 | 0.791 | - | - | - | 0.679 | 0.515 | 0.372 | - | - |
| MedHallu reported: with knowledge | paper | - | 0.797 | 0.775 | - | - | - | 0.880 | 0.796 | 0.722 | - | - |
| B3: llama alone (System A round 1) | ours (rerun) | 1 | 0.697 | 0.577 | 0.880 | 0.840 | 0.607 | 0.756 | 0.709 | 0.627 | 1.000 | 4.219 |
| B3: mistral-local alone (System A round 1) | ours (rerun) | 1 | 0.640 | 0.584 | 0.710 | 0.803 | 0.602 | 0.722 | 0.653 | 0.544 | 1.000 | 5.059 |
| B3: phi alone (System A round 1) | ours (rerun) | 1 | 0.692 | 0.680 | 0.703 | 0.975 | 0.689 | 0.768 | 0.683 | 0.620 | 1.000 | 2.517 |
| System A, round 1 (second round off) | ours (rerun) | 1 | 0.709 | 0.635 | 0.803 | 0.840 | 0.655 | 0.781 | 0.724 | 0.622 | 3.000 | 11.795 |
| System A, forced answer | ours (rerun) | 1 | 0.668 | 0.585 | 0.777 | 1.000 | 0.613 | 0.730 | 0.695 | 0.582 | 3.475 | 13.864 |

Paper rows: MedHallu paper (Pandit et al., EMNLP 2025), Table 2, Llama-3.1-8B-Instruct, reported; the paper gives no recall.
