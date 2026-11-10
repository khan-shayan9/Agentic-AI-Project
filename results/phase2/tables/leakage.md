# Source leakage in PubMed searches

Missing runs: b0-test-s1, b0-test-s2, b0-test-s3, b1-test-s1, b1-test-s2, b1-test-s3, b2-test-s1, b2-test-s2, b2-test-s3, sysA-test-s2, sysA-test-s3, sysB-test-s1, b4-test-s1, b4-test-s2, b4-test-s3, b5-test-s1, none-test-s1, noblock-test-s1, debate-test-s1

| run | round | items_searched | source_pmid_share | near_duplicate_share | leaked_into_evidence_share | mean_clean_abstracts | items_no_evidence | items_no_evidence_share | kept_leaks |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sysA-test-s1 | round1 | 600 | 0.833 | 0.000 | 0.000 | 2.547 | 12 | 0.020 | 0 |
| sysA-test-s1 | round2 | 96 | 0.396 | 0.000 | 0.000 | 1.865 | 1 | 0.010 | - |

kept_leaks counts kept abstracts with a leak mark or the item's own source PMID, over all rounds.
