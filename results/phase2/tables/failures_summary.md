# Failure analysis: sysA-test-s1

Missing runs: b0-test-s1, b0-test-s2, b0-test-s3, b1-test-s1, b1-test-s2, b1-test-s3, b2-test-s1, b2-test-s2, b2-test-s3, sysA-test-s2, sysA-test-s3, sysB-test-s1, b4-test-s1, b4-test-s2, b4-test-s3, b5-test-s1, none-test-s1, noblock-test-s1, debate-test-s1

600 items. Each item has one category, checked in the order below.

| category | count | share |
| --- | --- | --- |
| unreadable | 0 | 0.000 |
| abstained - majority would have been right | 4 | 0.007 |
| abstained - majority would have been wrong | 10 | 0.017 |
| abstained - votes tied | 30 | 0.050 |
| missed hallucination - all verifiers fooled | 19 | 0.032 |
| missed hallucination - majority fooled | 42 | 0.070 |
| false alarm - all verifiers | 49 | 0.082 |
| false alarm - majority | 88 | 0.147 |
| correct | 358 | 0.597 |

## By difficulty

| difficulty | n | abstained - majority would have been right | abstained - majority would have been wrong | abstained - votes tied | missed hallucination - all verifiers fooled | missed hallucination - majority fooled | false alarm - all verifiers | false alarm - majority | correct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| easy | 200 | 1 | 5 | 9 | 3 | 11 | 10 | 23 | 138 |
| hard | 200 | 0 | 4 | 13 | 12 | 18 | 18 | 37 | 98 |
| medium | 200 | 3 | 1 | 8 | 4 | 13 | 21 | 28 | 122 |

## By MedHallu hallucination category

| medhallu_category | n | abstained - majority would have been right | abstained - majority would have been wrong | abstained - votes tied | missed hallucination - all verifiers fooled | missed hallucination - majority fooled | false alarm - all verifiers | false alarm - majority | correct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Incomplete Information | 136 | 1 | 1 | 5 | 8 | 11 | 10 | 19 | 81 |
| Mechanism and Pathway Misattribution | 18 | 0 | 0 | 1 | 1 | 1 | 1 | 1 | 13 |
| Methodological and Evidence Fabrication | 2 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 |
| Misinterpretation of #Question# | 444 | 3 | 9 | 24 | 10 | 30 | 37 | 68 | 263 |

## By evidence state

| evidence_state | n | abstained - majority would have been right | abstained - majority would have been wrong | abstained - votes tied | missed hallucination - all verifiers fooled | missed hallucination - majority fooled | false alarm - all verifiers | false alarm - majority | correct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| decided in round 1 | 493 | 0 | 0 | 0 | 16 | 33 | 46 | 75 | 323 |
| no clean evidence | 11 | 0 | 0 | 0 | 1 | 2 | 1 | 0 | 7 |
| round 2 changed abstain->wrong | 24 | 0 | 0 | 0 | 2 | 7 | 2 | 13 | 0 |
| round 2 flipped abstain->right | 28 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 28 |
| round 2: no change | 43 | 4 | 10 | 29 | 0 | 0 | 0 | 0 | 0 |
| second round triggered, no new evidence | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |

Items with no clean evidence in round 1: 12 of 600.

## Round-2 flips

| round 1 \ final | right | wrong | abstain |
| --- | --- | --- | --- |
| right | 0 | 0 | 0 |
| wrong | 0 | 0 | 0 |
| abstain | 28 | 24 | 43 |

## Representative cases

- success: hard hallucination caught: [pqa_labeled-00001-hal](../../../logs/cases/pqa_labeled-00001-hal.md)
- success: fixed by the second round: [pqa_labeled-00009-hal](../../../logs/cases/pqa_labeled-00009-hal.md)
- failure: false alarm - majority: [pqa_labeled-00002-gt](../../../logs/cases/pqa_labeled-00002-gt.md)
- failure: false alarm - all verifiers: [pqa_labeled-00009-gt](../../../logs/cases/pqa_labeled-00009-gt.md)
