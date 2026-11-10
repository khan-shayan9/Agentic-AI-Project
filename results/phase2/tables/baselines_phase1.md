# Baselines B0-B2 (Phase 1 re-runs, test split)

These are the Phase 1 re-runs on the same 600 test answers (same split, prompt and settings). They have no per-item predictions.

| baseline | F1 | precision | recall | coverage | accuracy | easy F1 | medium F1 | hard F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0: no knowledge | 0.477 | 0.706 | 0.360 | 1.000 | 0.605 | 0.655 | 0.473 | 0.271 |
| B1: source given | 0.755 | 0.835 | 0.690 | 1.000 | 0.777 | 0.851 | 0.796 | 0.615 |
| B2: "not sure" allowed | 0.459 | 0.772 | 0.327 | 1.000 | 0.615 | 0.658 | 0.420 | 0.260 |

Source: `results/phase1/llama-test-{none,source,unsure}-metrics.json` (Llama-3.1-8B).
