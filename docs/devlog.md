# Development log

One entry per major change: what failed or was missing, what we changed, and the measured effect.
All numbers come from files in `results/phase1/` unless marked "not saved".

## Phase 1

| Date | Problem observed | Change made | Effect |
| --- | --- | --- | --- |
| 2026-10-17 | Baseline needed checking against the paper | Ran B0 and B1 (Llama 3.1 8B, authors' prompt) on the 600 test answers | Final saved run: B0 F1 0.477, B1 F1 0.755; paper 0.522 and 0.797. Knowledge gain 0.278 vs 0.275. A first run gave 0.479 and 0.761 (not saved), so repeated runs differ a little at temperature 0.3 |
| 2026-10-17 | Mistral's free API returned error 429 on the first call; the header showed 0 requests per minute allowed for our account | Verifier 2 now runs Mistral 7B locally through Ollama (same model family). The API option stays in the code. Clear stop message instead of a crash | Full system runs end to end with no rate limits |
| 2026-10-18 | With digit answers (0/1/2), Phi-4-mini often replied "2" (not sure) while its own explanation said the answer was wrong (first dev run, not saved) | Added a word-label prompt (FACTUAL / HALLUCINATED / UNSURE) | Phi stopped using "not sure", but the models then rarely said "hallucinated" (not saved) |
| 2026-10-18 | The unsure-word check matched "pass" inside words such as "passage" | Whole-word matching in the parser, with a test | Fewer replies wrongly read as "not sure" |
| 2026-10-18 | No prompt style worked well on the small dev split | Added a third style, `reason` (a short check, then a `Verdict:` line), and compared all three on the 200 validation answers | Vote F1: digits 0.595 (coverage 0.665), words 0.447 (0.795), reason 0.680 (0.890). `reason` became the default |
| 2026-10-18 | The vote rule had to be fixed before any test run | `tune` on validation: 1 of 3 gave accuracy 0.654 at coverage 0.955; 2 of 3 gave 0.646 at 0.890; 3 of 3 gave 0.704 at 0.405 | Kept 2 of 3 for the test run: almost the same accuracy, a true majority vote, and enough abstentions to measure gap 3 |

## Phase 2

| Date | Problem observed | Change made | Effect |
| --- | --- | --- | --- |
| 2026-10-21 | No seeds, so repeat runs could not be told apart; the trajectory was a short text log; an unreadable reply simply counted as no vote; no debate ablation or forced-answer comparison; resumed runs only checked models and prompt | Seeds in the cache key (`--seed`, `name@k` samples); structured steps and PubMed tool calls in every state; one re-ask on unreadable replies (`--no-reask` to switch off); `--debate` ablation; `evaluate --forced` and `--round`; full setup check and model digests in `config.json`. Phase 1 states still score F1 0.680 at coverage 0.890 | To be measured on val |
| 2026-10-21 | When the verifiers disagreed the system could only abstain, with no second look at the evidence; a baseline folder could be resumed with another seed or setting and silently return the old answers; no way to measure source leakage without blocking, or to run simple RAG (first query only) | Second evidence round (`--second-round split|no-majority`, `--k2`) with claim queries from the answer, the source still blocked and used abstracts skipped; baseline resume check; `--no-block` (leaks marked on `Evidence.leak`) and `--no-retry`; leakage and efficiency sections in `evaluate`. On the saved Phase 1 validation states the leakage section reproduces the README numbers (source returned for 81% of answers, 2.5 clean abstracts each) | To be measured on val |
