# Agent trajectories

Every pipeline run saves one **state** per item: the full trajectory of the agents for that answer.
This folder holds small, readable extracts; the complete states of every run are in
`results/phase2/runs/<run>/states.jsonl.gz` (one JSON object per line, gzipped).

- `samples/<run>.jsonl`: the first 20 states (by item id) of `sysA-test-s1`, the chosen validation run,
  `b4-test-s1` and `debate-test-s1`. Evidence abstracts are cut to 300 characters.
- `cases/<item_id>.md` and `.json`: representative cases picked by `python -m experiments.report`
  (two successes, two failures), as readable markdown and as the full JSON state.
  `cases/val/` holds the same for the validation run (`--split val`).

These files are written by `python -m experiments.report`; they appear once Phase 2 runs are exported.

## Fields of a state

| Field | Meaning |
| --- | --- |
| `item_id` | The question and one answer: `-gt` = the true answer (label 0), `-hal` = the hallucinated answer (label 1) |
| `evidence_mode` | `pubmed` (setting A, source blocked), `source` (setting B, source abstract given) or `none` |
| `query` | The PubMed query that found the round-1 evidence |
| `evidence` | Abstracts the verifiers saw: `pmid`, `title`, `text`, `query`, `rank`, `round` (2 = found by the second round), `leak` (set only with `--no-block`: the abstract is the source or a near copy) |
| `blocked` | Abstracts removed to stop source leakage: `pmid`, `reason` (`source_pmid` or `near_duplicate`), `overlap`, `round` |
| `verdicts` | One per verifier per round: `system` (`verifier:<name>`), `verdict` (0 factual, 1 hallucinated, 2 unsure, null unreadable), `raw_text` (the model's reply; two replies joined by `---` after a re-ask), `round`, `reasked`, `latency_s`, tokens, `cached` |
| `final` | The decision: 0, 1 or 2 (abstain) |
| `final_round1` | The decision from the round-1 votes alone (what the system would say with the second round off) |
| `rounds` | 1, or 2 when the verifiers judged again (second evidence round or debate) |
| `flags` | `human_review` (the system abstained), `reasked:<verifier>`, `no_new_evidence` |
| `steps` | The trajectory, step by step: `step`, `agent` (orchestrator, evidence, verifier, decision), `action`, `detail` (inputs, raw reply, parsed verdict, latency, tokens) and `t` (seconds since the item started) |
| `tool_calls` | Every PubMed request: `tool` (esearch or efetch), `input`, `n_results`, `cached`, `latency_s`, `round` |
| `trace` | The same story as short text lines (kept for older tools) |
