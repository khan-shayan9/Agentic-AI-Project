# MedVerify Agents

**Independent multi-agent verification of medical answers: detecting hallucinations with evidence retrieval and abstention.**

AI4015 Agentic AI semester project. Base paper: MedHallu (Pandit et al., EMNLP 2025). Everything here runs on free tools only.

## What it does

Given a medical question and an answer, the system decides whether the answer is **factual (0)**, **hallucinated (1)**, or says **not sure (2)**.

- **Baseline:** one LLM judges each answer, exactly as in the MedHallu paper (authors' prompt, settings and metric).
- **Proposed system:** an evidence agent searches PubMed (with the question's own source abstract blocked). Three verifiers from different model families (Llama 3.1 8B from Meta, Mistral 7B from Mistral AI, Phi-4-mini from Microsoft), all running locally, then judge separately (a short check, then a FACTUAL, HALLUCINATED or UNSURE verdict), and a vote rule accepts, rejects or abstains. When the votes split, the evidence agent can search again with a query built from the answer's own claim, and the verifiers judge once more, still independently.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the diagram and the data contracts.

## Use it

Give a medical question and an answer; MedVerify says **ACCEPT**, **REJECT** or **NOT SURE**, with the PubMed evidence, each verifier's vote and reason, and why it abstained. It needs Ollama with the three models (see [Setup](#setup)) and, for PubMed evidence, internet access (put `NCBI_EMAIL` in `.env`).

```bash
# one answer
python -m medverify verify --question "Does a daily walk after dinner lower fasting glucose in prediabetes?" \
    --answer "Walking after dinner raised fasting glucose." [--pmid 12345678] [--json]

# a CSV file (columns question, answer, optional pmid): results.csv plus results.states.jsonl
python -m medverify batch --input examples/sample_items.csv --output results.csv

# the web app
pip install -e ".[app]"
streamlit run app/streamlit_app.py
```

| Decision | Meaning |
| --- | --- |
| **ACCEPT** | Factual: at least 2 of the 3 verifiers found no hallucination |
| **REJECT** | Hallucinated: at least 2 of the 3 found wrong or invented information |
| **NOT SURE** | No verdict had 2 votes (even after a second search, when it runs): sent for human review, with the reason |

If you know the abstract the answer came from, pass its PubMed ID with `--pmid`: that abstract and near copies of it are kept out of the evidence (`--evidence source` uses it as the evidence instead). `--second-round auto` (the default) uses the trigger chosen on the validation split once `experiments/final_config.json` exists, and is off before that; the output says which. All three ways run the same pipeline as the experiments, with cached replies.

**Research prototype, not medical advice.**

## Repository layout

```
src/medverify/
  data.py         load MedHallu, attach PubMed IDs, make splits
  schemas.py      structured contracts (Row, Item, Verdict, Evidence, VerificationState)
  prompts.py      baseline prompts (copied from the MedHallu code) and verifier prompts
  parsing.py      reply -> verdict (strict parser + the authors' parser for reference)
  llm.py          one client for Ollama, Groq and Mistral, with caching and rate limits
  tools/pubmed.py PubMed search with source-leakage blocking
  baseline.py     single-LLM baseline
  pipeline.py     multi-agent system (evidence agent, verifiers, decision step)
  metrics.py      F1 per difficulty, coverage, selective accuracy, error overlap, cost
  cli.py          the `python -m medverify` commands (including verify and batch)
  service.py      verify one item or a CSV for a user (shared by the CLI and the web app)
  explain.py      plain-English views of a result (decision, votes, reasons, evidence)
experiments/      Phase 2 run plan (plan.py), runner (run.py), val-only trigger choice (choose.py), cache backup
                  (state.py), export to results/phase2 (export.py) and the report (report.py, gaps.py,
                  failures.py, figures.py, stats.py, facts.py)
results/          phase1/ (Phase 1 files) and phase2/ (exported runs, tables, figures; committed)
logs/             agent trajectory samples and representative cases (see logs/README.md)
notebooks/        Colab notebooks: Phase 1 baseline, and Phase 2 experiments (also runs on Kaggle)
app/              Streamlit web app (streamlit_app.py) and its pure helpers (helpers.py)
examples/         sample_items.csv: five made-up items to try `medverify batch`
tests/            offline tests (fake models, no internet needed)
docs/             architecture and development log (docs/devlog.md)
```

## Setup

### Option A: Google Colab (recommended, free GPU)

Open `notebooks/phase1_baseline_colab.ipynb` in Colab, choose a T4 GPU, add `NCBI_EMAIL` in Colab's Secrets panel, and run the cells in order.

### Option B: your own computer

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env                                # then fill in your keys
```

Install [Ollama](https://ollama.com/download) and run `ollama pull llama3.1:8b`, `ollama pull mistral:7b` and `ollama pull phi4-mini`. Together they need about 12 GB of free memory; without a GPU they are slow, so Colab is easier.

### Free accounts (only for the proposed system)

| Service | Used for | Key in `.env` | Free limit |
| --- | --- | --- | --- |
| Ollama (local) | Llama 3.1 8B (baseline and verifier), Mistral 7B and Phi-4-mini (verifiers) | none | no limit |
| [NCBI](https://www.ncbi.nlm.nih.gov/account/settings/) | PubMed evidence | `NCBI_EMAIL`, optional `NCBI_API_KEY` | 3 requests/s, 10 with key |

We first planned Mistral Small through Mistral's free API, but our free account allowed 0 requests per minute (error 429), so the Mistral verifier runs locally. The API version still works with `--verifiers llama,mistral,phi` and a `MISTRAL_API_KEY` (model `mistral-small-2603`, paced to its free limits of 1 request/second and 20,000 tokens/minute). Every reply is cached and every run resumes where it stopped, so long runs can be split across sessions. Groq (gpt-oss-120b) is supported as an optional extra verifier (`--verifiers llama,mistral,phi,gptoss`) if your account works there.

## Commands

```bash
python -m medverify prepare                         # data + splits (data/processed/)

# Baseline (Phase 1)
python -m medverify baseline --model llama --split test --knowledge none   --name llama-test-none
python -m medverify baseline --model llama --split test --knowledge source --name llama-test-source
#   add --unsure to allow the "2 = not sure" option

# Proposed system
for s in digits words reason; do                                               # pick the verifier prompt on val
  python -m medverify pipeline --split val --evidence pubmed --prompt $s --name sysA-val-$s
done
python -m medverify compare results/runs/sysA-val-digits results/runs/sysA-val-words results/runs/sysA-val-reason
python -m medverify pipeline --split val  --evidence pubmed --name sysA-val    # setting A: source blocked
python -m medverify tune results/runs/sysA-val --min-coverage 0.7              # pick the vote rule on val
python -m medverify pipeline --split test --evidence pubmed --min-agree 2 --name sysA-test
python -m medverify pipeline --split test --evidence source --min-agree 2 --name sysB-test   # setting B

# Analysis
python -m medverify evaluate results/runs/sysA-test          # system + each verifier alone
python -m medverify overlap  results/runs/sysA-test          # do the verifiers share mistakes?

# Phase 2 options
python -m medverify baseline --model llama --split test --seed 1 --name llama-test-none-s1    # repeat runs: one seed each
python -m medverify pipeline --split val --evidence pubmed --verifiers llama@1,llama@2,llama@3 --seed 1 --name sc-val
#   name@k = the same model again as sample k, with seed (seed or 0) * 100 + k (self-consistency control)
python -m medverify pipeline --split val --evidence pubmed --debate --name sysA-val-debate   # ablation: round 2 after seeing the others
python -m medverify pipeline --split val --evidence pubmed --no-reask --name sysA-val-noreask
#   by default an unreadable verifier reply is asked again once; --no-reask turns this off
#   --max-tokens N caps each verifier reply (default 512)
python -m medverify pipeline --split val --evidence pubmed --second-round split --k2 2 --name sysA-val-r2split
python -m medverify pipeline --split val --evidence pubmed --second-round no-majority --name sysA-val-r2nomaj
#   second evidence round: split = round-1 votes not all the same label; no-majority = round-1 decision is abstain
python -m medverify evaluate results/runs/sysA-val-r2split --round 1   # the same run with the second round off
python -m medverify pipeline --split val --evidence pubmed --no-block --name sysA-val-noblock    # ablation 2: let the source leak in
python -m medverify pipeline --split test --evidence pubmed --no-retry --name b5-test             # baseline B5, simple RAG: first query only
python -m medverify evaluate results/runs/sysA-val-debate --round 1   # decision from round 1 -> metrics-round1.json
python -m medverify evaluate results/runs/sysA-test --forced          # forced answer, no abstention -> metrics-forced.json
```

`--seed` is sent to Ollama (and Groq) and added to the reply cache key only when set, so all Phase 1 cache entries still match. A run folder can only be resumed with the same models, prompt, temperature, seed, evidence, debate and re-ask settings; folders from before these options existed count as Phase 1 settings (no seed, no debate, no re-ask). `config.json` also records each model's exact settings, the hash of the verifier instructions, the git commit and the Ollama model digests. `overlap`, `compare` and the single-verifier tables always use round-1 (independent) verdicts. Every pipeline state logs structured steps (`steps`), PubMed tool calls (`tool_calls`) and flags such as `human_review` (the system abstained), `reasked:phi` or `no_new_evidence`.

`--second-round` is `off` by default; its final setting is chosen on the validation split only. It works in PubMed mode only (for `none` and `source` it is logged as skipped) and cannot be combined with `--debate`. The claim query uses the words in the answer that are not in the question, plus the top two question words to keep the topic, and tries shorter versions (about 4+2, 3+1, 2+1 words). The source stays blocked and abstracts already used are skipped; if nothing new is found the round-1 decision stands (flag `no_new_evidence`, no extra model calls). `--no-block` keeps the source and near copies but marks them (`Evidence.leak`), to measure how often the source would leak in. `--no-retry` uses only the first PubMed query. All four are part of the run setup check. `evaluate` on a pipeline run also reports **leakage** (per round: how often PubMed returned the source or near copies, clean abstracts per item, items with no evidence) and **efficiency** (per item over all rounds: model calls, tokens, model seconds, PubMed requests, share sent to round 2, re-asks, abstentions, unreadable replies).

Each run folder holds `config.json`, the raw outputs (`predictions.jsonl` or `states.jsonl`), and `metrics.json`.

## Key design decisions

- **Splits:** from the 1,000 human-annotated rows (`pqa_labeled`). Test = 300 rows (100 easy, 100 medium, 100 hard); val = 100 rows (only for tuning); dev = 20 rows (debugging). Fixed seed 42.
- **Two items per row:** each row's true and hallucinated answers are both judged (600 test items, half hallucinated). The authors' code shows one random answer per row; ours gives the same balance without randomness.
- **Faithful baseline:** prompts, temperature (0.3) and top_p (0.95) match the MedHallu code. The paper's main table uses a version without "not sure" that is not in the released code; ours removes those sentences from their prompt.
- **Parsing:** the authors' parser reads any reply containing "not" as hallucinated, so "not sure" became 1. We use a stricter parser and count unparseable replies separately. `parsing.parse_authors` is kept for comparison. Verifier prompt styles (`--prompt`): `digits` (0/1/2), `words` (FACTUAL/HALLUCINATED/UNSURE) and `reason` (a short check, then `Verdict: ...`). On the 40 dev answers, digits made Phi-4-mini reply "2" while its own explanation called the answer wrong, and words made the models rarely say HALLUCINATED, so the style is chosen on the validation split with `compare`. On the 200 validation answers (setting A) the vote reached F1 0.595 with digits (coverage 0.665), 0.447 with words (0.795) and 0.680 with reason (0.890), so `reason` is the default.
- **Leakage control:** each question's source PubMed ID comes from PubMedQA. The evidence agent drops that PMID and any abstract sharing 30% or more of its 8-word sequences.
- **Metrics:** precision, recall and F1 with "hallucinated" as the positive class, after dropping "not sure" (as in MedHallu), overall and per difficulty, with a bootstrap 95% interval. Also coverage, accuracy on answered items, risk-coverage curve, error overlap between verifiers, tokens and latency.

## Phase 1 results

All numbers below come from the files in [`results/phase1/`](results/phase1/) (final Colab run, `notebooks/phase1_baseline_colab.ipynb`).

**Baselines on the test split** (600 answers: 300 true, 300 hallucinated; Llama 3.1 8B through Ollama, the authors' prompt, temperature 0.3):

| Run | Precision | Recall | F1 (95% interval) | Easy F1 | Medium F1 | Hard F1 |
| --- | --- | --- | --- | --- | --- | --- |
| B0: no knowledge (ours) | 0.706 | 0.360 | 0.477 (0.416-0.533) | 0.655 | 0.473 | 0.271 |
| B0: MedHallu paper | 0.791 | not reported | 0.522 | 0.679 | 0.515 | 0.372 |
| B1: source abstract given (ours) | 0.835 | 0.690 | 0.755 (0.714-0.794) | 0.851 | 0.796 | 0.615 |
| B1: MedHallu paper | 0.775 | not reported | 0.797 | 0.880 | 0.796 | 0.722 |
| B2: "not sure" allowed (ours) | 0.772 | 0.327 | 0.459 (0.401-0.514) | 0.658 | 0.420 | 0.260 |

The published B0 score (0.522) falls inside our 95% interval; the published B1 score (0.797) is just above it (upper end 0.794). Giving the source abstract raises F1 by 0.278 (paper: 0.275). Our hard-answer F1 is lower than the paper's, likely because of Ollama's 4-bit model and the smaller test split. In B2, Llama never chose "not sure" (coverage 1.000), so the authors' abstain option did not work for this model. An earlier run gave F1 0.479 (B0) and 0.761 (B1); the small gap comes from sampling at temperature 0.3, so Phase 2 reports the mean of three runs.

**Full system, end-to-end check on the dev split** (40 answers, setting A: PubMed with the source blocked; Llama 3.1 8B, Mistral 7B and Phi-4-mini; `reason` prompt; vote needs 2 of 3): coverage 0.950, precision 0.654, recall 0.895, F1 0.756 (0.595-0.889), accuracy on answered 0.711.

**Verifier prompt chosen on the validation split** (200 answers, setting A, vote needs 2 of 3):

| Prompt style | Vote coverage | Vote F1 (95% interval) | Vote accuracy on answered |
| --- | --- | --- | --- |
| digits (0 / 1 / 2) | 0.665 | 0.595 (0.476-0.694) | 0.662 |
| words (FACTUAL / HALLUCINATED / UNSURE) | 0.795 | 0.447 (0.322-0.556) | 0.642 |
| **reason** (short check, then `Verdict:`) | **0.890** | **0.680 (0.604-0.749)** | 0.646 |

In our first validation run (not saved), Phi-4-mini mostly replied "not sure" with digits, and the models rarely said "hallucinated" with words. `reason` is the default for all later runs. These numbers only pick the prompt; the test split is run once with the chosen setup.

**First look at the gaps on the validation split** (`reason` prompt):

| | Result | File |
| --- | --- | --- |
| Each verifier alone (F1, coverage) | Llama 0.690 (0.825), Mistral 7B 0.611 (0.830), Phi-4-mini 0.673 (0.975); vote 0.680 (0.890) | `sysA-val-reason-states.jsonl` |
| Gap 1: shared mistakes | Each pair is wrong together 1.6 to 1.8 times as often as independent errors would be (error correlation 0.41 to 0.47) | `sysA-val-reason-overlap.json` |
| Gap 2: source leakage | PubMed returned the answer's own source abstract for 162 of 200 answers (81%); all were blocked. 2.5 clean abstracts were kept per answer on average | `sysA-val-reason-states.jsonl` |
| Gap 3: vote rule | Among rules answering at least 70% of items, 1 of 3 agreeing gave the best accuracy on answered items (0.654, coverage 0.955); 2 of 3 gave 0.646 (0.890); 3 of 3 answered only 40.5% | `sysA-val-reason-tune.json` |
| Cost | About 12 seconds of model time per answer for the three verifiers on a T4 GPU | `sysA-val-reason-states.jsonl` |

These are early validation results. So far the three model families share many mistakes, and the vote does not beat the best single verifier. Phase 2 tests this on the test split against the self-consistency control (B4).

## Changes since Phase 1

- **Verifier 2 is Mistral 7B, run locally through Ollama**, because the free Mistral API allowed 0 requests per minute on our account (error 429 on the first call). Same model family, no rate limit.
- **The verifier prompt is `reason`** (a short check, then `Verdict: ...`), chosen on the validation split over `digits` and `words` (vote F1 0.680 vs 0.595 and 0.447).
- **The vote rule stays at 2 of 3**, although `tune` preferred 1 of 3: 2 of 3 is a true majority, it abstains more often so gap 3 can be measured, and its accuracy on answered items is within 1-2 answers of 1 of 3 (0.646 vs 0.654).
- **New in Phase 2:** a second evidence round with claim queries when the votes split (the agentic loop from the proposal), one re-ask when a reply has no readable verdict, a debate ablation, seeds for repeat runs, structured trajectories (`steps`) and PubMed tool-call logs (`tool_calls`), leakage and efficiency summaries, and a resume check for baseline folders. See [docs/devlog.md](docs/devlog.md).

## Phase 2 experiments

Every Phase 2 run is defined once in [`experiments/plan.py`](experiments/plan.py) and run in order by `experiments/run.py`; [`notebooks/phase2_colab.ipynb`](notebooks/phase2_colab.ipynb) runs it all on Colab or Kaggle. All runs use the pre-registered setup: vote 2 of 3, verifier temperature 0, the `reason` prompt, seeds 1-3 for the main configurations and seed 1 for the rest.

| Experiment | Runs (test split) | What it shows |
| --- | --- | --- |
| E1 Other baselines | `b4-test-s1..s3` (B4: Llama 3.1 8B sampled 3 times at temperature 0.7, vote 2 of 3), `b5-test-s1` (B5: simple RAG, first PubMed query only); B3 = each verifier alone, read from `sysA-test-s1` round 1 | Whether three model families beat three samples of one model, plain retrieval, and each verifier alone |
| E2 MedHallu baselines | `b0-`, `b1-`, `b2-test-s1..s3` (no knowledge, source abstract, "not sure" allowed) | The paper's settings reproduced, with the spread over three seeds |
| E3 Proposed system | `sysA-test-s1..s3` (PubMed, source blocked, chosen second-round trigger), `sysB-test-s1` (source abstract given) | The main result, against B0-B5 |
| E4 Gap-wise experiments (analysis of the E1, E3 and E5 runs; no extra runs) | **Gap 1 / H1:** round-1 three-family vote of `sysA-test-s1..s3` vs B3 (each verifier alone) and B4 (`b4-test-s1..s3`), plus `overlap` for every verifier pair: cross-family pairs vs the `llama@1..3` pairs.<br>**Gap 2 / H2:** System A vs B0 and System B vs B1 (F1 overall and per difficulty), share of searches where the source was found and blocked, and `noblock-test-s1` (how far leakage would raise F1).<br>**Gap 3 / H3:** test coverage of `sysA` vs the val coverage saved in `experiments/final_config.json` (within 5 points), accuracy on answered items vs the forced-answer rule (`evaluate --forced`), risk-coverage curve, and B2 | Tests the Phase 1 hypotheses H1, H2 and H3 (below), one per gap, on the test split |
| E5 Ablations | `none-test-s1` (no evidence), `noblock-test-s1` (source not blocked), `debate-test-s1` (verifiers see each other); second round off = `evaluate --round 1` on `sysA` | What each part adds, and how much source leakage would inflate the score (gap 2) |
| E6 Efficiency and reliability (analysis of every run) | The efficiency section of each run's `metrics.json`: model calls, prompt and completion tokens, model seconds and PubMed requests per item, share sent to round 2, re-asks, unreadable replies, abstention rate, max prompt tokens; mean and SD over seeds where there are 3 runs | What each configuration costs, and how reliably it runs |

```bash
python -m experiments.run --stage smoke             # one run per configuration on 6 dev answers: a crash check
python -m experiments.run --stage val               # val-sysA-off, val-sysA-split, val-sysA-nomaj
python -m experiments.choose                        # pre-registered rule -> experiments/final_config.json
python -m experiments.run --stage test --only E2    # test runs, one experiment at a time (or all)
python -m experiments.run --stage test --dry-run    # each command and how many items are left
python -m experiments.state save --to DIR          # back up the reply and PubMed caches (restore --from DIR)
#   add --backup-dir DIR to any stage to back up after every run
```

**Hypotheses** (from the Phase 1 proposal, also in [`experiments/facts.py`](experiments/facts.py)):

- **H1 (gap 1):** The three-family vote has a higher F1 than B4 (self-consistency), because cross-family pairs share fewer mistakes than samples of one model.
- **H2 (gap 2):** Setting A lands between the two baselines: its F1 is above B0 (no knowledge) but below B1 (source given).
- **H3 (gap 3):** Test coverage stays within 5 points of validation coverage, and accuracy on answered items is higher than when the same system must answer everything.

**Pre-registered rule** (fixed in [docs/devlog.md](docs/devlog.md) before any Phase 2 run): the second-round trigger is chosen on the validation split only, between `split` and `no-majority`: the highest accuracy on answered items among triggers with coverage >= 0.70, and `no-majority` (cheaper) if the two are within 0.005. "Second round off" is reported as an ablation, not chosen. The test stage refuses to start until `experiments/final_config.json` exists, and `choose` refuses to change it once any test run has results.

**Where results go:** each run in `results/runs/<name>/` (`config.json`, `states.jsonl` or `predictions.jsonl`, `metrics.json`); the chosen trigger with its validation numbers in `experiments/final_config.json` (commit it); the Ollama model versions in `results/runs/ollama_versions.txt`. On Colab, `results/runs` lives in `MyDrive/medverify_phase2`; the reply cache stays on local disk, because SQLite is not safe on network or cloud drives ([sqlite.org/useovernet.html](https://sqlite.org/useovernet.html)), and is backed up to `MyDrive/medverify_phase2/state` after every run (`--backup-dir`, `experiments/state.py`). On Kaggle the backup goes to `/kaggle/working/medverify_state`, and the last notebook cell zips it with `results/runs` as `medverify_state.zip`.

## Results

Every table and figure is rebuilt from files committed in [`results/phase2/`](results/phase2/), so each number can be reproduced from a fresh clone:

```bash
python -m experiments.export                 # after the runs: copy finished runs from results/runs to results/phase2
python -m experiments.report                 # tables, figures, failure analysis, cases and this gap table (test split)
python -m experiments.report --split val     # the same for the validation runs (tables/val, figures/val)
pip install -e ".[analysis]"                 # matplotlib, needed for the figures
```

- **Tables** in [`results/phase2/tables/`](results/phase2/tables/) (each as `.csv` and `.md`, missing runs listed at the top): `runs` (every run), `main` (mean ± SD over seeds, with the MedHallu reported numbers, B3 and System A round 1 / forced), `overlap`, `leakage`, `ablation`, `significance` (paired bootstrap with 10,000 resamples and exact McNemar), `gap_resolution`, and `failures` with `failures_summary.md`.
- **Figures** in [`results/phase2/figures/`](results/phase2/figures/) (`.png` at 300 dpi and `.pdf`, one IEEE column wide): `f1_by_difficulty`, `risk_coverage`, `error_overlap`, `ablation`, `cost`.
- **Trajectories** in [`logs/`](logs/): 20 sample states per run and the representative cases; the full states of every run are in `results/phase2/runs/<run>/states.jsonl.gz`.

How the hypotheses (see [Phase 2 experiments](#phase-2-experiments)) are tested: **H1** by `significance` (System A round-1 vote vs B4 and vs the best single verifier) and `overlap` (cross-family vs same-model pairs); **H2** by `main` and `significance` (System A vs B0 and B1, System B vs B1) and `leakage`; **H3** by the coverage in `main` against the val coverage in `experiments/final_config.json`, and System A vs its forced-answer version in `ablation`. The gap table below is written by `python -m experiments.report`; "Addressed?" follows the rule fixed in [docs/devlog.md](docs/devlog.md) before any test result existed.

<!-- GAP_TABLE_START -->
Pending: no Phase 2 test run has been exported yet. `python -m experiments.export`, then `python -m experiments.report`, fills this table.
<!-- GAP_TABLE_END -->

## Tests

```bash
pytest -q
```

All tests run offline with fake models and a fake PubMed, including a full run of every command.

## Working as a group

Both members should commit their own work from their own GitHub accounts (the rubric checks this). Work on separate branches or files, push often, and use clear commit messages, for example `add PubMed blocking test` or `run baseline on test split`.

## Reference

Pandit, S. et al. *MedHallu: A Comprehensive Benchmark for Detecting Medical Hallucinations in Large Language Models.* EMNLP 2025. Data: `UTAustin-AIHealth/MedHallu` (MIT licence). Code: https://github.com/MedHallu/MedHallu
