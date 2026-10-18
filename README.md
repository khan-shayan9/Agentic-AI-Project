# MedVerify Agents

**Independent multi-agent verification of medical answers: detecting hallucinations with evidence retrieval and abstention.**

AI4015 Agentic AI semester project. Base paper: MedHallu (Pandit et al., EMNLP 2025). Everything here runs on free tools only.

## What it does

Given a medical question and an answer, the system decides whether the answer is **factual (0)**, **hallucinated (1)**, or says **not sure (2)**.

- **Baseline:** one LLM judges each answer, exactly as in the MedHallu paper (authors' prompt, settings and metric).
- **Proposed system:** an evidence agent searches PubMed (with the question's own source abstract blocked). Three verifiers from different model families (Llama 3.1 8B from Meta, Mistral 7B from Mistral AI, Phi-4-mini from Microsoft), all running locally, then judge separately with a word label (FACTUAL, HALLUCINATED or UNSURE), and a vote rule accepts, rejects or abstains.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the diagram and the data contracts.

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
  cli.py          the `python -m medverify` commands
notebooks/        Colab notebook that runs the baseline on a free GPU
tests/            offline tests (fake models, no internet needed)
docs/             architecture
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
```

Each run folder holds `config.json`, the raw outputs (`predictions.jsonl` or `states.jsonl`), and `metrics.json`.

## Key design decisions

- **Splits:** from the 1,000 human-annotated rows (`pqa_labeled`). Test = 300 rows (100 easy, 100 medium, 100 hard); val = 100 rows (only for tuning); dev = 20 rows (debugging). Fixed seed 42.
- **Two items per row:** each row's true and hallucinated answers are both judged (600 test items, half hallucinated). The authors' code shows one random answer per row; ours gives the same balance without randomness.
- **Faithful baseline:** prompts, temperature (0.3) and top_p (0.95) match the MedHallu code. The paper's main table uses a version without "not sure" that is not in the released code; ours removes those sentences from their prompt.
- **Parsing:** the authors' parser reads any reply containing "not" as hallucinated, so "not sure" became 1. We use a stricter parser and count unparseable replies separately. `parsing.parse_authors` is kept for comparison. Verifier prompt styles (`--prompt`): `digits` (0/1/2), `words` (FACTUAL/HALLUCINATED/UNSURE) and `reason` (a short check, then `Verdict: ...`). On the 40 dev answers, digits made Phi-4-mini reply "2" while its own explanation called the answer wrong, and words made the models rarely say HALLUCINATED, so the style is chosen on the validation split with `compare`. On the 200 validation answers (setting A) the vote reached F1 0.595 with digits (coverage 0.66), 0.444 with words (0.78) and 0.683 with reason (0.90), so `reason` is the default.
- **Leakage control:** each question's source PubMed ID comes from PubMedQA. The evidence agent drops that PMID and any abstract sharing 30% or more of its 8-word sequences.
- **Metrics:** precision, recall and F1 with "hallucinated" as the positive class, after dropping "not sure" (as in MedHallu), overall and per difficulty, with a bootstrap 95% interval. Also coverage, accuracy on answered items, risk-coverage curve, error overlap between verifiers, tokens and latency.

## Phase 1 results

Filled in from the Colab run (`notebooks/phase1_baseline_colab.ipynb`, step 10). Test split = 600 answers (300 true, 300 hallucinated).

| Run | Precision | Recall | F1 (95% interval) | Easy F1 | Medium F1 | Hard F1 |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline B0: Llama 3.1 8B, no knowledge | 0.728 | 0.357 | 0.479 (0.419-0.533) | 0.683 | 0.428 | 0.281 |
| MedHallu paper, same setting | 0.791 | not reported | 0.522 | 0.679 | 0.515 | 0.372 |
| Baseline B1: Llama 3.1 8B, source abstract given | 0.833 | 0.700 | 0.761 (0.719-0.799) | 0.871 | 0.813 | 0.592 |
| MedHallu paper, same setting | 0.775 | not reported | 0.797 | 0.880 | 0.796 | 0.722 |

Both published F1 scores fall inside our 95% intervals. Giving the source abstract raises F1 by 0.282 (paper: 0.275). Our hard-answer F1 is lower than the paper's, likely because of Ollama's 4-bit model and the smaller test split. Coverage was 1.000 in both runs.

**Verifier prompt chosen on the validation split** (200 answers, setting A: PubMed with the source blocked; Llama 3.1 8B, Mistral 7B and Phi-4-mini; vote needs 2 of 3):

| Prompt style | Vote coverage | Vote F1 | Vote accuracy on answered | Best single verifier F1 |
| --- | --- | --- | --- | --- |
| digits (0 / 1 / 2) | 0.660 | 0.595 | 0.659 | Phi 0.740 (coverage 0.370) |
| words (FACTUAL / HALLUCINATED / UNSURE) | 0.780 | 0.444 | 0.647 | Llama 0.613 |
| **reason** (short check, then `Verdict:`) | **0.900** | **0.683** | 0.644 | Llama 0.689 |

These are validation numbers used to pick the prompt, not final results; the test split is run once with the chosen setup. With `reason`, the vote did not beat Llama alone on F1 (0.683 vs 0.689), which is exactly what gap 1 tests on the test split.

## Tests

```bash
pytest -q
```

All tests run offline with fake models and a fake PubMed, including a full run of every command.

## Working as a group

Both members should commit their own work from their own GitHub accounts (the rubric checks this). Work on separate branches or files, push often, and use clear commit messages, for example `add PubMed blocking test` or `run baseline on test split`.

## Reference

Pandit, S. et al. *MedHallu: A Comprehensive Benchmark for Detecting Medical Hallucinations in Large Language Models.* EMNLP 2025. Data: `UTAustin-AIHealth/MedHallu` (MIT licence). Code: https://github.com/MedHallu/MedHallu
