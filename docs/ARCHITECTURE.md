# Architecture

GitHub renders the diagram below. Boxes marked G1, G2 and G3 are where each research gap is addressed.

```mermaid
flowchart LR
    subgraph DATA[Data]
        MH[(MedHallu<br/>pqa_labeled)]
        PQ[(PubMedQA<br/>PMIDs)]
        SPL[Splits<br/>dev 20 / val 100 / test 300 rows]
        MH --> SPL
        PQ -->|question match| SPL
    end

    subgraph BASE[Baseline: MedHallu setup]
        B1[Single LLM<br/>Llama 3.1 8B<br/>authors' prompt]
    end

    subgraph SYS[Proposed system]
        EV[Evidence agent<br/>PubMed search<br/>G2: source PMID + near copies blocked]
        V1[Verifier: Llama 3.1 8B<br/>Meta, local Ollama]
        V2[Verifier: Mistral 7B<br/>Mistral AI, local Ollama]
        V3[Verifier: Phi-4-mini<br/>Microsoft, local Ollama]
        DEC[Decision step<br/>G3: vote with abstain<br/>threshold tuned on val]
        EV --> V1 & V2 & V3
        V1 & V2 & V3 -->|independent verdicts<br/>G1: 3 model families| DEC
    end

    PUB[(PubMed<br/>E-utilities)] <--> EV
    SPL --> B1
    SPL --> EV
    STATE[[VerificationState log<br/>evidence, blocked, votes, final, trace]]
    DEC --> STATE
    B1 --> M[Metrics<br/>F1 per tier, coverage,<br/>selective accuracy, error overlap,<br/>cost, latency]
    STATE --> M
```

## How the baseline differs from the proposed system

| | Baseline (MedHallu) | Proposed system |
| --- | --- | --- |
| Judges | 1 model | 3 models from different families, judging separately |
| Evidence | none, or the source abstract | PubMed search with the source blocked (A), or the source abstract (B) |
| Not sure | prompt option only | vote rule with abstention, tuned on the validation split |
| Reported | F1 after dropping "not sure" | F1, coverage, accuracy on answered items, error overlap, cost |

## Structured contracts (`src/medverify/schemas.py`)

| Contract | Holds |
| --- | --- |
| `Row` | One MedHallu row: question, source abstract, true answer, hallucinated answer, difficulty, category, PubMed ID |
| `Item` | One question plus one answer to verify, with its label (1 = hallucinated) |
| `LLMResult` | Raw reply, provider, model, latency, tokens, cache flag |
| `Evidence` / `BlockedEvidence` | Retrieved abstracts, and abstracts removed for leakage (with reason) |
| `Verdict` | One judge's verdict on one item: 0 factual, 1 hallucinated, 2 not sure, or None if unparseable |
| `VerificationState` | The full trajectory for one item in the proposed system |
