"""Command line entry point:  python -m medverify <command> ...

  prepare   download MedHallu, attach PubMed IDs, write dev/val/test splits
  baseline  run the single-LLM MedHallu baseline on a split
  pipeline  run the multi-agent system (verifiers + optional PubMed evidence)
  evaluate  score a run (F1 per difficulty, coverage, cost)
  overlap   error overlap between verifiers (independence check)
  tune      pick the vote rule on the validation split
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from itertools import combinations
from pathlib import Path

from dotenv import load_dotenv

from . import data as D
from .llm import DEFAULT_VERIFIERS
from .metrics import error_overlap, evaluate, format_table, risk_coverage
from .prompts import DEFAULT_PROMPT_STYLE, PROMPT_STYLES
from .schemas import Verdict, VerificationState

DATA_DIR = Path("data/processed")
RUNS_DIR = Path("results/runs")


def _run_dir(name: str | None, prefix: str) -> Path:
    name = name or f"{prefix}-{datetime.now():%Y%m%d-%H%M%S}"
    path = RUNS_DIR / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _items(split: str, limit: int | None):
    items = D.load_split_items(DATA_DIR, split)
    return items[:limit] if limit else items


# ----------------------------------------------------------------- commands

def cmd_prepare(args):
    try:
        rows = D.load_medhallu(args.subset)
    except Exception as err:  # network, or Hugging Face asking for a login
        raise SystemExit(
            f"Could not download MedHallu ({err}).\n"
            "Check your internet connection. If Hugging Face asks you to log in, create a free "
            "token at https://huggingface.co/settings/tokens and put HF_TOKEN=... in .env."
        )
    print(f"Loaded {len(rows)} MedHallu rows ({args.subset}).")
    pubmedqa = D.load_pubmedqa_labeled(Path("data/raw/ori_pqal.json"))
    stats = D.attach_pubids(rows, pubmedqa)
    print(f"PubMed ID mapping: {stats}")
    splits = D.make_splits(rows, args.test_per_level, args.val_size, args.dev_size, args.seed)
    for name, split_rows in splits.items():
        D.write_jsonl(DATA_DIR / f"{name}.jsonl", split_rows)
    summary = {"subset": args.subset, "seed": args.seed, "pubid_mapping": stats, "splits": D.split_summary(splits)}
    (DATA_DIR / "split_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def cmd_baseline(args):
    from .baseline import run_baseline
    from .llm import make_model

    items = _items(args.split, args.limit)
    model = make_model(args.model)
    out = _run_dir(args.name, f"baseline-{args.model}-{args.knowledge}-{args.split}")
    config = vars(args) | {"provider": model.provider, "model_id": model.model, "items": len(items)}
    (out / "config.json").write_text(json.dumps(config, indent=2, default=str))
    verdicts = run_baseline(items, model, args.knowledge, args.unsure, out / "predictions.jsonl")
    _score(out, items, verdicts)


def cmd_pipeline(args):
    from .llm import make_model
    from .pipeline import VerifierCallError, final_verdicts, run_pipeline
    from .tools.pubmed import PubMedSearch

    items = _items(args.split, args.limit)
    names = [n.strip() for n in args.verifiers.split(",") if n.strip()]
    verifiers = {n: make_model(n, temperature=args.temperature) for n in names}
    pubmed = PubMedSearch() if args.evidence == "pubmed" else None
    out = _run_dir(args.name, f"pipeline-{args.evidence}-{args.split}")
    models = {n: m.model for n, m in verifiers.items()}
    states_path = out / "states.jsonl"
    old_config = out / "config.json"
    if states_path.exists() and states_path.stat().st_size > 0 and old_config.exists():
        old = json.loads(old_config.read_text())
        old_setup = (old.get("models"), old.get("prompt", "words"))  # runs before --prompt existed used words
        if old.get("models") and old_setup != (models, args.prompt):
            raise SystemExit(
                f"{out} already holds results from other verifiers or another prompt "
                f"({old_setup[0]}, prompt '{old_setup[1]}'). Use a new --name so the two runs are not mixed."
            )
    config = vars(args) | {"items": len(items), "models": models}
    old_config.write_text(json.dumps(config, indent=2, default=str))
    try:
        states = run_pipeline(
            items, verifiers, args.evidence, pubmed, args.k, args.min_agree, states_path, prompt_style=args.prompt
        )
    except VerifierCallError as err:
        done = len(D.read_jsonl(states_path, VerificationState)) if states_path.exists() else 0
        hint = ""
        if err.provider == "mistral":
            hint = (
                "\nThe Mistral API keeps refusing calls (free-plan limit). Check your limits at "
                "https://admin.mistral.ai, wait and re-run (finished items are kept), or use the local "
                "verifier (the default): ollama pull mistral:7b, then --verifiers llama,mistral-local,phi with a new --name."
            )
        raise SystemExit(f"Run stopped: {err}\n{done} of {len(items)} items were saved in {states_path}.{hint}")
    _score(out, items, final_verdicts(states))


def _load_run(run: Path):
    run = Path(run)
    config = json.loads((run / "config.json").read_text())
    items = _items(config["split"], config.get("limit"))
    states_path, preds_path = run / "states.jsonl", run / "predictions.jsonl"
    if states_path.exists() and states_path.stat().st_size > 0:
        return config, items, D.read_jsonl(states_path, VerificationState), None
    if preds_path.exists() and preds_path.stat().st_size > 0:
        return config, items, None, D.read_jsonl(preds_path, Verdict)
    raise SystemExit(f"No results in {run}: the run did not finish any item. Run it again first.")


def _score(out: Path, items, verdicts):
    result = evaluate(items, verdicts)
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    print(format_table(result))
    print(f"Saved to {out}")


def cmd_evaluate(args):
    from .pipeline import agreement_confidence, final_verdicts, verifier_verdicts

    config, items, states, verdicts = _load_run(args.run)
    if states is None:
        _score(Path(args.run), items, verdicts)
        return
    _score(Path(args.run), items, final_verdicts(states, args.min_agree))
    for name, vs in verifier_verdicts(states).items():
        print(f"\n-- verifier {name} on its own --")
        print(format_table(evaluate(items, vs, n_boot=0)))
    curve = risk_coverage(items, agreement_confidence(states))
    (Path(args.run) / "risk_coverage.json").write_text(json.dumps(curve, indent=2))


def cmd_overlap(args):
    from .pipeline import verifier_verdicts

    sources: dict[str, list[Verdict]] = {}
    items = None
    for run in args.runs:
        config, run_items, states, verdicts = _load_run(run)
        items = items or run_items
        if states is not None:
            sources.update(verifier_verdicts(states))
        else:
            sources[Path(run).name] = verdicts
    report = {}
    for a, b in combinations(sorted(sources), 2):
        report[f"{a} vs {b}"] = error_overlap(items, sources[a], sources[b])
    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))


def cmd_compare(args):
    """One table across pipeline runs: the vote and each verifier, plus how often each said 0/1/2."""
    from collections import Counter

    from .pipeline import final_verdicts, verifier_verdicts

    header = f"{'who':<15}{'cover':>7}{'F1':>7}{'sel.acc':>9}   answers 0 / 1 / 2 / unreadable"
    lines = []
    for run in args.runs:
        config, items, states, _ = _load_run(run)
        if states is None:
            raise SystemExit(f"{run} is not a pipeline run")
        lines += ["", f"== {Path(run).name}  (prompt: {config.get('prompt', '?')}, {len(states)} items)", header]
        who = {"vote": final_verdicts(states)} | verifier_verdicts(states)
        for name, verdicts in who.items():
            m = evaluate(items, verdicts, n_boot=0)["overall"]
            c = Counter(v.verdict for v in verdicts)
            lines.append(
                f"{name:<15}{m['coverage']:>7.3f}{m['f1']:>7.3f}{m['selective_accuracy']:>9.3f}"
                f"   {c[0]} / {c[1]} / {c[2]} / {c[None]}"
            )
    print("\n".join(lines))


def cmd_tune(args):
    from .pipeline import final_verdicts

    config, items, states, _ = _load_run(args.run)
    if states is None:
        raise SystemExit("tune needs a pipeline run (states.jsonl)")
    n_verifiers = max(len(s.verdicts) for s in states)
    rows, best = [], None
    for min_agree in range(1, n_verifiers + 1):
        m = evaluate(items, final_verdicts(states, min_agree), n_boot=0)["overall"]
        rows.append({"min_agree": min_agree, "coverage": m["coverage"], "selective_accuracy": m["selective_accuracy"], "f1": m["f1"]})
        if m["coverage"] >= args.min_coverage and (best is None or m["selective_accuracy"] > best["selective_accuracy"]):
            best = rows[-1]
    print(json.dumps({"candidates": rows, "chosen": best, "min_coverage": args.min_coverage}, indent=2))
    if config.get("split") != "val":
        print("Warning: tune on the val split, then apply the chosen rule to test.")


# ------------------------------------------------------------------- parser

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="medverify")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("prepare", help="download data and make splits")
    s.add_argument("--subset", default="pqa_labeled")
    s.add_argument("--test-per-level", type=int, default=100)
    s.add_argument("--val-size", type=int, default=100)
    s.add_argument("--dev-size", type=int, default=20)
    s.add_argument("--seed", type=int, default=42)
    s.set_defaults(func=cmd_prepare)

    s = sub.add_parser("baseline", help="single-LLM MedHallu baseline")
    s.add_argument("--model", default="llama", help="llama | mistral | phi | provider:model")
    s.add_argument("--split", default="dev", choices=["dev", "val", "test"])
    s.add_argument("--knowledge", default="none", choices=["none", "source"])
    s.add_argument("--unsure", action="store_true", help="allow the '2 = not sure' answer")
    s.add_argument("--limit", type=int)
    s.add_argument("--name")
    s.set_defaults(func=cmd_baseline)

    s = sub.add_parser("pipeline", help="multi-agent verification")
    s.add_argument("--verifiers", default=",".join(DEFAULT_VERIFIERS), help="comma-separated: llama, mistral, phi, mistral-local, gptoss")
    s.add_argument("--evidence", default="none", choices=["none", "pubmed", "source"])
    s.add_argument("--split", default="dev", choices=["dev", "val", "test"])
    s.add_argument("--k", type=int, default=3, help="evidence abstracts per item")
    s.add_argument("--min-agree", type=int, default=2)
    s.add_argument("--prompt", default=DEFAULT_PROMPT_STYLE, choices=list(PROMPT_STYLES),
                   help="verifier prompt style; choose it on the val split")
    s.add_argument("--temperature", type=float, default=0.0)
    s.add_argument("--limit", type=int)
    s.add_argument("--name")
    s.set_defaults(func=cmd_pipeline)

    s = sub.add_parser("evaluate", help="score a finished run")
    s.add_argument("run")
    s.add_argument("--min-agree", type=int, help="re-apply a different vote rule")
    s.set_defaults(func=cmd_evaluate)

    s = sub.add_parser("overlap", help="error overlap between verifiers")
    s.add_argument("runs", nargs="+")
    s.add_argument("--out")
    s.set_defaults(func=cmd_overlap)

    s = sub.add_parser("compare", help="compare pipeline runs (e.g. prompt styles on val)")
    s.add_argument("runs", nargs="+")
    s.set_defaults(func=cmd_compare)

    s = sub.add_parser("tune", help="choose the vote rule on val")
    s.add_argument("run")
    s.add_argument("--min-coverage", type=float, default=0.7)
    s.set_defaults(func=cmd_tune)
    return p


def main(argv=None):
    load_dotenv()
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
