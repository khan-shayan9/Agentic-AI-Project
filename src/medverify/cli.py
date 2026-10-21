"""Command line entry point:  python -m medverify <command> ...

  prepare   download MedHallu, attach PubMed IDs, write dev/val/test splits
  baseline  run the single-LLM MedHallu baseline on a split
  pipeline  run the multi-agent system (verifiers + optional PubMed evidence)
  evaluate  score a run (F1 per difficulty, coverage, cost)
  overlap   error overlap between verifiers (independence check)
  compare   one table across pipeline runs (e.g. prompt styles on val)
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


# ------------------------------------------------------------- run setup

# What a run folder was made with. Runs saved before an option existed count as the Phase 1
# default for it (and before --prompt existed, the prompt was "words").
PHASE1_SETUP = {
    "prompt": "words", "temperature": 0.0, "seed": None, "evidence": "none",
    "debate": False, "reask": False, "block": True, "retry": True, "second_round": "off", "k2": 2,
}


def _pipeline_setup(args, models: dict[str, str]) -> dict:
    """The settings that must match before a run folder can be resumed."""
    return {
        "models": models,
        "prompt": args.prompt,
        "temperature": args.temperature,
        "seed": args.seed,
        "evidence": args.evidence,
        "debate": args.debate,
        "reask": not args.no_reask,
        "block": not args.no_block,
        "retry": not args.no_retry,
        "second_round": args.second_round,
        "k2": args.k2,
    }


def _saved_setup(config: dict) -> dict:
    """The setup stored in an older config.json, filling in Phase 1 defaults for missing options."""
    if "setup" in config:
        return PHASE1_SETUP | config["setup"]
    setup = {"models": config.get("models")}
    for key, default in PHASE1_SETUP.items():
        setup[key] = config.get(key, default)
    for flag in ("reask", "block", "retry"):
        if f"no_{flag}" in config:
            setup[flag] = not config[f"no_{flag}"]
    return setup


# Baseline folders from Phase 1 have no "setup" in config.json; they count as no seed.
BASELINE_DEFAULTS = {"knowledge": "none", "unsure": False, "seed": None}


def _baseline_setup(args) -> dict:
    """The baseline settings that must match before a run folder can be resumed."""
    return {"model": args.model, "knowledge": args.knowledge, "unsure": args.unsure, "seed": args.seed}


def _saved_baseline_setup(config: dict) -> dict:
    if "setup" in config:
        return config["setup"]
    return {"model": config.get("model")} | {key: config.get(key, d) for key, d in BASELINE_DEFAULTS.items()}


def _check_resume(out: Path, results: Path, saved: dict | None, setup: dict, what: str) -> None:
    """Stop when a run folder already holds results that were made with another setup."""
    if saved is None or not results.exists() or results.stat().st_size == 0:
        return
    changed = [f"{key} {saved.get(key)!r} -> {value!r}" for key, value in setup.items() if saved.get(key) != value]
    if changed:
        raise SystemExit(
            f"{out} already holds results from {what} ({'; '.join(changed)}). "
            "Use a new --name so the two runs are not mixed."
        )


def _git_commit() -> str | None:
    """Commit hash of the code that made the run (best effort)."""
    import subprocess

    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent,
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


def _ollama_digests(models: dict) -> dict[str, str | None]:
    """Exact Ollama model versions (digest from GET <host>/api/tags). Best effort: never fails a run."""
    import requests

    from .llm import PROVIDERS

    wanted = sorted({m.model for m in models.values() if getattr(m, "provider", "") == "ollama"})
    if not wanted:
        return {}
    host = PROVIDERS["ollama"].base_url.rstrip("/").removesuffix("/v1")
    try:
        tags = requests.get(f"{host}/api/tags", timeout=10).json().get("models", [])
        found = {t.get("name"): t.get("digest") for t in tags} | {t.get("model"): t.get("digest") for t in tags}
    except Exception:
        found = {}
    return {name: found.get(name) for name in wanted}


def _verifier_record(models: dict, prompt_style: str | None = None) -> dict:
    """Everything needed to say exactly which models and prompt made a run."""
    import hashlib

    from .prompts import verifier_instructions

    record = {
        "model_settings": {
            name: {
                "provider": getattr(m, "provider", None),
                "model_id": m.model,
                "temperature": getattr(m, "temperature", None),
                "top_p": getattr(m, "top_p", None),
                "max_tokens": getattr(m, "max_tokens", None),
                "seed": getattr(m, "seed", None),
            }
            for name, m in models.items()
        },
        "git_commit": _git_commit(),
        "ollama_digests": _ollama_digests(models),
    }
    if prompt_style:
        record["prompt_style"] = prompt_style
        record["verifier_instructions_sha256"] = hashlib.sha256(verifier_instructions(prompt_style).encode()).hexdigest()
    return record


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
    from .llm import make_model, model_seed

    items = _items(args.split, args.limit)
    model = make_model(args.model, seed=model_seed(args.model, args.seed))
    out = _run_dir(args.name, f"baseline-{args.model}-{args.knowledge}-{args.split}")
    setup = _baseline_setup(args)
    config_path, preds_path = out / "config.json", out / "predictions.jsonl"
    saved = _saved_baseline_setup(json.loads(config_path.read_text())) if config_path.exists() else None
    _check_resume(out, preds_path, saved, setup, "another model or setting")
    config = vars(args) | {"provider": model.provider, "model_id": model.model, "items": len(items), "setup": setup}
    config |= _verifier_record({args.model: model})
    config_path.write_text(json.dumps(config, indent=2, default=str))
    verdicts = run_baseline(items, model, args.knowledge, args.unsure, preds_path)
    _score(out, items, verdicts)


def _make_verifiers(args) -> dict:
    """One model per verifier name; 'llama@2' is Llama again with its own seed (sample 2)."""
    from .llm import make_model, model_seed

    names = [n.strip() for n in args.verifiers.split(",") if n.strip()]
    kwargs = {"temperature": args.temperature}
    if args.max_tokens is not None:
        kwargs["max_tokens"] = args.max_tokens
    return {n: make_model(n, seed=model_seed(n, args.seed), **kwargs) for n in names}


def cmd_pipeline(args):
    from .pipeline import VerifierCallError, final_verdicts, run_pipeline
    from .tools.pubmed import PubMedSearch

    if args.debate and args.second_round != "off":
        raise SystemExit(
            "--debate and --second-round cannot be used together: both add a round 2 "
            "(debate = see the others' verdicts; second round = new evidence, still independent). Run them separately."
        )
    items = _items(args.split, args.limit)
    verifiers = _make_verifiers(args)
    pubmed = PubMedSearch() if args.evidence == "pubmed" else None
    out = _run_dir(args.name, f"pipeline-{args.evidence}-{args.split}")
    models = {n: m.model for n, m in verifiers.items()}
    setup = _pipeline_setup(args, models)
    states_path = out / "states.jsonl"
    old_config = out / "config.json"
    saved = _saved_setup(json.loads(old_config.read_text())) if old_config.exists() else None
    if saved is not None and not saved["models"]:
        saved = None  # a folder with no record of its verifiers cannot be checked
    _check_resume(out, states_path, saved, setup, "other verifiers or another prompt or setting")
    config = vars(args) | {"items": len(items), "models": models, "setup": setup}
    config |= _verifier_record(verifiers, args.prompt)
    old_config.write_text(json.dumps(config, indent=2, default=str))
    try:
        states = run_pipeline(
            items, verifiers, args.evidence, pubmed, args.k, args.min_agree, states_path,
            prompt_style=args.prompt, reask=not args.no_reask, debate=args.debate,
            block=not args.no_block, retry=not args.no_retry,
            second_round_rule=args.second_round, k2=args.k2,
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


def _score(out: Path, items, verdicts, filename: str = "metrics.json"):
    result = evaluate(items, verdicts)
    (out / filename).write_text(json.dumps(result, indent=2))
    print(format_table(result))
    print(f"Saved to {out}")


def cmd_evaluate(args):
    from .pipeline import agreement_confidence, final_verdicts, verifier_verdicts

    config, items, states, verdicts = _load_run(args.run)
    if states is None:
        if args.forced or args.round:
            raise SystemExit("--forced and --round need a pipeline run (states.jsonl)")
        _score(Path(args.run), items, verdicts)
        return
    # The forced rule and other rounds are saved next to metrics.json, never over it.
    suffix = ("-forced" if args.forced else "") + (f"-round{args.round}" if args.round else "")
    system = final_verdicts(states, args.min_agree, round=args.round, forced=args.forced)
    _score(Path(args.run), items, system, f"metrics{suffix}.json")
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
    n_verifiers = max(len(s.last_round()) for s in states)
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
    s.add_argument("--seed", type=int, help="model seed for repeat runs (default: none, as in Phase 1)")
    s.add_argument("--limit", type=int)
    s.add_argument("--name")
    s.set_defaults(func=cmd_baseline)

    s = sub.add_parser("pipeline", help="multi-agent verification")
    s.add_argument("--verifiers", default=",".join(DEFAULT_VERIFIERS),
                   help="comma-separated: llama, mistral, phi, mistral-local, gptoss; 'llama@2' = Llama, sample 2")
    s.add_argument("--evidence", default="none", choices=["none", "pubmed", "source"])
    s.add_argument("--split", default="dev", choices=["dev", "val", "test"])
    s.add_argument("--k", type=int, default=3, help="evidence abstracts per item")
    s.add_argument("--no-block", action="store_true",
                   help="ablation: keep the source abstract and near copies (marked as leaks) instead of blocking them")
    s.add_argument("--second-round", default="off", choices=["off", "split", "no-majority"],
                   help="second evidence round (claim queries) when round 1 is split / has no majority; "
                        "PubMed mode only; choose on val")
    s.add_argument("--k2", type=int, default=2, help="new abstracts to add in the second round")
    s.add_argument("--no-retry", action="store_true",
                   help="use only the first PubMed query, no shorter retry (baseline B5, simple RAG)")
    s.add_argument("--min-agree", type=int, default=2)
    s.add_argument("--prompt", default=DEFAULT_PROMPT_STYLE, choices=list(PROMPT_STYLES),
                   help="verifier prompt style; choose it on the val split")
    s.add_argument("--temperature", type=float, default=0.0)
    s.add_argument("--seed", type=int, help="model seed; 'name@k' verifiers get (seed or 0) * 100 + k")
    s.add_argument("--debate", action="store_true",
                   help="ablation: a second round where each verifier sees the others' verdicts and reasoning")
    s.add_argument("--no-reask", action="store_true",
                   help="do not ask a verifier again when its reply has no readable verdict")
    s.add_argument("--max-tokens", type=int, help="reply length cap (default 512; 1024 for gpt-oss)")
    s.add_argument("--limit", type=int)
    s.add_argument("--name")
    s.set_defaults(func=cmd_pipeline)

    s = sub.add_parser("evaluate", help="score a finished run")
    s.add_argument("run")
    s.add_argument("--min-agree", type=int, help="re-apply a different vote rule")
    s.add_argument("--forced", action="store_true",
                   help="forced-answer rule: majority, no abstention, ties go to the first verifier")
    s.add_argument("--round", type=int, help="score one round's decision (1 = before the debate)")
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
