"""Pick the second-round trigger on the validation split only, with the pre-registered rule.

  python -m experiments.choose

Reads results/runs/val-sysA-split and val-sysA-nomaj (metrics.json), applies the rule fixed in
docs/devlog.md before any Phase 2 run, and writes experiments/final_config.json, which the test
stage needs. val-sysA-off ("second round off") is reported as an ablation, never chosen.
"""

from __future__ import annotations

import argparse
import json
from datetime import date

from medverify import cli

from . import run as runner
from .plan import TEST_RUNS

MIN_COVERAGE = 0.70
TIE_MARGIN = 0.005
CANDIDATES = {"split": "val-sysA-split", "no-majority": "val-sysA-nomaj"}
ABLATION = "val-sysA-off"
RULE = (
    "On the validation split only: among the triggers 'split' and 'no-majority', keep those with coverage >= 0.70; "
    "choose the one with the highest accuracy on answered items; if the two are within 0.005 of each other, choose "
    "'no-majority' (cheaper). 'Second round off' is reported as an ablation (evaluate --round 1), not chosen."
)


def choose_trigger(metrics: dict[str, dict]) -> str:
    """Apply the pre-registered rule to {trigger: overall val metrics}."""
    eligible = {t: m for t, m in metrics.items() if m["coverage"] >= MIN_COVERAGE}
    if not eligible:
        raise SystemExit(
            f"No trigger reaches coverage {MIN_COVERAGE:.2f} on the validation split, so the pre-registered rule "
            "cannot choose. Report this and decide in the devlog before any test run."
        )
    if len(eligible) == 2:
        accuracy = [m["selective_accuracy"] for m in eligible.values()]
        if abs(accuracy[0] - accuracy[1]) <= TIE_MARGIN + 1e-9:
            return "no-majority"
    return max(eligible, key=lambda t: eligible[t]["selective_accuracy"])


def _val_run(name: str) -> dict:
    """One finished full validation run: its overall metrics and efficiency section."""
    folder = cli.RUNS_DIR / name
    if not (folder / "metrics.json").exists():
        raise SystemExit(f"{folder.as_posix()} has no metrics.json. Run `python -m experiments.run --stage val` first.")
    config = json.loads((folder / "config.json").read_text())
    if config.get("split") != "val" or config.get("limit"):
        raise SystemExit(f"{name} is not a full validation run (split {config.get('split')}, limit {config.get('limit')}).")
    metrics = json.loads((folder / "metrics.json").read_text())
    return {"overall": metrics["overall"], "efficiency": metrics.get("efficiency", {})}


def _row(name: str, trigger: str, run: dict) -> dict:
    o, e = run["overall"], run["efficiency"]
    return {
        "run": name,
        "trigger": trigger,
        "coverage": o["coverage"],
        "selective_accuracy": o["selective_accuracy"],
        "f1": o["f1"],
        "eligible": o["coverage"] >= MIN_COVERAGE if trigger in CANDIDATES else None,
        "round2_share": e.get("round2_share"),
        "model_seconds_per_item": e.get("model_seconds_per_item"),
    }


def runs_with_test_results() -> list[str]:
    """Names of test runs that already saved results."""
    found = []
    for run in TEST_RUNS:
        for file in ("states.jsonl", "predictions.jsonl"):
            path = cli.RUNS_DIR / run.name / file
            if path.exists() and path.stat().st_size > 0:
                found.append(run.name)
    return found


def format_rows(rows: list[dict]) -> str:
    lines = [f"{'run':<16}{'trigger':<13}{'cover':>7}{'sel.acc':>9}{'F1':>7}{'eligible':>10}{'round 2':>9}{'model s/item':>14}"]
    for r in rows:
        eligible = "-" if r["eligible"] is None else ("yes" if r["eligible"] else "no")
        share = "-" if r["round2_share"] is None else f"{r['round2_share']:.3f}"
        seconds = "-" if r["model_seconds_per_item"] is None else f"{r['model_seconds_per_item']:.1f}"
        lines.append(f"{r['run']:<16}{r['trigger']:<13}{r['coverage']:>7.3f}{r['selective_accuracy']:>9.3f}"
                     f"{r['f1']:>7.3f}{eligible:>10}{share:>9}{seconds:>14}")
    return "\n".join(lines)


def main(argv=None) -> int:
    argparse.ArgumentParser(prog="experiments.choose", description=__doc__.splitlines()[0]).parse_args(argv)
    done = runs_with_test_results()
    if done:
        raise SystemExit(
            f"Test runs already have results ({', '.join(done)}), so the trigger can no longer be changed; "
            f"{runner.FINAL_CONFIG.as_posix()} is left as it is."
        )
    runs = {trigger: _val_run(name) for trigger, name in CANDIDATES.items()}
    chosen = choose_trigger({t: r["overall"] for t, r in runs.items()})
    rows = [_row(ABLATION, "off", _val_run(ABLATION))] + [_row(CANDIDATES[t], t, r) for t, r in runs.items()]
    best = runs[chosen]["overall"]
    config = {
        "trigger": chosen,
        "chosen_run": CANDIDATES[chosen],
        # needed later for H3: test coverage within 5 points of val coverage
        "val": {"coverage": best["coverage"], "selective_accuracy": best["selective_accuracy"], "f1": best["f1"]},
        "candidates": rows,
        "rule": RULE,
        "min_coverage": MIN_COVERAGE,
        "tie_margin": TIE_MARGIN,
        "date": date.today().isoformat(),
        "git_commit": cli._git_commit(),
    }
    runner.FINAL_CONFIG.write_text(json.dumps(config, indent=2) + "\n")
    print(format_rows(rows))
    print(f"\nRule: {RULE}")
    print(f"Chosen trigger: {chosen} (val coverage {best['coverage']:.3f}, accuracy on answered "
          f"{best['selective_accuracy']:.3f}). Saved to {runner.FINAL_CONFIG.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
