"""Turn the exported runs in results/phase2 into every table the paper and README need.

  python -m experiments.export          # once the runs are finished (copies results/runs -> results/phase2)
  python -m experiments.report [--split test|val]

Reads ONLY results/phase2, so a fresh clone rebuilds the same tables. Works with any subset of
runs: whatever is missing is skipped and listed at the top of every .md table, never guessed.
Tables go to results/phase2/tables/ (and tables/val/ for --split val), each as .csv and .md.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Callable, Optional

from medverify.llm import model_family, split_spec
from medverify.metrics import error_overlap, leakage_summary
from medverify.pipeline import verifier_verdicts

from . import facts
from .plan import TEST_RUNS, VAL_RUNS
from .results import (
    METRICS, PHASE2, RunData, Side, load_final_config, load_items, load_run, side_drop, side_final,
    side_forced, side_metrics, side_min_agree, side_round1, side_single, verifier_names,
)
from .stats import mcnemar, mean_sd, paired_bootstrap

LABELS = {
    "b0-test": "B0: no knowledge", "b1-test": "B1: source abstract given", "b2-test": 'B2: "not sure" allowed',
    "b4-test": "B4: self-consistency (Llama 3.1 8B x3, T 0.7)", "b5-test": "B5: simple RAG (first query only)",
    "sysA-test": "System A (PubMed, source blocked)", "sysB-test": "System B (source abstract given)",
    "none-test": "Ablation: no evidence", "noblock-test": "Ablation: source not blocked",
    "debate-test": "Ablation: debate", "val-sysA-off": "Val: second round off",
    "val-sysA-split": "Val: second round on split", "val-sysA-nomaj": "Val: second round on no-majority",
}
CHOSEN_VAL_RUN = {"split": "val-sysA-split", "no-majority": "val-sysA-nomaj"}


def config_key(name: str) -> str:
    return re.sub(r"-s\d+$", "", name)


@dataclass
class Report:
    split: str
    phase2: Path
    items: list
    runs: dict[str, RunData]
    missing: list[str]
    final_config: Optional[dict]
    chosen_val: Optional[RunData]
    out: Path
    figures: Path
    resamples: int = 10_000
    _cache: dict = field(default_factory=dict)

    def runs_with(self, prefix: str) -> list[RunData]:
        return sorted((r for n, r in self.runs.items() if config_key(n) == prefix), key=lambda r: r.seed or 0)

    def system_a(self) -> list[RunData]:
        """System A: the three test seeds, or the chosen val run with --split val."""
        if self.split == "val":
            return [self.chosen_val] if self.chosen_val else []
        return self.runs_with("sysA-test")


def build(split: str = "test", phase2: Path = PHASE2, resamples: int = 10_000) -> Report:
    phase2 = Path(phase2)
    expected = [r.name for r in (TEST_RUNS if split == "test" else VAL_RUNS)]
    runs = {n: r for n in expected if (r := load_run(n, phase2)) is not None}
    final = load_final_config(phase2)
    chosen = load_run(CHOSEN_VAL_RUN[final["trigger"]], phase2) if final else None
    missing = [n for n in expected if n not in runs]
    if final is None:
        missing.append("final_config.json")
    sub = "" if split == "test" else "val"
    return Report(split, phase2, load_items(split, phase2), runs, missing, final, chosen,
                  phase2 / "tables" / sub, phase2 / "figures" / sub, resamples)


# ------------------------------------------------------------- writing

def _fmt(value, column: str = "") -> str:
    if value is None or value == "":
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.4f}" if column.endswith("p") else f"{value:.3f}"
    return str(value)


def pm(mean, sd) -> str:
    if mean is None:
        return "-"
    return f"{mean:.3f}" if sd is None else f"{mean:.3f} ± {sd:.3f}"


def write_table(rep: Report, name: str, title: str, rows: list[dict], columns: list[str],
                md_rows: Optional[list[dict]] = None, md_columns: Optional[list[str]] = None,
                notes: tuple = ()) -> None:
    """name.csv with every column, name.md with the missing runs on top and the readable columns."""
    rep.out.mkdir(parents=True, exist_ok=True)
    with (rep.out / f"{name}.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: ("" if row.get(c) is None else row.get(c)) for c in columns})
    md_rows, md_columns = md_rows if md_rows is not None else rows, md_columns or columns
    lines = [f"# {title}", "", f"Missing runs: {', '.join(rep.missing) or 'none'}", ""]
    if md_rows:
        lines += ["| " + " | ".join(md_columns) + " |", "| " + " | ".join("---" for _ in md_columns) + " |"]
        lines += ["| " + " | ".join(_fmt(r.get(c), c) for c in md_columns) + " |" for r in md_rows]
    else:
        lines.append("No rows: the runs this table needs are missing.")
    lines += [""] + [f"{n}" for n in notes]
    (rep.out / f"{name}.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


# ------------------------------------------------------------- runs and main

def runs_table(rep: Report) -> list[dict]:
    plan_runs = {r.name: r for r in TEST_RUNS + VAL_RUNS}
    rows = []
    for name, run in rep.runs.items():
        rows.append({"run": name, "experiment": plan_runs[name].experiment, "seed": run.seed}
                    | side_metrics(rep.items, side_final(run)))
    columns = ["run", "experiment", "seed"] + METRICS
    md = ["run", "experiment", "seed", "n", "coverage", "precision", "recall", "f1", "f1_ci_low", "f1_ci_high",
          "selective_accuracy", "easy_f1", "medium_f1", "hard_f1", "model_calls", "model_seconds"]
    write_table(rep, "runs", "Every run", rows, columns, md_columns=md)
    return rows


def summarise(rep: Report, key: str, label: str, sides: list[Side], source: str = "ours (rerun)") -> dict:
    """Mean and SD over seeds of every metric for one configuration."""
    per_seed = [side_metrics(rep.items, s) for s in sides]
    row = {"config": key, "label": label, "source": source, "n_seeds": len(sides), "runs": " ".join(s.run for s in sides)}
    for m in METRICS:
        mean, sd, _ = mean_sd([r[m] for r in per_seed])
        row[f"{m}_mean"], row[f"{m}_sd"] = mean, sd
    return row


def main_table(rep: Report) -> list[dict]:
    rows = []
    for key in dict.fromkeys(config_key(n) for n in rep.runs):
        rows.append(summarise(rep, key, LABELS.get(key, key), [side_final(r) for r in rep.runs_with(key)]))
    if rep.split == "test":
        for key, paper in facts.MEDHALLU_REPORTED.items():
            row = {"config": f"paper-{key.lower()}", "label": paper["label"], "source": "paper", "n_seeds": None}
            for m in ("precision", "recall", "f1", "easy_f1", "medium_f1", "hard_f1"):
                row[f"{m}_mean"] = paper[m]
            rows.append(row)
    sys_a = rep.system_a()
    if sys_a:
        for name in verifier_names(sys_a[0]):
            rows.append(summarise(rep, f"b3-{name}", f"B3: {name} alone (System A round 1)",
                                  [side_single(r, name) for r in sys_a]))
        rows.append(summarise(rep, "sysA-round1", "System A, round 1 (second round off)", [side_round1(r) for r in sys_a]))
        rows.append(summarise(rep, "sysA-forced", "System A, forced answer", [side_forced(r) for r in sys_a]))
    columns = ["config", "label", "source", "n_seeds", "runs"] + [f"{m}_{s}" for m in METRICS for s in ("mean", "sd")]
    md_rows = [{
        "configuration": r["label"], "source": r["source"], "seeds": r["n_seeds"],
        "F1": pm(r.get("f1_mean"), r.get("f1_sd")), "precision": pm(r.get("precision_mean"), r.get("precision_sd")),
        "recall": pm(r.get("recall_mean"), r.get("recall_sd")), "coverage": pm(r.get("coverage_mean"), r.get("coverage_sd")),
        "acc. answered": pm(r.get("selective_accuracy_mean"), r.get("selective_accuracy_sd")),
        "easy F1": pm(r.get("easy_f1_mean"), r.get("easy_f1_sd")), "medium F1": pm(r.get("medium_f1_mean"), r.get("medium_f1_sd")),
        "hard F1": pm(r.get("hard_f1_mean"), r.get("hard_f1_sd")), "calls/item": pm(r.get("model_calls_mean"), None),
        "model s/item": pm(r.get("model_seconds_mean"), None),
    } for r in rows]
    write_table(rep, "main", "Main results (mean ± SD over seeds)", rows, columns, md_rows, list(md_rows[0]) if md_rows else ["configuration"],
                notes=(f"Paper rows: {facts.MEDHALLU_SOURCE}; the paper gives no recall.",) if rep.split == "test" else ())
    return rows


# ------------------------------------------------------------- overlap and leakage

def overlap_rows(rep: Report, runs: list[RunData]) -> list[dict]:
    """Error overlap for every pair of verifiers in each run (round-1 verdicts)."""
    rows = []
    for run in runs:
        per = verifier_verdicts(run.states)
        for a, b in combinations(sorted(per), 2):
            o = error_overlap(rep.items, per[a], per[b])
            rows.append({"level": "pair", "pair_type": _pair_type(a, b), "pair": f"{a} vs {b}", "run": run.name,
                         "seed": run.seed, "n": o.get("n"), "both_wrong": o.get("both_wrong"),
                         "expected_if_independent": o.get("both_wrong_if_independent"),
                         "overlap_ratio": o.get("overlap_ratio"), "phi": o.get("phi_error_correlation")})
    return rows


def _pair_type(name_a: str, name_b: str) -> str:
    a, b = split_spec(name_a)[0], split_spec(name_b)[0]
    if a == b:
        return "same model"
    return "same family" if model_family(a) == model_family(b) else "cross-family"


def overlap_table(rep: Report) -> list[dict]:
    rows = overlap_rows(rep, rep.system_a()) + overlap_rows(rep, rep.runs_with("b4-test"))
    summary = []
    for pair_type in dict.fromkeys(r["pair_type"] for r in rows):
        for pair in dict.fromkeys(r["pair"] for r in rows if r["pair_type"] == pair_type):
            vals = [r for r in rows if r["pair"] == pair]
            summary.append(_overlap_summary("pair (mean over seeds)", pair_type, pair, vals))
        by_seed = {}
        for r in rows:
            if r["pair_type"] == pair_type:
                by_seed.setdefault(r["run"], []).append(r)
        seed_means = [{k: mean_sd([r[k] for r in rs])[0] for k in ("both_wrong", "expected_if_independent", "overlap_ratio", "phi")}
                      for rs in by_seed.values()]
        summary.append(_overlap_summary("pair type (mean over seeds)", pair_type, f"all {pair_type} pairs", seed_means))
    all_rows = rows + summary
    columns = ["level", "pair_type", "pair", "run", "seed", "n_seeds", "n", "both_wrong", "both_wrong_sd", "expected_if_independent",
               "overlap_ratio", "overlap_ratio_sd", "phi", "phi_sd"]
    write_table(rep, "overlap", "Error overlap between verifiers (round 1)", all_rows, columns,
                notes=("Ratio = both wrong / expected if independent; 1.0 means independent errors.",))
    return all_rows


def _overlap_summary(level: str, pair_type: str, pair: str, vals: list[dict]) -> dict:
    row = {"level": level, "pair_type": pair_type, "pair": pair, "n_seeds": len(vals)}
    for k in ("both_wrong", "expected_if_independent", "overlap_ratio", "phi"):
        row[k], sd, _ = mean_sd([v.get(k) for v in vals])
        if k != "expected_if_independent":
            row[f"{k}_sd"] = sd
    return row


def kept_leaks(rep: Report, run: RunData) -> int:
    """Kept abstracts that carry a leak mark or are the item's own source PMID."""
    pubid = {i.item_id: str(i.pubid) for i in rep.items if i.pubid is not None}
    return sum(1 for s in run.states for e in s.evidence if e.leak or e.pmid == pubid.get(s.item_id))


def leakage_table(rep: Report) -> list[dict]:
    rows = []
    for name, run in rep.runs.items():
        if run.states is None or not any(s.evidence_mode == "pubmed" for s in run.states):
            continue
        summary = leakage_summary(run.states)
        for round_name, m in summary.items():
            rows.append({"run": name, "round": round_name, "kept_leaks": kept_leaks(rep, run) if round_name == "round1" else None} | m)
    columns = ["run", "round", "items_searched", "source_pmid_share", "near_duplicate_share", "leaked_into_evidence_share",
               "mean_clean_abstracts", "items_no_evidence", "items_no_evidence_share", "kept_leaks"]
    write_table(rep, "leakage", "Source leakage in PubMed searches", rows, columns,
                notes=("kept_leaks counts kept abstracts with a leak mark or the item's own source PMID, over all rounds.",))
    return rows


# ------------------------------------------------------------- comparisons

def compare(rep: Report, a_sides: list[Side], b_sides: list[Side]) -> dict:
    """Seed 1 vs seed 1 (paired bootstrap and McNemar), plus the mean difference over matching seeds."""
    if not a_sides or not b_sides:
        return {}
    a1 = next((s for s in a_sides if s.seed == 1), a_sides[0])
    b1 = next((s for s in b_sides if s.seed == 1), b_sides[0])
    key = (a1.label, a1.run, b1.label, b1.run)
    if key not in rep._cache:
        shared = [i for i in rep.items if i.item_id in a1.codes and i.item_id in b1.codes]
        labels = [i.label for i in shared]
        pa, pb = [a1.codes[i.item_id] for i in shared], [b1.codes[i.item_id] for i in shared]
        rep._cache[key] = paired_bootstrap(labels, pa, pb, n_resamples=rep.resamples) | {
            f"mcnemar_{k}": v for k, v in mcnemar(labels, pa, pb).items()}
    out = dict(rep._cache[key])
    seeds = sorted({s.seed for s in a_sides} & {s.seed for s in b_sides})
    diffs = {"f1": [], "selective_accuracy": [], "coverage": []}
    for seed in seeds:
        ma = side_metrics(rep.items, next(s for s in a_sides if s.seed == seed), n_boot=0)
        mb = side_metrics(rep.items, next(s for s in b_sides if s.seed == seed), n_boot=0)
        for k in diffs:
            diffs[k].append(ma[k] - mb[k])
    for k, values in diffs.items():
        out[f"mean_diff_{k}"], out[f"sd_diff_{k}"], out["n_seeds"] = mean_sd(values)
    out["seeds_a"], out["seeds_b"] = len(a_sides), len(b_sides)
    return out


def _sides(runs: list[RunData], view: Callable[[RunData], Side]) -> list[Side]:
    return [view(r) for r in runs]


def best_single_verifier(rep: Report) -> Optional[str]:
    """The verifier with the best round-1 F1 on the chosen val run (B3 chosen on val, not on test)."""
    if rep.chosen_val is None:
        return None
    items = _val_items(rep)
    scores = {n: side_metrics(items, side_single(rep.chosen_val, n), n_boot=0)["f1"] for n in verifier_names(rep.chosen_val)}
    return max(scores, key=scores.get) if scores else None


def _val_items(rep: Report):
    return rep.items if rep.split == "val" else load_items("val", rep.phase2)


def comparisons(rep: Report) -> list[tuple[str, list[Side], list[Side], str]]:
    """(name, A sides, B sides, what is missing) for every significance test of this split."""
    sys_a = rep.system_a()
    full, r1 = _sides(sys_a, side_final), _sides(sys_a, side_round1)
    run = lambda prefix: _sides(rep.runs_with(prefix), side_final)  # noqa: E731
    best = best_single_verifier(rep)
    single = _sides(sys_a, lambda r: side_single(r, best)) if best else []
    chosen_name = CHOSEN_VAL_RUN.get((rep.final_config or {}).get("trigger"), "the chosen val run")
    out = [
        ("System A vs B0", full, run("b0-test"), ""),
        ("System A vs B1", full, run("b1-test"), ""),
        ("System B vs B1", run("sysB-test"), run("b1-test"), ""),
        ("System A round-1 vote vs B4", r1, run("b4-test"), ""),
        (f"System A round-1 vote vs best single verifier ({best or 'unknown'}, chosen on val)", r1, single,
         "" if best else f"needs {chosen_name} and final_config.json"),
        ("System A vs System A round 1", full, r1, ""),
        ("System A vs B5", full, run("b5-test"), ""),
        ("System A vs no evidence", full, run("none-test"), ""),
        ("No blocking vs System A", run("noblock-test"), full, ""),
        ("Debate vs System A round 1", run("debate-test"), r1, ""),
    ]
    if rep.split == "val":
        out = [c for c in out if c[0].startswith("System A round-1 vote vs best") or c[0] == "System A vs System A round 1"]
        out.append(("Val: second round off run vs chosen run", run("val-sysA-off"), full, ""))
    return out


def significance_table(rep: Report) -> list[dict]:
    rows = []
    for name, a, b, note in comparisons(rep):
        result = compare(rep, a, b)
        missing = note or ("" if result else "runs missing")
        row = {"comparison": name, "a": a[0].run if a else None, "b": b[0].run if b else None, "missing": missing or None}
        if result:
            row |= {k: result.get(k) for k in SIG_COLUMNS if k in result}
            if not (result.get("seeds_a") == 3 and result.get("seeds_b") == 3):
                for k in ("mean_diff_f1", "sd_diff_f1", "mean_diff_selective_accuracy", "sd_diff_selective_accuracy", "n_seeds"):
                    row[k] = None  # the mean over seeds is reported only when both sides have 3 seeds
        rows.append(row)
    columns = ["comparison", "a", "b", "missing"] + SIG_COLUMNS
    md = ["comparison", "n", "f1_diff", "f1_ci_low", "f1_ci_high", "f1_p", "acc_diff", "acc_ci_low", "acc_ci_high", "acc_p",
          "mcnemar_b", "mcnemar_c", "mcnemar_p", "mean_diff_f1", "sd_diff_f1", "missing"]
    write_table(rep, "significance", "Paired significance tests (A - B, same items)", rows, columns, md_columns=md, notes=(
        "Seed 1 vs seed 1: paired bootstrap over items (10,000 resamples, seed 0; each side's F1 and accuracy on its own "
        "answered items) with 95% interval and two-sided p; McNemar exact test on items both sides answered "
        "(b = A right and B wrong, c = A wrong and B right). Mean difference over matching seeds when both sides have 3 seeds.",))
    return rows


SIG_COLUMNS = ["n", "f1_diff", "f1_ci_low", "f1_ci_high", "f1_p", "acc_diff", "acc_ci_low", "acc_ci_high", "acc_p",
               "mcnemar_both_answered", "mcnemar_b", "mcnemar_c", "mcnemar_p", "n_seeds", "mean_diff_f1", "sd_diff_f1",
               "mean_diff_selective_accuracy", "sd_diff_selective_accuracy"]


# ------------------------------------------------------------- ablation

def ablations(rep: Report) -> list[tuple[str, list[Side], list[Side], str]]:
    """(component, variant sides, reference sides, reference name): each component switched off in turn."""
    sys_a = rep.system_a()
    full, r1 = _sides(sys_a, side_final), _sides(sys_a, side_round1)
    run = lambda prefix: _sides(rep.runs_with(prefix), side_final)  # noqa: E731
    rows = [
        ("Second round off", r1, full, "System A"),
        ("No evidence", run("none-test"), full, "System A"),
        ("No source blocking", run("noblock-test"), full, "System A"),
        ("One-family panel (B4)", run("b4-test"), r1, "System A round 1"),
        ("Debate instead of the independent vote", run("debate-test"), r1, "System A round 1"),
    ]
    for name in verifier_names(sys_a[0]) if sys_a else []:
        rows.append((f"Drop {name} (2 of 2 must agree)", _sides(sys_a, lambda r, n=name: side_drop(r, n)), r1, "System A round 1"))
    rows.append(("Forced answer (no abstention)", _sides(sys_a, side_forced), full, "System A"))
    for k in (1, 2, 3):
        rows.append((f"Vote {k} of 3", _sides(sys_a, lambda r, k=k: side_min_agree(r, k)), full, "System A (2 of 3)"))
    return rows


def ablation_table(rep: Report) -> list[dict]:
    rows = []
    for component, variant, reference, ref_name in ablations(rep):
        if not variant or not reference:
            rows.append({"component": component, "reference": ref_name, "missing": "runs missing"})
            continue
        c = compare(rep, variant, reference)
        seeds = sorted({s.seed for s in variant} & {s.seed for s in reference})
        var_m = [side_metrics(rep.items, s, n_boot=0) for s in variant if s.seed in seeds]
        ref_m = [side_metrics(rep.items, s, n_boot=0) for s in reference if s.seed in seeds]
        rows.append({
            "component": component, "reference": ref_name, "n_seeds": c.get("n_seeds"),
            "f1_reference": mean_sd([m["f1"] for m in ref_m])[0], "f1_variant": mean_sd([m["f1"] for m in var_m])[0],
            "delta_f1": c.get("mean_diff_f1"), "delta_f1_sd": c.get("sd_diff_f1"),
            "delta_f1_ci_low": c.get("f1_ci_low"), "delta_f1_ci_high": c.get("f1_ci_high"), "delta_f1_p": c.get("f1_p"),
            "delta_acc": c.get("mean_diff_selective_accuracy"), "delta_acc_ci_low": c.get("acc_ci_low"),
            "delta_acc_ci_high": c.get("acc_ci_high"), "delta_coverage": c.get("mean_diff_coverage"),
            "calls_reference": mean_sd([m["model_calls"] for m in ref_m])[0], "calls_variant": mean_sd([m["model_calls"] for m in var_m])[0],
            "seconds_reference": mean_sd([m["model_seconds"] for m in ref_m])[0],
            "seconds_variant": mean_sd([m["model_seconds"] for m in var_m])[0], "missing": None,
        })
    columns = ["component", "reference", "n_seeds", "f1_reference", "f1_variant", "delta_f1", "delta_f1_sd", "delta_f1_ci_low",
               "delta_f1_ci_high", "delta_f1_p", "delta_acc", "delta_acc_ci_low", "delta_acc_ci_high", "delta_coverage",
               "calls_reference", "calls_variant", "seconds_reference", "seconds_variant", "missing"]
    write_table(rep, "ablation", "Ablations: each component switched off in turn (variant - reference)", rows, columns, notes=(
        "Deltas are means over matching seeds; intervals and p values are the seed-1 paired bootstrap. "
        "Offline views (round 1, dropped verifier, forced answer, vote thresholds) come from the System A runs themselves.",))
    return rows


# ------------------------------------------------------------- main

def write_all(rep: Report, readme: Optional[Path] = Path("README.md")) -> dict:
    tables = {
        "runs": runs_table(rep), "main": main_table(rep), "overlap": overlap_table(rep), "leakage": leakage_table(rep),
        "ablation": ablation_table(rep), "significance": significance_table(rep),
    }
    if rep.split == "test":  # the success criteria are about the test split
        from .gaps import write_gap_table

        tables["gap_resolution"] = write_gap_table(rep, readme)
    return tables


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="experiments.report", description="Build every Phase 2 table from results/phase2.")
    p.add_argument("--split", default="test", choices=["test", "val"])
    args = p.parse_args(argv)
    rep = build(args.split)
    if not rep.items:
        raise SystemExit(f"{(rep.phase2 / f'items_{args.split}.csv').as_posix()} is missing: run `python -m experiments.export` first.")
    write_all(rep)
    print(f"Tables written to {rep.out.as_posix()}. Missing runs: {', '.join(rep.missing) or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
