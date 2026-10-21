"""The gap-resolution table: one row per research gap, "Addressed?" decided by the rule in facts.py.

Columns follow the brief exactly: Gap | Method / component | Metric | Base paper result | Our result |
Addressed? (Fully / Partially / Not; Pending when runs are missing). The report writes it to
results/phase2/tables/gap_resolution.{csv,md} and into README.md between the GAP_TABLE markers.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

from . import facts
from .report import Report, _sides, best_single_verifier, compare, kept_leaks, overlap_rows, pm
from .results import side_final, side_forced, side_metrics, side_round1, side_single
from .stats import mean_sd

COLUMNS = ["Gap", "Method / component", "Metric", "Base paper result", "Our result", "Addressed?"]
START, END = "<!-- GAP_TABLE_START -->", "<!-- GAP_TABLE_END -->"
NOTE = ("Notes: \"Addressed?\" is set by the rule in `experiments/facts.py`, fixed in `docs/devlog.md` before any "
        "test result existed. The group reviewed the rule's output.")


def _metric(rep: Report, sides, key: str):
    return mean_sd([side_metrics(rep.items, s, n_boot=0)[key] for s in sides])[:2] if sides else (None, None)


MEAN_KEY = {"f1": "mean_diff_f1", "acc": "mean_diff_selective_accuracy"}


def _delta(result: dict, key: str = "f1") -> str:
    """Mean difference over seeds, with the seed-1 bootstrap interval."""
    mean = result.get(MEAN_KEY[key]) if result else None
    if mean is None:
        return "-"
    return f"{mean:+.3f} (seed-1 95% CI {result[f'{key}_ci_low']:+.3f} to {result[f'{key}_ci_high']:+.3f})"


def _comparison(result: dict, key: str = "f1") -> Optional[bool]:
    return facts.comparison_met(result.get(MEAN_KEY[key]), result.get(f"{key}_ci_low")) if result else None


def gap1(rep: Report) -> tuple[dict, list]:
    sys_a, b4_runs = rep.system_a(), rep.runs_with("b4-test")
    r1, b4 = _sides(sys_a, side_round1), _sides(b4_runs, side_final)
    best = best_single_verifier(rep)
    single = _sides(sys_a, lambda r: side_single(r, best)) if best else []
    vs_b4, vs_single = compare(rep, r1, b4), compare(rep, r1, single)
    pairs = overlap_rows(rep, sys_a) + overlap_rows(rep, b4_runs)
    cross = mean_sd([p["overlap_ratio"] for p in pairs if p["pair_type"] == "cross-family"])[0]
    same = mean_sd([p["overlap_ratio"] for p in pairs if p["pair_type"] == "same model"])[0]
    reported = None
    if sys_a and b4_runs:
        reported = any(p["pair_type"] == "cross-family" for p in pairs) and any(p["pair_type"] == "same model" for p in pairs) \
            and len({p["pair"] for p in pairs if p["pair_type"] == "cross-family"}) == 3
    parts = [reported, _comparison(vs_single), _comparison(vs_b4)]
    row = {
        "Gap": "1: independence",
        "Method / component": "Three verifiers from different model families vote independently (round 1); compared with "
                              "B4 (one model sampled 3 times, same number of calls) and each verifier alone (B3)",
        "Metric": "F1; error overlap ratio (both wrong / expected if independent)",
        "Base paper result": f"B4 (our rerun): F1 {pm(*_metric(rep, b4, 'f1'))}; best single verifier "
                             f"({best or 'unknown'}, our rerun, chosen on val): F1 {pm(*_metric(rep, single, 'f1'))}; "
                             "MedHallu reports no multi-model result",
        "Our result": f"Round-1 vote F1 {pm(*_metric(rep, r1, 'f1'))}; vs B4 {_delta(vs_b4)}; vs best single "
                      f"{_delta(vs_single)}; overlap ratio cross-family {pm(cross, None)} vs same model {pm(same, None)}",
    }
    return row, parts


def gap2(rep: Report) -> tuple[dict, list]:
    full = _sides(rep.system_a(), side_final)
    sys_b, b0, b1 = (_sides(rep.runs_with(p), side_final) for p in ("sysB-test", "b0-test", "b1-test"))
    vs_b0, b_vs_b1 = compare(rep, full, b0), compare(rep, sys_b, b1)
    setting_a = [r for r in rep.runs.values() if r.states and r.config.get("setup", {}).get("evidence") == "pubmed"
                 and r.config.get("setup", {}).get("block", True)]
    leaks = sum(kept_leaks(rep, r) for r in setting_a) if setting_a else None
    searched = [s for r in setting_a for s in r.states if s.evidence_mode == "pubmed"]
    found = sum(1 for s in searched if any(b.reason == "source_pmid" for b in s.blocked)) / len(searched) if searched else None
    paper = facts.MEDHALLU_REPORTED
    parts = [facts.leakage_met(leaks), _comparison(vs_b0), _comparison(b_vs_b1)]
    row = {
        "Gap": "2: source leakage",
        "Method / component": "Evidence agent searches PubMed with the source abstract and near copies blocked (setting A); "
                              "source abstract given (setting B)",
        "Metric": "F1 vs the MedHallu baselines rerun on the same items; kept leaks in the logs",
        "Base paper result": f"B0 (our rerun): F1 {pm(*_metric(rep, b0, 'f1'))} (reported {paper['B0']['f1']}); "
                             f"B1 (our rerun): F1 {pm(*_metric(rep, b1, 'f1'))} (reported {paper['B1']['f1']})",
        "Our result": f"System A: F1 {pm(*_metric(rep, full, 'f1'))}, vs B0 {_delta(vs_b0)}; System B: F1 "
                      f"{pm(*_metric(rep, sys_b, 'f1'))}, vs B1 {_delta(b_vs_b1)}; kept leaks in setting-A runs: "
                      f"{'-' if leaks is None else leaks}; source returned and blocked in "
                      f"{'-' if found is None else f'{found:.1%}'} of searches",
    }
    return row, parts


def gap3(rep: Report) -> tuple[dict, list]:
    full = _sides(rep.system_a(), side_final)
    forced = _sides(rep.system_a(), side_forced)
    b2 = _sides(rep.runs_with("b2-test"), side_final)
    vs_forced = compare(rep, full, forced)
    test_cov, sd = _metric(rep, full, "coverage")
    val_cov = (rep.final_config or {}).get("val", {}).get("coverage")
    parts = [facts.coverage_met(test_cov, val_cov), _comparison(vs_forced, "acc")]
    gap = "-" if test_cov is None or val_cov is None else f"{abs(test_cov - val_cov):.3f}"
    row = {
        "Gap": "3: abstention",
        "Method / component": "Vote with abstention (2 of 3) and a second evidence round when the votes split; trigger "
                              "chosen on the validation split only",
        "Metric": "Coverage (test vs val, within 5 points); accuracy on answered items vs the same system forced to answer",
        "Base paper result": f"B2 (our rerun, \"not sure\" allowed): coverage {pm(*_metric(rep, b2, 'coverage'))}, "
                             f"accuracy on answered {pm(*_metric(rep, b2, 'selective_accuracy'))}",
        "Our result": f"Test coverage {pm(test_cov, sd)} vs val {pm(val_cov, None)} (difference {gap}); accuracy on answered "
                      f"{pm(*_metric(rep, full, 'selective_accuracy'))} vs forced {pm(*_metric(rep, forced, 'selective_accuracy'))}, "
                      f"{_delta(vs_forced, 'acc')}",
    }
    return row, parts


def gap_table(rep: Report) -> tuple[list[dict], str]:
    """The rows and the markdown block (missing runs, table, the parts behind each verdict, notes)."""
    rows, details = [], []
    for gap_no, build in ((1, gap1), (2, gap2), (3, gap3)):
        row, parts = build(rep)
        row["Addressed?"] = facts.addressed(parts)
        rows.append(row)
        marks = {True: "met", False: "not met", None: "pending (runs missing)"}
        details.append(f"- Gap {gap_no}: " + "; ".join(f"{text}: {marks[p]}" for text, p in zip(facts.CRITERION_PARTS[gap_no], parts)))
    lines = [f"Missing runs: {', '.join(rep.missing) or 'none'}", "",
             "| " + " | ".join(COLUMNS) + " |", "| " + " | ".join("---" for _ in COLUMNS) + " |"]
    lines += ["| " + " | ".join(r[c].replace("|", "\\|") for c in COLUMNS) + " |" for r in rows]
    lines += ["", "Criterion parts behind each verdict:", *details, "", NOTE]
    return rows, "\n".join(lines)


def write_gap_table(rep: Report, readme: Optional[Path] = Path("README.md")) -> list[dict]:
    rows, block = gap_table(rep)
    rep.out.mkdir(parents=True, exist_ok=True)
    with (rep.out / "gap_resolution.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    (rep.out / "gap_resolution.md").write_text(f"# Gap resolution\n\n{block}\n", encoding="utf-8")
    if readme is not None and Path(readme).exists():
        update_readme(Path(readme), block)
    return rows


def update_readme(readme: Path, block: str) -> bool:
    """Replace the text between the GAP_TABLE markers; returns False if the markers are missing."""
    text = readme.read_text(encoding="utf-8")
    if START not in text or END not in text:
        return False
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    readme.write_text(f"{head}{START}\n{block}\n{END}{tail}", encoding="utf-8")
    return True
