"""Failure analysis of one System A run, representative case trajectories and trajectory samples.

Every item gets exactly one category, checked in this order:
  unreadable                    a verifier reply still unreadable after the re-ask
  abstained - ...               final decision 2; split by what the majority of 0/1 votes would have said
  missed hallucination - ...    a hallucinated answer accepted: all verifiers fooled / the majority fooled
  false alarm - ...             a true answer flagged: all verifiers / the majority
  correct
Written to tables/failures.csv and failures_summary.md; cases to logs/cases/; samples to logs/samples/.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Optional

from medverify.schemas import Item, VerificationState

from .export import EVIDENCE_CHARS
from .results import RunData, load_run

CATEGORIES = [
    "unreadable",
    "abstained - majority would have been right",
    "abstained - majority would have been wrong",
    "abstained - votes tied",
    "missed hallucination - all verifiers fooled",
    "missed hallucination - majority fooled",
    "false alarm - all verifiers",
    "false alarm - majority",
    "correct",
]
SAMPLE_RUNS = ("sysA-test-s1", "b4-test-s1", "debate-test-s1")
SAMPLE_SIZE = 20


def categorise(state: VerificationState, label: int) -> str:
    """The one failure category of an item (see the module docstring for the order)."""
    if any(v.verdict is None for v in state.verdicts):
        return "unreadable"
    last = [v.verdict for v in state.last_round()]
    if state.final == 2:
        zeros, ones = last.count(0), last.count(1)
        if zeros == ones:
            return "abstained - votes tied"
        right = (1 if ones > zeros else 0) == label
        return "abstained - majority would have been " + ("right" if right else "wrong")
    if state.final == label:
        return "correct"
    if label == 1:
        return "missed hallucination - " + ("all verifiers fooled" if all(v == 0 for v in last) else "majority fooled")
    return "false alarm - " + ("all verifiers" if all(v == 1 for v in last) else "majority")


def outcome(decision: Optional[int], label: int) -> str:
    if decision not in (0, 1):
        return "abstain"
    return "right" if decision == label else "wrong"


def evidence_state(state: VerificationState, label: int) -> str:
    """What happened to the evidence and the second round for this item (one label per item)."""
    if state.rounds >= 2:
        before, after = outcome(state.final_round1, label), outcome(state.final, label)
        if before == after:
            return "round 2: no change"
        if after == "right":
            return f"round 2 flipped {before}->right"
        if before == "right":
            return f"round 2 flipped right->{after}"
        return f"round 2 changed {before}->{after}"
    if "no_new_evidence" in state.flags:
        return "second round triggered, no new evidence"
    if any(s.action == "second_round_check" and s.detail.get("triggered") for s in state.steps):
        return "second round triggered, not run"
    if not clean_evidence(state):
        return "no clean evidence"
    return "decided in round 1"


def clean_evidence(state: VerificationState) -> int:
    return sum(1 for e in state.evidence if e.round == 1 and not e.leak)


def failure_rows(items: list[Item], run: RunData) -> list[dict]:
    by_id = {i.item_id: i for i in items}
    rows = []
    for s in sorted(run.states, key=lambda s: s.item_id):
        item = by_id.get(s.item_id)
        if item is None:
            continue
        rows.append({
            "item_id": s.item_id, "label": item.label, "difficulty": item.difficulty, "medhallu_category": item.category,
            "failure_category": categorise(s, item.label), "evidence_state": evidence_state(s, item.label),
            "clean_abstracts_round1": clean_evidence(s), "round1_decision": s.final_round1 if s.final_round1 is not None else s.final,
            "final_decision": s.final, "rounds": s.rounds,
            "votes_round1": " ".join(str(v.verdict) for v in s.first_round()),
            "votes_last": " ".join(str(v.verdict) for v in s.last_round()),
        })
    return rows


def flip_table(rows: list[dict]) -> dict[tuple[str, str], int]:
    """(round-1 outcome, final outcome) counts for the items judged again in round 2."""
    return Counter((outcome(r["round1_decision"], r["label"]), outcome(r["final_decision"], r["label"]))
                   for r in rows if r["rounds"] >= 2)


def _crosstab(rows: list[dict], key: str) -> list[str]:
    groups = list(dict.fromkeys(sorted(r[key] for r in rows)))
    cats = [c for c in CATEGORIES if any(r["failure_category"] == c for r in rows)]
    lines = [f"| {key} | n | " + " | ".join(cats) + " |", "| --- | --- | " + " | ".join("---" for _ in cats) + " |"]
    for g in groups:
        sub = [r for r in rows if r[key] == g]
        counts = Counter(r["failure_category"] for r in sub)
        lines.append(f"| {g} | {len(sub)} | " + " | ".join(str(counts.get(c, 0)) for c in cats) + " |")
    return lines


def pick_cases(rows: list[dict]) -> list[tuple[str, str]]:
    """(why, item_id): 2 successes and 2 failures, the lowest item_id in each group (deterministic)."""
    picks = []
    hard_caught = [r for r in rows if r["difficulty"] == "hard" and r["label"] == 1 and r["final_decision"] == 1]
    if hard_caught:
        picks.append(("success: hard hallucination caught", min(r["item_id"] for r in hard_caught)))
    fixed = [r for r in rows if r["rounds"] >= 2 and r["round1_decision"] != r["label"] and r["final_decision"] == r["label"]]
    if fixed:
        picks.append(("success: fixed by the second round", min(r["item_id"] for r in fixed)))
    failures = Counter(r["failure_category"] for r in rows if r["failure_category"] != "correct")
    top = sorted(failures, key=lambda c: (-failures[c], CATEGORIES.index(c)))[:2]
    for category in top:
        picks.append((f"failure: {category}", min(r["item_id"] for r in rows if r["failure_category"] == category)))
    return picks


def case_markdown(item: Item, state: VerificationState, why: str, category: str) -> str:
    """One full trajectory, readable."""
    label = "hallucinated (1)" if item.label == 1 else "factual (0)"
    lines = [f"# Case {item.item_id}", "", f"Why this case: {why}", f"Failure category: {category}", "",
             f"- **Question:** {item.question}", f"- **Answer:** {item.answer}", f"- **Label:** {label}",
             f"- **Difficulty:** {item.difficulty}", f"- **MedHallu category:** {item.category}",
             f"- **Final decision:** {state.final} (round 1: {state.final_round1}); rounds: {state.rounds}; "
             f"flags: {', '.join(state.flags) or 'none'}", "", "## Evidence", ""]
    lines += [f"- PMID {e.pmid} (round {e.round}{', leak: ' + e.leak if e.leak else ''}): {e.title}" for e in state.evidence] or ["- none"]
    lines += [f"- blocked PMID {b.pmid} (round {b.round}, {b.reason})" for b in state.blocked]
    lines += ["", "## Steps", ""]
    for step in state.steps:
        d = step.detail
        lines.append(f"### {step.step}. {step.agent}: {step.action} (t = {step.t:.1f} s)")
        if step.agent == "verifier":
            lines += [f"Verifier {d.get('verifier')}, round {d.get('round')}: verdict {d.get('verdict')}", "", "```text",
                      str(d.get("raw_reply", "")).strip(), "```"]
        else:
            lines.append("`" + json.dumps(d, ensure_ascii=False) + "`")
        lines.append("")
    if not state.steps:
        lines += ["(This run saved no structured steps; text trace below.)", "", *[f"- {t}" for t in state.trace]]
    return "\n".join(lines).rstrip() + "\n"


def write_cases(items: list[Item], run: RunData, rows: list[dict], folder: Path) -> list[str]:
    by_item, by_state = {i.item_id: i for i in items}, {s.item_id: s for s in run.states}
    category = {r["item_id"]: r["failure_category"] for r in rows}
    folder.mkdir(parents=True, exist_ok=True)
    written = []
    for why, item_id in pick_cases(rows):
        item, state = by_item[item_id], by_state[item_id]
        (folder / f"{item_id}.md").write_text(case_markdown(item, state, why, category[item_id]), encoding="utf-8")
        data = {"why": why, "failure_category": category[item_id], "run": run.name,
                "item": item.model_dump(exclude={"knowledge"}), "state": state.model_dump()}
        (folder / f"{item_id}.json").write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        written.append(item_id)
    return written


def summary_markdown(run: RunData, rows: list[dict], cases: list[tuple[str, str]], missing: list[str]) -> str:
    n = len(rows)
    counts = Counter(r["failure_category"] for r in rows)
    lines = [f"# Failure analysis: {run.name}", "", f"Missing runs: {', '.join(missing) or 'none'}", "",
             f"{n} items. Each item has one category, checked in the order below.", "",
             "| category | count | share |", "| --- | --- | --- |"]
    lines += [f"| {c} | {counts.get(c, 0)} | {counts.get(c, 0) / n:.3f} |" for c in CATEGORIES] if n else []
    for key, title in (("difficulty", "By difficulty"), ("medhallu_category", "By MedHallu hallucination category"),
                       ("evidence_state", "By evidence state")):
        lines += ["", f"## {title}", "", *_crosstab(rows, key)]
    no_clean = sum(1 for r in rows if r["clean_abstracts_round1"] == 0)
    lines += ["", f"Items with no clean evidence in round 1: {no_clean} of {n}.", "", "## Round-2 flips", ""]
    flips = flip_table(rows)
    if flips:
        outcomes = ["right", "wrong", "abstain"]
        lines += ["| round 1 \\ final | " + " | ".join(outcomes) + " |", "| --- | --- | --- | --- |"]
        lines += [f"| {a} | " + " | ".join(str(flips.get((a, b), 0)) for b in outcomes) + " |" for a in outcomes]
    else:
        lines.append("No item was judged again in round 2.")
    lines += ["", "## Representative cases", ""]
    lines += [f"- {why}: [{item_id}](../../../logs/cases/{item_id}.md)" for why, item_id in cases] or ["- none"]
    return "\n".join(lines) + "\n"


def write_failures(rep, run: Optional[RunData], cases_dir: Path = Path("logs/cases")) -> list[dict]:
    """failures.csv, failures_summary.md and the case trajectories for one run (System A seed 1, or the chosen val run)."""
    rep.out.mkdir(parents=True, exist_ok=True)
    if run is None or run.states is None:
        (rep.out / "failures_summary.md").write_text(
            f"# Failure analysis\n\nMissing runs: {', '.join(rep.missing) or 'none'}\n\nNo System A run to analyse.\n", encoding="utf-8")
        return []
    rows = failure_rows(rep.items, run)
    columns = list(rows[0]) if rows else ["item_id"]
    with (rep.out / "failures.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    folder = cases_dir if rep.split == "test" else cases_dir / "val"
    write_cases(rep.items, run, rows, folder)
    cases = pick_cases(rows)
    summary = summary_markdown(run, rows, cases, rep.missing)
    if rep.split == "val":
        summary = summary.replace("../../../logs/cases/", "../../../../logs/cases/val/")
    (rep.out / "failures_summary.md").write_text(summary, encoding="utf-8")
    return rows


def write_samples(rep, folder: Path = Path("logs/samples")) -> list[str]:
    """The first 20 states (by item_id) of a few runs, for reading the agent trajectories directly."""
    names = list(SAMPLE_RUNS)
    if rep.final_config:
        from .report import CHOSEN_VAL_RUN

        names.insert(1, CHOSEN_VAL_RUN[rep.final_config["trigger"]])
    written = []
    for name in names:
        run = load_run(name, rep.phase2)
        if run is None or run.states is None:
            continue
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for s in sorted(run.states, key=lambda s: s.item_id)[:SAMPLE_SIZE]:
                data = s.model_dump()
                for e in data["evidence"]:
                    e["text"] = e["text"][:EVIDENCE_CHARS]
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
        written.append(name)
    return written
