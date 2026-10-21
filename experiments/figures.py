"""Figures for the paper: .png (300 dpi) and .pdf in results/phase2/figures/, one IEEE column wide.

matplotlib with the Agg backend (no display needed), the Okabe-Ito colour-blind-safe palette and no
titles inside the images (captions go in the paper). A figure whose runs are missing is skipped.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from medverify.metrics import risk_coverage  # noqa: E402
from medverify.pipeline import agreement_confidence  # noqa: E402

from .results import side_final, side_forced, side_metrics, side_min_agree  # noqa: E402

WIDTH = 3.5  # inches: one IEEE column
OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]
SHORT = {"b0-test": "B0", "b1-test": "B1", "b2-test": "B2", "b4-test": "B4", "b5-test": "B5", "sysA-test": "System A",
         "sysB-test": "System B", "none-test": "No evidence", "noblock-test": "No blocking", "debate-test": "Debate",
         "sysA-round1": "Sys A round 1", "sysA-forced": "Sys A forced", "val-sysA-off": "Val off",
         "val-sysA-split": "Val split", "val-sysA-nomaj": "Val no-majority"}

plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                     "legend.fontsize": 6.5, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.bbox": "tight", "pdf.fonttype": 42})


def _save(fig, folder: Path, name: str) -> str:
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"{name}.png", dpi=300)
    fig.savefig(folder / f"{name}.pdf")
    plt.close(fig)
    return name


def f1_by_difficulty(rows: list[dict], folder: Path) -> Optional[str]:
    keys = [k for k in ("b0-test", "b1-test", "b4-test", "sysA-test", "sysB-test") if any(r["config"] == k for r in rows)]
    if not keys:
        return None
    by_key = {r["config"]: r for r in rows}
    levels, width = ("easy", "medium", "hard"), 0.8 / len(keys)
    fig, ax = plt.subplots(figsize=(WIDTH, 2.2))
    for i, key in enumerate(keys):
        r = by_key[key]
        means = [r.get(f"{lvl}_f1_mean") or 0 for lvl in levels]
        sds = [r.get(f"{lvl}_f1_sd") or 0 for lvl in levels]
        xs = [j + (i - (len(keys) - 1) / 2) * width for j in range(len(levels))]
        ax.bar(xs, means, width, yerr=sds, capsize=1.5, color=OKABE_ITO[i], label=SHORT[key], error_kw={"lw": 0.6})
    ax.set_xticks(range(len(levels)), [lvl.capitalize() for lvl in levels])
    ax.set_ylabel("F1 (mean ± SD over seeds)")
    ax.set_ylim(0, 1)
    ax.legend(ncol=len(keys), loc="upper center", bbox_to_anchor=(0.5, 1.16), frameon=False, handlelength=1, columnspacing=0.8)
    return _save(fig, folder, "f1_by_difficulty")


def risk_coverage_figure(rep, folder: Path) -> Optional[str]:
    runs = rep.system_a()
    run = next((r for r in runs if r.seed == 1), runs[0] if runs else None)
    if run is None:
        return None
    curve = risk_coverage(rep.items, agreement_confidence(run.states))
    fig, ax = plt.subplots(figsize=(WIDTH, 2.3))
    ax.plot([p["coverage"] for p in curve], [p["risk"] for p in curve], color=OKABE_ITO[0], lw=1, label="System A (agreement)")
    markers = {1: "v", 2: "o", 3: "^"}
    for k in (1, 2, 3):
        m = side_metrics(rep.items, side_min_agree(run, k), n_boot=0)
        ax.scatter(m["coverage"], 1 - m["selective_accuracy"], marker=markers[k], s=18, color=OKABE_ITO[1], zorder=3,
                   label=f"Vote {k} of 3")
    forced = side_metrics(rep.items, side_forced(run), n_boot=0)
    ax.scatter(forced["coverage"], 1 - forced["selective_accuracy"], marker="s", s=18, color=OKABE_ITO[3], zorder=3, label="Forced answer")
    b2 = next((r for r in rep.runs_with("b2-test") if r.seed == 1), None)
    if b2 is not None:
        m = side_metrics(rep.items, side_final(b2), n_boot=0)
        ax.scatter(m["coverage"], 1 - m["selective_accuracy"], marker="D", s=16, color=OKABE_ITO[2], zorder=3, label='B2 ("not sure")')
    ax.set_xlabel("Coverage (share of items answered)")
    ax.set_ylabel("Risk (error rate on answered)")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, loc="upper left")
    return _save(fig, folder, "risk_coverage")


VERIFIER = {"llama": "Llama", "mistral-local": "Mistral", "mistral": "Mistral", "phi": "Phi", "gptoss": "gpt-oss"}


def _pair_label(pair: str) -> str:
    """'llama@1 vs llama@2' -> 'Llama#1-Llama#2'; 'llama vs mistral-local' -> 'Llama-Mistral'."""
    def short(name):
        base, _, k = name.partition("@")
        return VERIFIER.get(base, base) + (f"#{k}" if k else "")
    a, b = pair.split(" vs ")
    return f"{short(a)}-{short(b)}"


def error_overlap_figure(rows: list[dict], folder: Path) -> Optional[str]:
    pairs = [r for r in rows if r["level"] == "pair (mean over seeds)" and r.get("overlap_ratio") is not None]
    if not pairs:
        return None
    colours = {"cross-family": OKABE_ITO[0], "same family": OKABE_ITO[5], "same model": OKABE_ITO[3]}
    fig, ax = plt.subplots(figsize=(WIDTH, 2.3))
    for i, r in enumerate(pairs):
        ax.bar(i, r["overlap_ratio"], 0.7, yerr=r.get("overlap_ratio_sd") or 0, capsize=1.5, color=colours[r["pair_type"]],
               error_kw={"lw": 0.6}, label=r["pair_type"])
    ax.axhline(1.0, color="black", lw=0.7, ls="--")
    ax.set_xticks(range(len(pairs)), [_pair_label(r["pair"]) for r in pairs], rotation=35, ha="right")
    ax.set_ylabel("Error overlap ratio\n(both wrong / expected)")
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), frameon=False)
    return _save(fig, folder, "error_overlap")


def ablation_figure(rows: list[dict], folder: Path) -> Optional[str]:
    rows = [r for r in rows if r.get("delta_f1_seed1") is not None]
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(WIDTH, 0.25 * len(rows) + 0.6))
    ys = list(range(len(rows)))[::-1]
    for y, r in zip(ys, rows):
        low, high, mid = r["delta_f1_ci_low"], r["delta_f1_ci_high"], r["delta_f1_seed1"]
        ax.errorbar(mid, y, xerr=[[mid - low], [high - mid]], fmt="o", ms=3, color=OKABE_ITO[0], ecolor=OKABE_ITO[0],
                    elinewidth=0.8, capsize=1.5)
    ax.axvline(0, color="black", lw=0.7, ls="--")
    ax.set_yticks(ys, [r["component"] for r in rows])
    ax.set_xlabel("Change in F1 vs reference (seed 1, 95% bootstrap interval)")
    return _save(fig, folder, "ablation")


def cost_figure(rows: list[dict], folder: Path) -> Optional[str]:
    points = [r for r in rows if (r.get("source") or "").startswith("ours") and r.get("model_seconds_mean") is not None
              and r["config"] in SHORT and r["config"] != "sysA-forced"]  # forced costs the same as System A
    if not points:
        return None
    fig, ax = plt.subplots(figsize=(WIDTH, 2.4))
    for i, r in enumerate(points):
        ax.scatter(r["model_seconds_mean"], r["f1_mean"], s=14, color=OKABE_ITO[i % len(OKABE_ITO)], zorder=3)
        ax.annotate(SHORT[r["config"]], (r["model_seconds_mean"], r["f1_mean"]), fontsize=5.5,
                    xytext=(3, 2), textcoords="offset points")
    ax.set_xlabel("Model seconds per item (all calls)")
    ax.set_ylabel("F1 (mean over seeds)")
    return _save(fig, folder, "cost")


def write_figures(rep, tables: dict) -> list[str]:
    folder = rep.figures
    made = [
        f1_by_difficulty(tables.get("main", []), folder),
        risk_coverage_figure(rep, folder),
        error_overlap_figure(tables.get("overlap", []), folder),
        ablation_figure(tables.get("ablation", []), folder),
        cost_figure(tables.get("main", []), folder),
    ]
    return [m for m in made if m]
