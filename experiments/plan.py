"""Every Phase 2 run, defined in one place.

Each Run holds the exact `python -m medverify` arguments except --split and --name, which
the runner adds for the stage (smoke, val or test). The pre-registered choices behind this
plan are in docs/devlog.md (Phase 2): vote rule 2 of 3, verifier temperature 0, B4 = Llama
sampled 3 times at temperature 0.7, seeds 1-3 for the main configurations, seed 1 otherwise.

Seed 1 is shared on purpose: runs whose round-1 messages match sysA-test-s1 (debate round 1,
B5 round 1 where the first query works) reuse its cached replies.
"""

from __future__ import annotations

from dataclasses import dataclass

# Stands for the second-round trigger saved in experiments/final_config.json (chosen on val).
TRIGGER = "TRIGGER"
SEEDS = (1, 2, 3)

# The system as pre-registered: default verifiers, reason prompt, 2 of 3, temperature 0.
SYSTEM = ("--verifiers", "llama,mistral-local,phi", "--prompt", "reason", "--min-agree", "2", "--temperature", "0")


@dataclass(frozen=True)
class Run:
    name: str
    experiment: str  # E1, E2, E3 or E5 (E4 and E6 are analyses of these runs)
    kind: str  # "baseline" or "pipeline"
    stage: str  # "val" or "test"
    args: tuple[str, ...]  # medverify arguments without --split and --name

    @property
    def seed(self) -> int | None:
        return int(self.args[self.args.index("--seed") + 1]) if "--seed" in self.args else None

    def argv(self, split: str, name: str, trigger: str, limit: int | None = None) -> list[str]:
        """Full argument list for medverify.cli.main, with TRIGGER replaced."""
        args = [trigger if a == TRIGGER else a for a in self.args]
        argv = [self.kind, *args, "--split", split, "--name", name]
        return argv + ["--limit", str(limit)] if limit else argv


def _baseline(name: str, experiment: str, seed: int, *extra: str) -> Run:
    return Run(name, experiment, "baseline", "test", ("--model", "llama", *extra, "--seed", str(seed)))


def _pipeline(name: str, experiment: str, stage: str, seed: int, *extra: str) -> Run:
    return Run(name, experiment, "pipeline", stage, (*SYSTEM, *extra, "--seed", str(seed)))


VAL_RUNS = [
    _pipeline("val-sysA-off", "E3", "val", 1, "--evidence", "pubmed", "--second-round", "off"),
    _pipeline("val-sysA-split", "E3", "val", 1, "--evidence", "pubmed", "--second-round", "split"),
    _pipeline("val-sysA-nomaj", "E3", "val", 1, "--evidence", "pubmed", "--second-round", "no-majority"),
]

TEST_RUNS = [
    # E2: the MedHallu baselines, three seeds each
    *[_baseline(f"b0-test-s{s}", "E2", s, "--knowledge", "none") for s in SEEDS],
    *[_baseline(f"b1-test-s{s}", "E2", s, "--knowledge", "source") for s in SEEDS],
    *[_baseline(f"b2-test-s{s}", "E2", s, "--knowledge", "none", "--unsure") for s in SEEDS],
    # E3: the proposed system (setting A: PubMed, source blocked) and setting B (source given)
    *[_pipeline(f"sysA-test-s{s}", "E3", "test", s, "--evidence", "pubmed", "--second-round", TRIGGER) for s in SEEDS],
    _pipeline("sysB-test-s1", "E3", "test", 1, "--evidence", "source", "--second-round", "off"),
    # E1: other baselines. B4 = self-consistency (one model sampled 3 times), B5 = simple RAG.
    # B3 = each verifier alone, read offline from sysA-test-s1 round 1 (no run needed).
    *[
        Run(f"b4-test-s{s}", "E1", "pipeline", "test", (
            "--verifiers", "llama@1,llama@2,llama@3", "--prompt", "reason", "--min-agree", "2",
            "--temperature", "0.7", "--evidence", "pubmed", "--second-round", "off", "--seed", str(s),
        ))
        for s in SEEDS
    ],
    _pipeline("b5-test-s1", "E1", "test", 1, "--evidence", "pubmed", "--no-retry", "--second-round", "off"),
    # E5: ablations (no evidence, no blocking, debate). "Second round off" = evaluate --round 1 on sysA.
    _pipeline("none-test-s1", "E5", "test", 1, "--evidence", "none"),
    _pipeline("noblock-test-s1", "E5", "test", 1, "--evidence", "pubmed", "--no-block", "--second-round", TRIGGER),
    _pipeline("debate-test-s1", "E5", "test", 1, "--evidence", "pubmed", "--debate", "--second-round", "off"),
]

ALL_RUNS = VAL_RUNS + TEST_RUNS
