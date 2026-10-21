"""Pure helpers for the Streamlit app, testable without Streamlit or models.

The plain-English views (decision_text, verifier_rows, evidence_rows, ...) come from
medverify.explain, so the app, `medverify verify` and batch mode say the same thing.
"""

from __future__ import annotations

import csv
import html
import io
import os
from typing import Callable, Optional

from medverify.explain import (  # noqa: F401  (re-exported for the app)
    DISCLAIMER,
    abstain_reason,
    agreement,
    blocked_rows,
    cost_summary,
    decision_label,
    decision_text,
    evidence_rows,
    second_round_text,
    verifier_rows,
)

REQUIRED_MODELS = ("llama3.1:8b", "mistral:7b", "phi4-mini")
# Dark enough for white text (contrast above 4.5:1), and not told apart by colour alone: the label is in the badge.
BADGE_COLOURS = {"ACCEPT": "#1b6e35", "REJECT": "#b3261e", "NOT SURE": "#7a5200", "UNKNOWN": "#555555"}
EVIDENCE_LABELS = {
    "pubmed": "PubMed search (the answer's source is blocked)",
    "none": "No evidence (the verifiers' own knowledge)",
    "source": "The source abstract (give its PubMed ID)",
}
RESULT_COLUMNS = ["question", "answer", "decision", "agreement", "evidence_pmids", "blocked_pmids", "second_round", "flags"]


def badge_html(code: Optional[int]) -> str:
    """A coloured decision badge (HTML) with the decision written out."""
    label = decision_label(code)
    meaning = decision_text(code).split("(", 1)[-1].rstrip(")")
    return (f'<div style="background:{BADGE_COLOURS.get(label, "#555555")};color:#ffffff;padding:0.6rem 1rem;'
            f'border-radius:0.5rem;font-size:1.3rem;font-weight:700;display:inline-block">{html.escape(label)}</div>'
            f'<div style="margin-top:0.3rem">{html.escape(meaning)}</div>')


def ollama_host() -> str:
    from medverify.llm import PROVIDERS

    return PROVIDERS["ollama"].base_url.rstrip("/").removesuffix("/v1")


def missing_models(tags: dict, required=REQUIRED_MODELS) -> list[str]:
    """Required models that are not in Ollama's /api/tags answer ("phi4-mini" matches "phi4-mini:latest")."""
    have = set()
    for model in tags.get("models", []):
        for name in (model.get("name"), model.get("model")):
            if name:
                have |= {name, name.removesuffix(":latest")}
    return [m for m in required if m not in have]


def setup_status(get: Optional[Callable] = None, env: Optional[dict] = None) -> dict:
    """Is Ollama reachable, which models are missing, is NCBI_EMAIL set. Never raises."""
    env = os.environ if env is None else env
    status = {"ollama": False, "ollama_error": "", "missing_models": list(REQUIRED_MODELS),
              "ncbi_email": bool(env.get("NCBI_EMAIL")), "host": ollama_host()}
    try:
        if get is None:
            import requests

            get = requests.get
        tags = get(f"{status['host']}/api/tags", timeout=3).json()
        status["ollama"], status["missing_models"] = True, missing_models(tags)
    except Exception as err:  # not running, wrong port, bad JSON: say so instead of crashing
        status["ollama_error"] = f"{type(err).__name__}: {err}"
    return status


def setup_messages(status: dict) -> list[tuple[str, str]]:
    """(level, text) lines for the sidebar: level is success, warning or error."""
    lines = []
    if status["ollama"]:
        lines.append(("success", f"Ollama is running at {status['host']}."))
        if status["missing_models"]:
            pulls = " ".join(status["missing_models"])
            lines.append(("warning", f"Missing models: {', '.join(status['missing_models'])}. Run `ollama pull {pulls}`."))
        else:
            lines.append(("success", "All three verifier models are installed."))
    else:
        lines.append(("error", f"Ollama is not reachable at {status['host']}. Start it with `ollama serve`, then reload this page."))
    if status["ncbi_email"]:
        lines.append(("success", "NCBI_EMAIL is set (PubMed)."))
    else:
        lines.append(("warning", "NCBI_EMAIL is not set: PubMed still works, but put NCBI_EMAIL=you@example.com in .env "
                                 "as NCBI asks."))
    return lines


def ready(status: dict) -> bool:
    return status["ollama"] and not status["missing_models"]


def model_error_text(err: Exception) -> str:
    return (f"A verifier could not answer: {err}. Check that Ollama is running and the three models are pulled "
            "(see Setup check in the sidebar).")


def evidence_markdown(state) -> list[str]:
    """Markdown lines: each abstract the verifiers saw, with its PubMed link."""
    lines = []
    for r in evidence_rows(state):
        title = r["title"] or "(no title)"
        link = f"[PMID {r['pmid']}]({r['url']})" if r["url"] else f"PMID {r['pmid']}"
        lines.append(f"- {link}: {title}" + (" *(second search)*" if r["round"] == 2 else ""))
    return lines


def blocked_markdown(state) -> list[str]:
    return [f"- PMID {r['pmid']}: {r['reason']}" + (" *(second search)*" if r["round"] == 2 else "") for r in blocked_rows(state)]


def cost_line(state) -> str:
    c = cost_summary(state)
    return (f"{c['model_calls']} model calls · {c['prompt_tokens']} prompt + {c['completion_tokens']} completion tokens · "
            f"{c['model_seconds']} model seconds · {c['pubmed_requests']} PubMed requests")


def results_csv(rows: list[dict]) -> bytes:
    """The batch results as CSV bytes for the download button."""
    columns = list(dict.fromkeys(c for r in rows for c in r))
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")
