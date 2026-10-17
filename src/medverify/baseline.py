"""Single-LLM baseline: the MedHallu detection setup (one model, one call per item)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from tqdm import tqdm

from .data import append_jsonl, read_jsonl
from .llm import ChatModel
from .parsing import parse_verdict
from .prompts import baseline_messages
from .schemas import Item, Verdict

KnowledgeMode = Literal["none", "source"]


def run_baseline(
    items: list[Item],
    model: ChatModel,
    knowledge: KnowledgeMode = "none",
    allow_unsure: bool = False,
    out_path: Path | None = None,
) -> list[Verdict]:
    """Judge every item with one model.

    knowledge="none"   -> the paper's "without knowledge" setting.
    knowledge="source" -> the paper's "with knowledge" setting (gives the source abstract).
    Results are appended to `out_path` as they arrive, so a stopped run can resume.
    """
    done: dict[str, Verdict] = {}
    if out_path and Path(out_path).exists():
        done = {v.item_id: v for v in read_jsonl(out_path, Verdict)}

    verdicts: list[Verdict] = []
    for item in tqdm(items, desc=f"baseline {model.model} ({knowledge})"):
        if item.item_id in done:
            verdicts.append(done[item.item_id])
            continue
        msgs = baseline_messages(
            item.question,
            item.answer,
            knowledge=item.knowledge if knowledge == "source" else None,
            allow_unsure=allow_unsure,
        )
        res = model.chat(msgs)
        v = Verdict(
            item_id=item.item_id,
            system="baseline",
            verdict=parse_verdict(res.text, allow_unsure=allow_unsure),
            raw_text=res.text,
            provider=res.provider,
            model=res.model,
            latency_s=res.latency_s,
            prompt_tokens=res.prompt_tokens,
            completion_tokens=res.completion_tokens,
            cached=res.cached,
        )
        verdicts.append(v)
        if out_path:
            append_jsonl(out_path, v)
    return verdicts
