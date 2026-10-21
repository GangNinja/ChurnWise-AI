"""Read ranked candidates from the Phase 2 discovery report.

The report is the durable boundary between phases. The loader accepts only
eligible, non-duplicate ranking entries, preserving their Phase 2 rank order.
Malformed entries fail with ``ValueError`` so the CLI can return a structured
configuration error instead of crashing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from churnwise.contracts import DatasetCandidate


def _candidate(data: dict[str, Any], context: str) -> DatasetCandidate:
    try:
        return DatasetCandidate.from_dict(data)
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{context} is invalid: {exc}") from exc


def _rank_value(entry: dict[str, Any]) -> int | None:
    rank = entry.get("rank")
    if isinstance(rank, int) and not isinstance(rank, bool) and rank >= 1:
        return rank
    if isinstance(rank, str) and rank.strip().isdigit():
        value = int(rank.strip())
        return value if value >= 1 else None
    return None


def load_candidates(report_path: Path | str) -> list[DatasetCandidate]:
    path = Path(report_path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("candidates.json root must be an object")

    ranking = payload.get("ranking") or []
    if not isinstance(ranking, list):
        raise ValueError("candidates.json ranking must be a list")

    ranked_entries: list[tuple[int, int, dict[str, Any]]] = []
    for order, entry in enumerate(ranking):
        if not isinstance(entry, dict):
            continue
        if not entry.get("eligible") or entry.get("duplicate_of"):
            continue
        rank = _rank_value(entry)
        # Missing ranks are still retained, after explicitly ranked entries,
        # in the order the report supplied them.
        ranked_entries.append((rank if rank is not None else 10**9, order, entry))
    ranked_entries.sort(key=lambda item: (item[0], item[1]))

    candidates: list[DatasetCandidate] = []
    seen: set[str] = set()
    for _, _, entry in ranked_entries:
        candidate = _candidate(entry, "candidates.json ranking entry")
        if candidate.candidate_id in seen:
            continue
        seen.add(candidate.candidate_id)
        candidates.append(candidate)

    selected = payload.get("selected")
    if not candidates and isinstance(selected, dict):
        snapshot = selected.get("snapshot")
        if isinstance(snapshot, dict):
            candidate = _candidate(snapshot, "candidates.json selected snapshot")
            if candidate.candidate_id not in seen:
                candidates.append(candidate)
    return candidates
