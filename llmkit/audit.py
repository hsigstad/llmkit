"""Audit sampling for LLM extractions.

Draws samples for human review from a set of extraction results.
Produces a CSV audit table that a researcher can fill in.

Usage:
    from llmkit.audit import audit_sample
    samples = audit_sample(results, n=50, strata={"conclusao": ["afastada"]})
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass
class AuditRow:
    """One row in the audit table."""
    doc_id: str
    sample_reason: str       # "random", "stratified:field=value", "invalid", "edge"
    extraction: dict         # the LLM extraction for this doc
    text_excerpt: str = ""   # first N chars of source text for quick reference


def audit_sample(
    results: list[dict],
    *,
    n: int = 50,
    id_field: str = "doc_id",
    invalid: list[dict] | None = None,
    strata: dict[str, list[str]] | None = None,
    edge_filter: Callable[[dict], bool] | None = None,
    text_field: str = "",
    excerpt_chars: int = 500,
    seed: int = 42,
) -> list[AuditRow]:
    """Draw a stratified audit sample.

    Parameters
    ----------
    results : list[dict]
        All extraction results (each must have `id_field`).
    n : int
        Total sample size target.
    id_field : str
        Key for the document identifier in each result dict.
    invalid : list[dict]
        Results that failed validation — all included automatically.
    strata : dict[str, list[str]]
        Field → values to oversample.  For each (field, value) pair,
        up to n//4 matching results are included.
    edge_filter : callable
        Predicate identifying edge cases to oversample.
    text_field : str
        Key for source text in each result (for excerpt).
    excerpt_chars : int
        Max chars for the text excerpt.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    list[AuditRow]
    """
    rng = random.Random(seed)
    seen: set[str] = set()
    rows: list[AuditRow] = []

    def _add(doc: dict, reason: str) -> None:
        did = doc.get(id_field, "")
        if did in seen:
            return
        seen.add(did)
        excerpt = ""
        if text_field and text_field in doc:
            excerpt = doc[text_field][:excerpt_chars]
        rows.append(AuditRow(
            doc_id=did, sample_reason=reason,
            extraction=doc, text_excerpt=excerpt,
        ))

    # 1. All invalid parses
    for doc in (invalid or []):
        _add(doc, "invalid")

    # 2. Stratified: oversample rare categories
    if strata:
        budget = max(1, n // 4)
        for field_name, values in strata.items():
            for val in values:
                matches = [
                    r for r in results
                    if r.get(field_name) == val and r.get(id_field, "") not in seen
                ]
                for doc in rng.sample(matches, min(budget, len(matches))):
                    _add(doc, f"stratified:{field_name}={val}")

    # 3. Edge cases
    if edge_filter:
        budget = max(1, n // 4)
        edges = [r for r in results if edge_filter(r) and r.get(id_field, "") not in seen]
        for doc in rng.sample(edges, min(budget, len(edges))):
            _add(doc, "edge")

    # 4. Fill remaining with random sample
    remaining = n - len(rows)
    if remaining > 0:
        pool = [r for r in results if r.get(id_field, "") not in seen]
        for doc in rng.sample(pool, min(remaining, len(pool))):
            _add(doc, "random")

    return rows


def write_audit_csv(
    rows: list[AuditRow],
    path: Path,
    extra_columns: list[str] | None = None,
) -> None:
    """Write audit sample to CSV for human review.

    Adds empty columns for human coding: ``human_correct``, ``human_notes``,
    plus any ``extra_columns``.
    """
    fieldnames = [
        "doc_id", "sample_reason", "text_excerpt",
        "human_correct", "human_notes",
    ]
    if extra_columns:
        fieldnames.extend(extra_columns)

    # Add all extraction fields as flattened columns
    all_extraction_keys: list[str] = []
    for row in rows:
        for k in row.extraction:
            if k not in all_extraction_keys and k not in fieldnames:
                all_extraction_keys.append(k)
    fieldnames.extend(all_extraction_keys)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            d: dict[str, Any] = {
                "doc_id": row.doc_id,
                "sample_reason": row.sample_reason,
                "text_excerpt": row.text_excerpt,
                "human_correct": "",
                "human_notes": "",
            }
            d.update(row.extraction)
            writer.writerow(d)
