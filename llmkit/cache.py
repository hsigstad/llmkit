"""Deterministic LLM response cache.

Cache key = hash(doc_id, text_hash, model).  Prompt and schema versions are
stored as metadata for auditing but do NOT affect the key — so editing a
prompt does not invalidate cached results.  Use ``is_stale()`` to check
whether a cached entry was produced with the current prompt/data, and
``--reextract`` before submission to bring everything in line.

Backward-compatible: reads old-style caches where the filename is the doc_id
and the file contains only the extraction (no metadata wrapper).

Usage:
    cache = LLMCache(Path("cache_dir"))
    key = cache.key(doc_id="ABC", text_hash="f3a1...", model="gpt-4o-mini")
    hit = cache.get(key)
    if hit is None:
        result = call_llm(...)
        cache.put(key, result, doc_id="ABC", ...)
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def text_hash(text: str) -> str:
    """Stable SHA-256 hex digest (first 16 chars) of document text."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def content_hash(text: str) -> str:
    """Hash of arbitrary text content (e.g. a prompt file)."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _cache_key(doc_id: str, text_hash: str, model: str) -> str:
    """Deterministic cache key — only content + model, not prompt version."""
    raw = f"{doc_id}|{text_hash}|{model}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def _git_head(repo_path: Path | None = None) -> str:
    """Return short HEAD commit hash, or empty string if not in a repo."""
    try:
        cmd = ["git", "rev-parse", "--short", "HEAD"]
        kw: dict[str, Any] = {"capture_output": True, "text": True}
        if repo_path:
            kw["cwd"] = repo_path
        return subprocess.run(cmd, **kw).stdout.strip()
    except Exception:
        return ""


@dataclass
class CacheEntry:
    """One cached LLM response with metadata."""
    key: str
    doc_id: str
    extraction: dict
    meta: dict = field(default_factory=dict)
    input_text: str = ""
    path: Path | None = None

    @property
    def is_valid(self) -> bool:
        """Whether the extraction passed schema validation."""
        return self.meta.get("validation_status", "") == "valid"

    def is_stale(
        self,
        *,
        current_text_hash: str = "",
        current_prompt_hash: str = "",
        current_model: str = "",
    ) -> bool:
        """Check whether this entry is out of date.

        An entry is stale if any of the provided current values differ
        from what was recorded when the entry was created.  Only checks
        fields that are passed (empty string = skip check).
        """
        if current_text_hash and self.meta.get("text_hash", "") != current_text_hash:
            return True
        if current_prompt_hash and self.meta.get("prompt_hash", "") != current_prompt_hash:
            return True
        if current_model and self.meta.get("model", "") != current_model:
            return True
        return False


class LLMCache:
    """File-backed LLM response cache.

    Each entry is a JSON file named ``{key}.json`` containing::

        {
          "_cache_meta": {
            "doc_id": "...",
            "text_hash": "...",
            "n_chars": 22666,
            "prompt_file": "irregularity_system.txt",
            "prompt_hash": "a3b4c5d6e7f8a9b0",
            "model": "gpt-4o-mini",
            "model_version": "gpt-4o-mini-2024-07-18",
            "temperature": 0,
            "max_tokens": 4000,
            "schema_name": "irregularity_extraction",
            "schema_version": "v1",
            "source_commit": "d540ee4",
            "validation_status": "valid",
            "finish_reason": "stop",
            "timestamp": "...",
            "usage": {...},
            "api_params": {"response_format": "json_object", "top_p": 1}
          },
          "input_text": "... full document text ...",
          "extraction": { ... raw LLM output ... }
        }
    """

    def __init__(self, directory: Path):
        self.directory = directory

    # ── Public helpers ────────────────────────────────────────────────

    @staticmethod
    def text_hash(text: str) -> str:
        return text_hash(text)

    @staticmethod
    def key(doc_id: str, text_hash: str, model: str) -> str:
        return _cache_key(doc_id, text_hash, model)

    # ── Read ─────────────────────────────────────────────────────────

    def get(self, key: str) -> CacheEntry | None:
        """Load a cached entry by composite key, or None."""
        p = self.directory / f"{key}.json"
        if not p.exists():
            return None
        return self._load(p, key)

    def get_by_doc(self, doc_id: str) -> CacheEntry | None:
        """Fallback: load old-style cache keyed by doc_id filename."""
        p = self.directory / f"{doc_id}.json"
        if not p.exists():
            return None
        return self._load(p, doc_id)

    def _load(self, p: Path, key: str) -> CacheEntry:
        with open(p) as f:
            data = json.load(f)
        if "_cache_meta" in data:
            meta = data["_cache_meta"]
            extraction = data.get("extraction", data)
            input_text = data.get("input_text", "")
        else:
            # Old format: entire file is the extraction
            meta = {}
            extraction = data
            input_text = ""
        return CacheEntry(
            key=key,
            doc_id=meta.get("doc_id", key),
            extraction=extraction,
            meta=meta,
            input_text=input_text,
            path=p,
        )

    # ── Write ────────────────────────────────────────────────────────

    def put(
        self,
        key: str,
        extraction: dict[str, Any],
        *,
        doc_id: str = "",
        text_hash: str = "",
        n_chars: int = 0,
        input_text: str = "",
        prompt_file: str = "",
        prompt_hash: str = "",
        model: str = "",
        model_version: str = "",
        temperature: float = 0,
        max_tokens: int = 0,
        finish_reason: str = "",
        schema_name: str = "",
        schema_version: str = "",
        source_commit: str = "",
        validation_status: str = "",
        usage: dict | None = None,
        api_params: dict | None = None,
    ) -> Path:
        """Write an extraction result to cache."""
        self.directory.mkdir(parents=True, exist_ok=True)
        if not source_commit:
            source_commit = _git_head()
        p = self.directory / f"{key}.json"
        envelope = {
            "_cache_meta": {
                "doc_id": doc_id,
                "text_hash": text_hash,
                "n_chars": n_chars,
                "prompt_file": prompt_file,
                "prompt_hash": prompt_hash,
                "model": model,
                "model_version": model_version,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "schema_name": schema_name,
                "schema_version": schema_version,
                "source_commit": source_commit,
                "validation_status": validation_status,
                "finish_reason": finish_reason,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "usage": usage or {},
                "api_params": api_params or {},
            },
            "input_text": input_text,
            "extraction": extraction,
        }
        with open(p, "w") as f:
            json.dump(envelope, f, ensure_ascii=False, indent=2)
        return p

    # ── Iteration ────────────────────────────────────────────────────

    def iter_entries(self) -> list[CacheEntry]:
        """Load all cached entries (both old and new format)."""
        if not self.directory.exists():
            return []
        entries = []
        for p in sorted(self.directory.glob("*.json")):
            try:
                entry = self._load(p, p.stem)
            except (json.JSONDecodeError, KeyError):
                continue
            entries.append(entry)
        return entries

    def __len__(self) -> int:
        if not self.directory.exists():
            return 0
        return sum(1 for _ in self.directory.glob("*.json"))
