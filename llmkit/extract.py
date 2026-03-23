"""Core extraction: call LLM, validate with Pydantic, cache result.

Cache key is (doc_id, text_hash, model) — prompt changes do NOT invalidate
cache.  Use ``reextract=True`` to force re-extraction of stale entries
(different prompt or data since cached).

Usage:
    from llmkit import extract, LLMCache
    from my_schemas import MySchema

    result = extract(
        doc_id="ABC",
        text="...",
        system_prompt="...",
        user_prompt="...",
        schema=MySchema,
        model="gpt-4o-mini",
        prompt_file="irregularity_system.txt",
        cache=LLMCache(Path("cache_dir")),
        client=openai_client,
    )
    if result.valid:
        print(result.parsed.some_field)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, ClassVar, TypeVar

from pydantic import BaseModel, ValidationError

from llmkit.cache import LLMCache, content_hash, text_hash


class ExtractionSchema(BaseModel):
    """Base class for extraction schemas.

    Subclass this and set ``schema_name`` and ``schema_version`` as ClassVars
    so that cache entries record which schema validated them.

        class MySchema(ExtractionSchema):
            schema_name: ClassVar[str] = "my_task"
            schema_version: ClassVar[str] = "v1"
            field_a: str = ""
    """
    schema_name: ClassVar[str] = ""
    schema_version: ClassVar[str] = ""


T = TypeVar("T", bound=BaseModel)


@dataclass
class ExtractionResult:
    """Result of one extraction attempt."""
    doc_id: str
    raw: dict                          # raw LLM JSON output
    parsed: BaseModel | None = None    # validated Pydantic object (None if invalid)
    valid: bool = False
    validation_errors: list[dict] = field(default_factory=list)
    cached: bool = False               # True if loaded from cache
    stale: bool = False                # True if cached but prompt/data changed
    usage: dict = field(default_factory=dict)


def extract(
    *,
    doc_id: str,
    text: str,
    system_prompt: str,
    user_prompt: str,
    schema: type[T],
    model: str,
    cache: LLMCache,
    client: Any,
    reextract: bool = False,
    temperature: float = 0,
    max_tokens: int = 4000,
) -> ExtractionResult:
    """Extract structured data from text via LLM.

    Parameters
    ----------
    doc_id : str
        Unique document identifier.
    text : str
        Document text to extract from.
    system_prompt : str
        System prompt for the LLM.
    user_prompt : str
        Rendered user prompt (with text already inserted).
    schema : type[BaseModel]
        Pydantic model to validate the LLM output against.
    model : str
        LLM model name (e.g. "gpt-4o-mini").
    cache : LLMCache
        Cache instance.
    client : openai.OpenAI
        OpenAI client instance.
    reextract : bool
        If True, re-extract documents whose cached entry has a different
        prompt hash or text hash than the current values.  Default False.
    temperature : float
        LLM temperature (default 0).
    max_tokens : int
        Max completion tokens.
    """
    t_hash = text_hash(text)
    p_hash = content_hash(system_prompt)
    key = cache.key(doc_id, t_hash, model)

    # Read schema identity if available (from ExtractionSchema subclass)
    s_name = getattr(schema, "schema_name", "") or ""
    s_version = getattr(schema, "schema_version", "") or ""

    # ── Check cache ──────────────────────────────────────────────────
    hit = cache.get(key)
    if hit is not None:
        stale = hit.is_stale(current_prompt_hash=p_hash)
        if not (reextract and stale):
            # Use cached result
            parsed, valid, errors = _validate(hit.extraction, schema)
            return ExtractionResult(
                doc_id=doc_id,
                raw=hit.extraction,
                parsed=parsed,
                valid=valid,
                validation_errors=errors,
                cached=True,
                stale=stale,
                usage=hit.meta.get("usage", {}),
            )

    # ── Call LLM ─────────────────────────────────────────────────────
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
        temperature=temperature,
        max_tokens=max_tokens,
    )
    content = response.choices[0].message.content
    raw = json.loads(content)
    usage = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
    }
    finish_reason = response.choices[0].finish_reason or ""
    # The API returns the actual model version used (e.g. "gpt-4o-mini-2024-07-18")
    model_version = response.model or ""

    # ── Validate ─────────────────────────────────────────────────────
    parsed, valid, errors = _validate(raw, schema)

    # ── Cache ────────────────────────────────────────────────────────
    cache.put(
        key,
        raw,
        doc_id=doc_id,
        text_hash=t_hash,
        messages=messages,
        prompt_hash=p_hash,
        model=model,
        model_version=model_version,
        temperature=temperature,
        max_tokens=max_tokens,
        finish_reason=finish_reason,
        schema_name=s_name,
        schema_version=s_version,
        validation_status="valid" if valid else "invalid",
        usage=usage,
        api_params={
            "response_format": "json_object",
            "top_p": 1,
        },
    )

    return ExtractionResult(
        doc_id=doc_id,
        raw=raw,
        parsed=parsed,
        valid=valid,
        validation_errors=errors,
        cached=False,
        stale=False,
        usage=usage,
    )


def _validate(
    raw: dict, schema: type[T]
) -> tuple[T | None, bool, list[dict]]:
    """Validate raw dict against Pydantic schema."""
    try:
        parsed = schema.model_validate(raw)
        return parsed, True, []
    except ValidationError as e:
        return None, False, e.errors()
