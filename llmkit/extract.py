"""Core extraction: call LLM, validate with Pydantic, cache result.

Cache key is (doc_id, text_hash, model[, schema_name]) — prompt changes do
NOT invalidate cache.  Use ``reextract=True`` to force re-extraction of
stale entries (different prompt or data since cached).

Two LLM-call modes are supported:

* Legacy ``json_object`` mode (default).  ``response_format`` is set to
  ``{"type": "json_object"}``.  The model returns a JSON document but the
  shape is enforced only by the prompt — typically the system prompt has
  to spell out the field names and the literal word "json" must appear in
  the messages.  Pydantic validation happens client-side after the call.
* Structured Outputs (``use_structured_outputs=True``).  ``response_format``
  is set to the Pydantic schema and the call uses
  ``client.chat.completions.parse``, so OpenAI enforces the schema
  server-side via constrained decoding.  No JSON-keyword requirement on
  the prompt and no field-name drift.  Available on ``gpt-4o``,
  ``gpt-4o-mini``, ``gpt-4.1``, and o-series models.

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
        cache=LLMCache(Path("cache_dir")),
        client=openai_client,
        use_structured_outputs=True,   # opt-in to schema enforcement
        schema_in_cache_key=True,      # opt-in to schema-aware cache key
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


def _schema_name(schema: type[BaseModel]) -> str:
    return getattr(schema, "schema_name", "") or ""


def _is_reasoning_model(model: str) -> bool:
    """Reasoning models (gpt-5 family, o-series) reject `temperature` != default
    and `max_tokens`; they require `max_completion_tokens` and no temperature
    override. `gpt-5-chat*` is a standard chat model and is NOT reasoning."""
    m = model.lower()
    if "chat" in m:
        return False
    return m.startswith(("o1", "o3", "o4")) or m.startswith("gpt-5")


def _sampling_kwargs(model: str, temperature: float, max_tokens: int) -> dict:
    """Per-family sampling kwargs. Reasoning models take `max_completion_tokens`
    and no `temperature`; standard chat models take `temperature` + `max_tokens`."""
    if _is_reasoning_model(model):
        return {"max_completion_tokens": max_tokens}
    return {"temperature": temperature, "max_tokens": max_tokens}


def _call_legacy_json_mode(
    *,
    client: Any,
    model: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
) -> tuple[dict, dict, str, str, dict]:
    """Legacy JSON-mode call. Returns (raw, usage, finish_reason,
    model_version, api_params)."""
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
        **_sampling_kwargs(model, temperature, max_tokens),
    )
    content = response.choices[0].message.content
    raw = json.loads(content)
    usage = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
    }
    finish_reason = response.choices[0].finish_reason or ""
    model_version = response.model or ""
    api_params = {"response_format": "json_object", "top_p": 1,
                  "reasoning_model": _is_reasoning_model(model)}
    return raw, usage, finish_reason, model_version, api_params


def _call_structured_outputs(
    *,
    client: Any,
    model: str,
    messages: list[dict],
    schema: type[BaseModel],
    temperature: float,
    max_tokens: int,
) -> tuple[dict, dict, str, str, dict]:
    """Structured-Outputs call (schema enforced server-side). Returns
    (raw, usage, finish_reason, model_version, api_params). The 'raw'
    dict is the Pydantic-instance's model_dump() — i.e. already
    schema-conformant, but we still feed it through ``_validate()``
    downstream as defense-in-depth (and to handle refusals)."""
    response = client.chat.completions.parse(
        model=model,
        messages=messages,
        response_format=schema,
        **_sampling_kwargs(model, temperature, max_tokens),
    )
    choice = response.choices[0]
    msg = choice.message
    usage = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
    }
    finish_reason = choice.finish_reason or ""
    model_version = response.model or ""
    api_params = {
        "response_format": "structured_outputs",
        "schema_name": _schema_name(schema),
        "top_p": 1,
        "reasoning_model": _is_reasoning_model(model),
    }
    # Refusal path: server declined to answer for safety reasons.
    if getattr(msg, "refusal", None):
        api_params["refusal"] = msg.refusal
        return {}, usage, finish_reason, model_version, api_params
    parsed = getattr(msg, "parsed", None)
    if parsed is not None:
        raw = parsed.model_dump()
    else:
        # Fallback: parsed missing (shouldn't happen on the happy path),
        # decode the raw content if any.
        raw = json.loads(msg.content) if msg.content else {}
    return raw, usage, finish_reason, model_version, api_params


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
    use_structured_outputs: bool = False,
    schema_in_cache_key: bool = False,
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
    use_structured_outputs : bool
        If True, call ``client.chat.completions.parse(response_format=schema)``
        so the schema is enforced server-side.  Requires a model that
        supports Structured Outputs (gpt-4o / gpt-4o-mini / gpt-4.1 /
        o-series).  Default False to preserve historical behavior for
        existing callers.
    schema_in_cache_key : bool
        If True, the schema's ``schema_name`` is mixed into the composite
        cache key, preventing cross-task collisions when multiple
        extraction tasks share a cache directory.  Default False to keep
        existing caches valid; new callers should set this True.
    """
    t_hash = text_hash(text)
    p_hash = content_hash(system_prompt)
    s_name = _schema_name(schema)
    s_version = getattr(schema, "schema_version", "") or ""
    key_schema = s_name if schema_in_cache_key else ""
    key = cache.key(doc_id, t_hash, model, schema_name=key_schema)

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
    if use_structured_outputs:
        raw, usage, finish_reason, model_version, api_params = \
            _call_structured_outputs(
                client=client, model=model, messages=messages,
                schema=schema, temperature=temperature,
                max_tokens=max_tokens,
            )
    else:
        raw, usage, finish_reason, model_version, api_params = \
            _call_legacy_json_mode(
                client=client, model=model, messages=messages,
                temperature=temperature, max_tokens=max_tokens,
            )

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
        api_params=api_params,
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
