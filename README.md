# llmkit

Minimal LLM extraction framework: structured outputs (Pydantic), a
file-backed deterministic cache, and an audit-sample helper for human
review of extracted fields.

The point is to make LLM extraction over a fixed set of documents
**reproducible and auditable**, not to wrap a model SDK. You bring your
own client (OpenAI, Anthropic, etc.); `llmkit` handles caching,
validation, and the audit workflow on top.

## Install

```bash
pip install git+https://github.com/hsigstad/llmkit.git
# or, with the OpenAI extra:
pip install "llmkit[openai] @ git+https://github.com/hsigstad/llmkit.git"
```

## Usage

```python
from pathlib import Path
from llmkit import extract, ExtractionSchema, LLMCache
from openai import OpenAI

class Irregularity(ExtractionSchema):
    schema_name = "irregularity"
    schema_version = "1.0"
    found: bool
    severity: str | None = None

cache = LLMCache(Path("cache/"))
client = OpenAI()

result = extract(
    doc_id="case-001",
    text=document_text,
    system_prompt="...",
    user_prompt="...",
    schema=Irregularity,
    model="gpt-4o-mini",
    prompt_file="prompts/irregularity_system.txt",
    cache=cache,
    client=client,
)
if result.valid:
    print(result.parsed.severity)
```

The cache key is `(doc_id, text_hash, model)` — editing a prompt does
**not** invalidate cached results. Use `cache.is_stale()` to detect
entries produced under a previous prompt/data version, and
`reextract=True` to force re-extraction. This separation lets you
iterate on prompts cheaply during development and force a clean
re-extraction before running analyses.

## Audit

```python
from llmkit import audit_sample

sample = audit_sample(cache, n=50, stratify_by=lambda meta: meta["model"])
# Review by hand; record agreement counts in a notebook.
```

## Status

Research tooling — used across my own projects. Not packaged for general
use; APIs may change as the projects evolve.

## Related repos

Part of a set of repositories I use across my research projects:

- [research-kit](https://github.com/hsigstad/research-kit) — Claude Code
  skills, conventions, methodology docs, tools (includes the `/llmkit`
  skill that documents conventional usage)
- [diarios](https://github.com/hsigstad/diarios) — Brazilian
  official-diary parsing
- [newsbr](https://github.com/hsigstad/newsbr) — Brazilian news collection
- [brazil-institutions](https://github.com/hsigstad/brazil-institutions) —
  institutional reference for Brazil-focused research

## License

MIT — see [`LICENSE`](LICENSE).
