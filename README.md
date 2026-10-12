# bhoot

> **Verified, grounded structured extraction for lending documents.**
> Status: **pre-alpha, under active development.** APIs will change.

`bhoot` extracts structured data from bank statements, salary slips and KYC documents. Every value comes back **with its source** (page, exact quote, position) and is **checked by rules**, such as running balances and totals. A value is either **verified** or **flagged for review**, never silently wrong.

## Goals

- **Zero undetected errors:** grounding + domain rules + targeted repair.
- **Low cost:** code first, cheap model first, re-ask only what failed, local OCR for scans.
- **Library first:** `pip install bhoot`, no server, no GPU. Bring your own LLM API key.
- **Code-first schemas and prompts:** versioned, selected by name (`bank_statement@prod`).
- **Scanned documents:** local CPU OCR (RapidOCR) with crop-based repair.

## Quickstart (target API, not yet implemented)

```python
from bhoot import Bhoot

bh = Bhoot()
r = bh.extract("statement.pdf", schema="pack:lending.bank_statement")
print(r.status, r.data["closing_balance"], r.fields["closing_balance"].quote)
```

## Development

```bash
uv sync --group dev          # create the virtualenv and install dev tools
uv run pytest                # tests
uv run ruff check .          # lint
uv run mypy                  # types
uv run lint-imports          # architecture layer rules
```

Optional extras: `bhoot[ocr]` (local OCR), `bhoot[otel]` (tracing), `bhoot[eval]` (reports).

## Licence

Copyright (c) 2026 Satyam. All rights reserved. No licence is granted at this time.
