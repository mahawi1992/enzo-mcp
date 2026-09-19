# Enzo MCP

Enzo applies decomposition pressure between an intelligent LLM and JEV:

```text
large question -> independently falsifiable atom -> evidence -> semantic result
```

The LLM remains responsible for strategy and deciding what to investigate. Enzo
only admits one operationalized predicate at a time, keeps evidence and dependency
history explicit, and refuses to convert missing information into confidence.

## Why Enzo?

The name is inspired by the Japanese Zen **ensō** (円相), the hand-drawn circle.
Enzo uses that image as a reasoning metaphor: begin with the large circle of a
problem, then reduce it into smaller circles until each one contains exactly one
independently falsifiable claim.

## Current scope

Version 0.1 provides exactly three MCP tools:

- `enzo_atomize` returns `ATOMIC`, `DECOMPOSE`, or `NEEDS_REFINEMENT`.
- `enzo_observe` applies typed evidence to an admitted atom.
- `enzo_state` returns canonical history plus computed status and frontier.

JEV is connected through Pydantic AI's official TypeSafe provider. When
`TYPESAFE_API_KEY` is present, Enzo sends the atom's supplied context and evidence
to `jev-latest` and asks one typed question. Without the key, the sensor fails
closed as `UNKNOWN`. Deterministic evidence can produce `VERIFIED` or
`CONTRADICTED` without calling Jev and cannot be overwritten by Jev.

## Why three atomicity outcomes?

`DECOMPOSE` is returned only when Enzo can identify safe, explicit child claims.
When prose looks composite but a deterministic split could change its meaning,
Enzo returns `NEEDS_REFINEMENT` and asks the LLM to propose the children. This keeps
the atomizer scientific without pretending that punctuation is semantics.

## Run

```bash
cd src/enzo
uv sync --group dev
export TYPESAFE_API_KEY="your-key"
uv run enzo-mcp
```

The default transport is stdio. For development with MCP Inspector:

```bash
uv run mcp dev src/enzo_mcp/server.py
```

## Test

```bash
uv run pytest
uv run ruff check .
uv run mypy
```

## JEV behavior

`PydanticJevSensor` uses the official `TypeSafeModel("jev-latest")` client and maps
Enzo's answer contracts to Jev primitives:

- `BOOLEAN` -> `Noul`
- `CHOICE` -> `Choice`
- `SCORE` -> `Score`

Jev's native probability/confidence is preserved in the sensor evidence payload,
but Enzo does not expose an invented aggregate confidence score. The default
conservative thresholds are `>= 0.8` for verification and `<= 0.2` for a Boolean
contradiction. Values between them remain unresolved.

Every semantic observation defaults to local-only processing. Set
`allow_external_jev=true` on that `enzo_observe` request to authorize sending its
context and evidence to TypeSafe/Jev. Without that explicit per-observation consent,
Enzo returns `UNKNOWN` and sends nothing externally.

Keep the API key in the MCP process environment or its secret manager; do not put
it in a request, source file, or committed configuration. Context and evidence
provided to a semantic observation are transmitted to the external TypeSafe API,
so callers must redact credentials, personal data, and unrelated sensitive content
before submitting them.
