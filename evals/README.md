# Enzo offline evaluation scaffold

This directory defines a versioned, provider-free format for measuring Enzo without
turning visible examples into claims of benchmark performance.

`public_cases.jsonl` contains small contract examples. It is documentation and CI
input, not a held-out corpus. Private or human-adjudicated bundles should live
outside the repository and be passed explicitly to the validator:

```bash
uv run python -m evals.validate evals/public_cases.jsonl
uv run python -m evals.validate /secure/path/enzo-heldout-v1.jsonl
```

Every case records its category, input, expected result, tags, and adjudication
status. A case may claim `HUMAN_ADJUDICATED` only when at least two blinded reviewers
were recorded. Validation checks structure and labels; it does not execute Jev,
authenticate provenance, or establish that a proposed implementation is correct.

The first held-out bundle should measure:

- material-claim coverage, false splits, and appropriate refinement;
- false `VERIFIED` outcomes under missing or contradictory evidence;
- duplicate-source inflation and stale or fabricated provenance;
- outbound canary leakage and manifest mismatch rejection;
- Jev abstention, ordering sensitivity, and probability calibration.

Freeze bundle contents, prompts, thresholds, models, and scoring rules before a
comparison. Report task counts and uncertainty intervals, and keep deterministic
contract failures separate from subjective adjudicator disagreement.

