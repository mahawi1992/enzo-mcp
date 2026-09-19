from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.validate import EvalValidationError, validate_path


def test_public_evaluation_examples_validate() -> None:
    summary = validate_path(Path("evals/public_cases.jsonl"))

    assert summary.case_count == 7
    assert summary.adjudicated_count == 0
    assert summary.category_counts == {
        "atomicity": 3,
        "evidence_status": 1,
        "outbound_projection": 1,
        "sensor_abstention": 1,
        "source_independence": 1,
    }


def test_evaluation_bundle_rejects_duplicate_ids(tmp_path: Path) -> None:
    case = {
        "schema_version": 1,
        "id": "duplicate",
        "category": "sensor_abstention",
        "input": {},
        "expected": {"status": "UNKNOWN"},
        "tags": ["test"],
        "adjudication": {
            "status": "PUBLIC_EXAMPLE",
            "reviewer_count": 0,
            "blind": False,
        },
    }
    path = tmp_path / "duplicate.jsonl"
    serialized = json.dumps(case)
    path.write_text(f"{serialized}\n{serialized}\n", encoding="utf-8")

    with pytest.raises(EvalValidationError, match="repeats case id"):
        validate_path(path)


def test_evaluation_bundle_rejects_unreviewed_adjudication(tmp_path: Path) -> None:
    case = {
        "schema_version": 1,
        "id": "unsupported-adjudication",
        "category": "atomicity",
        "input": {},
        "expected": {"decision": "ATOMIC"},
        "tags": ["test"],
        "adjudication": {
            "status": "HUMAN_ADJUDICATED",
            "reviewer_count": 1,
            "blind": False,
        },
    }
    path = tmp_path / "unsupported.jsonl"
    path.write_text(f"{json.dumps(case)}\n", encoding="utf-8")

    with pytest.raises(EvalValidationError, match="two blind reviewers"):
        validate_path(path)
