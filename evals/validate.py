"""Validate versioned Enzo JSONL evaluation bundles without provider calls."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_CATEGORIES = {
    "atomicity",
    "evidence_status",
    "source_independence",
    "outbound_projection",
    "sensor_abstention",
}
_DECISIONS = {"ATOMIC", "DECOMPOSE", "NEEDS_REFINEMENT"}
_STATUSES = {"VERIFIED", "CONTRADICTED", "UNKNOWN", "INSUFFICIENT_EVIDENCE"}


class EvalValidationError(ValueError):
    """Raised when an offline evaluation bundle violates its public contract."""


@dataclass(frozen=True)
class EvalSummary:
    """Deterministic summary emitted after validating an evaluation bundle."""

    case_count: int
    category_counts: dict[str, int]
    adjudicated_count: int

    def as_json(self) -> dict[str, object]:
        """Return a stable JSON-serializable metrics summary."""

        return {
            "schema_version": 1,
            "case_count": self.case_count,
            "category_counts": self.category_counts,
            "adjudicated_count": self.adjudicated_count,
        }


def _require_object(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EvalValidationError(f"{label} must be an object")
    return value


def _validate_case(case: object, *, line_number: int) -> tuple[str, bool]:
    item = _require_object(case, label=f"line {line_number}")
    required = {
        "schema_version",
        "id",
        "category",
        "input",
        "expected",
        "tags",
        "adjudication",
    }
    unknown = set(item) - required
    missing = required - set(item)
    if missing or unknown:
        raise EvalValidationError(
            f"line {line_number} has missing={sorted(missing)} unknown={sorted(unknown)}"
        )
    if item["schema_version"] != 1:
        raise EvalValidationError(f"line {line_number} uses an unsupported schema_version")
    if not isinstance(item["id"], str) or not item["id"].strip():
        raise EvalValidationError(f"line {line_number} requires a non-empty id")
    category = item["category"]
    if category not in _CATEGORIES:
        raise EvalValidationError(f"line {line_number} has an unknown category")

    inputs = _require_object(item["input"], label=f"line {line_number} input")
    expected = _require_object(item["expected"], label=f"line {line_number} expected")
    tags = item["tags"]
    if (
        not isinstance(tags, list)
        or any(not isinstance(tag, str) or not tag.strip() for tag in tags)
        or len(tags) != len(set(tags))
    ):
        raise EvalValidationError(f"line {line_number} tags must be unique non-empty strings")

    if category == "atomicity" and expected.get("decision") not in _DECISIONS:
        raise EvalValidationError(f"line {line_number} requires a valid atomicity decision")
    if category in {"evidence_status", "sensor_abstention"} and (
        expected.get("status") not in _STATUSES
    ):
        raise EvalValidationError(f"line {line_number} requires a valid semantic status")
    if category == "source_independence" and not isinstance(
        expected.get("distinct_source_count"), int
    ):
        raise EvalValidationError(f"line {line_number} requires distinct_source_count")
    if category == "outbound_projection":
        if not isinstance(inputs.get("selected_paths"), list):
            raise EvalValidationError(f"line {line_number} requires selected_paths")
        if not isinstance(expected.get("sent_canaries"), list):
            raise EvalValidationError(f"line {line_number} requires sent_canaries")

    adjudication = _require_object(item["adjudication"], label=f"line {line_number} adjudication")
    if set(adjudication) != {"status", "reviewer_count", "blind"}:
        raise EvalValidationError(f"line {line_number} has an invalid adjudication shape")
    status = adjudication["status"]
    reviewer_count = adjudication["reviewer_count"]
    blind = adjudication["blind"]
    if status not in {"PUBLIC_EXAMPLE", "HUMAN_ADJUDICATED"}:
        raise EvalValidationError(f"line {line_number} has an invalid adjudication status")
    if type(reviewer_count) is not int or reviewer_count < 0 or type(blind) is not bool:
        raise EvalValidationError(f"line {line_number} has invalid adjudication metadata")
    if status == "HUMAN_ADJUDICATED" and (reviewer_count < 2 or not blind):
        raise EvalValidationError(
            f"line {line_number} cannot claim human adjudication without two blind reviewers"
        )
    return str(category), status == "HUMAN_ADJUDICATED"


def validate_path(path: Path) -> EvalSummary:
    """Validate one JSONL bundle and return deterministic aggregate counts."""

    ids: set[str] = set()
    category_counts: Counter[str] = Counter()
    adjudicated_count = 0
    case_count = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                case = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise EvalValidationError(f"line {line_number} is not valid JSON") from error
            category, adjudicated = _validate_case(case, line_number=line_number)
            case_id = str(case["id"])
            if case_id in ids:
                raise EvalValidationError(f"line {line_number} repeats case id {case_id!r}")
            ids.add(case_id)
            category_counts[category] += 1
            adjudicated_count += int(adjudicated)
            case_count += 1
    if case_count == 0:
        raise EvalValidationError("evaluation bundle must contain at least one case")
    return EvalSummary(
        case_count=case_count,
        category_counts=dict(sorted(category_counts.items())),
        adjudicated_count=adjudicated_count,
    )


def main() -> None:
    """Validate a JSONL bundle and print a machine-readable summary."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    summary = validate_path(args.path)
    print(json.dumps(summary.as_json(), sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
