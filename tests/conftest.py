from __future__ import annotations

from typing import Any

import pytest

from enzo_mcp.models import (
    AtomizeRequest,
    EvidenceDirection,
    EvidenceKind,
    EvidenceRecord,
    EvidenceRequirement,
    ExpectedAnswerType,
    PredicateOperator,
    Provenance,
    VerificationMethod,
)


@pytest.fixture
def requirement() -> EvidenceRequirement:
    return EvidenceRequirement(
        id="required-config",
        description="Read the production cookie configuration",
        accepted_kinds=(EvidenceKind.SCHEMA_VALIDATION,),
        deterministic_required=True,
    )


@pytest.fixture
def make_request(requirement: EvidenceRequirement):
    def factory(**updates: Any) -> AtomizeRequest:
        values: dict[str, Any] = {
            "atom_id": "atom-root",
            "root_goal": "Determine whether the production session cookie is hardened",
            "question": "Does the production session cookie set Secure=true?",
            "subject": "the production session cookie",
            "predicate": "sets Secure=true",
            "scope": "production session configuration",
            "operator": PredicateOperator.EQUALS,
            "expected_value": True,
            "expected_answer_type": ExpectedAnswerType.BOOLEAN,
            "evidence_requirements": (requirement,),
            "verification_method": VerificationMethod.SCHEMA_VALIDATION,
        }
        values.update(updates)
        return AtomizeRequest(**values)

    return factory


@pytest.fixture
def make_evidence():
    def factory(
        *,
        atom_id: str,
        requirement_id: str,
        direction: EvidenceDirection = EvidenceDirection.SUPPORTS,
        evidence_id: str = "evidence-1",
        deterministic: bool = True,
    ) -> EvidenceRecord:
        return EvidenceRecord(
            id=evidence_id,
            atom_id=atom_id,
            direction=direction,
            kind=(EvidenceKind.SCHEMA_VALIDATION if deterministic else EvidenceKind.USER_PROVIDED),
            summary="Production configuration contains cookie.secure=true",
            deterministic=deterministic,
            verification_method=VerificationMethod.SCHEMA_VALIDATION,
            requirement_ids=(requirement_id,),
            provenance=Provenance(
                provider="pytest",
                locator="tests/fixtures/production-config.json",
                tool="schema-validator",
                run_id=evidence_id,
            ),
        )

    return factory
