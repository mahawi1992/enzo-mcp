from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta
from itertools import product

import pytest

from enzo_mcp.engine import EnzoEngine
from enzo_mcp.models import (
    AtomicityDecision,
    DependencyLogic,
    DependencyRole,
    EvidenceDirection,
    EvidenceKind,
    EvidenceRecord,
    EvidenceRequirement,
    ObserveRequest,
    PredicateOperator,
    ProposedAtom,
    Provenance,
    SemanticStatus,
    VerificationMethod,
)


async def test_seeded_exact_observation_retries_are_idempotent(
    make_request,
    make_evidence,
) -> None:
    engine = EnzoEngine()
    atom = (await engine.atomize(make_request())).atom
    requirement_id = atom.evidence_requirements[0].id
    rng = random.Random(7341)
    directions = [
        rng.choice((EvidenceDirection.SUPPORTS, EvidenceDirection.CONTRADICTS)) for _ in range(12)
    ]
    observed_ids: list[str] = []

    for index, direction in enumerate(directions):
        evidence = make_evidence(
            atom_id=atom.id,
            requirement_id=requirement_id,
            direction=direction,
            evidence_id=f"generated-evidence-{index}",
        )
        request = ObserveRequest(
            investigation_id=atom.investigation_id,
            atom_id=atom.id,
            evidence=(evidence,),
        )

        first_result = await engine.observe(request)
        before_retry = await engine.state(atom.investigation_id)
        retry_result = await engine.observe(request)
        after_retry = await engine.state(atom.investigation_id)

        observed_ids.append(evidence.id)
        assert retry_result is first_result
        assert after_retry == before_retry
        assert tuple(item.id for item in first_result.evidence) == tuple(observed_ids)
        assert first_result.status in {
            SemanticStatus.VERIFIED,
            SemanticStatus.CONTRADICTED,
        }


async def test_timestamp_only_provenance_change_is_an_exact_retry(
    make_request,
    make_evidence,
) -> None:
    engine = EnzoEngine()
    atom = (await engine.atomize(make_request())).atom
    evidence = make_evidence(
        atom_id=atom.id,
        requirement_id=atom.evidence_requirements[0].id,
        evidence_id="timestamped-evidence",
    )
    first_time = datetime(2026, 9, 19, 12, 0, tzinfo=UTC)
    retry_time = first_time + timedelta(minutes=5)
    evidence_at_first_time = evidence.model_copy(
        update={"provenance": evidence.provenance.model_copy(update={"observed_at": first_time})}
    )
    evidence_at_retry_time = evidence.model_copy(
        update={"provenance": evidence.provenance.model_copy(update={"observed_at": retry_time})}
    )

    first_result = await engine.observe(
        ObserveRequest(
            investigation_id=atom.investigation_id,
            atom_id=atom.id,
            evidence=(evidence_at_first_time,),
        )
    )
    retry_result = await engine.observe(
        ObserveRequest(
            investigation_id=atom.investigation_id,
            atom_id=atom.id,
            evidence=(evidence_at_retry_time,),
        )
    )

    assert (
        evidence_at_first_time.provenance.observed_at
        != evidence_at_retry_time.provenance.observed_at
    )
    assert retry_result is first_result
    assert retry_result.evidence[0].provenance.observed_at == first_time


async def test_same_evidence_id_with_changed_content_is_rejected(
    make_request,
    make_evidence,
) -> None:
    engine = EnzoEngine()
    atom = (await engine.atomize(make_request())).atom
    evidence = make_evidence(
        atom_id=atom.id,
        requirement_id=atom.evidence_requirements[0].id,
        evidence_id="stable-evidence-id",
    )
    first_result = await engine.observe(
        ObserveRequest(
            investigation_id=atom.investigation_id,
            atom_id=atom.id,
            evidence=(evidence,),
        )
    )
    changed_evidence = evidence.model_copy(update={"summary": "Different evidence content"})
    before_rejection = await engine.state(atom.investigation_id)

    with pytest.raises(ValueError, match="was reused with different content"):
        await engine.observe(
            ObserveRequest(
                investigation_id=atom.investigation_id,
                atom_id=atom.id,
                evidence=(changed_evidence,),
            )
        )

    assert (await engine.state(atom.investigation_id)) == before_rejection
    assert before_rejection.atoms[0].record.result is first_result


def _reference_definition_status(
    logic: DependencyLogic,
    statuses: tuple[SemanticStatus, ...],
) -> SemanticStatus:
    if logic is DependencyLogic.ALL_OF:
        if SemanticStatus.CONTRADICTED in statuses:
            return SemanticStatus.CONTRADICTED
        if all(status is SemanticStatus.VERIFIED for status in statuses):
            return SemanticStatus.VERIFIED
        if SemanticStatus.INSUFFICIENT_EVIDENCE in statuses:
            return SemanticStatus.INSUFFICIENT_EVIDENCE
        return SemanticStatus.UNKNOWN

    if SemanticStatus.VERIFIED in statuses:
        return SemanticStatus.VERIFIED
    if all(status is SemanticStatus.CONTRADICTED for status in statuses):
        return SemanticStatus.CONTRADICTED
    if SemanticStatus.INSUFFICIENT_EVIDENCE in statuses:
        return SemanticStatus.INSUFFICIENT_EVIDENCE
    return SemanticStatus.UNKNOWN


def _evidence_for_child_status(
    atom_id: str,
    requirement_id: str,
    status: SemanticStatus,
) -> tuple[EvidenceRecord, ...]:
    if status is SemanticStatus.UNKNOWN:
        return ()

    if status is SemanticStatus.INSUFFICIENT_EVIDENCE:
        return (
            EvidenceRecord(
                id=f"{atom_id}-neutral-evidence",
                atom_id=atom_id,
                direction=EvidenceDirection.NEUTRAL,
                kind=EvidenceKind.USER_PROVIDED,
                summary="Neutral evidence leaves the required proof unresolved",
                verification_method=VerificationMethod.HUMAN_OBSERVATION,
                provenance=Provenance(provider="pytest", locator="generated-neutral-case"),
            ),
        )

    direction = (
        EvidenceDirection.SUPPORTS
        if status is SemanticStatus.VERIFIED
        else EvidenceDirection.CONTRADICTS
    )
    return (
        EvidenceRecord(
            id=f"{atom_id}-{status.value.lower()}-evidence",
            atom_id=atom_id,
            direction=direction,
            kind=EvidenceKind.SCHEMA_VALIDATION,
            summary=f"Deterministic {status.value.lower()} test evidence",
            deterministic=True,
            verification_method=VerificationMethod.SCHEMA_VALIDATION,
            requirement_ids=(requirement_id,),
            provenance=Provenance(
                provider="pytest",
                locator=f"generated-{status.value.lower()}-case",
            ),
        ),
    )


@pytest.mark.parametrize("logic", (DependencyLogic.ALL_OF, DependencyLogic.ANY_OF))
async def test_derived_parent_status_matches_reference_for_generated_child_statuses(
    make_request,
    logic: DependencyLogic,
) -> None:
    statuses = (
        SemanticStatus.VERIFIED,
        SemanticStatus.CONTRADICTED,
        SemanticStatus.UNKNOWN,
        SemanticStatus.INSUFFICIENT_EVIDENCE,
    )
    operator_word = "and" if logic is DependencyLogic.ALL_OF else "or"

    for child_statuses in product(statuses, repeat=2):
        requirements = tuple(
            EvidenceRequirement(
                id=f"child-requirement-{index}",
                description=f"Verify child {index}",
                accepted_kinds=(EvidenceKind.SCHEMA_VALIDATION,),
                deterministic_required=True,
            )
            for index in range(2)
        )
        children = tuple(
            ProposedAtom(
                question=f"Does criterion {index} hold?",
                subject=f"criterion {index}",
                predicate=f"sets criterion-{index}=true",
                scope="generated dependency case",
                operator=PredicateOperator.EQUALS,
                expected_value=True,
                evidence_requirements=(requirements[index],),
                verification_method=VerificationMethod.SCHEMA_VALIDATION,
            )
            for index in range(2)
        )
        engine = EnzoEngine()
        atomized = await engine.atomize(
            make_request(
                predicate=f"sets criterion-0=true {operator_word} sets criterion-1=true",
                proposed_children=children,
            )
        )

        assert atomized.decision is AtomicityDecision.DECOMPOSE
        definition = next(
            group
            for group in atomized.atom.dependencies
            if group.role is DependencyRole.DEFINES_PARENT
        )
        assert definition.logic is logic

        for child, child_status in zip(atomized.children, child_statuses, strict=True):
            await engine.observe(
                ObserveRequest(
                    investigation_id=atomized.investigation_id,
                    atom_id=child.id,
                    evidence=_evidence_for_child_status(
                        child.id,
                        child.evidence_requirements[0].id,
                        child_status,
                    ),
                )
            )

        view = await engine.state(atomized.investigation_id)
        effective_statuses = {item.record.atom.id: item.effective_status for item in view.atoms}
        assert tuple(effective_statuses[child.id] for child in atomized.children) == child_statuses
        assert effective_statuses[atomized.atom.id] is _reference_definition_status(
            logic,
            child_statuses,
        )


async def test_revision_preserves_observed_history_and_blocks_superseded_observations(
    make_request,
    make_evidence,
) -> None:
    engine = EnzoEngine()
    original = await engine.atomize(make_request())
    original_evidence = make_evidence(
        atom_id=original.atom.id,
        requirement_id=original.atom.evidence_requirements[0].id,
        evidence_id="original-history-evidence",
    )
    original_result = await engine.observe(
        ObserveRequest(
            investigation_id=original.investigation_id,
            atom_id=original.atom.id,
            evidence=(original_evidence,),
        )
    )
    revised = await engine.atomize(
        make_request(
            investigation_id=original.investigation_id,
            atom_id="atom-revised-history",
            revision_of=original.atom.id,
            question="Does the production session cookie explicitly set Secure=true?",
            predicate="explicitly sets Secure=true",
        )
    )
    before_rejected_observation = await engine.state(original.investigation_id)
    records = {item.record.atom.id: item.record for item in before_rejected_observation.atoms}

    assert records[original.atom.id].replaced_by == revised.atom.id
    assert records[original.atom.id].result is original_result
    assert records[original.atom.id].result.evidence == (original_evidence,)
    assert records[revised.atom.id].result is None

    with pytest.raises(ValueError, match="superseded atom cannot receive new observations"):
        await engine.observe(
            ObserveRequest(
                investigation_id=original.investigation_id,
                atom_id=original.atom.id,
            )
        )

    assert await engine.state(original.investigation_id) == before_rejected_observation
