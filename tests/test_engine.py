from __future__ import annotations

import pytest
from pydantic import ValidationError

from enzo_mcp.engine import EnzoEngine
from enzo_mcp.models import (
    AtomicityDecision,
    DependencyGroup,
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


async def test_deterministic_evidence_verifies_atom(make_request, make_evidence) -> None:
    engine = EnzoEngine()
    atomized = await engine.atomize(make_request())
    atom = atomized.atom
    evidence = make_evidence(
        atom_id=atom.id,
        requirement_id=atom.evidence_requirements[0].id,
    )

    result = await engine.observe(
        ObserveRequest(
            investigation_id=atom.investigation_id,
            atom_id=atom.id,
            evidence=(evidence,),
        )
    )
    view = await engine.state(atom.investigation_id)

    assert result.status is SemanticStatus.VERIFIED
    assert result.source.value == "DETERMINISTIC"
    assert view.verified_atoms == (atom.id,)
    assert view.current_frontier == ()


async def test_deterministic_contradiction_cannot_be_overwritten(
    make_request,
    make_evidence,
) -> None:
    engine = EnzoEngine()
    atom = (await engine.atomize(make_request())).atom
    requirement_id = atom.evidence_requirements[0].id
    supporting = make_evidence(atom_id=atom.id, requirement_id=requirement_id)
    contradicting = make_evidence(
        atom_id=atom.id,
        requirement_id=requirement_id,
        direction=EvidenceDirection.CONTRADICTS,
        evidence_id="evidence-contradiction",
    )

    await engine.observe(
        ObserveRequest(
            investigation_id=atom.investigation_id,
            atom_id=atom.id,
            evidence=(supporting,),
        )
    )
    result = await engine.observe(
        ObserveRequest(
            investigation_id=atom.investigation_id,
            atom_id=atom.id,
            evidence=(contradicting,),
            satisfied_constraints=("An LLM says the configuration is acceptable",),
        )
    )

    assert result.status is SemanticStatus.CONTRADICTED
    assert {item.id for item in result.evidence} == {
        "evidence-1",
        "evidence-contradiction",
    }


async def test_nondeterministic_contradiction_does_not_override_deterministic_support(
    make_request,
    make_evidence,
) -> None:
    engine = EnzoEngine()
    atom = (await engine.atomize(make_request())).atom
    requirement_id = atom.evidence_requirements[0].id
    supporting = make_evidence(atom_id=atom.id, requirement_id=requirement_id)
    low_trust_contradiction = make_evidence(
        atom_id=atom.id,
        requirement_id=requirement_id,
        direction=EvidenceDirection.CONTRADICTS,
        evidence_id="user-contradiction",
        deterministic=False,
    )

    result = await engine.observe(
        ObserveRequest(
            investigation_id=atom.investigation_id,
            atom_id=atom.id,
            evidence=(supporting, low_trust_contradiction),
        )
    )

    assert result.status is SemanticStatus.VERIFIED
    assert {item.id for item in result.evidence} == {"evidence-1", "user-contradiction"}


async def test_out_of_contract_deterministic_contradiction_is_not_authoritative(
    make_request,
) -> None:
    engine = EnzoEngine()
    atom = (await engine.atomize(make_request())).atom
    no_requirement = EvidenceRecord(
        id="no-requirement",
        atom_id=atom.id,
        direction=EvidenceDirection.CONTRADICTS,
        kind=EvidenceKind.SCHEMA_VALIDATION,
        summary="A contradiction with no declared requirement",
        deterministic=True,
        verification_method=VerificationMethod.SCHEMA_VALIDATION,
        provenance=Provenance(provider="pytest", locator="unbound-result"),
    )
    excluded_kind = EvidenceRecord(
        id="excluded-kind",
        atom_id=atom.id,
        direction=EvidenceDirection.CONTRADICTS,
        kind=EvidenceKind.TEST_RESULT,
        summary="A contradiction from an excluded evidence kind",
        deterministic=True,
        verification_method=VerificationMethod.TEST,
        requirement_ids=(atom.evidence_requirements[0].id,),
        provenance=Provenance(provider="pytest", locator="excluded-result"),
    )

    result = await engine.observe(
        ObserveRequest(
            investigation_id=atom.investigation_id,
            atom_id=atom.id,
            evidence=(no_requirement, excluded_kind),
        )
    )

    assert result.status is SemanticStatus.INSUFFICIENT_EVIDENCE
    assert result.status is not SemanticStatus.CONTRADICTED


async def test_out_of_contract_support_cannot_satisfy_optional_requirement(
    make_request,
) -> None:
    optional_requirement = EvidenceRequirement(
        id="optional-schema",
        description="Optional schema observation",
        accepted_kinds=(EvidenceKind.SCHEMA_VALIDATION,),
        required=False,
    )
    engine = EnzoEngine()
    atom = (await engine.atomize(make_request(evidence_requirements=(optional_requirement,)))).atom
    excluded_support = EvidenceRecord(
        id="excluded-support",
        atom_id=atom.id,
        direction=EvidenceDirection.SUPPORTS,
        kind=EvidenceKind.TEST_RESULT,
        summary="Support from an excluded evidence kind",
        deterministic=True,
        verification_method=VerificationMethod.TEST,
        provenance=Provenance(provider="pytest", locator="excluded-support"),
    )

    result = await engine.observe(
        ObserveRequest(
            investigation_id=atom.investigation_id,
            atom_id=atom.id,
            evidence=(excluded_support,),
        )
    )

    assert result.status is SemanticStatus.INSUFFICIENT_EVIDENCE
    assert result.status is not SemanticStatus.VERIFIED


async def test_no_jev_and_no_evidence_is_unknown(make_request, monkeypatch) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    engine = EnzoEngine()
    atom = (await engine.atomize(make_request())).atom

    result = await engine.observe(
        ObserveRequest(
            allow_external_jev=True,
            investigation_id=atom.investigation_id,
            atom_id=atom.id,
        )
    )

    assert result.status is SemanticStatus.UNKNOWN
    assert "JEV semantic sensor is not configured" in result.missing_information


async def test_parent_is_derived_from_complete_children(make_request, make_evidence) -> None:
    first_requirement = EvidenceRequirement(
        id="secure-flag",
        description="Inspect cookie.secure",
        accepted_kinds=(EvidenceKind.SCHEMA_VALIDATION,),
        deterministic_required=True,
    )
    second_requirement = EvidenceRequirement(
        id="http-only-flag",
        description="Inspect cookie.httpOnly",
        accepted_kinds=(EvidenceKind.SCHEMA_VALIDATION,),
        deterministic_required=True,
    )
    children = (
        ProposedAtom(
            question="Does the production cookie set Secure=true?",
            subject="the production cookie",
            predicate="sets Secure=true",
            scope="production",
            operator=PredicateOperator.EQUALS,
            expected_value=True,
            evidence_requirements=(first_requirement,),
            verification_method=VerificationMethod.SCHEMA_VALIDATION,
        ),
        ProposedAtom(
            question="Does the production cookie set HttpOnly=true?",
            subject="the production cookie",
            predicate="sets HttpOnly=true",
            scope="production",
            operator=PredicateOperator.EQUALS,
            expected_value=True,
            evidence_requirements=(second_requirement,),
            verification_method=VerificationMethod.SCHEMA_VALIDATION,
        ),
    )
    engine = EnzoEngine()
    atomized = await engine.atomize(
        make_request(
            question="Is the production cookie secure?",
            subject="the production cookie",
            predicate="is secure",
            proposed_children=children,
        )
    )

    assert atomized.decision is AtomicityDecision.DECOMPOSE
    for index, child in enumerate(atomized.children, start=1):
        await engine.observe(
            ObserveRequest(
                investigation_id=atomized.investigation_id,
                atom_id=child.id,
                evidence=(
                    make_evidence(
                        atom_id=child.id,
                        requirement_id=child.evidence_requirements[0].id,
                        evidence_id=f"child-evidence-{index}",
                    ),
                ),
            )
        )

    view = await engine.state(atomized.investigation_id)
    assert atomized.atom.id in view.verified_atoms
    assert set(view.verified_atoms) == {
        atomized.atom.id,
        *(child.id for child in atomized.children),
    }


async def test_revision_is_append_only(make_request, make_evidence) -> None:
    engine = EnzoEngine()
    original = await engine.atomize(make_request())
    revised_request = make_request(
        investigation_id=original.investigation_id,
        atom_id="atom-revision",
        revision_of=original.atom.id,
        question="Does the production session cookie explicitly set Secure=true?",
        predicate="explicitly sets Secure=true",
    )
    revised = await engine.atomize(revised_request)
    view = await engine.state(original.investigation_id)

    records = {item.record.atom.id: item.record for item in view.atoms}
    assert records[original.atom.id].replaced_by == revised.atom.id
    assert records[original.atom.id].result is None
    assert records[revised.atom.id].result is None
    assert revised.atom.revision_of == original.atom.id


async def test_revising_decomposed_parent_supersedes_old_children(make_request) -> None:
    requirement = EvidenceRequirement(id="child-proof", description="Inspect one flag")
    children = tuple(
        ProposedAtom(
            question=f"Does the cookie set {flag}=true?",
            subject="the cookie",
            predicate=f"sets {flag}=true",
            scope="production",
            expected_value=True,
            evidence_requirements=(requirement,),
            verification_method=VerificationMethod.SCHEMA_VALIDATION,
        )
        for flag in ("Secure", "HttpOnly")
    )
    engine = EnzoEngine()
    original = await engine.atomize(
        make_request(predicate="is hardened", proposed_children=children)
    )

    revised = await engine.atomize(
        make_request(
            investigation_id=original.investigation_id,
            atom_id="atom-revision",
            revision_of=original.atom.id,
            predicate="sets Secure=true",
        )
    )
    view = await engine.state(original.investigation_id)
    records = {item.record.atom.id: item.record for item in view.atoms}

    assert all(records[child.id].replaced_by == revised.atom.id for child in original.children)
    child_ids = {child.id for child in original.children}
    assert not child_ids & set(view.unresolved_atoms)
    assert not child_ids & set(view.current_frontier)


async def test_revised_child_still_resolves_parent_definition(
    make_request,
    make_evidence,
) -> None:
    requirement = EvidenceRequirement(
        id="child-proof",
        description="Inspect one flag",
        accepted_kinds=(EvidenceKind.SCHEMA_VALIDATION,),
        deterministic_required=True,
    )
    children = tuple(
        ProposedAtom(
            question=f"Does the cookie set {flag}=true?",
            subject="the cookie",
            predicate=f"sets {flag}=true",
            scope="production",
            expected_value=True,
            evidence_requirements=(requirement,),
            verification_method=VerificationMethod.SCHEMA_VALIDATION,
        )
        for flag in ("Secure", "HttpOnly")
    )
    engine = EnzoEngine()
    original = await engine.atomize(
        make_request(predicate="is hardened", proposed_children=children)
    )
    old_child, unchanged_child = original.children
    revised_child = await engine.atomize(
        make_request(
            investigation_id=original.investigation_id,
            atom_id="atom-child-revision",
            parent_id=original.atom.id,
            revision_of=old_child.id,
            depth=1,
            predicate="explicitly sets Secure=true",
        )
    )

    for evidence_id, atom in (
        ("revised-child-evidence", revised_child.atom),
        ("unchanged-child-evidence", unchanged_child),
    ):
        await engine.observe(
            ObserveRequest(
                investigation_id=original.investigation_id,
                atom_id=atom.id,
                evidence=(
                    make_evidence(
                        atom_id=atom.id,
                        requirement_id=atom.evidence_requirements[0].id,
                        evidence_id=evidence_id,
                    ),
                ),
            )
        )

    view = await engine.state(original.investigation_id)
    assert original.atom.id in view.verified_atoms
    assert revised_child.atom.id in view.verified_atoms
    assert old_child.id not in view.verified_atoms


async def test_revision_cannot_create_effective_replacement_cycle(make_request) -> None:
    engine = EnzoEngine()
    original = await engine.atomize(make_request())
    circular_definition = DependencyGroup(
        id="circular-definition",
        role=DependencyRole.PREREQUISITE,
        atom_ids=(original.atom.id,),
        complete=True,
    )

    with pytest.raises(
        ValidationError,
        match="dependency and replacement relationships must be acyclic",
    ):
        await engine.atomize(
            make_request(
                investigation_id=original.investigation_id,
                atom_id="atom-circular-revision",
                revision_of=original.atom.id,
                dependencies=(circular_definition,),
            )
        )


async def test_child_cannot_be_added_to_superseded_parent(make_request) -> None:
    requirement = EvidenceRequirement(id="child-proof", description="Inspect one flag")
    children = tuple(
        ProposedAtom(
            question=f"Does the cookie set {flag}=true?",
            subject="the cookie",
            predicate=f"sets {flag}=true",
            scope="production",
            expected_value=True,
            evidence_requirements=(requirement,),
            verification_method=VerificationMethod.SCHEMA_VALIDATION,
        )
        for flag in ("Secure", "HttpOnly")
    )
    engine = EnzoEngine()
    original = await engine.atomize(
        make_request(predicate="is hardened", proposed_children=children)
    )
    await engine.atomize(
        make_request(
            investigation_id=original.investigation_id,
            atom_id="atom-parent-revision",
            revision_of=original.atom.id,
            predicate="sets Secure=true",
        )
    )

    with pytest.raises(ValueError, match="cannot add a child to a superseded parent"):
        await engine.atomize(
            make_request(
                investigation_id=original.investigation_id,
                atom_id="orphan-child",
                parent_id=original.atom.id,
                depth=1,
            )
        )
