from __future__ import annotations

import pytest
from pydantic import ValidationError

from enzo_mcp.engine import EnzoEngine
from enzo_mcp.models import (
    AtomicityDecision,
    AtomizationResult,
    DependencyLogic,
    EvidenceKind,
    EvidenceRequirement,
    ProposedAtom,
    VerificationMethod,
)


async def test_operationalized_single_predicate_is_atomic(make_request) -> None:
    result = await EnzoEngine().atomize(make_request())

    assert result.decision is AtomicityDecision.ATOMIC
    assert result.children == ()
    assert result.validation_basis == "STRUCTURAL"


async def test_broad_quality_requires_refinement(make_request) -> None:
    result = await EnzoEngine().atomize(
        make_request(
            question="Is authentication secure?",
            subject="authentication",
            predicate="is secure",
            expected_value=True,
        )
    )

    assert result.decision is AtomicityDecision.NEEDS_REFINEMENT
    assert result.children == ()
    assert result.refinement_questions


async def test_explicit_predicates_are_safely_decomposed(make_request) -> None:
    result = await EnzoEngine().atomize(
        make_request(
            question="Does the cookie set Secure and HttpOnly?",
            predicate="sets Secure=true; sets HttpOnly=true",
        )
    )

    assert result.decision is AtomicityDecision.DECOMPOSE
    assert len(result.children) == 2
    assert {child.parent_id for child in result.children} == {result.atom.id}
    assert {child.depth for child in result.children} == {1}


async def test_ambiguous_and_is_not_mechanically_split(make_request) -> None:
    result = await EnzoEngine().atomize(
        make_request(
            question="Does the policy grant read and write permission as one role?",
            subject="the editor role",
            predicate="grants read and write permission as one named role",
        )
    )

    assert result.decision is AtomicityDecision.NEEDS_REFINEMENT
    assert result.children == ()


async def test_between_range_is_one_predicate(make_request) -> None:
    result = await EnzoEngine().atomize(
        make_request(
            question="Is latency between 5 and 10 milliseconds?",
            subject="request latency",
            predicate="is between 5 and 10 milliseconds",
            operational_definition="A measured latency in the closed interval [5, 10] ms",
        )
    )

    assert result.decision is AtomicityDecision.ATOMIC


async def test_enum_alternatives_do_not_create_false_compound(make_request) -> None:
    result = await EnzoEngine().atomize(
        make_request(
            question="Does Jev retain the context item's epistemic label?",
            subject="the state sent to Jev",
            predicate="preserves the context item's FACT or ASSUMPTION label",
            operational_definition=(
                "Inspect one serialized context item and compare its kind with the input kind"
            ),
        )
    )

    assert result.decision is AtomicityDecision.ATOMIC
    assert result.children == ()


async def test_uppercase_disjunctive_claim_remains_ambiguous(make_request) -> None:
    result = await EnzoEngine().atomize(
        make_request(
            question="Does either service accept production traffic?",
            subject="the production router",
            predicate="SERVICE_A or SERVICE_B accepts production traffic",
            operational_definition="Inspect routing behavior for both named services",
        )
    )

    assert result.decision is AtomicityDecision.NEEDS_REFINEMENT
    assert result.children == ()


async def test_comma_formatted_between_range_is_one_predicate(make_request) -> None:
    result = await EnzoEngine().atomize(
        make_request(
            question="Is latency between 1,000 and 2,000 milliseconds?",
            subject="request latency",
            predicate="is between 1,000 and 2,000 milliseconds",
            operational_definition="A measured latency in the closed interval [1000, 2000] ms",
        )
    )

    assert result.decision is AtomicityDecision.ATOMIC
    assert result.children == ()


async def test_range_does_not_hide_separate_semicolon_claim(make_request) -> None:
    result = await EnzoEngine().atomize(
        make_request(
            question="Is latency in range and is the feature enabled?",
            subject="the service",
            predicate="is between 1 and 2 milliseconds; sets enabled=true",
            operational_definition="Measure latency and inspect the enabled flag",
        )
    )

    assert result.decision is AtomicityDecision.DECOMPOSE
    assert len(result.children) == 2


async def test_range_does_not_hide_trailing_conjunctive_claim(make_request) -> None:
    result = await EnzoEngine().atomize(
        make_request(
            question="Is latency in range and is telemetry enabled?",
            subject="the service",
            predicate="is between 1 and 2 milliseconds and is telemetry enabled",
            operational_definition="Measure latency and inspect telemetry configuration",
        )
    )

    assert result.decision is AtomicityDecision.NEEDS_REFINEMENT
    assert result.children == ()


async def test_generated_children_must_be_operationalized(make_request) -> None:
    result = await EnzoEngine().atomize(
        make_request(
            question="Is the service secure and enabled?",
            subject="the service",
            predicate="is secure; is enabled",
            operational_definition="Assess the two named qualities",
        )
    )

    assert result.decision is AtomicityDecision.NEEDS_REFINEMENT
    assert result.children == ()
    assert result.reasons[0].code == "generated_child_not_atomic"


async def test_automatic_decomposition_rejects_duplicate_children(make_request) -> None:
    result = await EnzoEngine().atomize(
        make_request(predicate="sets Secure=true; sets Secure=true")
    )

    assert result.decision is AtomicityDecision.NEEDS_REFINEMENT
    assert result.reasons[0].code == "duplicate_generated_children"


async def test_atomization_result_rejects_inconsistent_next_flag(make_request) -> None:
    result = await EnzoEngine().atomize(make_request())
    payload = result.model_dump(mode="python")
    payload["next_atom_needed"] = True

    with pytest.raises(ValidationError, match="next_atom_needed"):
        AtomizationResult.model_validate(payload)


async def test_disjunction_decomposition_uses_any_of(make_request) -> None:
    result = await EnzoEngine().atomize(
        make_request(
            question="Does either cookie flag equal true?",
            predicate="sets Secure=true || sets HttpOnly=true",
        )
    )

    assert result.decision is AtomicityDecision.DECOMPOSE
    definition = result.atom.dependencies[-1]
    assert definition.role.value == "DEFINES_PARENT"
    assert definition.logic is DependencyLogic.ANY_OF


async def test_compound_proposed_child_is_not_admitted_as_atomic(make_request) -> None:
    requirement = EvidenceRequirement(id="child-proof", description="Inspect both flags")
    compound = ProposedAtom(
        question="Does the cookie set both flags?",
        subject="the cookie",
        predicate="sets Secure=true; sets HttpOnly=true",
        scope="production",
        expected_value=True,
        evidence_requirements=(requirement,),
        verification_method=VerificationMethod.SCHEMA_VALIDATION,
    )
    precise = ProposedAtom(
        question="Does the cookie set SameSite=Strict?",
        subject="the cookie",
        predicate="sets SameSite=Strict",
        scope="production",
        expected_value=True,
        evidence_requirements=(requirement,),
        verification_method=VerificationMethod.SCHEMA_VALIDATION,
    )

    result = await EnzoEngine().atomize(
        make_request(
            predicate="is hardened",
            proposed_children=(compound, precise),
        )
    )

    assert result.decision is AtomicityDecision.NEEDS_REFINEMENT
    assert result.children == ()


async def test_duplicate_proposed_children_require_refinement(make_request) -> None:
    requirement = EvidenceRequirement(id="child-proof", description="Inspect the flag")
    first = ProposedAtom(
        question="Does the cookie set Secure=true?",
        subject="the cookie",
        predicate="sets Secure=true",
        scope="production",
        expected_value=True,
        evidence_requirements=(requirement,),
        verification_method=VerificationMethod.SCHEMA_VALIDATION,
    )
    duplicate = ProposedAtom(
        question="Is Secure enabled on the cookie?",
        subject="  THE cookie ",
        predicate="sets   secure=TRUE",
        scope="Production",
        expected_value=True,
        evidence_requirements=(requirement,),
        verification_method=VerificationMethod.SCHEMA_VALIDATION,
    )

    result = await EnzoEngine().atomize(
        make_request(predicate="is hardened", proposed_children=(first, duplicate))
    )

    assert result.decision is AtomicityDecision.NEEDS_REFINEMENT
    assert result.reasons[0].code == "duplicate_proposed_children"


async def test_automatic_split_refuses_ambiguous_evidence_allocation(make_request) -> None:
    requirements = (
        EvidenceRequirement(
            id="secure-proof",
            description="Inspect Secure",
            accepted_kinds=(EvidenceKind.SCHEMA_VALIDATION,),
        ),
        EvidenceRequirement(
            id="http-only-proof",
            description="Inspect HttpOnly",
            accepted_kinds=(EvidenceKind.SCHEMA_VALIDATION,),
        ),
    )

    result = await EnzoEngine().atomize(
        make_request(
            predicate="sets Secure=true; sets HttpOnly=true",
            evidence_requirements=requirements,
        )
    )

    assert result.decision is AtomicityDecision.NEEDS_REFINEMENT
    assert result.children == ()
    assert result.reasons[0].code == "ambiguous_evidence_allocation"
