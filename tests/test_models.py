from __future__ import annotations

import pytest
from pydantic import ValidationError

from enzo_mcp.models import (
    AtomizeRequest,
    ContextItem,
    ContextKind,
    DependencyGroup,
    DependencyRole,
    JEVResult,
    ParentImpact,
    Provenance,
    ResultSource,
    SemanticStatus,
    VerificationMethod,
)


def test_contracts_forbid_unknown_fields(make_request) -> None:
    payload = make_request().model_dump(mode="python")
    payload["invented"] = True

    with pytest.raises(ValidationError, match="invented"):
        AtomizeRequest.model_validate(payload)

    assert AtomizeRequest.model_json_schema()["additionalProperties"] is False


def test_contracts_do_not_coerce_semantic_types(make_request) -> None:
    payload = make_request().model_dump(mode="python")
    payload["depth"] = "0"

    with pytest.raises(ValidationError, match="depth"):
        AtomizeRequest.model_validate(payload)


def test_facts_require_provenance() -> None:
    with pytest.raises(ValidationError, match="FACT context requires provenance"):
        ContextItem(key="cookie.secure", value=True, kind=ContextKind.FACT)


def test_verified_result_cannot_exist_without_evidence() -> None:
    with pytest.raises(ValidationError, match="VERIFIED requires supporting evidence"):
        JEVResult(
            atom_id="atom-1",
            status=SemanticStatus.VERIFIED,
            evidence=(),
            provenance=(Provenance(provider="test", locator="unit-test"),),
            verification_method=VerificationMethod.TEST,
            parent_impact=ParentImpact(summary="No parent"),
            source=ResultSource.DETERMINISTIC,
        )


def test_non_root_requires_parent(make_request) -> None:
    with pytest.raises(ValidationError, match="non-root atoms require parent_id"):
        make_request(depth=1)


def test_boolean_claim_requires_boolean_expected_value(make_request) -> None:
    with pytest.raises(ValidationError, match="BOOLEAN atoms require a boolean expected_value"):
        make_request(expected_value="true")


def test_callers_cannot_inject_defining_dependencies(make_request) -> None:
    definition = DependencyGroup(
        id="forged-definition",
        role=DependencyRole.DEFINES_PARENT,
        atom_ids=("unrelated-atom",),
        complete=True,
    )

    with pytest.raises(ValidationError, match="DEFINES_PARENT dependencies are engine-managed"):
        make_request(dependencies=(definition,))
