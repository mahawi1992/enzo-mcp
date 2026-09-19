from __future__ import annotations

import hashlib
import json
from collections.abc import Callable

import httpx2
import pytest
from pydantic import ValidationError
from typesafe_sdk import (
    Answer,
    AsyncTypeSafeClient,
    JSONContent,
    Noul,
    NoulAnswer,
    Questions,
    RetryPolicy,
    SystemOneResponse,
    TypeSafeError,
    Usage,
)

from enzo_mcp.engine import EnzoEngine
from enzo_mcp.models import (
    AtomizeRequest,
    ContextItem,
    ContextKind,
    EvidenceDirection,
    EvidenceKind,
    EvidenceRecord,
    EvidenceRequirement,
    JevDispatchSelection,
    ObserveRequest,
    Provenance,
    SemanticStatus,
    VerificationMethod,
)
from enzo_mcp.sensor import PydanticJevSensor, UnconfiguredJevSensor


class RecordingClient:
    def __init__(self, answer: Answer | None = None) -> None:
        self.answer = answer if answer is not None else NoulAnswer(noul=0.95)
        self.calls = 0
        self.state: JSONContent | None = None
        self.questions: Questions | None = None
        self.model: str | None = None

    async def system_one(
        self,
        state: JSONContent,
        questions: Questions,
        *,
        model: str | None = None,
        retry: RetryPolicy | None = None,
    ) -> SystemOneResponse:
        del retry
        self.calls += 1
        self.state = state
        self.questions = questions
        self.model = model
        return SystemOneResponse(
            model="jev-test",
            usage=Usage(input_tokens=1, output_tokens=1),
            answers={"claim": self.answer},
        )


class MissingAnswerClient(RecordingClient):
    async def system_one(
        self,
        state: JSONContent,
        questions: Questions,
        *,
        model: str | None = None,
        retry: RetryPolicy | None = None,
    ) -> SystemOneResponse:
        del retry
        self.calls += 1
        self.state = state
        self.questions = questions
        self.model = model
        return SystemOneResponse(
            model="jev-test",
            usage=Usage(input_tokens=1, output_tokens=1),
            answers={},
        )


class FailingClient(RecordingClient):
    async def system_one(
        self,
        state: JSONContent,
        questions: Questions,
        *,
        model: str | None = None,
        retry: RetryPolicy | None = None,
    ) -> SystemOneResponse:
        del retry
        self.calls += 1
        self.state = state
        self.questions = questions
        self.model = model
        raise TypeSafeError("simulated provider failure")


class ExplodingClient(RecordingClient):
    async def system_one(
        self,
        state: JSONContent,
        questions: Questions,
        *,
        model: str | None = None,
        retry: RetryPolicy | None = None,
    ) -> SystemOneResponse:
        del retry
        self.calls += 1
        self.state = state
        self.questions = questions
        self.model = model
        raise RuntimeError("simulated unexpected client failure")


def _semantic_requirement() -> EvidenceRequirement:
    return EvidenceRequirement(
        id="jev-observation",
        description="Jev evaluates the caller-selected state",
        accepted_kinds=(EvidenceKind.SENSOR_OUTPUT,),
    )


def _deterministic_requirement() -> EvidenceRequirement:
    return EvidenceRequirement(
        id="deterministic-observation",
        description="A deterministic instrument verifies the configuration",
        accepted_kinds=(EvidenceKind.SCHEMA_VALIDATION,),
        deterministic_required=True,
    )


async def test_dispatch_preview_and_approved_send_are_selected_only(
    make_request: Callable[..., AtomizeRequest],
) -> None:
    client = RecordingClient()
    engine = EnzoEngine(sensor=PydanticJevSensor(client=client))
    atom = (
        await engine.atomize(
            make_request(
                context=(
                    ContextItem(
                        key="approved-config",
                        value={"secure": True},
                        kind=ContextKind.ASSUMPTION,
                        provenance=Provenance(provider="caller", locator="request:config"),
                    ),
                    ContextItem(
                        key="secret-token",
                        value="never-send",
                        kind=ContextKind.ASSUMPTION,
                    ),
                ),
                evidence_requirements=(_semantic_requirement(),),
                verification_method=VerificationMethod.SEMANTIC_SENSOR,
            )
        )
    ).atom
    selected_evidence = EvidenceRecord(
        id="selected-evidence",
        atom_id=atom.id,
        direction=EvidenceDirection.NEUTRAL,
        kind=EvidenceKind.USER_PROVIDED,
        summary="Selected evidence is relevant",
        payload={"secret": "send-only-if-opted-in"},
        verification_method=VerificationMethod.HUMAN_OBSERVATION,
        provenance=Provenance(provider="caller", locator="request:selected"),
        assumptions=("caller supplied this interpretation",),
    )
    unselected_evidence = EvidenceRecord(
        id="unselected-evidence",
        atom_id=atom.id,
        direction=EvidenceDirection.NEUTRAL,
        kind=EvidenceKind.USER_PROVIDED,
        summary="UNSELECTED-EVIDENCE-CANARY",
        payload={"secret": "never-send"},
        verification_method=VerificationMethod.HUMAN_OBSERVATION,
        provenance=Provenance(provider="caller", locator="request:unselected"),
    )
    strict = await engine.observe(
        ObserveRequest(
            allow_external_jev=True,
            investigation_id=atom.investigation_id,
            atom_id=atom.id,
        )
    )
    assert strict.dispatch_manifest is None
    assert client.calls == 0
    selection = JevDispatchSelection(
        context_keys=("approved-config",), evidence_ids=("selected-evidence",)
    )
    preview = await engine.observe(
        ObserveRequest(
            allow_external_jev=True,
            investigation_id=atom.investigation_id,
            atom_id=atom.id,
            dispatch_selection=selection,
            evidence=(selected_evidence, unselected_evidence),
            approval_reference="host-ticket-42",
        )
    )

    assert client.calls == 0
    assert preview.dispatch_manifest is not None
    manifest = preview.dispatch_manifest
    assert manifest.approval_reference == "host-ticket-42"
    logical_request = manifest.logical_request.model_dump(mode="json")
    state = logical_request["state"]
    assert state["context"] == [
        {
            "key": "approved-config",
            "value": {"secure": True},
            "kind": "ASSUMPTION",
            "provenance": None,
        }
    ]
    assert "expected_value" not in state["claim"]
    assert "investigation_id" not in str(logical_request)
    assert "secret-token" not in str(logical_request)
    assert "UNSELECTED-EVIDENCE-CANARY" not in str(logical_request)
    assert state["evidence"] == [
        {
            "direction": "NEUTRAL",
            "kind": "USER_PROVIDED",
            "summary": "Selected evidence is relevant",
            "payload": None,
            "deterministic": False,
            "verification_method": "HUMAN_OBSERVATION",
            "provenance": None,
            "assumptions": [],
        }
    ]

    fully_opted_in = await engine.observe(
        ObserveRequest(
            investigation_id=atom.investigation_id,
            atom_id=atom.id,
            evidence=(selected_evidence, unselected_evidence),
            dispatch_selection=JevDispatchSelection(
                context_keys=("approved-config",),
                evidence_ids=("selected-evidence",),
                include_context_provenance=True,
                include_evidence_payloads=True,
                include_assumptions=True,
                include_evidence_provenance=True,
            ),
        )
    )
    assert fully_opted_in.dispatch_manifest is not None
    opted_state = fully_opted_in.dispatch_manifest.logical_request.state.model_dump(mode="json")
    assert opted_state["context"][0]["provenance"]["locator"] == "request:config"
    assert opted_state["evidence"][0]["payload"] == {"secret": "send-only-if-opted-in"}
    assert opted_state["evidence"][0]["assumptions"] == ["caller supplied this interpretation"]
    assert opted_state["evidence"][0]["provenance"]["locator"] == "request:selected"

    approved = await engine.observe(
        ObserveRequest(
            allow_external_jev=True,
            investigation_id=atom.investigation_id,
            atom_id=atom.id,
            dispatch_selection=selection,
            evidence=(selected_evidence, unselected_evidence),
            approved_dispatch_sha256=manifest.canonical_sha256,
            approval_reference="host-ticket-42",
        )
    )

    assert approved.status is SemanticStatus.VERIFIED
    assert client.calls == 1
    assert client.model == "jev-latest"
    assert approved.dispatch_manifest is not None
    approved_logical = approved.dispatch_manifest.logical_request
    assert client.model == approved_logical.model
    assert client.state == approved_logical.state.model_dump(mode="json")
    assert client.questions is not None
    captured_question = client.questions["claim"]
    assert isinstance(captured_question, Noul)
    assert (
        captured_question.model_dump(mode="json")
        == (approved_logical.questions.model_dump(mode="json")["claim"])
    )
    canonical = json.dumps(
        approved_logical.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    assert approved.dispatch_manifest.byte_length == len(canonical)
    assert approved.dispatch_manifest.canonical_sha256 == hashlib.sha256(canonical).hexdigest()

    replay = await engine.observe(
        ObserveRequest(
            allow_external_jev=True,
            investigation_id=atom.investigation_id,
            atom_id=atom.id,
            dispatch_selection=selection,
            evidence=(selected_evidence, unselected_evidence),
            approved_dispatch_sha256=manifest.canonical_sha256,
            approval_reference="host-ticket-42",
        )
    )
    assert replay == approved
    assert client.calls == 1


async def test_unusable_provider_results_are_idempotent_until_explicit_retry(
    make_request: Callable[..., AtomizeRequest],
) -> None:
    clients = (
        MissingAnswerClient(),
        RecordingClient(NoulAnswer(noul=2.0)),
        FailingClient(),
    )
    for index, client in enumerate(clients):
        engine = EnzoEngine(sensor=PydanticJevSensor(client=client))
        atom = (
            await engine.atomize(
                make_request(
                    atom_id=f"provider-failure-{index}",
                    context=(
                        ContextItem(
                            key="selected",
                            value=True,
                            kind=ContextKind.ASSUMPTION,
                        ),
                    ),
                    evidence_requirements=(_semantic_requirement(),),
                    verification_method=VerificationMethod.SEMANTIC_SENSOR,
                )
            )
        ).atom
        selection = JevDispatchSelection(context_keys=("selected",))
        preview = await engine.observe(
            ObserveRequest(
                investigation_id=atom.investigation_id,
                atom_id=atom.id,
                dispatch_selection=selection,
            )
        )
        assert preview.dispatch_manifest is not None
        approved_request = ObserveRequest(
            allow_external_jev=True,
            investigation_id=atom.investigation_id,
            atom_id=atom.id,
            dispatch_selection=selection,
            approved_dispatch_sha256=preview.dispatch_manifest.canonical_sha256,
        )

        first = await engine.observe(approved_request)
        retry = await engine.observe(approved_request)

        assert first.status is SemanticStatus.UNKNOWN
        assert retry is first
        assert client.calls == 1

        await engine.observe(
            approved_request.model_copy(update={"approval_reference": f"explicit-retry-{index}"})
        )
        assert client.calls == 2


async def test_real_typesafe_client_makes_one_transport_attempt_per_approval(
    make_request: Callable[..., AtomizeRequest],
) -> None:
    attempts = 0

    async def fail_transport(request: httpx2.Request) -> httpx2.Response:
        nonlocal attempts
        attempts += 1
        raise httpx2.ReadTimeout("simulated ambiguous read timeout", request=request)

    client = AsyncTypeSafeClient(
        api_key="test-key",
        base_url="https://typesafe.invalid",
        retry=RetryPolicy(max_retries=2, backoff_initial=0, backoff_max=0),
        transport=httpx2.MockTransport(fail_transport),
    )
    try:
        engine = EnzoEngine(sensor=PydanticJevSensor(client=client))
        atom = (
            await engine.atomize(
                make_request(
                    atom_id="transport-retry-atom",
                    context=(
                        ContextItem(
                            key="selected",
                            value=True,
                            kind=ContextKind.ASSUMPTION,
                        ),
                    ),
                    evidence_requirements=(_semantic_requirement(),),
                    verification_method=VerificationMethod.SEMANTIC_SENSOR,
                )
            )
        ).atom
        selection = JevDispatchSelection(context_keys=("selected",))
        preview = await engine.observe(
            ObserveRequest(
                investigation_id=atom.investigation_id,
                atom_id=atom.id,
                dispatch_selection=selection,
            )
        )
        assert preview.dispatch_manifest is not None
        approved_request = ObserveRequest(
            allow_external_jev=True,
            investigation_id=atom.investigation_id,
            atom_id=atom.id,
            dispatch_selection=selection,
            approved_dispatch_sha256=preview.dispatch_manifest.canonical_sha256,
        )

        first = await engine.observe(approved_request)
        retry = await engine.observe(approved_request)

        assert first.status is SemanticStatus.UNKNOWN
        assert retry is first
        assert attempts == 1
    finally:
        await client.aclose()


async def test_approval_is_reserved_before_an_unexpected_client_exception(
    make_request: Callable[..., AtomizeRequest],
) -> None:
    client = ExplodingClient()
    engine = EnzoEngine(sensor=PydanticJevSensor(client=client))
    atom = (
        await engine.atomize(
            make_request(
                atom_id="unexpected-provider-failure",
                context=(
                    ContextItem(
                        key="selected",
                        value=True,
                        kind=ContextKind.ASSUMPTION,
                    ),
                ),
                evidence_requirements=(_semantic_requirement(),),
                verification_method=VerificationMethod.SEMANTIC_SENSOR,
            )
        )
    ).atom
    selection = JevDispatchSelection(context_keys=("selected",))
    preview = await engine.observe(
        ObserveRequest(
            investigation_id=atom.investigation_id,
            atom_id=atom.id,
            dispatch_selection=selection,
        )
    )
    assert preview.dispatch_manifest is not None
    approved_request = ObserveRequest(
        allow_external_jev=True,
        investigation_id=atom.investigation_id,
        atom_id=atom.id,
        dispatch_selection=selection,
        approved_dispatch_sha256=preview.dispatch_manifest.canonical_sha256,
    )

    with pytest.raises(RuntimeError, match="unexpected client failure"):
        await engine.observe(approved_request)
    retry = await engine.observe(approved_request)

    assert retry.status is SemanticStatus.UNKNOWN
    assert "already used" in retry.missing_information[0]
    assert client.calls == 1


async def test_mismatched_or_mutated_selected_data_never_sends(
    make_request: Callable[..., AtomizeRequest],
) -> None:
    client = RecordingClient()
    engine = EnzoEngine(sensor=PydanticJevSensor(client=client))
    atom = (
        await engine.atomize(
            make_request(
                context=(
                    ContextItem(
                        key="selected",
                        value="first",
                        kind=ContextKind.FACT,
                        provenance=Provenance(provider="caller", locator="request:selected"),
                    ),
                ),
                evidence_requirements=(_deterministic_requirement(),),
                verification_method=VerificationMethod.SEMANTIC_SENSOR,
            )
        )
    ).atom
    selection = JevDispatchSelection(context_keys=("selected",))
    preview = await engine.observe(
        ObserveRequest(
            investigation_id=atom.investigation_id,
            atom_id=atom.id,
            dispatch_selection=selection,
        )
    )
    assert preview.dispatch_manifest is not None

    mismatch = await engine.observe(
        ObserveRequest(
            allow_external_jev=True,
            investigation_id=atom.investigation_id,
            atom_id=atom.id,
            dispatch_selection=selection,
            approved_dispatch_sha256="0" * 64,
        )
    )
    assert mismatch.dispatch_manifest == preview.dispatch_manifest
    assert client.calls == 0

    mutated_selection = JevDispatchSelection(
        context_keys=("selected",), include_context_provenance=True
    )
    mutated = await engine.observe(
        ObserveRequest(
            allow_external_jev=True,
            investigation_id=atom.investigation_id,
            atom_id=atom.id,
            dispatch_selection=mutated_selection,
            approved_dispatch_sha256=preview.dispatch_manifest.canonical_sha256,
        )
    )
    assert mutated.dispatch_manifest is not None
    assert mutated.dispatch_manifest.canonical_sha256 != preview.dispatch_manifest.canonical_sha256
    assert client.calls == 0


async def test_limits_and_sensor_feedback_fail_closed(
    make_request: Callable[..., AtomizeRequest],
    make_evidence: Callable[..., EvidenceRecord],
) -> None:
    client = RecordingClient()
    engine = EnzoEngine(sensor=PydanticJevSensor(client=client))
    atom = (
        await engine.atomize(
            make_request(
                context=tuple(
                    ContextItem(key=f"entry-{index}", value="x" * 4000, kind=ContextKind.ASSUMPTION)
                    for index in range(5)
                ),
                evidence_requirements=(_deterministic_requirement(),),
                verification_method=VerificationMethod.SEMANTIC_SENSOR,
            )
        )
    ).atom
    oversized = await engine.observe(
        ObserveRequest(
            allow_external_jev=True,
            investigation_id=atom.investigation_id,
            atom_id=atom.id,
            dispatch_selection=JevDispatchSelection(
                context_keys=tuple(f"entry-{index}" for index in range(5))
            ),
        )
    )
    assert oversized.dispatch_manifest is None
    assert "16KiB" in oversized.missing_information[0]
    assert client.calls == 0

    with pytest.raises(ValidationError, match="at most 8 items"):
        JevDispatchSelection(context_keys=tuple(f"key-{index}" for index in range(9)))

    deterministic = make_evidence(
        atom_id=atom.id,
        requirement_id=atom.evidence_requirements[0].id,
    )
    bypass = await engine.observe(
        ObserveRequest(
            allow_external_jev=True,
            investigation_id=atom.investigation_id,
            atom_id=atom.id,
            evidence=(deterministic,),
            dispatch_selection=JevDispatchSelection(context_keys=("entry-0",)),
            approved_dispatch_sha256="0" * 64,
        )
    )
    assert bypass.status is SemanticStatus.VERIFIED
    assert client.calls == 0

    per_value_atom = (
        await engine.atomize(
            make_request(
                atom_id="oversized-selected-value",
                context=(ContextItem(key="large", value="x" * 4097, kind=ContextKind.ASSUMPTION),),
                evidence_requirements=(_semantic_requirement(),),
                verification_method=VerificationMethod.SEMANTIC_SENSOR,
            )
        )
    ).atom
    per_value = await engine.observe(
        ObserveRequest(
            investigation_id=per_value_atom.investigation_id,
            atom_id=per_value_atom.id,
            dispatch_selection=JevDispatchSelection(context_keys=("large",)),
        )
    )
    assert per_value.dispatch_manifest is None
    assert "4KiB" in per_value.missing_information[0]
    assert client.calls == 0


async def test_selected_sensor_output_is_rejected_without_feedback(
    make_request: Callable[..., AtomizeRequest],
) -> None:
    client = RecordingClient()
    engine = EnzoEngine(sensor=PydanticJevSensor(client=client))
    atom = (
        await engine.atomize(
            make_request(
                context=(ContextItem(key="selected", value=True, kind=ContextKind.ASSUMPTION),),
                evidence_requirements=(_semantic_requirement(),),
                verification_method=VerificationMethod.SEMANTIC_SENSOR,
            )
        )
    ).atom
    selection = JevDispatchSelection(context_keys=("selected",))
    preview = await engine.observe(
        ObserveRequest(
            investigation_id=atom.investigation_id,
            atom_id=atom.id,
            dispatch_selection=selection,
        )
    )
    assert preview.dispatch_manifest is not None
    sent = await engine.observe(
        ObserveRequest(
            allow_external_jev=True,
            investigation_id=atom.investigation_id,
            atom_id=atom.id,
            dispatch_selection=selection,
            approved_dispatch_sha256=preview.dispatch_manifest.canonical_sha256,
        )
    )
    sensor_evidence_id = next(
        item.id for item in sent.evidence if item.kind is EvidenceKind.SENSOR_OUTPUT
    )
    blocked = await engine.observe(
        ObserveRequest(
            allow_external_jev=True,
            investigation_id=atom.investigation_id,
            atom_id=atom.id,
            dispatch_selection=JevDispatchSelection(evidence_ids=(sensor_evidence_id,)),
        )
    )
    assert blocked.dispatch_manifest is None
    assert "SENSOR_OUTPUT" in blocked.missing_information[0]
    assert client.calls == 1


async def test_unconfigured_sensor_never_dispatches(
    make_request: Callable[..., AtomizeRequest],
) -> None:
    engine = EnzoEngine(sensor=UnconfiguredJevSensor())
    atom = (
        await engine.atomize(
            make_request(
                context=(ContextItem(key="selected", value=True, kind=ContextKind.ASSUMPTION),),
                evidence_requirements=(_semantic_requirement(),),
                verification_method=VerificationMethod.SEMANTIC_SENSOR,
            )
        )
    ).atom
    selection = JevDispatchSelection(context_keys=("selected",))
    preview = await engine.observe(
        ObserveRequest(
            investigation_id=atom.investigation_id,
            atom_id=atom.id,
            dispatch_selection=selection,
        )
    )
    assert preview.dispatch_manifest is not None
    blocked = await engine.observe(
        ObserveRequest(
            allow_external_jev=True,
            investigation_id=atom.investigation_id,
            atom_id=atom.id,
            dispatch_selection=selection,
            approved_dispatch_sha256=preview.dispatch_manifest.canonical_sha256,
        )
    )
    assert "not configured" in blocked.missing_information[0]
