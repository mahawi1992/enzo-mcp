from __future__ import annotations

from typing import cast

from typesafe_sdk import (
    Answer,
    ChoiceAnswer,
    JSONContent,
    Noul,
    NoulAnswer,
    Questions,
    ScoreAnswer,
    SystemOneResponse,
    Usage,
)

from enzo_mcp.engine import EnzoEngine
from enzo_mcp.models import (
    ContextItem,
    ContextKind,
    DependencyGroup,
    DependencyRole,
    EvidenceDirection,
    EvidenceKind,
    EvidenceRecord,
    EvidenceRequirement,
    ExpectedAnswerType,
    ObserveRequest,
    PredicateOperator,
    Provenance,
    SemanticStatus,
    VerificationMethod,
)
from enzo_mcp.sensor import PydanticJevSensor


class FakeSystemOneClient:
    def __init__(self, answer: Answer) -> None:
        self.answer = answer
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
    ) -> SystemOneResponse:
        self.calls += 1
        self.state = state
        self.questions = questions
        self.model = model
        return SystemOneResponse(
            model="jev-test",
            usage=Usage(input_tokens=12, output_tokens=1),
            answers={"claim": self.answer},
        )


async def test_pydantic_jev_verifies_supported_boolean_atom(make_request) -> None:
    requirement = EvidenceRequirement(
        id="jev-observation",
        description="Jev evaluates the supplied configuration context",
        accepted_kinds=(EvidenceKind.SENSOR_OUTPUT,),
    )
    client = FakeSystemOneClient(NoulAnswer(noul=0.93))
    engine = EnzoEngine(sensor=PydanticJevSensor(client=client))
    atomized = await engine.atomize(
        make_request(
            context=(
                ContextItem(
                    key="production-cookie-config",
                    value={"secure": True},
                    kind=ContextKind.ASSUMPTION,
                ),
            ),
            evidence_requirements=(requirement,),
            verification_method=VerificationMethod.SEMANTIC_SENSOR,
        )
    )

    result = await engine.observe(
        ObserveRequest(
            allow_external_jev=True,
            investigation_id=atomized.investigation_id,
            atom_id=atomized.atom.id,
        )
    )

    assert result.status is SemanticStatus.VERIFIED
    assert result.evidence[0].kind is EvidenceKind.SENSOR_OUTPUT
    assert client.model == "jev-latest"
    assert isinstance(cast(dict[str, object], client.questions)["claim"], Noul)
    state = cast(dict[str, object], client.state)
    assert "question" not in state
    claim = cast(dict[str, object], state["claim"])
    assert claim["question"] == "Does the production session cookie set Secure=true?"
    assert claim["subject"] == "the production session cookie"
    assert claim["predicate"] == "sets Secure=true"
    assert claim["scope"] == "production session configuration"
    assert claim["operator"] == "EQUALS"
    assert claim["expected_value"] is True
    assert claim["expected_answer_type"] == "BOOLEAN"
    requirements = cast(list[dict[str, object]], claim["evidence_requirements"])
    assert requirements[0]["id"] == "jev-observation"
    assert state["scope"] == "production session configuration"
    context = cast(list[dict[str, object]], state["context"])
    assert context[0]["kind"] == "ASSUMPTION"


async def test_pydantic_jev_sends_complete_atomic_claim_contract(make_request) -> None:
    requirement = EvidenceRequirement(
        id="semantic-choice",
        description="Jev selects the deployment state from supplied evidence",
        accepted_kinds=(EvidenceKind.SENSOR_OUTPUT,),
    )
    answer = ChoiceAnswer(
        choice="enabled",
        confidence=0.92,
        probabilities={"enabled": 0.92, "disabled": 0.08},
    )
    client = FakeSystemOneClient(answer)
    engine = EnzoEngine(sensor=PydanticJevSensor(client=client))
    atomized = await engine.atomize(
        make_request(
            question="Which deployment state is supported?",
            subject="the deployment",
            predicate="has the selected state",
            scope="the supplied deployment record",
            operator=PredicateOperator.EQUALS,
            expected_answer_type=ExpectedAnswerType.CHOICE,
            answer_options=("enabled", "disabled"),
            expected_value="enabled",
            operational_definition="Select the state explicitly named in the record.",
            context=(
                ContextItem(
                    key="deployment-state",
                    value="enabled",
                    kind=ContextKind.ASSUMPTION,
                ),
            ),
            evidence_requirements=(requirement,),
            verification_method=VerificationMethod.SEMANTIC_SENSOR,
        )
    )

    await engine.observe(
        ObserveRequest(
            allow_external_jev=True,
            investigation_id=atomized.investigation_id,
            atom_id=atomized.atom.id,
        )
    )

    state = cast(dict[str, object], client.state)
    claim = cast(dict[str, object], state["claim"])
    assert claim == {
        "question": "Which deployment state is supported?",
        "subject": "the deployment",
        "predicate": "has the selected state",
        "scope": "the supplied deployment record",
        "operator": "EQUALS",
        "expected_value": "enabled",
        "quantifier": "ONE",
        "expected_answer_type": "CHOICE",
        "answer_options": ["enabled", "disabled"],
        "score_criteria": [],
        "operational_definition": "Select the state explicitly named in the record.",
        "verification_method": "SEMANTIC_SENSOR",
        "evidence_requirements": [
            {
                "id": "semantic-choice",
                "description": "Jev selects the deployment state from supplied evidence",
                "accepted_kinds": ["SENSOR_OUTPUT"],
                "minimum_items": 1,
                "required": True,
                "deterministic_required": False,
            }
        ],
    }


async def test_pydantic_jev_keeps_uncertain_boolean_unresolved(make_request) -> None:
    requirement = EvidenceRequirement(
        id="jev-observation",
        description="Jev evaluates the supplied configuration context",
        accepted_kinds=(EvidenceKind.SENSOR_OUTPUT,),
    )
    engine = EnzoEngine(sensor=PydanticJevSensor(client=FakeSystemOneClient(NoulAnswer(noul=0.55))))
    atomized = await engine.atomize(
        make_request(
            context=(
                ContextItem(
                    key="production-cookie-config",
                    value="Secure may be inherited",
                    kind=ContextKind.ASSUMPTION,
                ),
            ),
            evidence_requirements=(requirement,),
            verification_method=VerificationMethod.SEMANTIC_SENSOR,
        )
    )

    result = await engine.observe(
        ObserveRequest(
            allow_external_jev=True,
            investigation_id=atomized.investigation_id,
            atom_id=atomized.atom.id,
        )
    )

    assert result.status is SemanticStatus.INSUFFICIENT_EVIDENCE
    assert "did not cross a decision threshold" in result.missing_information[0]


async def test_pydantic_jev_honors_false_boolean_expectation(make_request) -> None:
    requirement = EvidenceRequirement(
        id="jev-observation",
        description="Jev evaluates the supplied configuration context",
        accepted_kinds=(EvidenceKind.SENSOR_OUTPUT,),
    )
    engine = EnzoEngine(sensor=PydanticJevSensor(client=FakeSystemOneClient(NoulAnswer(noul=0.95))))
    atomized = await engine.atomize(
        make_request(
            expected_value=False,
            context=(
                ContextItem(
                    key="production-cookie-config",
                    value={"secure": True},
                    kind=ContextKind.ASSUMPTION,
                ),
            ),
            evidence_requirements=(requirement,),
            verification_method=VerificationMethod.SEMANTIC_SENSOR,
        )
    )

    result = await engine.observe(
        ObserveRequest(
            allow_external_jev=True,
            investigation_id=atomized.investigation_id,
            atom_id=atomized.atom.id,
        )
    )

    assert result.status is SemanticStatus.CONTRADICTED


async def test_pydantic_jev_maps_high_confidence_choice_mismatch(make_request) -> None:
    requirement = EvidenceRequirement(
        id="jev-observation",
        description="Jev selects the observed deployment state",
        accepted_kinds=(EvidenceKind.SENSOR_OUTPUT,),
    )
    answer = ChoiceAnswer(
        choice="disabled",
        confidence=0.92,
        probabilities={"enabled": 0.08, "disabled": 0.92},
    )
    engine = EnzoEngine(sensor=PydanticJevSensor(client=FakeSystemOneClient(answer)))
    atomized = await engine.atomize(
        make_request(
            question="Which deployment state is shown?",
            expected_answer_type=ExpectedAnswerType.CHOICE,
            answer_options=("enabled", "disabled"),
            expected_value="enabled",
            context=(
                ContextItem(
                    key="deployment-state",
                    value="disabled",
                    kind=ContextKind.ASSUMPTION,
                ),
            ),
            evidence_requirements=(requirement,),
            verification_method=VerificationMethod.SEMANTIC_SENSOR,
        )
    )

    result = await engine.observe(
        ObserveRequest(
            allow_external_jev=True,
            investigation_id=atomized.investigation_id,
            atom_id=atomized.atom.id,
        )
    )

    assert result.status is SemanticStatus.CONTRADICTED


async def test_pydantic_jev_maps_high_confidence_score_match(make_request) -> None:
    requirement = EvidenceRequirement(
        id="jev-observation",
        description="Jev scores the supplied rubric",
        accepted_kinds=(EvidenceKind.SENSOR_OUTPUT,),
    )
    answer = ScoreAnswer(
        score=1.85,
        confidence=0.9,
        legend={0: "absent", 1: "partial", 2: "complete"},
        probabilities={0: 0.02, 1: 0.11, 2: 0.87},
    )
    engine = EnzoEngine(sensor=PydanticJevSensor(client=FakeSystemOneClient(answer)))
    atomized = await engine.atomize(
        make_request(
            question="How completely is the control implemented?",
            operator=PredicateOperator.ON_SCALE,
            expected_answer_type=ExpectedAnswerType.SCORE,
            score_criteria=("absent", "partial", "complete"),
            expected_value=2,
            context=(
                ContextItem(
                    key="control-description",
                    value="The control is implemented end to end",
                    kind=ContextKind.ASSUMPTION,
                ),
            ),
            evidence_requirements=(requirement,),
            verification_method=VerificationMethod.SEMANTIC_SENSOR,
        )
    )

    result = await engine.observe(
        ObserveRequest(
            allow_external_jev=True,
            investigation_id=atomized.investigation_id,
            atom_id=atomized.atom.id,
        )
    )

    assert result.status is SemanticStatus.VERIFIED


async def test_pydantic_jev_rejects_choice_outside_requested_options(make_request) -> None:
    requirement = EvidenceRequirement(
        id="jev-observation",
        description="Jev selects the observed deployment state",
        accepted_kinds=(EvidenceKind.SENSOR_OUTPUT,),
    )
    answer = ChoiceAnswer(
        choice="unknown-provider-option",
        confidence=0.99,
        probabilities={"unknown-provider-option": 0.99, "enabled": 0.01},
    )
    engine = EnzoEngine(sensor=PydanticJevSensor(client=FakeSystemOneClient(answer)))
    atomized = await engine.atomize(
        make_request(
            question="Which deployment state is shown?",
            expected_answer_type=ExpectedAnswerType.CHOICE,
            answer_options=("enabled", "disabled"),
            expected_value="enabled",
            context=(
                ContextItem(
                    key="deployment-state",
                    value="enabled",
                    kind=ContextKind.ASSUMPTION,
                ),
            ),
            evidence_requirements=(requirement,),
            verification_method=VerificationMethod.SEMANTIC_SENSOR,
        )
    )

    result = await engine.observe(
        ObserveRequest(
            allow_external_jev=True,
            investigation_id=atomized.investigation_id,
            atom_id=atomized.atom.id,
        )
    )

    assert result.status is SemanticStatus.UNKNOWN
    assert "outside the requested answer options" in result.missing_information[0]


async def test_pydantic_jev_rejects_score_outside_requested_rubric(make_request) -> None:
    requirement = EvidenceRequirement(
        id="jev-observation",
        description="Jev scores the supplied rubric",
        accepted_kinds=(EvidenceKind.SENSOR_OUTPUT,),
    )
    answer = ScoreAnswer(
        score=3.0,
        confidence=0.99,
        legend={0: "absent", 1: "partial", 2: "complete", 3: "unexpected"},
        probabilities={0: 0.0, 1: 0.0, 2: 0.01, 3: 0.99},
    )
    engine = EnzoEngine(sensor=PydanticJevSensor(client=FakeSystemOneClient(answer)))
    atomized = await engine.atomize(
        make_request(
            question="How completely is the control implemented?",
            operator=PredicateOperator.ON_SCALE,
            expected_answer_type=ExpectedAnswerType.SCORE,
            score_criteria=("absent", "partial", "complete"),
            expected_value=2,
            context=(
                ContextItem(
                    key="control-description",
                    value="The control is implemented end to end",
                    kind=ContextKind.ASSUMPTION,
                ),
            ),
            evidence_requirements=(requirement,),
            verification_method=VerificationMethod.SEMANTIC_SENSOR,
        )
    )

    result = await engine.observe(
        ObserveRequest(
            allow_external_jev=True,
            investigation_id=atomized.investigation_id,
            atom_id=atomized.atom.id,
        )
    )

    assert result.status is SemanticStatus.UNKNOWN
    assert "outside the requested rubric" in result.missing_information[0]


async def test_pydantic_jev_rejects_out_of_range_noul_probability(make_request) -> None:
    requirement = EvidenceRequirement(
        id="jev-observation",
        description="Jev evaluates the supplied configuration context",
        accepted_kinds=(EvidenceKind.SENSOR_OUTPUT,),
    )
    engine = EnzoEngine(sensor=PydanticJevSensor(client=FakeSystemOneClient(NoulAnswer(noul=2.0))))
    atomized = await engine.atomize(
        make_request(
            context=(
                ContextItem(
                    key="production-cookie-config",
                    value={"secure": True},
                    kind=ContextKind.ASSUMPTION,
                ),
            ),
            evidence_requirements=(requirement,),
            verification_method=VerificationMethod.SEMANTIC_SENSOR,
        )
    )

    result = await engine.observe(
        ObserveRequest(
            allow_external_jev=True,
            investigation_id=atomized.investigation_id,
            atom_id=atomized.atom.id,
        )
    )

    assert result.status is SemanticStatus.UNKNOWN
    assert "finite [0, 1] range" in result.missing_information[0]


async def test_pydantic_jev_rejects_non_normalized_choice_distribution(make_request) -> None:
    requirement = EvidenceRequirement(
        id="jev-observation",
        description="Jev selects the observed deployment state",
        accepted_kinds=(EvidenceKind.SENSOR_OUTPUT,),
    )
    answer = ChoiceAnswer(
        choice="enabled",
        confidence=0.9,
        probabilities={"enabled": 0.9, "disabled": 0.9},
    )
    engine = EnzoEngine(sensor=PydanticJevSensor(client=FakeSystemOneClient(answer)))
    atomized = await engine.atomize(
        make_request(
            question="Which deployment state is shown?",
            expected_answer_type=ExpectedAnswerType.CHOICE,
            answer_options=("enabled", "disabled"),
            expected_value="enabled",
            context=(
                ContextItem(
                    key="deployment-state",
                    value="enabled",
                    kind=ContextKind.ASSUMPTION,
                ),
            ),
            evidence_requirements=(requirement,),
            verification_method=VerificationMethod.SEMANTIC_SENSOR,
        )
    )

    result = await engine.observe(
        ObserveRequest(
            allow_external_jev=True,
            investigation_id=atomized.investigation_id,
            atom_id=atomized.atom.id,
        )
    )

    assert result.status is SemanticStatus.UNKNOWN
    assert "invalid choice probability distribution" in result.missing_information[0]


async def test_pydantic_jev_rejects_out_of_range_score_confidence(make_request) -> None:
    requirement = EvidenceRequirement(
        id="jev-observation",
        description="Jev scores the supplied rubric",
        accepted_kinds=(EvidenceKind.SENSOR_OUTPUT,),
    )
    answer = ScoreAnswer(
        score=2.0,
        confidence=1.2,
        legend={0: "absent", 1: "partial", 2: "complete"},
        probabilities={0: 0.01, 1: 0.04, 2: 0.95},
    )
    engine = EnzoEngine(sensor=PydanticJevSensor(client=FakeSystemOneClient(answer)))
    atomized = await engine.atomize(
        make_request(
            question="How completely is the control implemented?",
            operator=PredicateOperator.ON_SCALE,
            expected_answer_type=ExpectedAnswerType.SCORE,
            score_criteria=("absent", "partial", "complete"),
            expected_value=2,
            context=(
                ContextItem(
                    key="control-description",
                    value="The control is implemented end to end",
                    kind=ContextKind.ASSUMPTION,
                ),
            ),
            evidence_requirements=(requirement,),
            verification_method=VerificationMethod.SEMANTIC_SENSOR,
        )
    )

    result = await engine.observe(
        ObserveRequest(
            allow_external_jev=True,
            investigation_id=atomized.investigation_id,
            atom_id=atomized.atom.id,
        )
    )

    assert result.status is SemanticStatus.UNKNOWN
    assert "score confidence outside" in result.missing_information[0]


async def test_external_jev_send_requires_per_observation_consent(make_request) -> None:
    requirement = EvidenceRequirement(
        id="jev-observation",
        description="Jev evaluates the supplied configuration context",
        accepted_kinds=(EvidenceKind.SENSOR_OUTPUT,),
    )
    client = FakeSystemOneClient(NoulAnswer(noul=0.99))
    engine = EnzoEngine(sensor=PydanticJevSensor(client=client))
    atomized = await engine.atomize(
        make_request(
            context=(
                ContextItem(
                    key="secret-token",
                    value="must-not-leave-process",
                    kind=ContextKind.ASSUMPTION,
                ),
            ),
            evidence_requirements=(requirement,),
            verification_method=VerificationMethod.SEMANTIC_SENSOR,
        )
    )

    result = await engine.observe(
        ObserveRequest(
            investigation_id=atomized.investigation_id,
            atom_id=atomized.atom.id,
        )
    )

    assert result.status is SemanticStatus.UNKNOWN
    assert "requires allow_external_jev=true" in result.missing_information[0]
    assert client.calls == 0
    assert client.state is None


async def test_unconsented_observation_can_be_retried_with_consent(make_request) -> None:
    requirement = EvidenceRequirement(
        id="jev-observation",
        description="Jev evaluates the supplied configuration context",
        accepted_kinds=(EvidenceKind.SENSOR_OUTPUT,),
    )
    client = FakeSystemOneClient(NoulAnswer(noul=0.99))
    engine = EnzoEngine(sensor=PydanticJevSensor(client=client))
    atomized = await engine.atomize(
        make_request(
            context=(
                ContextItem(
                    key="production-cookie-config",
                    value={"secure": True},
                    kind=ContextKind.ASSUMPTION,
                ),
            ),
            evidence_requirements=(requirement,),
            verification_method=VerificationMethod.SEMANTIC_SENSOR,
        )
    )

    first = await engine.observe(
        ObserveRequest(
            investigation_id=atomized.investigation_id,
            atom_id=atomized.atom.id,
        )
    )
    second = await engine.observe(
        ObserveRequest(
            allow_external_jev=True,
            investigation_id=atomized.investigation_id,
            atom_id=atomized.atom.id,
        )
    )

    assert first.status is SemanticStatus.UNKNOWN
    assert second.status is SemanticStatus.VERIFIED
    assert client.calls == 1


async def test_pydantic_jev_rejects_score_inconsistent_with_distribution(make_request) -> None:
    requirement = EvidenceRequirement(
        id="jev-observation",
        description="Jev scores the supplied rubric",
        accepted_kinds=(EvidenceKind.SENSOR_OUTPUT,),
    )
    answer = ScoreAnswer(
        score=0.0,
        confidence=0.9,
        legend={0: "absent", 1: "partial", 2: "complete"},
        probabilities={0: 0.05, 1: 0.15, 2: 0.8},
    )
    engine = EnzoEngine(sensor=PydanticJevSensor(client=FakeSystemOneClient(answer)))
    atomized = await engine.atomize(
        make_request(
            question="How completely is the control implemented?",
            operator=PredicateOperator.ON_SCALE,
            expected_answer_type=ExpectedAnswerType.SCORE,
            score_criteria=("absent", "partial", "complete"),
            expected_value=2,
            context=(
                ContextItem(
                    key="control-description",
                    value="The control is implemented end to end",
                    kind=ContextKind.ASSUMPTION,
                ),
            ),
            evidence_requirements=(requirement,),
            verification_method=VerificationMethod.SEMANTIC_SENSOR,
        )
    )

    result = await engine.observe(
        ObserveRequest(
            allow_external_jev=True,
            investigation_id=atomized.investigation_id,
            atom_id=atomized.atom.id,
        )
    )

    assert result.status is SemanticStatus.UNKNOWN
    assert "probability-weighted rubric value" in result.missing_information[0]


async def test_repeated_semantic_observation_is_idempotent(make_request) -> None:
    requirement = EvidenceRequirement(
        id="jev-observation",
        description="Jev evaluates the supplied configuration context",
        accepted_kinds=(EvidenceKind.SENSOR_OUTPUT,),
    )
    client = FakeSystemOneClient(NoulAnswer(noul=0.93))
    engine = EnzoEngine(sensor=PydanticJevSensor(client=client))
    atomized = await engine.atomize(
        make_request(
            context=(
                ContextItem(
                    key="production-cookie-config",
                    value={"secure": True},
                    kind=ContextKind.ASSUMPTION,
                ),
            ),
            evidence_requirements=(requirement,),
            verification_method=VerificationMethod.SEMANTIC_SENSOR,
        )
    )
    observation = ObserveRequest(
        allow_external_jev=True,
        investigation_id=atomized.investigation_id,
        atom_id=atomized.atom.id,
    )

    first = await engine.observe(observation)
    second = await engine.observe(observation)

    assert second == first
    assert client.calls == 1
    assert sum(item.kind is EvidenceKind.SENSOR_OUTPUT for item in second.evidence) == 1


async def test_repeated_evidence_bearing_observation_is_idempotent(make_request) -> None:
    requirement = EvidenceRequirement(
        id="jev-observation",
        description="Jev evaluates the supplied configuration context",
        accepted_kinds=(EvidenceKind.SENSOR_OUTPUT,),
    )
    client = FakeSystemOneClient(NoulAnswer(noul=0.93))
    engine = EnzoEngine(sensor=PydanticJevSensor(client=client))
    atomized = await engine.atomize(
        make_request(
            evidence_requirements=(requirement,),
            verification_method=VerificationMethod.SEMANTIC_SENSOR,
        )
    )
    caller_evidence = EvidenceRecord(
        id="caller-observation",
        atom_id=atomized.atom.id,
        direction=EvidenceDirection.NEUTRAL,
        kind=EvidenceKind.USER_PROVIDED,
        summary="The caller supplied semantic context for Jev to assess",
        verification_method=VerificationMethod.HUMAN_OBSERVATION,
        provenance=Provenance(provider="caller", locator="request:caller-observation"),
    )
    observation = ObserveRequest(
        allow_external_jev=True,
        investigation_id=atomized.investigation_id,
        atom_id=atomized.atom.id,
        evidence=(caller_evidence,),
        satisfied_constraints=("Caller evidence was collected",),
    )

    first = await engine.observe(observation)
    retry_payload = observation.model_dump(mode="json")
    retry_evidence = cast(list[dict[str, object]], retry_payload["evidence"])
    retry_provenance = cast(dict[str, object], retry_evidence[0]["provenance"])
    del retry_provenance["observed_at"]
    second = await engine.observe(ObserveRequest.model_validate(retry_payload))

    assert second == first
    assert client.calls == 1
    assert sum(item.kind is EvidenceKind.SENSOR_OUTPUT for item in second.evidence) == 1


async def test_dependency_change_accepts_transport_retry_evidence(
    make_request,
    make_evidence,
) -> None:
    client = FakeSystemOneClient(NoulAnswer(noul=0.93))
    engine = EnzoEngine(sensor=PydanticJevSensor(client=client))
    prerequisite = (await engine.atomize(make_request())).atom
    prerequisite_requirement = prerequisite.evidence_requirements[0].id
    await engine.observe(
        ObserveRequest(
            investigation_id=prerequisite.investigation_id,
            atom_id=prerequisite.id,
            evidence=(
                make_evidence(
                    atom_id=prerequisite.id,
                    requirement_id=prerequisite_requirement,
                ),
            ),
        )
    )
    sensor_requirement = EvidenceRequirement(
        id="jev-observation",
        description="Jev evaluates the caller evidence",
        accepted_kinds=(EvidenceKind.SENSOR_OUTPUT,),
    )
    dependent = (
        await engine.atomize(
            make_request(
                atom_id="dependent-atom",
                investigation_id=prerequisite.investigation_id,
                parent_id=prerequisite.id,
                depth=1,
                dependencies=(
                    DependencyGroup(
                        id="verified-prerequisite",
                        role=DependencyRole.PREREQUISITE,
                        atom_ids=(prerequisite.id,),
                    ),
                ),
                evidence_requirements=(sensor_requirement,),
                verification_method=VerificationMethod.SEMANTIC_SENSOR,
            )
        )
    ).atom
    caller_evidence = EvidenceRecord(
        id="dependency-retry-evidence",
        atom_id=dependent.id,
        direction=EvidenceDirection.NEUTRAL,
        kind=EvidenceKind.USER_PROVIDED,
        summary="Caller evidence requiring semantic assessment",
        verification_method=VerificationMethod.HUMAN_OBSERVATION,
        provenance=Provenance(provider="caller", locator="request:dependency-retry"),
    )
    observation = ObserveRequest(
        allow_external_jev=True,
        investigation_id=dependent.investigation_id,
        atom_id=dependent.id,
        evidence=(caller_evidence,),
    )
    first = await engine.observe(observation)

    await engine.observe(
        ObserveRequest(
            investigation_id=prerequisite.investigation_id,
            atom_id=prerequisite.id,
            evidence=(
                make_evidence(
                    atom_id=prerequisite.id,
                    requirement_id=prerequisite_requirement,
                    direction=EvidenceDirection.CONTRADICTS,
                    evidence_id="prerequisite-contradiction",
                ),
            ),
        )
    )
    retry_payload = observation.model_dump(mode="json")
    retry_evidence = cast(list[dict[str, object]], retry_payload["evidence"])
    retry_provenance = cast(dict[str, object], retry_evidence[0]["provenance"])
    del retry_provenance["observed_at"]

    second = await engine.observe(ObserveRequest.model_validate(retry_payload))

    assert first.status is SemanticStatus.VERIFIED
    assert second.status is SemanticStatus.INSUFFICIENT_EVIDENCE
    assert client.calls == 2


async def test_prior_sensor_output_is_not_sent_back_to_jev(make_request) -> None:
    requirement = EvidenceRequirement(
        id="jev-observation",
        description="Jev evaluates the supplied configuration context",
        accepted_kinds=(EvidenceKind.SENSOR_OUTPUT,),
    )
    client = FakeSystemOneClient(NoulAnswer(noul=0.93))
    engine = EnzoEngine(sensor=PydanticJevSensor(client=client))
    atomized = await engine.atomize(
        make_request(
            context=(
                ContextItem(
                    key="production-cookie-config",
                    value={"secure": True},
                    kind=ContextKind.ASSUMPTION,
                ),
            ),
            evidence_requirements=(requirement,),
            verification_method=VerificationMethod.SEMANTIC_SENSOR,
        )
    )
    await engine.observe(
        ObserveRequest(
            allow_external_jev=True,
            investigation_id=atomized.investigation_id,
            atom_id=atomized.atom.id,
        )
    )

    reevaluation = ObserveRequest(
        allow_external_jev=True,
        investigation_id=atomized.investigation_id,
        atom_id=atomized.atom.id,
        missing_information=("A new caller-supplied gap requires reevaluation",),
    )
    second = await engine.observe(reevaluation)
    replay = await engine.observe(reevaluation)

    assert client.calls == 2
    assert replay == second
    state = cast(dict[str, object], client.state)
    assert state["evidence"] == []
