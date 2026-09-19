"""Pydantic AI TypeSafe/Jev semantic sensor boundary."""

from __future__ import annotations

import os
from collections.abc import Iterable
from math import isclose, isfinite
from typing import Protocol, cast

from pydantic import Field
from pydantic_ai.models.typesafe import TypeSafeModel
from typesafe_sdk import (
    Choice,
    ChoiceAnswer,
    JSONContent,
    Noul,
    NoulAnswer,
    Question,
    Questions,
    Score,
    ScoreAnswer,
    SystemOneResponse,
    TypeSafeError,
)

from .models import (
    AtomicRequest,
    ContractModel,
    EvidenceDirection,
    EvidenceKind,
    EvidenceRecord,
    ExpectedAnswerType,
    Provenance,
    SemanticStatus,
    VerificationMethod,
    new_id,
)

_QUESTION_KEY = "claim"
_DISTRIBUTION_TOLERANCE = 1e-3


class SensorAssessment(ContractModel):
    status: SemanticStatus
    evidence: tuple[EvidenceRecord, ...] = ()
    missing_information: tuple[str, ...] = ()
    satisfied_constraints: tuple[str, ...] = ()
    violated_constraints: tuple[str, ...] = ()
    provenance: Provenance
    explanation: str = Field(min_length=1)


class SemanticSensor(Protocol):
    async def evaluate(
        self,
        atom: AtomicRequest,
        evidence: tuple[EvidenceRecord, ...],
    ) -> SensorAssessment:
        """Evaluate one already-admitted atomic claim against typed evidence."""


class SystemOneClient(Protocol):
    async def system_one(
        self,
        state: JSONContent,
        questions: Questions,
        *,
        model: str | None = None,
    ) -> SystemOneResponse:
        """Return Jev answers for typed questions about state."""


class PydanticJevSensor:
    """Use Pydantic AI's official TypeSafe model provider as a Jev sensor."""

    def __init__(
        self,
        *,
        model_name: str = "jev-latest",
        verify_threshold: float = 0.8,
        contradiction_threshold: float = 0.2,
        client: SystemOneClient | None = None,
    ) -> None:
        if not 0 < contradiction_threshold < verify_threshold < 1:
            raise ValueError("thresholds must satisfy 0 < contradiction < verify < 1")
        self._model_name = model_name
        self._verify_threshold = verify_threshold
        self._contradiction_threshold = contradiction_threshold
        self._client = client if client is not None else TypeSafeModel(model_name).client

    async def evaluate(
        self,
        atom: AtomicRequest,
        evidence: tuple[EvidenceRecord, ...],
    ) -> SensorAssessment:
        if not atom.context and not evidence:
            return self._unknown(
                "Jev needs context or evidence to evaluate this atom",
                locator="sensor:jev:no-state",
            )

        questions: Questions = {_QUESTION_KEY: self._question(atom)}
        try:
            response = await self._client.system_one(
                self._state(atom, evidence),
                questions,
                model=self._model_name,
            )
        except TypeSafeError as error:
            return self._unknown(
                f"Jev request failed: {type(error).__name__}",
                locator="sensor:jev:request-failed",
            )

        answer = response.answers.get(_QUESTION_KEY)
        if answer is None:
            return self._unknown(
                "Jev returned no answer for the atomic claim",
                locator=f"model:{response.model}",
            )

        if atom.expected_answer_type is ExpectedAnswerType.BOOLEAN:
            if not isinstance(answer, NoulAnswer):
                return self._unexpected_answer(response)
            if not self._is_probability(answer.noul):
                return self._invalid_answer_contract(
                    response,
                    "Jev returned a yes/no probability outside the finite [0, 1] range",
                )
            status, direction, gap = self._boolean_outcome(
                answer.noul,
                cast(bool, atom.expected_value),
            )
        elif atom.expected_answer_type is ExpectedAnswerType.CHOICE:
            if not isinstance(answer, ChoiceAnswer):
                return self._unexpected_answer(response)
            contract_gap = self._choice_contract_gap(atom, answer)
            if contract_gap is not None:
                return self._invalid_answer_contract(response, contract_gap)
            status, direction, gap = self._choice_outcome(
                answer.choice,
                answer.confidence,
                cast(str, atom.expected_value),
            )
        else:
            if not isinstance(answer, ScoreAnswer):
                return self._unexpected_answer(response)
            contract_gap = self._score_contract_gap(atom, answer)
            if contract_gap is not None:
                return self._invalid_answer_contract(response, contract_gap)
            status, direction, gap = self._score_outcome(
                answer,
                cast(int, atom.expected_value),
            )

        evidence_id = new_id("evidence")
        provenance = Provenance(
            provider="pydantic-ai/typesafe",
            locator=f"model:{response.model}",
            tool="TypeSafeModel.system_one",
            run_id=evidence_id,
        )
        sensor_evidence = EvidenceRecord(
            id=evidence_id,
            atom_id=atom.id,
            direction=direction,
            kind=EvidenceKind.SENSOR_OUTPUT,
            summary=self._summary(status),
            payload={
                "model": response.model,
                "answer": answer.model_dump(mode="json"),
                "usage": response.usage.model_dump(mode="json"),
                "thresholds": {
                    "verify": self._verify_threshold,
                    "contradiction": self._contradiction_threshold,
                },
            },
            deterministic=False,
            verification_method=VerificationMethod.SEMANTIC_SENSOR,
            requirement_ids=self._compatible_requirements(atom),
            provenance=provenance,
        )
        return SensorAssessment(
            status=status,
            evidence=(sensor_evidence,),
            missing_information=(() if gap is None else (gap,)),
            satisfied_constraints=(
                ("Jev's typed answer matched the atom's expected outcome",)
                if status is SemanticStatus.VERIFIED
                else ()
            ),
            violated_constraints=(
                ("Jev's typed answer contradicted the atom's expected outcome",)
                if status is SemanticStatus.CONTRADICTED
                else ()
            ),
            provenance=provenance,
            explanation=(
                "Jev evaluated one typed question against only the supplied context and evidence."
            ),
        )

    def _question(self, atom: AtomicRequest) -> Question:
        if atom.expected_answer_type is ExpectedAnswerType.BOOLEAN:
            return Noul(instructions=atom.question)
        if atom.expected_answer_type is ExpectedAnswerType.CHOICE:
            return Choice(
                instructions=atom.question,
                criteria={option: option for option in atom.answer_options},
            )
        return Score(instructions=atom.question, criteria=atom.score_criteria)

    @staticmethod
    def _state(
        atom: AtomicRequest,
        evidence: tuple[EvidenceRecord, ...],
    ) -> JSONContent:
        payload = {
            "scope": atom.scope,
            "context": [item.model_dump(mode="json") for item in atom.context],
            "evidence": [
                {
                    "direction": item.direction.value,
                    "kind": item.kind.value,
                    "summary": item.summary,
                    "payload": item.payload,
                    "assumptions": list(item.assumptions),
                    "provenance": {
                        "provider": item.provenance.provider,
                        "locator": item.provenance.locator,
                        "tool": item.provenance.tool,
                        "content_hash": item.provenance.content_hash,
                    },
                }
                for item in evidence
            ],
        }
        return cast(JSONContent, payload)

    def _boolean_outcome(
        self,
        probability_yes: float,
        expected: bool,
    ) -> tuple[SemanticStatus, EvidenceDirection, str | None]:
        probability_expected = probability_yes if expected else 1 - probability_yes
        if probability_expected >= self._verify_threshold:
            return SemanticStatus.VERIFIED, EvidenceDirection.SUPPORTS, None
        if probability_expected <= self._contradiction_threshold:
            return SemanticStatus.CONTRADICTED, EvidenceDirection.CONTRADICTS, None
        return (
            SemanticStatus.UNKNOWN,
            EvidenceDirection.NEUTRAL,
            "Jev's yes/no probability did not cross a decision threshold",
        )

    def _choice_outcome(
        self,
        choice: str,
        confidence: float,
        expected: str,
    ) -> tuple[SemanticStatus, EvidenceDirection, str | None]:
        if confidence < self._verify_threshold:
            return (
                SemanticStatus.UNKNOWN,
                EvidenceDirection.NEUTRAL,
                "Jev's choice confidence did not cross the verification threshold",
            )
        if choice == expected:
            return SemanticStatus.VERIFIED, EvidenceDirection.SUPPORTS, None
        return SemanticStatus.CONTRADICTED, EvidenceDirection.CONTRADICTS, None

    def _score_outcome(
        self,
        answer: ScoreAnswer,
        expected: int,
    ) -> tuple[SemanticStatus, EvidenceDirection, str | None]:
        if answer.confidence < self._verify_threshold:
            return (
                SemanticStatus.UNKNOWN,
                EvidenceDirection.NEUTRAL,
                "Jev's score confidence did not cross the verification threshold",
            )
        highest = max(answer.probabilities.values())
        modes = tuple(
            level for level, probability in answer.probabilities.items() if probability == highest
        )
        if len(modes) != 1:
            return (
                SemanticStatus.UNKNOWN,
                EvidenceDirection.NEUTRAL,
                "Jev's score distribution has no unique most likely rubric level",
            )
        if modes[0] == expected:
            return SemanticStatus.VERIFIED, EvidenceDirection.SUPPORTS, None
        return SemanticStatus.CONTRADICTED, EvidenceDirection.CONTRADICTS, None

    @staticmethod
    def _choice_contract_gap(
        atom: AtomicRequest,
        answer: ChoiceAnswer,
    ) -> str | None:
        if not PydanticJevSensor._is_probability(answer.confidence):
            return "Jev returned choice confidence outside the finite [0, 1] range"
        expected_options = set(atom.answer_options)
        if answer.choice not in expected_options:
            return "Jev selected a choice outside the requested answer options"
        if set(answer.probabilities) != expected_options:
            return "Jev returned a probability distribution for unexpected choice options"
        if not PydanticJevSensor._is_distribution(answer.probabilities.values()):
            return "Jev returned an invalid choice probability distribution"
        highest = max(answer.probabilities.values())
        modes = tuple(
            option for option, probability in answer.probabilities.items() if probability == highest
        )
        if modes != (answer.choice,):
            return "Jev's selected choice does not uniquely match its probability distribution"
        return None

    @staticmethod
    def _score_contract_gap(
        atom: AtomicRequest,
        answer: ScoreAnswer,
    ) -> str | None:
        if not PydanticJevSensor._is_probability(answer.confidence):
            return "Jev returned score confidence outside the finite [0, 1] range"
        expected_levels = set(range(len(atom.score_criteria)))
        if set(answer.probabilities) != expected_levels or set(answer.legend) != expected_levels:
            return "Jev returned score levels outside the requested rubric"
        if not PydanticJevSensor._is_distribution(answer.probabilities.values()):
            return "Jev returned an invalid score probability distribution"
        if any(
            answer.legend[index] != criterion for index, criterion in enumerate(atom.score_criteria)
        ):
            return "Jev returned a score legend that does not match the requested rubric"
        if not isfinite(answer.score) or not 0 <= answer.score <= len(atom.score_criteria) - 1:
            return "Jev returned a score outside the requested rubric range"
        weighted_score = sum(
            level * probability for level, probability in answer.probabilities.items()
        )
        if not isclose(
            answer.score,
            weighted_score,
            rel_tol=_DISTRIBUTION_TOLERANCE,
            abs_tol=_DISTRIBUTION_TOLERANCE,
        ):
            return "Jev's score does not match its probability-weighted rubric value"
        return None

    @staticmethod
    def _is_probability(value: float) -> bool:
        return isfinite(value) and 0 <= value <= 1

    @classmethod
    def _is_distribution(cls, values: Iterable[float]) -> bool:
        probabilities = tuple(values)
        return (
            bool(probabilities)
            and all(cls._is_probability(value) for value in probabilities)
            and isclose(
                sum(probabilities),
                1.0,
                rel_tol=_DISTRIBUTION_TOLERANCE,
                abs_tol=_DISTRIBUTION_TOLERANCE,
            )
        )

    @staticmethod
    def _compatible_requirements(atom: AtomicRequest) -> tuple[str, ...]:
        return tuple(
            requirement.id
            for requirement in atom.evidence_requirements
            if not requirement.deterministic_required
            and (
                not requirement.accepted_kinds
                or EvidenceKind.SENSOR_OUTPUT in requirement.accepted_kinds
            )
        )

    @staticmethod
    def _summary(status: SemanticStatus) -> str:
        if status is SemanticStatus.VERIFIED:
            return "Jev's typed answer supports the atomic claim"
        if status is SemanticStatus.CONTRADICTED:
            return "Jev's typed answer contradicts the atomic claim"
        return "Jev evaluated the atomic claim but did not resolve it"

    @staticmethod
    def _unknown(message: str, *, locator: str) -> SensorAssessment:
        return SensorAssessment(
            status=SemanticStatus.UNKNOWN,
            missing_information=(message,),
            provenance=Provenance(
                provider="pydantic-ai/typesafe",
                locator=locator,
                tool="TypeSafeModel.system_one",
            ),
            explanation="No semantic conclusion was produced.",
        )

    def _unexpected_answer(self, response: SystemOneResponse) -> SensorAssessment:
        return self._unknown(
            "Jev returned an answer primitive that does not match the atom contract",
            locator=f"model:{response.model}",
        )

    def _invalid_answer_contract(
        self,
        response: SystemOneResponse,
        message: str,
    ) -> SensorAssessment:
        return self._unknown(message, locator=f"model:{response.model}")


class UnconfiguredJevSensor:
    """Fail closed when TYPESAFE_API_KEY is absent."""

    async def evaluate(
        self,
        atom: AtomicRequest,
        evidence: tuple[EvidenceRecord, ...],
    ) -> SensorAssessment:
        del atom, evidence
        return SensorAssessment(
            status=SemanticStatus.UNKNOWN,
            missing_information=("JEV semantic sensor is not configured",),
            provenance=Provenance(
                provider="enzo",
                locator="sensor:jev:unconfigured",
                tool="UnconfiguredJevSensor",
            ),
            explanation=(
                "Set TYPESAFE_API_KEY in the MCP runtime environment. Deterministic evidence "
                "may still resolve the atom without Jev."
            ),
        )


def default_sensor() -> SemanticSensor:
    """Select the official Jev adapter only when its credential is available."""

    if os.getenv("TYPESAFE_API_KEY", "").strip():
        return PydanticJevSensor()
    return UnconfiguredJevSensor()
