"""Pydantic AI TypeSafe/Jev semantic sensor boundary."""

from __future__ import annotations

import os
from collections.abc import Iterable
from contextlib import suppress
from datetime import UTC, datetime
from math import isclose, isfinite
from typing import Protocol, cast

from pydantic import Field, JsonValue
from pydantic_ai.models.typesafe import TypeSafeModel
from typesafe_sdk import (
    Choice,
    ChoiceAnswer,
    JSONContent,
    Noul,
    NoulAnswer,
    Question,
    Questions,
    RetryPolicy,
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
    JevDispatchManifest,
    JevProviderQuestion,
    Provenance,
    SemanticStatus,
    VerificationMethod,
    new_id,
)
from .response_cache import JevCacheHit, JevResponseCache, response_cache_from_env

_QUESTION_KEY = "claim"
_DISTRIBUTION_TOLERANCE = 1e-3
# One reviewed manifest authorizes one transport attempt; retries need a new assertion.
_NO_TRANSPORT_RETRIES = RetryPolicy(max_retries=0)


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
        manifest: JevDispatchManifest,
    ) -> SensorAssessment:
        """Evaluate one atom against a prepared, approved outbound request."""


class SystemOneClient(Protocol):
    async def system_one(
        self,
        state: JSONContent,
        questions: Questions,
        *,
        model: str | None = None,
        retry: RetryPolicy | None = None,
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
        cache: JevResponseCache | None = None,
    ) -> None:
        if not 0 < contradiction_threshold < verify_threshold < 1:
            raise ValueError("thresholds must satisfy 0 < contradiction < verify < 1")
        self._model_name = model_name
        self._verify_threshold = verify_threshold
        self._contradiction_threshold = contradiction_threshold
        self._client = (
            client
            if client is not None
            else (
                TypeSafeModel(model_name).client
                if os.getenv("TYPESAFE_API_KEY", "").strip()
                else None
            )
        )
        self._cache = cache

    @property
    def model_name(self) -> str:
        """Expose the exact model covered by a dispatch manifest."""

        return self._model_name

    async def evaluate(
        self,
        atom: AtomicRequest,
        manifest: JevDispatchManifest,
    ) -> SensorAssessment:
        logical_request = manifest.logical_request
        if not logical_request.state.context and not logical_request.state.evidence:
            return self._unknown(
                "Jev needs context or evidence to evaluate this atom",
                locator="sensor:jev:no-state",
            )

        questions: Questions = {_QUESTION_KEY: self._question(logical_request.questions.claim)}
        if self._cache is not None:
            try:
                cache_hit = await self._cache.get(manifest)
            except Exception:
                cache_hit = None
            if cache_hit is not None:
                assessment, valid = self._assessment_from_response(
                    atom,
                    cache_hit.response,
                    cache_hit=cache_hit,
                )
                if valid:
                    return assessment

        if self._client is None:
            return self._unknown(
                "JEV semantic sensor is not configured and no cached response matched",
                locator="sensor:jev:unconfigured-cache-miss",
            )
        try:
            response = await self._client.system_one(
                cast(JSONContent, logical_request.state.model_dump(mode="json")),
                questions,
                model=logical_request.model,
                retry=_NO_TRANSPORT_RETRIES,
            )
        except TypeSafeError as error:
            return self._unknown(
                f"Jev request failed: {type(error).__name__}",
                locator="sensor:jev:request-failed",
            )

        assessment, valid = self._assessment_from_response(atom, response)
        if valid and self._cache is not None:
            with suppress(Exception):
                await self._cache.put(manifest, response)
        return assessment

    def _assessment_from_response(
        self,
        atom: AtomicRequest,
        response: SystemOneResponse,
        *,
        cache_hit: JevCacheHit | None = None,
    ) -> tuple[SensorAssessment, bool]:
        """Validate one typed response and map it to Enzo's semantic contract."""

        answer = response.answers.get(_QUESTION_KEY)
        if answer is None:
            return (
                self._unknown(
                    "Jev returned no answer for the atomic claim",
                    locator=f"model:{response.model}",
                ),
                False,
            )

        if atom.expected_answer_type is ExpectedAnswerType.BOOLEAN:
            if not isinstance(answer, NoulAnswer):
                return self._unexpected_answer(response), False
            if not self._is_probability(answer.noul):
                return (
                    self._invalid_answer_contract(
                        response,
                        "Jev returned a yes/no probability outside the finite [0, 1] range",
                    ),
                    False,
                )
            status, direction, gap = self._boolean_outcome(
                answer.noul,
                cast(bool, atom.expected_value),
            )
        elif atom.expected_answer_type is ExpectedAnswerType.CHOICE:
            if not isinstance(answer, ChoiceAnswer):
                return self._unexpected_answer(response), False
            contract_gap = self._choice_contract_gap(atom, answer)
            if contract_gap is not None:
                return self._invalid_answer_contract(response, contract_gap), False
            status, direction, gap = self._choice_outcome(
                answer.choice,
                answer.confidence,
                cast(str, atom.expected_value),
            )
        else:
            if not isinstance(answer, ScoreAnswer):
                return self._unexpected_answer(response), False
            contract_gap = self._score_contract_gap(atom, answer)
            if contract_gap is not None:
                return self._invalid_answer_contract(response, contract_gap), False
            status, direction, gap = self._score_outcome(
                answer,
                cast(int, atom.expected_value),
            )

        evidence_id = new_id("evidence")
        if cache_hit is None:
            provenance = Provenance(
                provider="pydantic-ai/typesafe",
                locator=f"model:{response.model}",
                tool="TypeSafeModel.system_one",
                run_id=evidence_id,
            )
            cache_payload: dict[str, JsonValue] = {"hit": False}
        else:
            provenance = Provenance(
                provider="enzo/jev-response-cache",
                locator=(
                    f"cache:{cache_hit.namespace}:{cache_hit.model_epoch}:{cache_hit.cache_key}"
                ),
                tool="JevResponseCache.get",
                run_id=evidence_id,
                content_hash=cache_hit.response_sha256,
                observed_at=cache_hit.created_at,
            )
            cache_payload = {
                "hit": True,
                "namespace": cache_hit.namespace,
                "model_epoch": cache_hit.model_epoch,
                "key_sha256": cache_hit.cache_key,
                "response_sha256": cache_hit.response_sha256,
                "created_at": cache_hit.created_at.isoformat(),
                "expires_at": cache_hit.expires_at.isoformat(),
                "retrieved_at": datetime.now(UTC).isoformat(),
                "original_provider": "pydantic-ai/typesafe",
                "usage_is_original": True,
            }
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
                "cache": cache_payload,
            },
            deterministic=False,
            verification_method=VerificationMethod.SEMANTIC_SENSOR,
            requirement_ids=self._compatible_requirements(atom),
            provenance=provenance,
        )
        return (
            SensorAssessment(
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
                    "Enzo reused one cached typed Jev response for the exact approved request."
                    if cache_hit is not None
                    else (
                        "Jev evaluated one typed question against only the supplied context "
                        "and evidence."
                    )
                ),
            ),
            True,
        )

    def _question(self, question: JevProviderQuestion) -> Question:
        if question.type == "noul":
            return Noul(instructions=question.instructions)
        if question.type == "choice":
            return Choice(
                instructions=question.instructions,
                criteria=question.criteria,
            )
        return Score(
            instructions=question.instructions,
            criteria=cast(list[str], question.criteria),
        )

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

    @property
    def model_name(self) -> str:
        """Keep previews deterministic without making an external client available."""

        return "jev-latest"

    async def evaluate(
        self,
        atom: AtomicRequest,
        manifest: JevDispatchManifest,
    ) -> SensorAssessment:
        del atom, manifest
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

    cache = response_cache_from_env()
    if os.getenv("TYPESAFE_API_KEY", "").strip() or cache is not None:
        return PydanticJevSensor(cache=cache)
    return UnconfiguredJevSensor()
