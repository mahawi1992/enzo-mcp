"""Pydantic contracts for Enzo's legal semantic state space."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal, Self
from uuid import uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StrictBool,
    StrictInt,
    model_validator,
)


def new_id(prefix: str) -> str:
    """Return a human-readable, globally unique identifier."""

    return f"{prefix}_{uuid4().hex}"


class ContractModel(BaseModel):
    """Strict immutable base for values crossing a semantic boundary."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        use_enum_values=False,
    )


class AtomicityDecision(StrEnum):
    ATOMIC = "ATOMIC"
    DECOMPOSE = "DECOMPOSE"
    NEEDS_REFINEMENT = "NEEDS_REFINEMENT"


class ExpectedAnswerType(StrEnum):
    BOOLEAN = "BOOLEAN"
    CHOICE = "CHOICE"
    SCORE = "SCORE"


class PredicateOperator(StrEnum):
    IS_TRUE = "IS_TRUE"
    EQUALS = "EQUALS"
    NOT_EQUALS = "NOT_EQUALS"
    CONTAINS = "CONTAINS"
    MATCHES = "MATCHES"
    LESS_THAN = "LESS_THAN"
    LESS_THAN_OR_EQUAL = "LESS_THAN_OR_EQUAL"
    GREATER_THAN = "GREATER_THAN"
    GREATER_THAN_OR_EQUAL = "GREATER_THAN_OR_EQUAL"
    EXISTS = "EXISTS"
    NOT_EXISTS = "NOT_EXISTS"
    ON_SCALE = "ON_SCALE"


class Quantifier(StrEnum):
    ONE = "ONE"
    ALL = "ALL"
    ANY = "ANY"
    NONE = "NONE"


class VerificationMethod(StrEnum):
    TEST = "TEST"
    PROPERTY_TEST = "PROPERTY_TEST"
    STATIC_ANALYSIS = "STATIC_ANALYSIS"
    SCHEMA_VALIDATION = "SCHEMA_VALIDATION"
    AST_INSPECTION = "AST_INSPECTION"
    REPOSITORY_SEARCH = "REPOSITORY_SEARCH"
    RUNTIME_MEASUREMENT = "RUNTIME_MEASUREMENT"
    HUMAN_OBSERVATION = "HUMAN_OBSERVATION"
    SEMANTIC_SENSOR = "SEMANTIC_SENSOR"
    COMPOSITE = "COMPOSITE"


class ContextKind(StrEnum):
    FACT = "FACT"
    ASSUMPTION = "ASSUMPTION"


class EvidenceKind(StrEnum):
    TEST_RESULT = "TEST_RESULT"
    PROPERTY_TEST_RESULT = "PROPERTY_TEST_RESULT"
    STATIC_ANALYSIS = "STATIC_ANALYSIS"
    SCHEMA_VALIDATION = "SCHEMA_VALIDATION"
    AST_INSPECTION = "AST_INSPECTION"
    SOURCE_CODE = "SOURCE_CODE"
    REPOSITORY_SEARCH = "REPOSITORY_SEARCH"
    RUNTIME_MEASUREMENT = "RUNTIME_MEASUREMENT"
    HUMAN_OBSERVATION = "HUMAN_OBSERVATION"
    USER_PROVIDED = "USER_PROVIDED"
    SENSOR_OUTPUT = "SENSOR_OUTPUT"


class EvidenceDirection(StrEnum):
    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    NEUTRAL = "NEUTRAL"


class SemanticStatus(StrEnum):
    VERIFIED = "VERIFIED"
    CONTRADICTED = "CONTRADICTED"
    UNKNOWN = "UNKNOWN"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class ResultSource(StrEnum):
    DETERMINISTIC = "DETERMINISTIC"
    JEV = "JEV"
    MIXED = "MIXED"
    UNCONFIGURED = "UNCONFIGURED"
    DERIVED = "DERIVED"


class DependencyRole(StrEnum):
    PREREQUISITE = "PREREQUISITE"
    DEFINES_PARENT = "DEFINES_PARENT"
    INFORMATIONAL = "INFORMATIONAL"


class DependencyLogic(StrEnum):
    ALL_OF = "ALL_OF"
    ANY_OF = "ANY_OF"


class Provenance(ContractModel):
    provider: str = Field(min_length=1)
    locator: str = Field(min_length=1)
    tool: str | None = Field(default=None, min_length=1)
    command: str | None = Field(default=None, min_length=1)
    run_id: str | None = Field(default=None, min_length=1)
    content_hash: str | None = Field(default=None, min_length=1)
    observed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ContextItem(ContractModel):
    key: str = Field(min_length=1)
    value: JsonValue
    kind: ContextKind
    provenance: Provenance | None = None

    @model_validator(mode="after")
    def fact_requires_provenance(self) -> Self:
        if self.kind is ContextKind.FACT and self.provenance is None:
            raise ValueError("FACT context requires provenance")
        return self


class EvidenceRequirement(ContractModel):
    id: str = Field(default_factory=lambda: new_id("req"), min_length=1)
    description: str = Field(min_length=1)
    accepted_kinds: tuple[EvidenceKind, ...] = ()
    minimum_items: StrictInt = Field(default=1, ge=1)
    required: StrictBool = True
    deterministic_required: StrictBool = False

    @model_validator(mode="after")
    def kinds_are_unique(self) -> Self:
        if len(self.accepted_kinds) != len(set(self.accepted_kinds)):
            raise ValueError("accepted_kinds must not contain duplicates")
        return self


class DependencyGroup(ContractModel):
    id: str = Field(default_factory=lambda: new_id("dep"), min_length=1)
    role: DependencyRole
    logic: DependencyLogic = DependencyLogic.ALL_OF
    atom_ids: tuple[str, ...] = Field(min_length=1)
    complete: StrictBool = False

    @model_validator(mode="after")
    def atom_ids_are_unique(self) -> Self:
        if len(self.atom_ids) != len(set(self.atom_ids)):
            raise ValueError("dependency atom_ids must be unique")
        return self


class ClaimSpec(ContractModel):
    question: str = Field(min_length=1)
    subject: str = Field(min_length=1)
    predicate: str = Field(min_length=1)
    scope: str = Field(min_length=1)
    operator: PredicateOperator = PredicateOperator.IS_TRUE
    expected_value: JsonValue | None = None
    quantifier: Quantifier = Quantifier.ONE
    expected_answer_type: ExpectedAnswerType = ExpectedAnswerType.BOOLEAN
    answer_options: tuple[str, ...] = ()
    score_criteria: tuple[str, ...] = ()
    operational_definition: str | None = Field(default=None, min_length=1)
    evidence_requirements: tuple[EvidenceRequirement, ...] = Field(min_length=1)
    verification_method: VerificationMethod

    @model_validator(mode="after")
    def answer_contract_is_coherent(self) -> Self:
        if len(self.answer_options) != len(set(self.answer_options)):
            raise ValueError("answer_options must be unique")
        if len(self.score_criteria) != len(set(self.score_criteria)):
            raise ValueError("score_criteria must be unique")
        requirement_ids = [item.id for item in self.evidence_requirements]
        if len(requirement_ids) != len(set(requirement_ids)):
            raise ValueError("evidence requirement ids must be unique within an atom")

        if self.expected_answer_type is ExpectedAnswerType.BOOLEAN:
            if type(self.expected_value) is not bool:
                raise ValueError("BOOLEAN atoms require a boolean expected_value")
            if self.answer_options or self.score_criteria:
                raise ValueError("BOOLEAN atoms cannot define choice or score criteria")
            if self.operator is PredicateOperator.ON_SCALE:
                raise ValueError("BOOLEAN atoms cannot use ON_SCALE")
        elif self.expected_answer_type is ExpectedAnswerType.CHOICE:
            if len(self.answer_options) < 2:
                raise ValueError("CHOICE atoms require at least two answer_options")
            if not isinstance(self.expected_value, str):
                raise ValueError("CHOICE atoms require a string expected_value")
            if self.expected_value not in self.answer_options:
                raise ValueError("CHOICE expected_value must be one of answer_options")
            if self.score_criteria:
                raise ValueError("CHOICE atoms cannot define score_criteria")
            if self.operator is PredicateOperator.ON_SCALE:
                raise ValueError("CHOICE atoms cannot use ON_SCALE")
        else:
            if len(self.score_criteria) < 2:
                raise ValueError("SCORE atoms require at least two score_criteria")
            if type(self.expected_value) is not int:
                raise ValueError("SCORE atoms require an integer expected_value")
            if not 0 <= self.expected_value < len(self.score_criteria):
                raise ValueError("SCORE expected_value must index score_criteria")
            if self.answer_options:
                raise ValueError("SCORE atoms cannot define answer_options")
            if self.operator is not PredicateOperator.ON_SCALE:
                raise ValueError("SCORE atoms must use ON_SCALE")
        return self


class ProposedAtom(ClaimSpec):
    """A child claim proposed by the caller, normally the intelligent LLM."""


class AtomizeRequest(ClaimSpec):
    investigation_id: str | None = Field(default=None, min_length=1)
    atom_id: str = Field(default_factory=lambda: new_id("atom"), min_length=1)
    parent_id: str | None = Field(default=None, min_length=1)
    root_goal: str = Field(min_length=1)
    context: tuple[ContextItem, ...] = ()
    dependencies: tuple[DependencyGroup, ...] = ()
    depth: StrictInt = Field(default=0, ge=0)
    revision_of: str | None = Field(default=None, min_length=1)
    branch_from_id: str | None = Field(default=None, min_length=1)
    branch_id: str | None = Field(default=None, min_length=1)
    proposed_children: tuple[ProposedAtom, ...] = ()

    @model_validator(mode="after")
    def trace_is_coherent(self) -> Self:
        if self.depth == 0 and self.parent_id is not None:
            raise ValueError("root atoms cannot have parent_id")
        if self.depth > 0 and self.parent_id is None:
            raise ValueError("non-root atoms require parent_id")
        if (self.branch_from_id is None) != (self.branch_id is None):
            raise ValueError("branch_from_id and branch_id must be provided together")
        referenced = {atom_id for group in self.dependencies for atom_id in group.atom_ids}
        if self.atom_id in referenced:
            raise ValueError("an atom cannot depend on itself")
        dependency_ids = [group.id for group in self.dependencies]
        if len(dependency_ids) != len(set(dependency_ids)):
            raise ValueError("dependency group ids must be unique")
        if any(group.role is DependencyRole.DEFINES_PARENT for group in self.dependencies):
            raise ValueError("DEFINES_PARENT dependencies are engine-managed")
        if self.atom_id in {self.parent_id, self.revision_of, self.branch_from_id}:
            raise ValueError("an atom cannot reference itself")
        if len(self.proposed_children) == 1:
            raise ValueError("a decomposition requires at least two proposed children")
        return self


class AtomicRequest(ClaimSpec):
    id: str = Field(min_length=1)
    investigation_id: str = Field(min_length=1)
    atom_number: StrictInt = Field(ge=1)
    parent_id: str | None = Field(default=None, min_length=1)
    root_goal: str = Field(min_length=1)
    context: tuple[ContextItem, ...] = ()
    dependencies: tuple[DependencyGroup, ...] = ()
    depth: StrictInt = Field(ge=0)
    revision_of: str | None = Field(default=None, min_length=1)
    branch_from_id: str | None = Field(default=None, min_length=1)
    branch_id: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def identity_is_coherent(self) -> Self:
        if self.depth == 0 and self.parent_id is not None:
            raise ValueError("root atoms cannot have parent_id")
        if self.depth > 0 and self.parent_id is None:
            raise ValueError("non-root atoms require parent_id")
        if (self.branch_from_id is None) != (self.branch_id is None):
            raise ValueError("branch_from_id and branch_id must be provided together")
        referenced = {atom_id for group in self.dependencies for atom_id in group.atom_ids}
        if self.id in referenced | {self.parent_id, self.revision_of, self.branch_from_id}:
            raise ValueError("an atom cannot reference itself")
        return self


class AtomicityReason(ContractModel):
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    fragment: str | None = Field(default=None, min_length=1)


class AtomizationResult(ContractModel):
    decision: AtomicityDecision
    investigation_id: str
    atom: AtomicRequest
    children: tuple[AtomicRequest, ...] = ()
    reasons: tuple[AtomicityReason, ...] = Field(min_length=1)
    refinement_questions: tuple[str, ...] = ()
    next_atom_needed: StrictBool
    validation_basis: str = "STRUCTURAL"

    @model_validator(mode="after")
    def decision_shape_is_valid(self) -> Self:
        if self.decision is AtomicityDecision.ATOMIC:
            if self.children or self.refinement_questions:
                raise ValueError("ATOMIC cannot return children or refinement questions")
        elif self.decision is AtomicityDecision.DECOMPOSE:
            if len(self.children) < 2:
                raise ValueError("DECOMPOSE requires at least two children")
            if any(child.parent_id != self.atom.id for child in self.children):
                raise ValueError("every child must trace to the decomposed atom")
            if self.refinement_questions:
                raise ValueError("DECOMPOSE cannot return refinement questions")
        else:
            if self.children:
                raise ValueError("NEEDS_REFINEMENT cannot return children")
            if not self.refinement_questions:
                raise ValueError("NEEDS_REFINEMENT requires refinement questions")
        expected_next = self.decision is not AtomicityDecision.ATOMIC
        if self.next_atom_needed is not expected_next:
            raise ValueError("next_atom_needed must be false only for ATOMIC")
        return self


class EvidenceRecord(ContractModel):
    id: str = Field(default_factory=lambda: new_id("evidence"), min_length=1)
    atom_id: str = Field(min_length=1)
    direction: EvidenceDirection
    kind: EvidenceKind
    summary: str = Field(min_length=1)
    payload: JsonValue | None = None
    deterministic: StrictBool = False
    verification_method: VerificationMethod
    requirement_ids: tuple[str, ...] = ()
    provenance: Provenance
    assumptions: tuple[str, ...] = ()

    @model_validator(mode="after")
    def evidence_is_coherent(self) -> Self:
        if len(self.requirement_ids) != len(set(self.requirement_ids)):
            raise ValueError("requirement_ids must be unique")
        if self.deterministic and self.kind in {
            EvidenceKind.HUMAN_OBSERVATION,
            EvidenceKind.USER_PROVIDED,
            EvidenceKind.SENSOR_OUTPUT,
        }:
            raise ValueError(f"{self.kind} cannot be marked deterministic")
        if self.deterministic and self.assumptions:
            raise ValueError("evidence with assumptions cannot be marked deterministic")
        return self


class JevDispatchSelection(ContractModel):
    """The caller-selected, least-privilege state eligible for one Jev request."""

    context_keys: tuple[str, ...] = Field(default=(), max_length=8)
    evidence_ids: tuple[str, ...] = Field(default=(), max_length=8)
    include_context_provenance: StrictBool = False
    include_evidence_payloads: StrictBool = False
    include_assumptions: StrictBool = False
    include_evidence_provenance: StrictBool = False

    @model_validator(mode="after")
    def selectors_are_explicit_and_unique(self) -> Self:
        if len(self.context_keys) != len(set(self.context_keys)):
            raise ValueError("context_keys must be unique")
        if len(self.evidence_ids) != len(set(self.evidence_ids)):
            raise ValueError("evidence_ids must be unique")
        if not self.context_keys and not self.evidence_ids:
            raise ValueError("a Jev dispatch selection must select context or evidence")
        return self


class JevQuestion(ContractModel):
    """The typed question primitive that is safe to disclose to the semantic sensor."""

    question: str = Field(min_length=1)
    subject: str = Field(min_length=1)
    predicate: str = Field(min_length=1)
    scope: str = Field(min_length=1)
    operator: PredicateOperator
    quantifier: Quantifier
    expected_answer_type: ExpectedAnswerType
    answer_options: tuple[str, ...] = ()
    score_criteria: tuple[str, ...] = ()


class JevContextState(ContractModel):
    key: str = Field(min_length=1)
    value: JsonValue
    kind: ContextKind
    provenance: Provenance | None = None


class JevEvidenceState(ContractModel):
    direction: EvidenceDirection
    kind: EvidenceKind
    summary: str = Field(min_length=1)
    payload: JsonValue | None = None
    deterministic: StrictBool
    verification_method: VerificationMethod
    provenance: Provenance | None = None
    assumptions: tuple[str, ...] = ()


class JevState(ContractModel):
    """The exact JSON state argument supplied to TypeSafe/Jev."""

    claim: JevQuestion
    scope: str = Field(min_length=1)
    context: tuple[JevContextState, ...] = ()
    evidence: tuple[JevEvidenceState, ...] = ()


class JevNoulQuestion(ContractModel):
    type: Literal["noul"] = "noul"
    instructions: str = Field(min_length=1)


class JevChoiceQuestion(ContractModel):
    type: Literal["choice"] = "choice"
    instructions: str = Field(min_length=1)
    criteria: dict[str, str] = Field(min_length=2)


class JevScoreQuestion(ContractModel):
    type: Literal["score"] = "score"
    instructions: str = Field(min_length=1)
    criteria: tuple[str, ...] = Field(min_length=2)


JevProviderQuestion = JevNoulQuestion | JevChoiceQuestion | JevScoreQuestion


class JevQuestions(ContractModel):
    claim: JevProviderQuestion


class JevLogicalRequest(ContractModel):
    """The exact logical provider inputs: model, state, and question primitive."""

    model: str = Field(min_length=1)
    state: JevState
    questions: JevQuestions

    @model_validator(mode="after")
    def question_matches_state(self) -> Self:
        question = self.questions.claim
        claim = self.state.claim
        expected_type = {
            ExpectedAnswerType.BOOLEAN: "noul",
            ExpectedAnswerType.CHOICE: "choice",
            ExpectedAnswerType.SCORE: "score",
        }[claim.expected_answer_type]
        if question.type != expected_type or question.instructions != claim.question:
            raise ValueError("Jev question primitive must match the selected typed claim")
        if question.type == "choice" and question.criteria != {
            option: option for option in claim.answer_options
        }:
            raise ValueError("choice question criteria must match answer_options")
        if question.type == "score" and question.criteria != claim.score_criteria:
            raise ValueError("score question criteria must match score_criteria")
        return self


class JevDispatchManifest(ContractModel):
    """Canonical preview and approval record for a logical outbound request.

    A matching digest records the host/caller's approval assertion for this logical
    request. It intentionally does not prove a human-consent event.
    """

    schema_version: StrictInt = Field(default=1, ge=1)
    canonical_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    byte_length: StrictInt = Field(ge=1)
    logical_request: JevLogicalRequest
    selection: JevDispatchSelection
    approval_reference: str | None = Field(default=None, min_length=1)


class ObserveRequest(ContractModel):
    investigation_id: str = Field(min_length=1)
    atom_id: str = Field(min_length=1)
    evidence: tuple[EvidenceRecord, ...] = ()
    allow_external_jev: StrictBool = False
    dispatch_selection: JevDispatchSelection | None = None
    approved_dispatch_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    approval_reference: str | None = Field(default=None, min_length=1)
    satisfied_constraints: tuple[str, ...] = ()
    violated_constraints: tuple[str, ...] = ()
    missing_information: tuple[str, ...] = ()

    @model_validator(mode="after")
    def evidence_targets_atom(self) -> Self:
        if any(item.atom_id != self.atom_id for item in self.evidence):
            raise ValueError("all evidence must target atom_id")
        evidence_ids = [item.id for item in self.evidence]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("evidence ids must be unique within an observation")
        return self


class ParentImpact(ContractModel):
    affected_atom_ids: tuple[str, ...] = ()
    summary: str = Field(min_length=1)


class JEVResult(ContractModel):
    atom_id: str = Field(min_length=1)
    status: SemanticStatus
    evidence: tuple[EvidenceRecord, ...]
    missing_information: tuple[str, ...] = ()
    violated_constraints: tuple[str, ...] = ()
    satisfied_constraints: tuple[str, ...] = ()
    provenance: tuple[Provenance, ...] = Field(min_length=1)
    verification_method: VerificationMethod
    parent_impact: ParentImpact
    source: ResultSource
    dispatch_manifest: JevDispatchManifest | None = None

    @model_validator(mode="after")
    def status_has_required_support(self) -> Self:
        supporting = any(item.direction is EvidenceDirection.SUPPORTS for item in self.evidence)
        contradicting = any(
            item.direction is EvidenceDirection.CONTRADICTS for item in self.evidence
        )
        deterministic_contradiction = any(
            item.deterministic and item.direction is EvidenceDirection.CONTRADICTS
            for item in self.evidence
        )
        if self.status is SemanticStatus.VERIFIED:
            if not supporting:
                raise ValueError("VERIFIED requires supporting evidence")
            if deterministic_contradiction or self.violated_constraints or self.missing_information:
                raise ValueError(
                    "VERIFIED cannot contain deterministic contradictions or unresolved gaps"
                )
        elif self.status is SemanticStatus.CONTRADICTED:
            if not contradicting:
                raise ValueError("CONTRADICTED requires contradicting evidence")
        elif not self.missing_information:
            raise ValueError("UNKNOWN and INSUFFICIENT_EVIDENCE require missing_information")
        if self.status is SemanticStatus.INSUFFICIENT_EVIDENCE and not self.evidence:
            raise ValueError("INSUFFICIENT_EVIDENCE requires evaluated evidence")
        return self


class AtomRecord(ContractModel):
    atom: AtomicRequest
    atomicity: AtomicityDecision
    child_ids: tuple[str, ...] = ()
    replaced_by: str | None = Field(default=None, min_length=1)
    result: JEVResult | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def record_shape_is_valid(self) -> Self:
        if self.atomicity is AtomicityDecision.ATOMIC and self.child_ids:
            raise ValueError("atomic records cannot have child_ids")
        if self.atomicity is AtomicityDecision.DECOMPOSE and len(self.child_ids) < 2:
            raise ValueError("decomposed records require at least two child_ids")
        if self.atomicity is AtomicityDecision.NEEDS_REFINEMENT and self.child_ids:
            raise ValueError("refinement records cannot have invented children")
        if self.result is not None and self.atomicity is not AtomicityDecision.ATOMIC:
            raise ValueError("only atomic records may be observed directly")
        return self


class InvestigationState(ContractModel):
    investigation_id: str = Field(min_length=1)
    root_goal: str = Field(min_length=1)
    atoms: dict[str, AtomRecord] = Field(default_factory=dict)
    next_atom_number: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def state_is_relationally_valid(self) -> Self:
        atom_ids = set(self.atoms)
        numbers: set[int] = set()
        for atom_id, record in self.atoms.items():
            atom = record.atom
            if atom.id != atom_id:
                raise ValueError("atom map keys must match atom ids")
            if atom.investigation_id != self.investigation_id:
                raise ValueError("all atoms must belong to this investigation")
            if atom.root_goal != self.root_goal:
                raise ValueError("all atoms must share the investigation root_goal")
            if atom.atom_number in numbers:
                raise ValueError("atom_number must be unique within an investigation")
            numbers.add(atom.atom_number)
            if atom.parent_id is not None:
                parent = self.atoms.get(atom.parent_id)
                if parent is None:
                    raise ValueError("parent_id must exist in this investigation")
                if atom.depth != parent.atom.depth + 1:
                    raise ValueError("child depth must equal parent depth + 1")
            for reference in (atom.revision_of, atom.branch_from_id):
                if reference is not None and reference not in atom_ids:
                    raise ValueError("revision and branch references must exist")
            for group in atom.dependencies:
                if any(dep_id not in atom_ids for dep_id in group.atom_ids):
                    raise ValueError("dependency references must exist")
            if any(child_id not in atom_ids for child_id in record.child_ids):
                raise ValueError("child_ids must exist")
            if record.replaced_by is not None and record.replaced_by not in atom_ids:
                raise ValueError("replaced_by must exist")
        self._assert_acyclic_relationships()
        return self

    def _assert_acyclic_relationships(self) -> None:
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(atom_id: str) -> None:
            if atom_id in visiting:
                raise ValueError("dependency and replacement relationships must be acyclic")
            if atom_id in visited:
                return
            visiting.add(atom_id)
            record = self.atoms[atom_id]
            for group in record.atom.dependencies:
                for dependency_id in group.atom_ids:
                    visit(dependency_id)
            if record.replaced_by is not None:
                visit(record.replaced_by)
            visiting.remove(atom_id)
            visited.add(atom_id)

        for atom_id in self.atoms:
            visit(atom_id)


class AtomStateView(ContractModel):
    record: AtomRecord
    effective_status: SemanticStatus


class InvestigationView(ContractModel):
    investigation_id: str
    root_goal: str
    atoms: tuple[AtomStateView, ...]
    verified_atoms: tuple[str, ...]
    contradicted_atoms: tuple[str, ...]
    unknown_atoms: tuple[str, ...]
    insufficient_evidence_atoms: tuple[str, ...]
    unresolved_atoms: tuple[str, ...]
    contradictions: tuple[str, ...]
    current_frontier: tuple[str, ...]
    branch_ids: tuple[str, ...]
