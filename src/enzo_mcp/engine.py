"""Transactional in-memory investigation engine for Enzo."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Mapping

from .atomicity import AtomicityAnalyzer
from .evidence import EvidenceEvaluator
from .models import (
    AtomicityDecision,
    AtomicRequest,
    AtomizationResult,
    AtomizeRequest,
    AtomRecord,
    AtomStateView,
    ClaimSpec,
    DependencyGroup,
    DependencyLogic,
    DependencyRole,
    EvidenceKind,
    EvidenceRecord,
    InvestigationState,
    InvestigationView,
    JevChoiceQuestion,
    JevContextState,
    JevDispatchManifest,
    JevDispatchSelection,
    JevEvidenceState,
    JevLogicalRequest,
    JevNoulQuestion,
    JevProviderQuestion,
    JevQuestion,
    JevQuestions,
    JEVResult,
    JevScoreQuestion,
    JevState,
    ObserveRequest,
    ParentImpact,
    ProposedAtom,
    Provenance,
    ResultSource,
    SemanticStatus,
    new_id,
)
from .sensor import SemanticSensor, SensorAssessment, default_sensor

_CLAIM_FIELDS = (
    "question",
    "subject",
    "predicate",
    "scope",
    "operator",
    "expected_value",
    "quantifier",
    "expected_answer_type",
    "answer_options",
    "score_criteria",
    "operational_definition",
    "evidence_requirements",
    "verification_method",
)
_MAX_JEV_CONTEXT_ENTRIES = 8
_MAX_JEV_EVIDENCE_ENTRIES = 8
_MAX_JEV_SELECTED_VALUE_BYTES = 4 * 1024
_MAX_JEV_CANONICAL_REQUEST_BYTES = 16 * 1024
_JEV_MANIFEST_SCHEMA_VERSION = 1


def _claim_values(spec: ClaimSpec) -> dict[str, object]:
    return {field: getattr(spec, field) for field in _CLAIM_FIELDS}


class EnzoEngine:
    """Own atomization, evidence observation, and derived investigation state."""

    def __init__(
        self,
        *,
        sensor: SemanticSensor | None = None,
        max_depth: int = 8,
    ) -> None:
        self._sensor = sensor if sensor is not None else default_sensor()
        self._atomizer = AtomicityAnalyzer(max_depth=max_depth)
        self._evaluator = EvidenceEvaluator()
        self._states: dict[str, InvestigationState] = {}
        self._atomize_requests: dict[str, AtomizeRequest] = {}
        self._atomize_results: dict[str, AtomizationResult] = {}
        self._observation_replays: dict[
            tuple[str, str],
            tuple[str, tuple[tuple[str, SemanticStatus], ...]],
        ] = {}
        self._jev_dispatch_assessments: dict[
            tuple[str, str, str, str | None], SensorAssessment
        ] = {}
        self._lock = asyncio.Lock()

    async def atomize(self, request: AtomizeRequest) -> AtomizationResult:
        """Admit, decompose, or request refinement for one proposed claim."""

        async with self._lock:
            replay = self._atomize_results.get(request.atom_id)
            if replay is not None:
                if self._atomize_requests[request.atom_id] != request:
                    raise ValueError(
                        f"atom id {request.atom_id!r} was reused with different content"
                    )
                return replay

            investigation_id = request.investigation_id or new_id("investigation")
            state = self._states.get(investigation_id)
            if state is None:
                if request.depth != 0 or request.parent_id is not None:
                    raise ValueError("a new investigation must begin with a root atom")
                state = InvestigationState(
                    investigation_id=investigation_id,
                    root_goal=request.root_goal,
                )
            elif state.root_goal != request.root_goal:
                raise ValueError("root_goal cannot change within an investigation")

            self._validate_references(state, request)
            parent_atom = self._build_atom(
                request,
                investigation_id=investigation_id,
                atom_number=state.next_atom_number,
            )
            plan = self._atomizer.assess(parent_atom, request.proposed_children)

            children: tuple[AtomicRequest, ...] = ()
            if plan.decision is AtomicityDecision.DECOMPOSE:
                children = tuple(
                    self._build_child(
                        parent_atom,
                        child,
                        atom_number=state.next_atom_number + index,
                    )
                    for index, child in enumerate(plan.children, start=1)
                )
                definition = DependencyGroup(
                    id=f"decomposition_{parent_atom.id}",
                    role=DependencyRole.DEFINES_PARENT,
                    logic=plan.dependency_logic,
                    atom_ids=tuple(child.id for child in children),
                    complete=True,
                )
                parent_atom = AtomicRequest.model_validate(
                    {
                        **parent_atom.model_dump(mode="python"),
                        "dependencies": (*parent_atom.dependencies, definition),
                    }
                )

            result = AtomizationResult(
                decision=plan.decision,
                investigation_id=investigation_id,
                atom=parent_atom,
                children=children,
                reasons=plan.reasons,
                refinement_questions=plan.refinement_questions,
                next_atom_needed=plan.decision is not AtomicityDecision.ATOMIC,
            )

            new_atoms = dict(state.atoms)
            new_atoms[parent_atom.id] = AtomRecord(
                atom=parent_atom,
                atomicity=plan.decision,
                child_ids=tuple(child.id for child in children),
            )
            for child in children:
                new_atoms[child.id] = AtomRecord(
                    atom=child,
                    atomicity=AtomicityDecision.ATOMIC,
                )

            if parent_atom.revision_of is not None:
                for replaced_id in self._subtree_ids(state, parent_atom.revision_of):
                    revised = new_atoms[replaced_id]
                    if revised.replaced_by is not None:
                        continue
                    new_atoms[replaced_id] = AtomRecord(
                        atom=revised.atom,
                        atomicity=revised.atomicity,
                        child_ids=revised.child_ids,
                        replaced_by=parent_atom.id,
                        result=revised.result,
                        created_at=revised.created_at,
                    )

            next_number = state.next_atom_number + 1 + len(children)
            next_state = InvestigationState(
                investigation_id=state.investigation_id,
                root_goal=state.root_goal,
                atoms=new_atoms,
                next_atom_number=next_number,
            )
            self._states[investigation_id] = next_state
            self._atomize_requests[request.atom_id] = request
            self._atomize_results[request.atom_id] = result
            return result

    async def observe(self, request: ObserveRequest) -> JEVResult:
        """Evaluate typed evidence for one atomic leaf and update its state."""

        async with self._lock:
            state = self._require_state(request.investigation_id)
            record = state.atoms.get(request.atom_id)
            if record is None:
                raise ValueError(f"unknown atom {request.atom_id!r}")
            if record.atomicity is not AtomicityDecision.ATOMIC:
                raise ValueError("only an ATOMIC leaf may be observed directly")
            if record.replaced_by is not None:
                raise ValueError("a superseded atom cannot receive new observations")
            statuses = self._effective_statuses(state)
            dependency_statuses = {
                atom_id: statuses[atom_id]
                for group in record.atom.dependencies
                for atom_id in group.atom_ids
            }
            dependency_fingerprint = tuple(sorted(dependency_statuses.items()))
            request_fingerprint = self._observation_fingerprint(request)
            replay_key = (request.investigation_id, request.atom_id)
            if (
                record.result is not None
                and self._is_stable_result(record.result)
                and self._observation_replays.get(replay_key)
                == (request_fingerprint, dependency_fingerprint)
            ):
                return record.result

            prior_evidence = record.result.evidence if record.result is not None else ()
            evidence = self._merge_evidence(prior_evidence, request.evidence)

            should_use_sensor = self._evaluator.should_use_sensor(
                atom=record.atom,
                evidence=evidence,
                dependency_statuses=dependency_statuses,
            )
            dispatch_manifest: JevDispatchManifest | None = None
            if should_use_sensor and request.dispatch_selection is not None:
                try:
                    dispatch_manifest = self._prepare_jev_manifest(
                        atom=record.atom,
                        evidence=evidence,
                        selection=request.dispatch_selection,
                        approval_reference=request.approval_reference,
                    )
                except ValueError as error:
                    sensor_assessment = self._dispatch_blocked_assessment(str(error))
                else:
                    if not request.allow_external_jev:
                        sensor_assessment = self._dispatch_blocked_assessment(
                            "external Jev evaluation requires allow_external_jev=true"
                        )
                    elif request.approved_dispatch_sha256 != dispatch_manifest.canonical_sha256:
                        sensor_assessment = self._dispatch_blocked_assessment(
                            "external Jev evaluation requires a matching approved_dispatch_sha256"
                        )
                    else:
                        dispatch_key = (
                            request.investigation_id,
                            request.atom_id,
                            dispatch_manifest.canonical_sha256,
                            request.approval_reference,
                        )
                        prior_assessment = self._jev_dispatch_assessments.get(dispatch_key)
                        if prior_assessment is None:
                            # Reserve the approval before I/O so an unexpected exception cannot
                            # make the same in-process assertion spendable twice.
                            sensor_assessment = self._dispatch_attempted_assessment()
                            self._jev_dispatch_assessments[dispatch_key] = sensor_assessment
                            sensor_assessment = await self._sensor.evaluate(
                                record.atom, dispatch_manifest
                            )
                            self._jev_dispatch_assessments[dispatch_key] = sensor_assessment
                        else:
                            sensor_assessment = prior_assessment
            elif should_use_sensor:
                sensor_assessment = self._dispatch_blocked_assessment(
                    "external Jev evaluation requires an explicit dispatch_selection"
                )
            else:
                sensor_assessment = SensorAssessment(
                    status=SemanticStatus.UNKNOWN,
                    provenance=Provenance(
                        provider="enzo",
                        locator="sensor:skipped:deterministic-result",
                        tool="EvidenceEvaluator",
                    ),
                    explanation=(
                        "The semantic sensor was skipped because deterministic evidence "
                        "already resolved the atom."
                    ),
                )

            affected = self._affected_ancestors(state, record.atom.id)
            result = self._evaluator.evaluate(
                atom=record.atom,
                evidence=evidence,
                dependency_statuses=dependency_statuses,
                sensor=sensor_assessment,
                parent_impact=ParentImpact(
                    affected_atom_ids=affected,
                    summary=(
                        "Ancestor and dependent conclusions are recomputed from this result."
                        if affected
                        else "This atom currently has no parent or dependent conclusion."
                    ),
                ),
                supplied_satisfied_constraints=request.satisfied_constraints,
                supplied_violated_constraints=request.violated_constraints,
                supplied_missing_information=request.missing_information,
                dispatch_manifest=dispatch_manifest,
            )

            new_atoms = dict(state.atoms)
            new_atoms[record.atom.id] = AtomRecord(
                atom=record.atom,
                atomicity=record.atomicity,
                child_ids=record.child_ids,
                replaced_by=record.replaced_by,
                result=result,
                created_at=record.created_at,
            )
            self._states[state.investigation_id] = InvestigationState(
                investigation_id=state.investigation_id,
                root_goal=state.root_goal,
                atoms=new_atoms,
                next_atom_number=state.next_atom_number,
            )
            if self._is_stable_result(result):
                self._observation_replays[replay_key] = (
                    request_fingerprint,
                    dependency_fingerprint,
                )
            else:
                self._observation_replays.pop(replay_key, None)
            return result

    async def state(self, investigation_id: str) -> InvestigationView:
        """Return canonical records plus categories computed from current evidence."""

        async with self._lock:
            state = self._require_state(investigation_id)
            statuses = self._effective_statuses(state)
            ordered = tuple(sorted(state.atoms.values(), key=lambda item: item.atom.atom_number))
            active_ids = tuple(record.atom.id for record in ordered if record.replaced_by is None)

            def ids_with(status: SemanticStatus) -> tuple[str, ...]:
                return tuple(atom_id for atom_id in active_ids if statuses[atom_id] is status)

            unresolved = tuple(
                atom_id
                for atom_id in active_ids
                if statuses[atom_id]
                in {SemanticStatus.UNKNOWN, SemanticStatus.INSUFFICIENT_EVIDENCE}
            )
            frontier = tuple(
                atom_id for atom_id in unresolved if self._is_frontier(state, atom_id, statuses)
            )
            branches = tuple(
                sorted(
                    {
                        record.atom.branch_id
                        for record in ordered
                        if record.atom.branch_id is not None
                    }
                )
            )
            contradicted = ids_with(SemanticStatus.CONTRADICTED)
            return InvestigationView(
                investigation_id=state.investigation_id,
                root_goal=state.root_goal,
                atoms=tuple(
                    AtomStateView(record=record, effective_status=statuses[record.atom.id])
                    for record in ordered
                ),
                verified_atoms=ids_with(SemanticStatus.VERIFIED),
                contradicted_atoms=contradicted,
                unknown_atoms=ids_with(SemanticStatus.UNKNOWN),
                insufficient_evidence_atoms=ids_with(SemanticStatus.INSUFFICIENT_EVIDENCE),
                unresolved_atoms=unresolved,
                contradictions=contradicted,
                current_frontier=frontier,
                branch_ids=branches,
            )

    def _require_state(self, investigation_id: str) -> InvestigationState:
        try:
            return self._states[investigation_id]
        except KeyError as error:
            raise ValueError(f"unknown investigation {investigation_id!r}") from error

    @staticmethod
    def _build_atom(
        request: AtomizeRequest,
        *,
        investigation_id: str,
        atom_number: int,
    ) -> AtomicRequest:
        return AtomicRequest.model_validate(
            {
                **_claim_values(request),
                "id": request.atom_id,
                "investigation_id": investigation_id,
                "atom_number": atom_number,
                "parent_id": request.parent_id,
                "root_goal": request.root_goal,
                "context": request.context,
                "dependencies": request.dependencies,
                "depth": request.depth,
                "revision_of": request.revision_of,
                "branch_from_id": request.branch_from_id,
                "branch_id": request.branch_id,
            }
        )

    @staticmethod
    def _build_child(
        parent: AtomicRequest,
        proposal: ProposedAtom,
        *,
        atom_number: int,
    ) -> AtomicRequest:
        return AtomicRequest.model_validate(
            {
                **_claim_values(proposal),
                "id": new_id("atom"),
                "investigation_id": parent.investigation_id,
                "atom_number": atom_number,
                "parent_id": parent.id,
                "root_goal": parent.root_goal,
                "context": parent.context,
                "dependencies": (),
                "depth": parent.depth + 1,
                "branch_from_id": parent.branch_from_id,
                "branch_id": parent.branch_id,
            }
        )

    @staticmethod
    def _validate_references(state: InvestigationState, request: AtomizeRequest) -> None:
        if request.atom_id in state.atoms:
            raise ValueError(f"atom {request.atom_id!r} already exists")
        references = {
            item
            for item in (request.parent_id, request.revision_of, request.branch_from_id)
            if item is not None
        }
        references.update(atom_id for group in request.dependencies for atom_id in group.atom_ids)
        unknown = references - set(state.atoms)
        if unknown:
            raise ValueError(f"references do not exist in this investigation: {sorted(unknown)}")

        if request.parent_id is not None:
            parent = state.atoms[request.parent_id]
            if parent.replaced_by is not None:
                raise ValueError("cannot add a child to a superseded parent")
            if request.depth != parent.atom.depth + 1:
                raise ValueError("depth must equal parent depth + 1")
        if request.revision_of is not None:
            revised = state.atoms[request.revision_of]
            if revised.replaced_by is not None:
                raise ValueError("cannot revise an atom that is already superseded")
            if request.parent_id != revised.atom.parent_id or request.depth != revised.atom.depth:
                raise ValueError("a revision must preserve parent and depth")

    @staticmethod
    def _merge_evidence(
        existing: tuple[EvidenceRecord, ...],
        additional: tuple[EvidenceRecord, ...],
    ) -> tuple[EvidenceRecord, ...]:
        merged: dict[str, EvidenceRecord] = {item.id: item for item in existing}
        for item in additional:
            previous = merged.get(item.id)
            if previous is not None:
                if EnzoEngine._evidence_fingerprint(previous) != EnzoEngine._evidence_fingerprint(
                    item
                ):
                    raise ValueError(f"evidence id {item.id!r} was reused with different content")
                continue
            merged[item.id] = item
        return tuple(merged.values())

    def _prepare_jev_manifest(
        self,
        *,
        atom: AtomicRequest,
        evidence: tuple[EvidenceRecord, ...],
        selection: JevDispatchSelection,
        approval_reference: str | None,
    ) -> JevDispatchManifest:
        """Project selected state, enforce outbound limits, and canonicalize it.

        The digest is an approval check for this logical request only. It does not
        imply that a particular person reviewed or consented to the dispatch.
        """

        if len(selection.context_keys) > _MAX_JEV_CONTEXT_ENTRIES:
            raise ValueError("Jev dispatch exceeds the maximum of 8 context entries")
        if len(selection.evidence_ids) > _MAX_JEV_EVIDENCE_ENTRIES:
            raise ValueError("Jev dispatch exceeds the maximum of 8 evidence entries")

        context_by_key = {item.key: item for item in atom.context}
        if len(context_by_key) != len(atom.context):
            raise ValueError("Jev dispatch cannot select ambiguous duplicate context keys")
        unknown_context = set(selection.context_keys) - set(context_by_key)
        if unknown_context:
            raise ValueError(
                f"Jev dispatch selects unknown context keys: {sorted(unknown_context)!r}"
            )

        evidence_by_id = {item.id: item for item in evidence}
        unknown_evidence = set(selection.evidence_ids) - set(evidence_by_id)
        if unknown_evidence:
            raise ValueError(
                f"Jev dispatch selects unknown evidence ids: {sorted(unknown_evidence)!r}"
            )

        selected_context: list[JevContextState] = []
        for key in selection.context_keys:
            context_item = context_by_key[key]
            self._require_jev_size(context_item.value, "selected context value")
            selected_context.append(
                JevContextState(
                    key=context_item.key,
                    value=context_item.value,
                    kind=context_item.kind,
                    provenance=(
                        context_item.provenance if selection.include_context_provenance else None
                    ),
                )
            )

        selected_evidence: list[JevEvidenceState] = []
        for evidence_id in selection.evidence_ids:
            evidence_item = evidence_by_id[evidence_id]
            if evidence_item.kind is EvidenceKind.SENSOR_OUTPUT:
                raise ValueError("Jev dispatch cannot include prior SENSOR_OUTPUT evidence")
            outbound_evidence = JevEvidenceState(
                direction=evidence_item.direction,
                kind=evidence_item.kind,
                summary=evidence_item.summary,
                payload=(evidence_item.payload if selection.include_evidence_payloads else None),
                deterministic=evidence_item.deterministic,
                verification_method=evidence_item.verification_method,
                provenance=(
                    evidence_item.provenance if selection.include_evidence_provenance else None
                ),
                assumptions=(evidence_item.assumptions if selection.include_assumptions else ()),
            )
            self._require_jev_size(
                outbound_evidence.model_dump(mode="json"), "selected evidence record"
            )
            selected_evidence.append(outbound_evidence)

        logical_request = JevLogicalRequest(
            model=self._jev_model_name(),
            state=JevState(
                claim=JevQuestion(
                    question=atom.question,
                    subject=atom.subject,
                    predicate=atom.predicate,
                    scope=atom.scope,
                    operator=atom.operator,
                    quantifier=atom.quantifier,
                    expected_answer_type=atom.expected_answer_type,
                    answer_options=atom.answer_options,
                    score_criteria=atom.score_criteria,
                ),
                scope=atom.scope,
                context=tuple(selected_context),
                evidence=tuple(selected_evidence),
            ),
            questions=JevQuestions(claim=self._jev_provider_question(atom)),
        )
        canonical_bytes = self._canonical_json_bytes(logical_request.model_dump(mode="json"))
        if len(canonical_bytes) > _MAX_JEV_CANONICAL_REQUEST_BYTES:
            raise ValueError("Jev dispatch exceeds the 16KiB canonical request limit")
        return JevDispatchManifest(
            schema_version=_JEV_MANIFEST_SCHEMA_VERSION,
            canonical_sha256=hashlib.sha256(canonical_bytes).hexdigest(),
            byte_length=len(canonical_bytes),
            logical_request=logical_request,
            selection=selection,
            approval_reference=approval_reference,
        )

    @staticmethod
    def _jev_provider_question(atom: AtomicRequest) -> JevProviderQuestion:
        if atom.expected_answer_type.value == "BOOLEAN":
            return JevNoulQuestion(instructions=atom.question)
        if atom.expected_answer_type.value == "CHOICE":
            return JevChoiceQuestion(
                instructions=atom.question,
                criteria={option: option for option in atom.answer_options},
            )
        return JevScoreQuestion(
            instructions=atom.question,
            criteria=atom.score_criteria,
        )

    def _jev_model_name(self) -> str:
        model_name = getattr(self._sensor, "model_name", None)
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("Jev sensor does not expose a configured model name")
        return model_name

    @staticmethod
    def _canonical_json_bytes(value: object) -> bytes:
        try:
            return json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Jev dispatch state cannot be represented as canonical JSON"
            ) from error

    @classmethod
    def _require_jev_size(cls, value: object, description: str) -> None:
        if len(cls._canonical_json_bytes(value)) > _MAX_JEV_SELECTED_VALUE_BYTES:
            raise ValueError(f"Jev dispatch {description} exceeds the 4KiB limit")

    @staticmethod
    def _dispatch_blocked_assessment(message: str) -> SensorAssessment:
        return SensorAssessment(
            status=SemanticStatus.UNKNOWN,
            missing_information=(message,),
            provenance=Provenance(
                provider="enzo",
                locator="sensor:jev:external-send-not-authorized",
                tool="EnzoEngine",
            ),
            explanation=(
                "No context or evidence was transmitted to TypeSafe/Jev because the outbound "
                "dispatch policy was not satisfied."
            ),
        )

    @staticmethod
    def _dispatch_attempted_assessment() -> SensorAssessment:
        return SensorAssessment(
            status=SemanticStatus.UNKNOWN,
            missing_information=(
                "this manifest approval was already used for an external Jev attempt",
            ),
            provenance=Provenance(
                provider="enzo",
                locator="sensor:jev:dispatch-already-attempted",
                tool="EnzoEngine",
            ),
            explanation=(
                "Enzo did not repeat an external Jev call for the same manifest hash and "
                "approval reference. Use a new approval_reference for an intentional retry."
            ),
        )

    @staticmethod
    def _is_stable_result(result: JEVResult) -> bool:
        """Return whether an exact retry must reuse this observation result.

        A manifest means the outbound decision was fully determined. Reusing the
        result prevents a transport retry from duplicating a Jev call even when the
        provider returned no usable sensor evidence or raised a handled error.
        """

        return (
            result.source is ResultSource.DETERMINISTIC
            or result.dispatch_manifest is not None
            or any(item.kind is EvidenceKind.SENSOR_OUTPUT for item in result.evidence)
        )

    @staticmethod
    def _observation_fingerprint(request: ObserveRequest) -> str:
        """Normalize server-generated provenance timestamps for transport retries."""

        payload = request.model_dump(mode="json")
        for item in payload["evidence"]:
            item["provenance"].pop("observed_at", None)
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def _evidence_fingerprint(evidence: EvidenceRecord) -> str:
        """Compare evidence identity without its server-generated observation time."""

        payload = evidence.model_dump(mode="json")
        payload["provenance"].pop("observed_at", None)
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def _subtree_ids(state: InvestigationState, root_id: str) -> tuple[str, ...]:
        """Return the revision root and every atom whose parent chain reaches it."""

        subtree = {root_id}
        changed = True
        while changed:
            changed = False
            for atom_id, record in state.atoms.items():
                if atom_id not in subtree and record.atom.parent_id in subtree:
                    subtree.add(atom_id)
                    changed = True
        return tuple(subtree)

    def _effective_statuses(
        self,
        state: InvestigationState,
    ) -> dict[str, SemanticStatus]:
        memo: dict[str, SemanticStatus] = {}

        def resolve(atom_id: str) -> SemanticStatus:
            if atom_id in memo:
                return memo[atom_id]
            record = state.atoms[atom_id]
            if record.replaced_by is not None:
                memo[atom_id] = resolve(record.replaced_by)
                return memo[atom_id]

            definition_statuses: list[SemanticStatus] = []
            for group in record.atom.dependencies:
                if group.role is not DependencyRole.DEFINES_PARENT or not group.complete:
                    continue
                children = [resolve(child_id) for child_id in group.atom_ids]
                if group.logic is DependencyLogic.ALL_OF:
                    if any(item is SemanticStatus.CONTRADICTED for item in children):
                        group_status = SemanticStatus.CONTRADICTED
                    elif all(item is SemanticStatus.VERIFIED for item in children):
                        group_status = SemanticStatus.VERIFIED
                    elif any(item is SemanticStatus.INSUFFICIENT_EVIDENCE for item in children):
                        group_status = SemanticStatus.INSUFFICIENT_EVIDENCE
                    else:
                        group_status = SemanticStatus.UNKNOWN
                else:
                    if any(item is SemanticStatus.VERIFIED for item in children):
                        group_status = SemanticStatus.VERIFIED
                    elif all(item is SemanticStatus.CONTRADICTED for item in children):
                        group_status = SemanticStatus.CONTRADICTED
                    elif any(item is SemanticStatus.INSUFFICIENT_EVIDENCE for item in children):
                        group_status = SemanticStatus.INSUFFICIENT_EVIDENCE
                    else:
                        group_status = SemanticStatus.UNKNOWN
                definition_statuses.append(group_status)

            if definition_statuses:
                if any(item is SemanticStatus.CONTRADICTED for item in definition_statuses):
                    status = SemanticStatus.CONTRADICTED
                elif all(item is SemanticStatus.VERIFIED for item in definition_statuses):
                    status = SemanticStatus.VERIFIED
                elif any(
                    item is SemanticStatus.INSUFFICIENT_EVIDENCE for item in definition_statuses
                ):
                    status = SemanticStatus.INSUFFICIENT_EVIDENCE
                else:
                    status = SemanticStatus.UNKNOWN
            elif record.result is not None:
                status = record.result.status
            else:
                status = SemanticStatus.UNKNOWN

            if status is SemanticStatus.VERIFIED and not self._prerequisites_satisfied(
                record.atom,
                {
                    dependency_id: resolve(dependency_id)
                    for dependency_id in self._dependency_ids(record.atom)
                },
            ):
                status = SemanticStatus.UNKNOWN
            memo[atom_id] = status
            return status

        for atom_id in state.atoms:
            resolve(atom_id)
        return memo

    @staticmethod
    def _dependency_ids(atom: AtomicRequest) -> tuple[str, ...]:
        return tuple(
            dependency_id
            for group in atom.dependencies
            if group.role is DependencyRole.PREREQUISITE
            for dependency_id in group.atom_ids
        )

    @staticmethod
    def _prerequisites_satisfied(
        atom: AtomicRequest,
        statuses: Mapping[str, SemanticStatus],
    ) -> bool:
        for group in atom.dependencies:
            if group.role is not DependencyRole.PREREQUISITE:
                continue
            values = [statuses[dependency_id] for dependency_id in group.atom_ids]
            if group.logic is DependencyLogic.ALL_OF:
                if not all(value is SemanticStatus.VERIFIED for value in values):
                    return False
            elif not any(value is SemanticStatus.VERIFIED for value in values):
                return False
        return True

    def _is_frontier(
        self,
        state: InvestigationState,
        atom_id: str,
        statuses: Mapping[str, SemanticStatus],
    ) -> bool:
        record = state.atoms[atom_id]
        if record.atomicity is AtomicityDecision.NEEDS_REFINEMENT:
            return True
        if record.atomicity is not AtomicityDecision.ATOMIC:
            return False
        dependency_statuses = {
            dependency_id: statuses[dependency_id]
            for dependency_id in self._dependency_ids(record.atom)
        }
        return self._prerequisites_satisfied(record.atom, dependency_statuses)

    @staticmethod
    def _affected_ancestors(
        state: InvestigationState,
        atom_id: str,
    ) -> tuple[str, ...]:
        reverse_edges: dict[str, set[str]] = {key: set() for key in state.atoms}
        for candidate_id, record in state.atoms.items():
            if record.atom.parent_id is not None:
                reverse_edges[candidate_id].add(record.atom.parent_id)
            for group in record.atom.dependencies:
                for dependency_id in group.atom_ids:
                    reverse_edges[dependency_id].add(candidate_id)

        affected: set[str] = set()
        frontier = list(reverse_edges[atom_id])
        while frontier:
            candidate = frontier.pop()
            if candidate in affected:
                continue
            affected.add(candidate)
            frontier.extend(reverse_edges[candidate])
        return tuple(sorted(affected, key=lambda item: state.atoms[item].atom.atom_number))
