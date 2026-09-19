"""Evidence coverage and semantic result derivation."""

from __future__ import annotations

from collections.abc import Mapping

from .models import (
    AtomicRequest,
    DependencyLogic,
    DependencyRole,
    EvidenceDirection,
    EvidenceRecord,
    EvidenceRequirement,
    JEVResult,
    ParentImpact,
    ResultSource,
    SemanticStatus,
    VerificationMethod,
)
from .sensor import SensorAssessment


class EvidenceEvaluator:
    """Combine deterministic instruments and a semantic sensor without fake precision."""

    def should_use_sensor(
        self,
        *,
        atom: AtomicRequest,
        evidence: tuple[EvidenceRecord, ...],
        dependency_statuses: Mapping[str, SemanticStatus],
    ) -> bool:
        """Return false when deterministic evidence already resolves the atom."""

        deterministic_contradiction = any(
            item.deterministic
            and item.direction is EvidenceDirection.CONTRADICTS
            and self._matches_any_requirement(atom, item)
            for item in evidence
        )
        if deterministic_contradiction:
            return False
        deterministic_support = any(
            item.deterministic
            and item.direction is EvidenceDirection.SUPPORTS
            and self._matches_any_requirement(atom, item)
            for item in evidence
        )
        return not (
            deterministic_support
            and not self._coverage_gaps(atom, evidence)
            and not self._prerequisite_gaps(atom, dependency_statuses)
        )

    def evaluate(
        self,
        *,
        atom: AtomicRequest,
        evidence: tuple[EvidenceRecord, ...],
        dependency_statuses: Mapping[str, SemanticStatus],
        sensor: SensorAssessment,
        parent_impact: ParentImpact,
        supplied_satisfied_constraints: tuple[str, ...] = (),
        supplied_violated_constraints: tuple[str, ...] = (),
        supplied_missing_information: tuple[str, ...] = (),
    ) -> JEVResult:
        self._validate_requirement_references(atom, evidence)

        combined_evidence = self._merge_evidence(evidence, sensor.evidence)
        satisfied = list(
            dict.fromkeys((*supplied_satisfied_constraints, *sensor.satisfied_constraints))
        )
        violated = list(
            dict.fromkeys((*supplied_violated_constraints, *sensor.violated_constraints))
        )
        missing = list(dict.fromkeys((*supplied_missing_information, *sensor.missing_information)))

        deterministic_contradictions = tuple(
            item
            for item in combined_evidence
            if item.deterministic
            and item.direction is EvidenceDirection.CONTRADICTS
            and self._matches_any_requirement(atom, item)
        )
        deterministic_support = tuple(
            item
            for item in combined_evidence
            if item.deterministic
            and item.direction is EvidenceDirection.SUPPORTS
            and self._matches_any_requirement(atom, item)
        )

        prerequisite_gaps = self._prerequisite_gaps(atom, dependency_statuses)
        if prerequisite_gaps:
            missing.extend(prerequisite_gaps)
            violated.append("required dependencies are not verified")

        coverage_gaps = self._coverage_gaps(atom, combined_evidence)
        if coverage_gaps:
            missing.extend(coverage_gaps)

        if deterministic_contradictions:
            status = SemanticStatus.CONTRADICTED
            source = ResultSource.DETERMINISTIC
        elif deterministic_support and not coverage_gaps and not prerequisite_gaps:
            status = SemanticStatus.VERIFIED
            source = ResultSource.DETERMINISTIC
            missing = []
        elif sensor.status is SemanticStatus.CONTRADICTED and any(
            item.direction is EvidenceDirection.CONTRADICTS
            and self._matches_any_requirement(atom, item)
            for item in sensor.evidence
        ):
            status = SemanticStatus.CONTRADICTED
            source = ResultSource.JEV
        elif (
            sensor.status is SemanticStatus.VERIFIED
            and any(
                item.direction is EvidenceDirection.SUPPORTS
                and self._matches_any_requirement(atom, item)
                for item in sensor.evidence
            )
            and not coverage_gaps
            and not prerequisite_gaps
            and not violated
        ):
            status = SemanticStatus.VERIFIED
            source = ResultSource.JEV if not deterministic_support else ResultSource.MIXED
            missing = []
        elif combined_evidence:
            status = SemanticStatus.INSUFFICIENT_EVIDENCE
            source = ResultSource.UNCONFIGURED if not sensor.evidence else ResultSource.MIXED
            if not missing:
                missing.append("available evidence does not resolve the predicate")
        else:
            status = SemanticStatus.UNKNOWN
            source = ResultSource.UNCONFIGURED
            if not missing:
                missing.append("no evidence is available")

        if status is SemanticStatus.VERIFIED:
            violated = []

        provenance = tuple(
            dict.fromkeys([*(item.provenance for item in combined_evidence), sensor.provenance])
        )
        methods = {item.verification_method for item in combined_evidence}
        verification_method = (
            next(iter(methods))
            if len(methods) == 1
            else atom.verification_method
            if not methods
            else VerificationMethod.COMPOSITE
        )

        return JEVResult(
            atom_id=atom.id,
            status=status,
            evidence=combined_evidence,
            missing_information=tuple(dict.fromkeys(missing)),
            violated_constraints=tuple(dict.fromkeys(violated)),
            satisfied_constraints=tuple(satisfied),
            provenance=provenance,
            verification_method=verification_method,
            parent_impact=parent_impact,
            source=source,
        )

    @staticmethod
    def _merge_evidence(
        existing: tuple[EvidenceRecord, ...],
        additional: tuple[EvidenceRecord, ...],
    ) -> tuple[EvidenceRecord, ...]:
        merged: dict[str, EvidenceRecord] = {}
        for item in (*existing, *additional):
            previous = merged.get(item.id)
            if previous is not None and previous != item:
                raise ValueError(f"evidence id {item.id!r} was reused with different content")
            merged[item.id] = item
        return tuple(merged.values())

    @staticmethod
    def _validate_requirement_references(
        atom: AtomicRequest,
        evidence: tuple[EvidenceRecord, ...],
    ) -> None:
        requirement_ids = {item.id for item in atom.evidence_requirements}
        for item in evidence:
            unknown = set(item.requirement_ids) - requirement_ids
            if unknown:
                raise ValueError(
                    f"evidence {item.id!r} references unknown requirements: {sorted(unknown)}"
                )

    @staticmethod
    def _coverage_gaps(
        atom: AtomicRequest,
        evidence: tuple[EvidenceRecord, ...],
    ) -> tuple[str, ...]:
        gaps: list[str] = []
        supporting = tuple(
            item for item in evidence if item.direction is EvidenceDirection.SUPPORTS
        )
        for requirement in atom.evidence_requirements:
            if not requirement.required:
                continue
            matched = [
                item
                for item in supporting
                if EvidenceEvaluator._matches_requirement(requirement, item)
            ]
            if len(matched) < requirement.minimum_items:
                gaps.append(
                    f"requirement {requirement.id!r} needs {requirement.minimum_items} "
                    f"supporting item(s); found {len(matched)}"
                )
        return tuple(gaps)

    @staticmethod
    def _matches_requirement(
        requirement: EvidenceRequirement,
        evidence: EvidenceRecord,
    ) -> bool:
        return (
            requirement.id in evidence.requirement_ids
            and (not requirement.accepted_kinds or evidence.kind in requirement.accepted_kinds)
            and (not requirement.deterministic_required or evidence.deterministic)
        )

    @classmethod
    def _matches_any_requirement(
        cls,
        atom: AtomicRequest,
        evidence: EvidenceRecord,
    ) -> bool:
        return any(
            cls._matches_requirement(requirement, evidence)
            for requirement in atom.evidence_requirements
        )

    @staticmethod
    def _prerequisite_gaps(
        atom: AtomicRequest,
        statuses: Mapping[str, SemanticStatus],
    ) -> tuple[str, ...]:
        gaps: list[str] = []
        for group in atom.dependencies:
            if group.role is not DependencyRole.PREREQUISITE:
                continue
            group_statuses = [statuses[atom_id] for atom_id in group.atom_ids]
            satisfied = (
                all(status is SemanticStatus.VERIFIED for status in group_statuses)
                if group.logic is DependencyLogic.ALL_OF
                else any(status is SemanticStatus.VERIFIED for status in group_statuses)
            )
            if not satisfied:
                gaps.append(f"prerequisite dependency group {group.id!r} is unresolved")
        return tuple(gaps)
