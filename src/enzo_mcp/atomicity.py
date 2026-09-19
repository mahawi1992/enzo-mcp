"""Conservative structural atomicity checks.

This module deliberately does not claim semantic intelligence. It admits a claim
only when one falsifiable predicate is structurally explicit, decomposes only safe
lists, and otherwise asks the caller (normally an LLM) for a refinement.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .models import (
    AtomicityDecision,
    AtomicityReason,
    AtomicRequest,
    DependencyLogic,
    EvidenceRequirement,
    ProposedAtom,
)

_BROAD_TERMS = re.compile(
    r"\b(secure|safe|scalable|production[- ]ready|robust|correct|good|"
    r"reliable|properly|works?|high[- ]quality)\b",
    flags=re.IGNORECASE,
)
_AMBIGUOUS_CONNECTOR = re.compile(r"\b(?:and|or)\b", flags=re.IGNORECASE)
_ENUM_ALTERNATIVE = re.compile(
    r"\b[A-Z][A-Z0-9_]*\s+or\s+[A-Z][A-Z0-9_]*"
    r"(?=\s+(?i:labels?|kinds?|types?|values?|statuses|modes?|options?)\b)"
)
_SAFE_SEPARATORS = re.compile(r"\s*(?:;|&&|\|\|)\s*")
_COPULA_LIST = re.compile(
    r"^(?P<prefix>is|are|was|were|has|have|can|does|do|contains?|supports?|enforces?)\s+"
    r"(?P<body>.+)$",
    flags=re.IGNORECASE,
)
_NUMBER = r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
_BETWEEN_RANGE = re.compile(
    rf"^(?P<prefix>.*?)\bbetween\s+{_NUMBER}\s+and\s+{_NUMBER}(?P<suffix>.*?)$",
    flags=re.IGNORECASE,
)
_LOGICAL_CONNECTOR = re.compile(r"\b(?:and|or)\b|;|&&|\|\|", flags=re.IGNORECASE)


def _is_single_numeric_range(predicate: str) -> bool:
    match = _BETWEEN_RANGE.fullmatch(predicate.strip())
    if match is None:
        return False
    outside_range = f"{match.group('prefix')} {match.group('suffix')}"
    return _LOGICAL_CONNECTOR.search(outside_range) is None


def _without_enum_alternatives(predicate: str) -> str:
    """Mask typed enum alternatives that name one value domain, not two claims."""

    return _ENUM_ALTERNATIVE.sub("ENUM_VALUE", predicate)


@dataclass(frozen=True, slots=True)
class AtomicityPlan:
    decision: AtomicityDecision
    children: tuple[ProposedAtom, ...]
    reasons: tuple[AtomicityReason, ...]
    refinement_questions: tuple[str, ...] = ()
    dependency_logic: DependencyLogic = DependencyLogic.ALL_OF


class AtomicityAnalyzer:
    """Apply deterministic decomposition pressure without inventing semantics."""

    def __init__(self, *, max_depth: int = 8) -> None:
        if max_depth < 1:
            raise ValueError("max_depth must be at least 1")
        self.max_depth = max_depth

    def assess(
        self,
        atom: AtomicRequest,
        proposed_children: tuple[ProposedAtom, ...] = (),
    ) -> AtomicityPlan:
        diagnostics = self._diagnose(atom.predicate, atom.operational_definition)

        if proposed_children:
            if atom.depth >= self.max_depth:
                return self._limit_plan(atom)
            if self._has_duplicate_children(proposed_children):
                return AtomicityPlan(
                    decision=AtomicityDecision.NEEDS_REFINEMENT,
                    children=(),
                    reasons=(
                        AtomicityReason(
                            code="duplicate_proposed_children",
                            message=(
                                "Two or more proposed children express the same semantic claim."
                            ),
                        ),
                    ),
                    refinement_questions=(
                        "Remove duplicate children or replace them with independently "
                        "falsifiable claims.",
                    ),
                )
            child_gaps = [(child, self._child_diagnostic(child)) for child in proposed_children]
            invalid = [(child, gap) for child, gap in child_gaps if gap is not None]
            if invalid:
                details = tuple(
                    AtomicityReason(
                        code="child_not_atomic",
                        message=f"Proposed child needs refinement: {gap}",
                        fragment=child.predicate,
                    )
                    for child, gap in invalid
                )
                return AtomicityPlan(
                    decision=AtomicityDecision.NEEDS_REFINEMENT,
                    children=(),
                    reasons=details,
                    refinement_questions=(
                        "Rewrite each proposed child as one operationalized predicate.",
                    ),
                )
            dependency_logic = self._decomposition_logic(atom.predicate)
            if dependency_logic is None:
                return AtomicityPlan(
                    decision=AtomicityDecision.NEEDS_REFINEMENT,
                    children=(),
                    reasons=(
                        AtomicityReason(
                            code="mixed_dependency_logic",
                            message=(
                                "The parent mixes conjunction and disjunction that cannot be "
                                "represented by one dependency group."
                            ),
                            fragment=atom.predicate,
                        ),
                    ),
                    refinement_questions=(
                        "Rewrite the parent or create nested atoms with one logical operator each.",
                    ),
                )
            return AtomicityPlan(
                decision=AtomicityDecision.DECOMPOSE,
                children=proposed_children,
                reasons=(
                    AtomicityReason(
                        code="caller_supplied_independent_claims",
                        message=(
                            "The caller supplied multiple independently testable child claims."
                        ),
                    ),
                ),
                dependency_logic=dependency_logic,
            )

        if diagnostics is not None and diagnostics[0] == "unoperationalized_predicate":
            return AtomicityPlan(
                decision=AtomicityDecision.NEEDS_REFINEMENT,
                children=(),
                reasons=(
                    AtomicityReason(
                        code=diagnostics[0],
                        message=diagnostics[1],
                        fragment=atom.predicate,
                    ),
                ),
                refinement_questions=(diagnostics[2],),
            )

        safe_fragments = self._safe_fragments(atom.predicate)
        if len(safe_fragments) >= 2:
            if atom.depth >= self.max_depth:
                return self._limit_plan(atom)
            if len(atom.evidence_requirements) != 1:
                return AtomicityPlan(
                    decision=AtomicityDecision.NEEDS_REFINEMENT,
                    children=(),
                    reasons=(
                        AtomicityReason(
                            code="ambiguous_evidence_allocation",
                            message=(
                                "Automatic decomposition cannot safely allocate multiple "
                                "evidence requirements across child claims."
                            ),
                            fragment=atom.predicate,
                        ),
                    ),
                    refinement_questions=(
                        "Propose each child atom with its own complete evidence requirements.",
                    ),
                )
            dependency_logic = self._decomposition_logic(atom.predicate)
            if dependency_logic is None:
                return AtomicityPlan(
                    decision=AtomicityDecision.NEEDS_REFINEMENT,
                    children=(),
                    reasons=(
                        AtomicityReason(
                            code="mixed_dependency_logic",
                            message=(
                                "The predicate mixes conjunction and disjunction that cannot be "
                                "safely decomposed into one dependency group."
                            ),
                            fragment=atom.predicate,
                        ),
                    ),
                    refinement_questions=(
                        "Rewrite the claim as nested atoms with one logical operator each.",
                    ),
                )
            children = tuple(self._child_for_fragment(atom, part) for part in safe_fragments)
            if self._has_duplicate_children(children):
                return AtomicityPlan(
                    decision=AtomicityDecision.NEEDS_REFINEMENT,
                    children=(),
                    reasons=(
                        AtomicityReason(
                            code="duplicate_generated_children",
                            message="Automatic decomposition produced duplicate child claims.",
                            fragment=atom.predicate,
                        ),
                    ),
                    refinement_questions=(
                        "Remove the duplicate claim or propose distinct child atoms.",
                    ),
                )
            invalid_children = tuple(
                (child, diagnostic)
                for child in children
                if (diagnostic := self._child_diagnostic(child)) is not None
            )
            if invalid_children:
                return AtomicityPlan(
                    decision=AtomicityDecision.NEEDS_REFINEMENT,
                    children=(),
                    reasons=tuple(
                        AtomicityReason(
                            code="generated_child_not_atomic",
                            message=f"Generated child needs refinement: {diagnostic}",
                            fragment=child.predicate,
                        )
                        for child, diagnostic in invalid_children
                    ),
                    refinement_questions=(
                        "Propose operationalized child atoms for the unresolved fragments.",
                    ),
                )
            return AtomicityPlan(
                decision=AtomicityDecision.DECOMPOSE,
                children=children,
                reasons=(
                    AtomicityReason(
                        code="explicit_independent_predicates",
                        message="Multiple structurally explicit predicates can vary independently.",
                        fragment=atom.predicate,
                    ),
                ),
                dependency_logic=dependency_logic,
            )

        if diagnostics is not None:
            return AtomicityPlan(
                decision=AtomicityDecision.NEEDS_REFINEMENT,
                children=(),
                reasons=(
                    AtomicityReason(
                        code=diagnostics[0],
                        message=diagnostics[1],
                        fragment=atom.predicate,
                    ),
                ),
                refinement_questions=(diagnostics[2],),
            )

        return AtomicityPlan(
            decision=AtomicityDecision.ATOMIC,
            children=(),
            reasons=(
                AtomicityReason(
                    code="one_operationalized_predicate",
                    message=(
                        "One structurally explicit predicate is admissible for independent "
                        "evaluation. This is a structural determination, not proof of truth."
                    ),
                ),
            ),
        )

    def _diagnose(
        self,
        predicate: str,
        operational_definition: str | None,
    ) -> tuple[str, str, str] | None:
        explicit_measurement = re.search(
            r"(?:=|!=|<=|>=|<|>|\btrue\b|\bfalse\b|\bexists?\b)",
            predicate,
            flags=re.IGNORECASE,
        )
        if (
            _BROAD_TERMS.search(predicate)
            and operational_definition is None
            and explicit_measurement is None
        ):
            return (
                "unoperationalized_predicate",
                "The predicate uses a broad quality whose falsifying observation is undefined.",
                "What single observable condition would make this predicate true or false?",
            )
        if _is_single_numeric_range(predicate):
            return None
        if _AMBIGUOUS_CONNECTOR.search(_without_enum_alternatives(predicate)):
            return (
                "possible_compound_predicate",
                (
                    "The predicate may contain independently variable claims, "
                    "but a safe split is ambiguous."
                ),
                "List each independently falsifiable predicate as a proposed child atom.",
            )
        return None

    @staticmethod
    def _has_duplicate_children(children: tuple[ProposedAtom, ...]) -> bool:
        signatures: set[str] = set()
        for child in children:
            signature = json.dumps(
                {
                    "subject": " ".join(child.subject.casefold().split()),
                    "predicate": " ".join(child.predicate.casefold().split()),
                    "scope": " ".join(child.scope.casefold().split()),
                    "operator": child.operator.value,
                    "expected_value": child.expected_value,
                    "quantifier": child.quantifier.value,
                    "expected_answer_type": child.expected_answer_type.value,
                    "answer_options": child.answer_options,
                    "score_criteria": child.score_criteria,
                    "operational_definition": child.operational_definition,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            if signature in signatures:
                return True
            signatures.add(signature)
        return False

    def _child_diagnostic(
        self,
        child: ProposedAtom,
    ) -> tuple[str, str, str] | None:
        if len(self._safe_fragments(child.predicate)) >= 2:
            return (
                "explicit_independent_predicates",
                "The proposed child still contains multiple explicit predicates.",
                "Split this child again so each proposal contains one predicate.",
            )
        return self._diagnose(child.predicate, child.operational_definition)

    def _safe_fragments(self, predicate: str) -> tuple[str, ...]:
        explicit = tuple(part.strip() for part in _SAFE_SEPARATORS.split(predicate) if part.strip())
        if len(explicit) >= 2:
            return explicit

        if _is_single_numeric_range(predicate):
            return ()

        match = _COPULA_LIST.match(predicate)
        if match is None or "," not in match.group("body"):
            return ()

        prefix = match.group("prefix")
        body = re.sub(r",?\s+(?:and|or)\s+", ",", match.group("body"), flags=re.IGNORECASE)
        parts = tuple(part.strip() for part in body.split(",") if part.strip())
        if len(parts) < 2:
            return ()
        return tuple(f"{prefix} {part}" for part in parts)

    @staticmethod
    def _decomposition_logic(predicate: str) -> DependencyLogic | None:
        has_disjunction = bool(re.search(r"\bor\b|\|\|", predicate, flags=re.IGNORECASE))
        has_conjunction = bool(re.search(r"\band\b|&&", predicate, flags=re.IGNORECASE))
        if has_disjunction and has_conjunction and not _is_single_numeric_range(predicate):
            return None
        return DependencyLogic.ANY_OF if has_disjunction else DependencyLogic.ALL_OF

    def _child_for_fragment(self, parent: AtomicRequest, predicate: str) -> ProposedAtom:
        requirement = parent.evidence_requirements[0]
        child_requirement = EvidenceRequirement(
            description=(
                f"Evidence that can independently determine whether {parent.subject} {predicate}."
            ),
            accepted_kinds=requirement.accepted_kinds,
            minimum_items=requirement.minimum_items,
            required=requirement.required,
            deterministic_required=requirement.deterministic_required,
        )
        return ProposedAtom(
            question=self._question(parent.subject, predicate),
            subject=parent.subject,
            predicate=predicate,
            scope=parent.scope,
            operator=parent.operator,
            expected_value=parent.expected_value,
            quantifier=parent.quantifier,
            expected_answer_type=parent.expected_answer_type,
            answer_options=parent.answer_options,
            score_criteria=parent.score_criteria,
            operational_definition=None,
            evidence_requirements=(child_requirement,),
            verification_method=parent.verification_method,
        )

    @staticmethod
    def _question(subject: str, predicate: str) -> str:
        words = predicate.split(maxsplit=1)
        auxiliary = words[0].lower()
        rest = words[1] if len(words) == 2 else ""
        if auxiliary in {"is", "are", "was", "were", "can", "does", "do"}:
            return f"{words[0].capitalize()} {subject} {rest}?".replace("  ", " ")
        if auxiliary in {"has", "have"}:
            return f"Does {subject} have {rest}?".replace("  ", " ")
        return f"Does {subject} {predicate}?"

    def _limit_plan(self, atom: AtomicRequest) -> AtomicityPlan:
        return AtomicityPlan(
            decision=AtomicityDecision.NEEDS_REFINEMENT,
            children=(),
            reasons=(
                AtomicityReason(
                    code="maximum_depth_reached",
                    message=f"Maximum decomposition depth {self.max_depth} has been reached.",
                    fragment=atom.predicate,
                ),
            ),
            refinement_questions=(
                "Should the LLM terminate, replace this atom, or raise the configured depth limit?",
            ),
        )
