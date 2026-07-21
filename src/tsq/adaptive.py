# SPDX-License-Identifier: MPL-2.0

"""Evidence-constrained recursive planning over a learner's knowledge graph.

The live algorithm deliberately stays interpretable.  It combines the current
Bayesian learner projection with the immutable prerequisite DAG, but it does
not learn global graph edges or item parameters from a single learner.  Those
population-level changes belong in an offline, release-gated calibration loop.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from math import exp, sqrt
from typing import Mapping

from .graph import KnowledgeGraph
from .learner import LearnerModel
from .models import SkillState


BOUNDARY_ALGORITHM_VERSION = "recursive-evidence-boundary-v1"


def _clamp(value: float) -> float:
    return min(1.0, max(0.0, value))


@dataclass(frozen=True, slots=True)
class ConceptReadiness:
    """Interpretable readiness projection for one learner/concept pair.

    ``intrinsic_readiness`` is evidence about the concept itself.
    ``prerequisite_support`` is the weakest weighted incoming prerequisite
    route after recursively accounting for its own foundations.
    ``effective_readiness`` conservatively combines both quantities.
    """

    concept_id: str
    intrinsic_readiness: float
    prerequisite_support: float
    effective_readiness: float
    mastery_probability: float
    expected_competence: float
    uncertainty: float
    evidence_mass: float
    exposures: int
    bottleneck_concept_id: str | None

    def terms(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BoundaryCandidate:
    """One direct prerequisite route considered for a focused fallback."""

    concept_id: str
    edge_weight: float
    score: float
    need: float
    uncertainty_value: float
    evidence_gap: float
    recent_failure_rate: float
    prerequisite_support: float
    effective_readiness: float
    recursive_bottleneck_concept_id: str

    def terms(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BoundaryDecision:
    """Deterministic choice of the next prerequisite learning boundary."""

    focus_concept_id: str
    selected_concept_id: str
    algorithm_version: str
    candidates: tuple[BoundaryCandidate, ...]

    @property
    def selected(self) -> BoundaryCandidate:
        return next(
            candidate
            for candidate in self.candidates
            if candidate.concept_id == self.selected_concept_id
        )

    def terms(self) -> dict[str, object]:
        return {
            "focus_concept_id": self.focus_concept_id,
            "selected_concept_id": self.selected_concept_id,
            "algorithm_version": self.algorithm_version,
            "selected": self.selected.terms(),
            "candidates": [candidate.terms() for candidate in self.candidates],
        }


class RecursiveEvidenceBoundary:
    """Find a personalized prerequisite boundary without a black-box policy.

    The graph recursion makes an upstream weakness influence readiness for all
    dependent concepts.  Fallback is still only one edge at a time: after a
    learner also struggles on that prerequisite, the same calculation can
    recurse again.  This keeps the path explainable and allows every step to be
    checked with an independently authored question family.
    """

    def __init__(self, learner_model: LearnerModel):
        self.learner_model = learner_model

    def readiness_map(
        self,
        *,
        learner_id: str,
        graph: KnowledgeGraph,
        stored_states: Mapping[str, SkillState],
        now: datetime,
        concept_ids: set[str] | None = None,
    ) -> dict[str, ConceptReadiness]:
        requested = set(concept_ids or graph.concepts)
        unknown = requested - set(graph.concepts)
        if unknown:
            raise ValueError(
                "Unknown readiness concepts: " + ", ".join(sorted(unknown))
            )
        memo: dict[str, ConceptReadiness] = {}

        def project(concept_id: str) -> ConceptReadiness:
            existing = memo.get(concept_id)
            if existing is not None:
                return existing
            concept = graph.concepts[concept_id]
            state = stored_states.get(concept_id) or self.learner_model.initial_state(
                learner_id, concept
            )
            state = self.learner_model.project_state(state, concept, now)
            mastery = state.mastery_probability
            competence = state.expected_competence
            # Certification probability is deliberately prominent while the
            # posterior mean prevents a cold learner from collapsing to zero.
            intrinsic = _clamp(0.55 * mastery + 0.45 * competence)

            prerequisite_rows: list[tuple[str, float, ConceptReadiness]] = []
            for prerequisite_id, weight in graph.direct_prerequisites(concept_id):
                prerequisite_rows.append(
                    (prerequisite_id, _clamp(weight), project(prerequisite_id))
                )
            if prerequisite_rows:
                weighted_support = [
                    1.0 - weight * (1.0 - readiness.effective_readiness)
                    for _, weight, readiness in prerequisite_rows
                ]
                prerequisite_support = min(weighted_support)
                bottleneck_index = weighted_support.index(prerequisite_support)
                direct_id, _, direct_readiness = prerequisite_rows[bottleneck_index]
                bottleneck = (
                    direct_readiness.bottleneck_concept_id or direct_id
                )
            else:
                prerequisite_support = 1.0
                bottleneck = None

            # A learner cannot be treated as ready for an advanced concept
            # solely because of a noisy direct success when a required
            # foundation remains weak.  The non-zero floor avoids pretending
            # that prerequisites are perfectly deterministic causes.
            effective = _clamp(
                intrinsic * (0.40 + 0.60 * prerequisite_support)
            )
            result = ConceptReadiness(
                concept_id=concept_id,
                intrinsic_readiness=intrinsic,
                prerequisite_support=prerequisite_support,
                effective_readiness=effective,
                mastery_probability=mastery,
                expected_competence=competence,
                uncertainty=sqrt(state.variance),
                evidence_mass=state.evidence_mass,
                exposures=state.exposures,
                bottleneck_concept_id=bottleneck,
            )
            memo[concept_id] = result
            return result

        for concept_id in sorted(requested):
            project(concept_id)
        return {concept_id: memo[concept_id] for concept_id in requested}

    def choose_direct_boundary(
        self,
        *,
        learner_id: str,
        focus_concept_id: str,
        graph: KnowledgeGraph,
        stored_states: Mapping[str, SkillState],
        now: datetime,
        recent_performance: Mapping[str, tuple[int, int]] | None = None,
        excluded_concept_ids: set[str] | None = None,
    ) -> BoundaryDecision | None:
        excluded = excluded_concept_ids or set()
        direct = [
            (concept_id, weight)
            for concept_id, weight in graph.direct_prerequisites(focus_concept_id)
            if concept_id not in excluded
        ]
        if not direct:
            return None
        relevant = {concept_id for concept_id, _ in direct}
        readiness = self.readiness_map(
            learner_id=learner_id,
            graph=graph,
            stored_states=stored_states,
            now=now,
            concept_ids=relevant,
        )
        performance = recent_performance or {}
        candidates: list[BoundaryCandidate] = []
        for concept_id, edge_weight in direct:
            state = readiness[concept_id]
            attempted, incorrect = performance.get(concept_id, (0, 0))
            failure_rate = incorrect / attempted if attempted else 0.0
            uncertainty_value = 1.0 - exp(-state.uncertainty)
            evidence_gap = 1.0 / (1.0 + state.evidence_mass)
            need = 1.0 - state.effective_readiness
            # The score balances a likely gap with the ability to teach at the
            # next boundary.  Recent direct failures strengthen learner-local
            # evidence; an unobserved prerequisite retains an uncertainty and
            # evidence-gap signal rather than being declared failed.
            score = (
                0.34 * need
                + 0.18 * (1.0 - state.mastery_probability)
                + 0.14 * uncertainty_value
                + 0.12 * evidence_gap
                + 0.12 * failure_rate
                + 0.10 * _clamp(edge_weight)
            )
            candidates.append(
                BoundaryCandidate(
                    concept_id=concept_id,
                    edge_weight=_clamp(edge_weight),
                    score=score,
                    need=need,
                    uncertainty_value=uncertainty_value,
                    evidence_gap=evidence_gap,
                    recent_failure_rate=failure_rate,
                    prerequisite_support=state.prerequisite_support,
                    effective_readiness=state.effective_readiness,
                    recursive_bottleneck_concept_id=(
                        state.bottleneck_concept_id or concept_id
                    ),
                )
            )
        candidates.sort(key=lambda item: (-item.score, item.concept_id))
        return BoundaryDecision(
            focus_concept_id=focus_concept_id,
            selected_concept_id=candidates[0].concept_id,
            algorithm_version=BOUNDARY_ALGORITHM_VERSION,
            candidates=tuple(candidates),
        )
