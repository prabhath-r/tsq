# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable

from .errors import ValidationError
from .models import Concept, ConceptEdge, RelationType


class KnowledgeGraph:
    """Typed concept graph with strict handling of readiness edges."""

    def __init__(self, concepts: Iterable[Concept], edges: Iterable[ConceptEdge]):
        self.concepts = {concept.id: concept for concept in concepts}
        self.edges = tuple(edges)
        self.outgoing: dict[str, list[ConceptEdge]] = defaultdict(list)
        self.incoming: dict[str, list[ConceptEdge]] = defaultdict(list)
        for edge in self.edges:
            if edge.source_id not in self.concepts or edge.target_id not in self.concepts:
                raise ValidationError(
                    f"Edge {edge.source_id} -> {edge.target_id} references an unknown concept."
                )
            self.outgoing[edge.source_id].append(edge)
            self.incoming[edge.target_id].append(edge)
        self._validate_prerequisite_dag()
        self._validate_part_hierarchy()
        self._validate_learning_dag()

    def _validate_prerequisite_dag(self) -> None:
        adjacency: dict[str, list[str]] = defaultdict(list)
        indegree = {concept_id: 0 for concept_id in self.concepts}
        for edge in self.edges:
            if edge.relation.is_strict_prerequisite:
                adjacency[edge.source_id].append(edge.target_id)
                indegree[edge.target_id] += 1
        queue = deque(node for node, degree in indegree.items() if degree == 0)
        visited = 0
        while queue:
            node = queue.popleft()
            visited += 1
            for downstream in adjacency[node]:
                indegree[downstream] -= 1
                if indegree[downstream] == 0:
                    queue.append(downstream)
        if visited != len(self.concepts):
            cyclic = sorted(node for node, degree in indegree.items() if degree > 0)
            raise ValidationError(f"Strict prerequisite edges contain a cycle: {', '.join(cyclic)}")

    def _validate_part_hierarchy(self) -> None:
        """Reject topic-containment cycles independently of readiness edges."""
        adjacency: dict[str, list[str]] = defaultdict(list)
        indegree = {concept_id: 0 for concept_id in self.concepts}
        for edge in self.edges:
            if edge.relation == RelationType.PART_OF:
                adjacency[edge.source_id].append(edge.target_id)
                indegree[edge.target_id] += 1
        queue = deque(node for node, degree in indegree.items() if degree == 0)
        visited = 0
        while queue:
            node = queue.popleft()
            visited += 1
            for downstream in adjacency[node]:
                indegree[downstream] -= 1
                if indegree[downstream] == 0:
                    queue.append(downstream)
        if visited != len(self.concepts):
            cyclic = sorted(node for node, degree in indegree.items() if degree > 0)
            raise ValidationError(
                f"Part-of edges contain a cycle: {', '.join(cyclic)}"
            )

    def _validate_learning_dag(self) -> None:
        """Reject cycles formed by mixing containment and readiness edges."""
        adjacency: dict[str, list[str]] = defaultdict(list)
        indegree = {concept_id: 0 for concept_id in self.concepts}
        for edge in self.edges:
            if edge.relation == RelationType.PART_OF or edge.relation.is_strict_prerequisite:
                adjacency[edge.source_id].append(edge.target_id)
                indegree[edge.target_id] += 1
        queue = deque(node for node, degree in indegree.items() if degree == 0)
        visited = 0
        while queue:
            node = queue.popleft()
            visited += 1
            for downstream in adjacency[node]:
                indegree[downstream] -= 1
                if indegree[downstream] == 0:
                    queue.append(downstream)
        if visited != len(self.concepts):
            cyclic = sorted(node for node, degree in indegree.items() if degree > 0)
            raise ValidationError(
                "Combined part-of/prerequisite learning graph contains a cycle: "
                + ", ".join(cyclic)
            )

    def prerequisites(self, concept_id: str, *, transitive: bool = True) -> set[str]:
        if concept_id not in self.concepts:
            raise ValidationError(f"Unknown concept: {concept_id}")
        result: set[str] = set()
        frontier = deque([concept_id])
        while frontier:
            target = frontier.popleft()
            for edge in self.incoming[target]:
                if not edge.relation.is_strict_prerequisite or edge.source_id in result:
                    continue
                result.add(edge.source_id)
                if transitive:
                    frontier.append(edge.source_id)
        return result

    def parts(self, concept_id: str, *, transitive: bool = True) -> set[str]:
        """Return concepts declared as parts of the supplied topic container."""
        if concept_id not in self.concepts:
            raise ValidationError(f"Unknown concept: {concept_id}")
        result: set[str] = set()
        frontier = deque([concept_id])
        while frontier:
            whole = frontier.popleft()
            for edge in self.incoming[whole]:
                if edge.relation != RelationType.PART_OF or edge.source_id in result:
                    continue
                result.add(edge.source_id)
                if transitive:
                    frontier.append(edge.source_id)
        return result

    def learning_scope(self, root_concept_id: str) -> set[str]:
        if root_concept_id not in self.concepts:
            raise ValidationError(f"Unknown concept: {root_concept_id}")
        # Close over containment and readiness together.  This matters when an
        # assessable concept requires a topic container: the container's parts
        # are then part of the actual learning scope as well.
        scope = {root_concept_id}
        frontier = deque([root_concept_id])
        while frontier:
            target = frontier.popleft()
            for edge in self.incoming[target]:
                if not (
                    edge.relation == RelationType.PART_OF
                    or edge.relation.is_strict_prerequisite
                ):
                    continue
                if edge.source_id in scope:
                    continue
                scope.add(edge.source_id)
                frontier.append(edge.source_id)
        return scope

    def direct_prerequisites(self, concept_id: str) -> list[tuple[str, float]]:
        if concept_id not in self.concepts:
            raise ValidationError(f"Unknown concept: {concept_id}")
        return sorted(
            [
            (edge.source_id, edge.weight)
            for edge in self.incoming[concept_id]
            if edge.relation.is_strict_prerequisite
            ],
            key=lambda item: item[0],
        )

    def prerequisite_distances_to(self, concept_id: str) -> dict[str, int]:
        """Return all strict prerequisite distances to one downstream target."""
        if concept_id not in self.concepts:
            raise ValidationError(f"Unknown concept: {concept_id}")
        distances = {concept_id: 0}
        queue = deque([concept_id])
        while queue:
            target = queue.popleft()
            next_distance = distances[target] + 1
            for edge in self.incoming[target]:
                if not edge.relation.is_strict_prerequisite:
                    continue
                current = distances.get(edge.source_id)
                if current is None or next_distance < current:
                    distances[edge.source_id] = next_distance
                    queue.append(edge.source_id)
        return distances

    def learning_distances_to(self, concept_id: str) -> dict[str, int]:
        """Typed distance to a topic: containment is free, prerequisites cost one.

        All assessable parts of a topic therefore sit at distance zero, while
        their direct prerequisites sit at distance one even when the requested
        root is a container.
        """
        if concept_id not in self.concepts:
            raise ValidationError(f"Unknown concept: {concept_id}")
        distances = {concept_id: 0}
        frontier = deque([concept_id])
        while frontier:
            target = frontier.popleft()
            base = distances[target]
            for edge in self.incoming[target]:
                if edge.relation == RelationType.PART_OF:
                    cost = 0
                elif edge.relation.is_strict_prerequisite:
                    cost = 1
                else:
                    continue
                candidate = base + cost
                current = distances.get(edge.source_id)
                if current is not None and current <= candidate:
                    continue
                distances[edge.source_id] = candidate
                if cost == 0:
                    frontier.appendleft(edge.source_id)
                else:
                    frontier.append(edge.source_id)
        return distances

    def distance_to(self, source_id: str, target_id: str) -> int | None:
        """Shortest strict-prerequisite distance from source to target."""
        if source_id == target_id:
            return 0
        queue = deque([(source_id, 0)])
        seen = {source_id}
        while queue:
            node, distance = queue.popleft()
            for edge in self.outgoing[node]:
                if not edge.relation.is_strict_prerequisite or edge.target_id in seen:
                    continue
                if edge.target_id == target_id:
                    return distance + 1
                seen.add(edge.target_id)
                queue.append((edge.target_id, distance + 1))
        return None
