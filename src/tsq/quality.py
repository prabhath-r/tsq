# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import re
from collections import Counter, defaultdict
from math import isfinite
from statistics import median
from typing import Iterable

from .graph import KnowledgeGraph
from .models import ConceptRole, Misconception, QualityIssue, Question


_BANNED_OPTION_PATTERNS = (
    re.compile(r"\ball of the above\b", re.IGNORECASE),
    re.compile(r"\bnone of the above\b", re.IGNORECASE),
    re.compile(r"\bboth [a-d] and [a-d]\b", re.IGNORECASE),
)
_ABSOLUTE_WORDS = re.compile(
    r"\b(always|never|only|impossible|guarantees?|must|cannot|every|entirely|"
    r"automatically|universally|regardless)\b",
    re.IGNORECASE,
)
_TOKEN = re.compile(r"[a-z0-9]+")
_MIN_CORPUS_METRIC_ITEMS = 12
_LONGEST_HEURISTIC_WARNING = 0.45
_ABSOLUTE_RATE_GAP_WARNING = 0.15
_VERIFICATION_KINDS = frozenset(
    {"application", "calculation", "comparison", "counterfactual", "debugging", "transfer"}
)


def _normalized(text: str) -> str:
    return " ".join(_TOKEN.findall(text.casefold()))


def _finite_number(value: object) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def validate_question(question: Question) -> list[QualityIssue]:
    """Run deterministic item checks before an item can enter the corpus.

    These checks catch structural and clue-bearing failures. They deliberately do
    not claim to prove factual correctness or one-best-answer validity; those are
    separate review gates.
    """

    issues: list[QualityIssue] = []

    def add(code: str, severity: str, message: str) -> None:
        issues.append(QualityIssue(code, severity, message, question.id))

    if not question.id or not question.family_id:
        add("missing_identity", "error", "Question and family IDs are required.")
    if not isinstance(question.version, int) or isinstance(question.version, bool) or question.version < 1:
        add("invalid_version", "error", "Question version must be positive.")
    if len(question.stem.strip()) < 35:
        add("thin_stem", "error", "Stem is too short to establish a precise problem.")
    if len(question.options) != 4:
        add("option_count", "error", "Exactly four options are required for this corpus.")
    correct = [option for option in question.options if option.correct]
    if len(correct) != 1:
        add("answer_count", "error", "Exactly one option must be marked correct.")
    if len({option.id for option in question.options}) != len(question.options):
        add("duplicate_option_id", "error", "Option IDs must be unique within an item.")
    normalized_options = [_normalized(option.text) for option in question.options]
    if len(set(normalized_options)) != len(normalized_options):
        add("duplicate_option", "error", "Options collapse to duplicate normalized text.")
    if any(len(text) < 4 for text in normalized_options):
        add("thin_option", "error", "Every option must be a substantive response.")
    for option in question.options:
        if len(option.rationale.strip()) < 18:
            add("missing_rationale", "error", f"Option {option.id} needs a specific rationale.")
        if option.correct and option.misconception_id:
            add("correct_misconception", "error", f"Correct option {option.id} cannot signal a misconception.")
        if not option.correct and not option.misconception_id:
            add("unmodeled_distractor", "error", f"Distractor {option.id} must map to a misconception.")
        if any(pattern.search(option.text) for pattern in _BANNED_OPTION_PATTERNS):
            add("meta_option", "error", f"Option {option.id} uses a banned meta-answer pattern.")

    if not question.concepts:
        add("missing_concepts", "error", "At least one concept mapping is required.")
    else:
        if len({c.concept_id for c in question.concepts}) != len(question.concepts):
            add("duplicate_concept_mapping", "error", "A concept may be mapped only once per item.")
        invalid_roles = sorted(
            {str(c.role) for c in question.concepts if not isinstance(c.role, ConceptRole)}
        )
        if invalid_roles:
            add(
                "invalid_concept_role",
                "error",
                f"Unknown concept roles: {', '.join(invalid_roles)}.",
            )
        if sum(c.role == ConceptRole.PRIMARY for c in question.concepts) != 1:
            add("primary_concept", "error", "Exactly one concept mapping must have role 'primary'.")
        if any(not _finite_number(c.weight) for c in question.concepts):
            add("concept_weight_finite", "error", "Concept weights must be finite numbers.")
        elif any(c.weight <= 0.0 for c in question.concepts):
            add("concept_weight", "error", "Concept weights must be positive.")
        total_weight = sum(c.weight for c in question.concepts) if all(
            _finite_number(c.weight) for c in question.concepts
        ) else float("nan")
        if isfinite(total_weight) and abs(total_weight - 1.0) > 0.02:
            add("concept_weight_sum", "error", f"Concept weights sum to {total_weight:.3f}, not 1.0.")

    if not _finite_number(question.difficulty):
        add("difficulty_finite", "error", "Difficulty must be a finite number.")
    elif not (-4.0 <= question.difficulty <= 4.0):
        add("difficulty_range", "error", "Difficulty must be on the [-4, 4] latent scale.")
    if not _finite_number(question.discrimination):
        add("discrimination_finite", "error", "Discrimination must be a finite number.")
    elif not (0.25 <= question.discrimination <= 3.0):
        add("discrimination_range", "error", "Discrimination must be in [0.25, 3.0].")
    if not _finite_number(question.guess_rate):
        add("guess_finite", "error", "Guess rate must be a finite number.")
    elif not (0.0 <= question.guess_rate <= 0.35):
        add("guess_range", "error", "Guess rate must be in [0, 0.35].")
    elif question.options:
        chance_rate = 1.0 / len(question.options)
        if question.guess_rate + 1e-9 < chance_rate:
            add(
                "guess_below_forced_choice_chance",
                "error",
                f"Guess rate {question.guess_rate:.3f} is below the forced-choice chance "
                f"floor {chance_rate:.3f} for {len(question.options)} options; use a nominal "
                "option model for systematic below-chance misconceptions.",
            )
    if not _finite_number(question.slip_rate):
        add("slip_finite", "error", "Slip rate must be a finite number.")
    elif not (0.0 <= question.slip_rate <= 0.25):
        add("slip_range", "error", "Slip rate must be in [0, 0.25].")
    if (
        _finite_number(question.guess_rate)
        and _finite_number(question.slip_rate)
        and question.guess_rate + question.slip_rate >= 0.8
    ):
        add("uninformative_item", "error", "Guess and slip rates make the item nearly uninformative.")
    if not question.source_ids:
        add("missing_source", "error", "At least one provenance source is required.")

    if len(correct) == 1 and len(question.options) >= 3:
        correct_len = len(_TOKEN.findall(correct[0].text))
        distractor_lens = [len(_TOKEN.findall(o.text)) for o in question.options if not o.correct]
        typical = max(1.0, float(median(distractor_lens)))
        ratio = correct_len / typical
        if ratio > 1.70 or ratio < 0.55:
            add("answer_length_leak", "error", f"Correct-option length ratio {ratio:.2f} is strongly clue-bearing.")
        elif ratio > 1.35 or ratio < 0.70:
            add("answer_length_warning", "warning", f"Correct-option length ratio {ratio:.2f} may be clue-bearing.")

        absolute_flags = [_ABSOLUTE_WORDS.search(o.text) is not None for o in question.options]
        if sum(absolute_flags) in {1, len(question.options) - 1}:
            add(
                "absolute_word_asymmetry",
                "warning",
                "Exactly one option differs from the others in its use of absolute qualifiers.",
            )

    endings = Counter(option.text.rstrip()[-1:] for option in question.options if option.text.rstrip())
    if len(endings) > 2:
        add("punctuation_asymmetry", "warning", "Option punctuation is inconsistent.")
    return issues


def audit_corpus(
    questions: Iterable[Question],
    *,
    expected_primary_concept_ids: Iterable[str] | None = None,
    minimum_primary_families: int = 3,
    knowledge_graph: KnowledgeGraph | None = None,
    misconceptions: Iterable[Misconception] | None = None,
) -> list[QualityIssue]:
    all_items = list(questions)
    issues = [issue for item in all_items for issue in validate_question(item)]
    # Draft, quarantined, pilot, and retired material cannot rescue the coverage
    # or option-key statistics of the bank actually served by the policy.
    items = [item for item in all_items if item.status.eligible_for_adaptation]
    stems: dict[str, str] = {}
    family_counts: Counter[str] = Counter()
    answer_positions: Counter[int] = Counter()
    primary_families: dict[str, set[str]] = defaultdict(set)
    mapped_concepts: set[str] = set()
    longest_expected_correct = 0.0
    keyed_item_count = 0
    absolute_options = 0
    absolute_keys = 0
    keyed_options = 0
    keyed_answers = 0

    for item in items:
        normalized_stem = _normalized(item.stem)
        if normalized_stem in stems:
            issues.append(
                QualityIssue(
                    "duplicate_stem",
                    "error",
                    f"Stem duplicates question {stems[normalized_stem]}.",
                    item.id,
                )
            )
        stems[normalized_stem] = item.id
        family_counts[item.family_id] += 1
        for mapping in item.concepts:
            mapped_concepts.add(mapping.concept_id)
            if mapping.role == ConceptRole.PRIMARY:
                primary_families[mapping.concept_id].add(item.family_id)

        correct_count = sum(option.correct for option in item.options)
        keyed_item = len(item.options) == 4 and correct_count == 1
        if keyed_item:
            keyed_item_count += 1
            lengths = [len(_TOKEN.findall(option.text)) for option in item.options]
            longest = max(lengths)
            longest_indices = [index for index, length in enumerate(lengths) if length == longest]
            correct_index = next(index for index, option in enumerate(item.options) if option.correct)
            if correct_index in longest_indices:
                longest_expected_correct += 1.0 / len(longest_indices)

        for index, option in enumerate(item.options):
            if keyed_item:
                if option.correct:
                    answer_positions[index] += 1
                has_absolute = _ABSOLUTE_WORDS.search(option.text) is not None
                keyed_options += 1
                keyed_answers += int(option.correct)
                absolute_options += int(has_absolute)
                absolute_keys += int(has_absolute and option.correct)

    if keyed_item_count >= _MIN_CORPUS_METRIC_ITEMS:
        expected = keyed_item_count / 4.0
        for index in range(4):
            if abs(answer_positions[index] - expected) > max(2.0, expected * 0.45):
                issues.append(
                    QualityIssue(
                        "answer_position_imbalance",
                        "warning",
                        f"Source position {index + 1} holds {answer_positions[index]} of "
                        f"{keyed_item_count} keys.",
                    )
                )
    for family_id, count in family_counts.items():
        if count > 8:
            issues.append(
                QualityIssue(
                    "large_item_family",
                    "warning",
                    f"Family {family_id} has {count} locally dependent items.",
                )
            )

    if keyed_item_count >= _MIN_CORPUS_METRIC_ITEMS:
        longest_accuracy = longest_expected_correct / keyed_item_count
        if longest_accuracy >= _LONGEST_HEURISTIC_WARNING:
            issues.append(
                QualityIssue(
                    "longest_option_key_leak",
                    "error",
                    "An option-only longest-answer heuristic has expected accuracy "
                    f"{longest_accuracy:.1%} across {keyed_item_count} items (chance is 25.0%).",
                    path="questions",
                )
            )

        non_absolute_options = keyed_options - absolute_options
        non_absolute_keys = keyed_answers - absolute_keys
        if absolute_options and non_absolute_options:
            absolute_key_rate = absolute_keys / absolute_options
            non_absolute_key_rate = non_absolute_keys / non_absolute_options
            rate_gap = abs(absolute_key_rate - non_absolute_key_rate)
            if absolute_options >= 4 and rate_gap >= _ABSOLUTE_RATE_GAP_WARNING:
                issues.append(
                    QualityIssue(
                        "absolute_qualifier_key_leak",
                        "error",
                        "Absolute-qualified options have key rate "
                        f"{absolute_key_rate:.1%} versus {non_absolute_key_rate:.1%} for other options "
                        f"({absolute_options} absolute-qualified options).",
                        path="questions",
                    )
                )

    expected_primary = (
        set(mapped_concepts)
        if expected_primary_concept_ids is None
        else set(expected_primary_concept_ids)
    )
    missing_primary = sorted(expected_primary - set(primary_families))
    if missing_primary:
        issues.append(
            QualityIssue(
                "missing_primary_mapping_coverage",
                "warning",
                f"{len(missing_primary)} mapped concepts have no primary item: "
                + ", ".join(missing_primary),
                path="questions[].concepts",
            )
        )
    thin_primary = sorted(
        (concept_id, len(families))
        for concept_id, families in primary_families.items()
        if len(families) < minimum_primary_families
    )
    if thin_primary:
        summary = ", ".join(f"{concept_id}={count}" for concept_id, count in thin_primary)
        issues.append(
            QualityIssue(
                "insufficient_primary_family_coverage",
                "warning",
                f"{len(thin_primary)} primary concepts have fewer than "
                f"{minimum_primary_families} independent families: {summary}",
                path="questions[].family_id",
            )
        )
    if knowledge_graph is not None and misconceptions is not None:
        issues.extend(
            _audit_contextual_serviceability(
                items,
                knowledge_graph=knowledge_graph,
                misconceptions=misconceptions,
                minimum_primary_families=minimum_primary_families,
            )
        )
    return issues


def _audit_contextual_serviceability(
    questions: list[Question],
    *,
    knowledge_graph: KnowledgeGraph,
    misconceptions: Iterable[Misconception],
    minimum_primary_families: int,
) -> list[QualityIssue]:
    """Check that count-covered roots can actually preserve repair paths.

    Primary-family counts alone can include an item whose distractor belongs to
    a supporting concept outside the root's learning scope.  The live policy
    correctly withholds such an item, so the release audit must count only
    families that can leave both an independent repair and a distinct
    verification family in that same root scope.
    """

    misconception_owners = {item.id: item.concept_id for item in misconceptions}
    primary_roots = sorted({question.primary_concept_id for question in questions})
    issues: list[QualityIssue] = []
    for root_id in primary_roots:
        scope = knowledge_graph.learning_scope(root_id)
        pool = [question for question in questions if question.primary_concept_id in scope]
        families_by_concept: dict[str, set[str]] = defaultdict(set)
        families_by_misconception: dict[str, set[str]] = defaultdict(set)
        verification_by_concept: dict[str, set[str]] = defaultdict(set)
        for question in pool:
            families_by_concept[question.primary_concept_id].add(question.family_id)
            if question.kind.value in _VERIFICATION_KINDS:
                verification_by_concept[question.primary_concept_id].add(
                    question.family_id
                )
            for misconception_id in question.misconception_ids:
                families_by_misconception[misconception_id].add(question.family_id)

        serviceable_families: set[str] = set()
        blocked: list[str] = []
        for question in pool:
            if question.primary_concept_id != root_id:
                continue
            reasons: list[str] = []
            remaining_primary = (
                families_by_concept[root_id] - {question.family_id}
            )
            remaining_verification = (
                verification_by_concept[root_id] - {question.family_id}
            )
            if len(remaining_primary) < minimum_primary_families - 1:
                reasons.append("independent-main-families")
            if not any(
                remaining_verification - {repair_family}
                for repair_family in remaining_primary
            ):
                reasons.append("generic-focus-pair")
            for misconception_id in question.misconception_ids:
                owner = misconception_owners.get(misconception_id)
                if owner is None:
                    reasons.append(misconception_id)
                    continue
                repair_families = (
                    families_by_misconception[misconception_id]
                    - {question.family_id}
                )
                verification_families = (
                    verification_by_concept[owner] - {question.family_id}
                )
                if not any(
                    verification_families - {repair_family}
                    for repair_family in repair_families
                ):
                    reasons.append(misconception_id)
            if reasons:
                blocked.append(f"{question.id}({','.join(sorted(set(reasons)))})")
            else:
                serviceable_families.add(question.family_id)

        if len(serviceable_families) < minimum_primary_families:
            detail = "; ".join(blocked[:4])
            suffix = f" (+{len(blocked) - 4} more)" if len(blocked) > 4 else ""
            issues.append(
                QualityIssue(
                    "insufficient_contextual_family_coverage",
                    "warning",
                    f"Root {root_id} has {len(serviceable_families)} safely serviceable "
                    f"primary families; {minimum_primary_families} are required. "
                    f"Blocked paths: {detail}{suffix}",
                    path="questions[].options[].misconception_id",
                )
            )
    return issues
