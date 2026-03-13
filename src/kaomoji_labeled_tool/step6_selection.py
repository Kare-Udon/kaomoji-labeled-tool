from __future__ import annotations

import re

from .models import QualityScoredRecord, Step6Candidate


def select_step6_candidates(
    records: list[QualityScoredRecord],
    *,
    limit: int,
    exclude_kaomoji: set[str] | None = None,
) -> list[Step6Candidate]:
    rule_tag_candidates: list[Step6Candidate] = []
    empty_rule_tag_candidates: list[Step6Candidate] = []
    excluded = exclude_kaomoji or set()

    for record in records:
        if record.kaomoji in excluded:
            continue
        if record.hard_filter_reason is not None:
            continue
        if record.dedup.cluster_role == "duplicate_candidate":
            continue
        if not _looks_like_step6_candidate(record.kaomoji):
            continue

        candidate_reason: str | None = None
        priority = 99

        if record.decision.review:
            candidate_reason = "review_decision"
            priority = 1
        elif not record.canonical_tags_final and (
            record.unmapped_tags or record.quality.structure_score >= 0.8
        ):
            candidate_reason = "sparse_or_unmapped_tags"
            priority = 2
        elif (
            len(record.canonical_tags_final) <= 1
            and len(record.unmapped_tags) >= 1
            and record.quality.final_score < 0.9
        ):
            candidate_reason = "sparse_or_unmapped_tags"
            priority = 3
        elif record.canonical_tags_final:
            candidate_reason = "rule_tags_present"
            priority = 4

        if candidate_reason is None:
            continue

        candidate = Step6Candidate(
            sample_id="",
            record=record,
            candidate_reason=candidate_reason,
            priority=priority,
        )
        if record.canonical_tags_final:
            rule_tag_candidates.append(candidate)
        else:
            empty_rule_tag_candidates.append(candidate)

    _sort_candidates(rule_tag_candidates)
    _sort_candidates(empty_rule_tag_candidates)

    selected = _select_mixed_candidates(
        rule_tag_candidates=rule_tag_candidates,
        empty_rule_tag_candidates=empty_rule_tag_candidates,
        limit=limit,
    )
    return [
        Step6Candidate(
            sample_id=f"step6-{index + 1:04d}",
            record=candidate.record,
            candidate_reason=candidate.candidate_reason,
            priority=candidate.priority,
        )
        for index, candidate in enumerate(selected)
    ]


def _sort_candidates(candidates: list[Step6Candidate]) -> None:
    candidates.sort(
        key=lambda candidate: (
            candidate.priority,
            -candidate.record.quality.structure_score,
            -candidate.record.quality.final_score,
            candidate.record.kaomoji,
        )
    )


def _select_mixed_candidates(
    *,
    rule_tag_candidates: list[Step6Candidate],
    empty_rule_tag_candidates: list[Step6Candidate],
    limit: int,
) -> list[Step6Candidate]:
    if limit <= 0:
        return []
    if limit < 50:
        combined = list(rule_tag_candidates) + list(empty_rule_tag_candidates)
        _sort_candidates(combined)
        return combined[:limit]

    target_rule_tag_count = min(len(rule_tag_candidates), round(limit * 0.6))
    target_empty_count = min(len(empty_rule_tag_candidates), limit - target_rule_tag_count)

    selected = list(rule_tag_candidates[:target_rule_tag_count])
    selected.extend(empty_rule_tag_candidates[:target_empty_count])

    remaining_slots = limit - len(selected)
    if remaining_slots <= 0:
        return selected

    selected_rule_ids = {id(candidate) for candidate in selected}
    remainder: list[Step6Candidate] = []
    for candidate in rule_tag_candidates[target_rule_tag_count:]:
        if id(candidate) not in selected_rule_ids:
            remainder.append(candidate)
    for candidate in empty_rule_tag_candidates[target_empty_count:]:
        if id(candidate) not in selected_rule_ids:
            remainder.append(candidate)

    return selected + remainder[:remaining_slots]


def _looks_like_step6_candidate(kaomoji: str) -> bool:
    if len(kaomoji) <= 24:
        return True
    if len(re.findall(r"[A-Za-z]{2,}", kaomoji)) >= 3:
        return False
    return True
