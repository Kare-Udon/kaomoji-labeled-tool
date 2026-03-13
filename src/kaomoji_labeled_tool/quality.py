from __future__ import annotations

import re
import unicodedata

from .models import (
    DedupInfo,
    QualityMetrics,
    QualityScoredRecord,
    RecordDecision,
    NormalizedRecord,
)

_MIN_LENGTH = 2
_MAX_LENGTH = 80
_NORMALIZED_DUPLICATE_PENALTY = 0.25
_NEAR_DUPLICATE_PENALTY = 0.10
_KEEP_THRESHOLD = 0.75
_REVIEW_THRESHOLD = 0.55


def score_records(records: list[NormalizedRecord]) -> list[QualityScoredRecord]:
    if not records:
        return []

    exact_counts = _build_group_counts(record.kaomoji for record in records)
    normalized_keys = [_normalized_key(record.kaomoji) for record in records]
    near_dup_keys = [_near_duplicate_key(record.kaomoji) for record in records]

    raw_items: list[dict[str, object]] = []
    seen_exact: set[str] = set()

    for index, record in enumerate(records):
        exact_key = record.kaomoji
        normalized_key = normalized_keys[index]
        near_dup_key = near_dup_keys[index]
        hard_filter_reason = _intrinsic_hard_filter_reason(record.kaomoji)
        if hard_filter_reason is None and exact_key in seen_exact:
            hard_filter_reason = "exact_duplicate"
        if hard_filter_reason is None:
            seen_exact.add(exact_key)

        raw_items.append(
            {
                "record": record,
                "exact_key": exact_key,
                "normalized_key": normalized_key,
                "near_dup_key": near_dup_key,
                "hard_filter_reason": hard_filter_reason,
                "base_score": _base_score(record),
            }
        )

    valid_normalized_counts = _build_group_counts(
        str(item["normalized_key"])
        for item in raw_items
        if item["hard_filter_reason"] is None
    )
    valid_near_dup_counts = _build_group_counts(
        str(item["near_dup_key"])
        for item in raw_items
        if item["hard_filter_reason"] is None and item["near_dup_key"] is not None
    )

    normalized_primary = _select_primary_indexes(
        raw_items,
        "normalized_key",
        valid_normalized_counts,
    )
    near_dup_primary = _select_primary_indexes(
        raw_items,
        "near_dup_key",
        valid_near_dup_counts,
    )

    scored_records: list[QualityScoredRecord] = []
    for index, item in enumerate(raw_items):
        record = item["record"]
        assert isinstance(record, NormalizedRecord)
        exact_key = str(item["exact_key"])
        normalized_key = str(item["normalized_key"])
        near_dup_key = item["near_dup_key"]
        if near_dup_key is not None:
            near_dup_key = str(near_dup_key)
        hard_filter_reason = item["hard_filter_reason"]
        if hard_filter_reason is not None:
            hard_filter_reason = str(hard_filter_reason)
        base_score = float(item["base_score"])

        normalized_duplicate = (
            valid_normalized_counts.get(normalized_key, 0) > 1
            and normalized_primary.get(normalized_key) != index
        )
        near_duplicate = (
            near_dup_key is not None
            and valid_near_dup_counts.get(near_dup_key, 0) > 1
            and near_dup_primary.get(near_dup_key) != index
        )
        dedup_penalty = max(
            _NORMALIZED_DUPLICATE_PENALTY if normalized_duplicate else 0.0,
            _NEAR_DUPLICATE_PENALTY if near_duplicate else 0.0,
        )
        final_score = max(0.0, base_score - dedup_penalty)
        quality = _build_quality_metrics(record, dedup_penalty, final_score)
        cluster_role = _cluster_role(
            index=index,
            hard_filter_reason=hard_filter_reason,
            exact_key=exact_key,
            exact_counts=exact_counts,
            normalized_key=normalized_key,
            normalized_counts=valid_normalized_counts,
            normalized_primary=normalized_primary,
            near_dup_key=near_dup_key,
            near_dup_counts=valid_near_dup_counts,
            near_dup_primary=near_dup_primary,
        )
        decision = _build_decision(hard_filter_reason, final_score)

        scored_records.append(
            QualityScoredRecord(
                kaomoji=record.kaomoji,
                source=record.source,
                merged_tags_raw=list(record.merged_tags_raw),
                rule_removed_tags=list(record.rule_removed_tags),
                rule_kept_tags=list(record.rule_kept_tags),
                normalization_candidates=list(record.normalization_candidates),
                normalized_tags=list(record.normalized_tags),
                canonical_tags_final=list(record.canonical_tags_final),
                unmapped_tags=list(record.unmapped_tags),
                quality=quality,
                decision=decision,
                dedup=DedupInfo(
                    exact_key=exact_key,
                    normalized_key=normalized_key,
                    near_dup_key=near_dup_key,
                    cluster_role=cluster_role,
                ),
                hard_filter_reason=hard_filter_reason,
            )
        )

    return scored_records


def _build_group_counts(keys) -> dict[str, int]:
    counts: dict[str, int] = {}
    for key in keys:
        counts[key] = counts.get(key, 0) + 1
    return counts


def _intrinsic_hard_filter_reason(kaomoji: str) -> str | None:
    if not kaomoji.strip():
        return "empty_kaomoji"
    if len(kaomoji) < _MIN_LENGTH or len(kaomoji) > _MAX_LENGTH:
        return "length_out_of_range"
    if any(unicodedata.category(char).startswith("C") for char in kaomoji):
        return "control_character"
    if _is_plain_text(kaomoji):
        return "plain_text"
    return None


def _is_plain_text(value: str) -> bool:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9\s'!?,.-]*", value) is None:
        return False
    if not any(char.isspace() for char in value):
        return False
    tokens = [token for token in re.split(r"\s+", value.strip()) if token]
    if len(tokens) < 2:
        return False
    if any(not re.fullmatch(r"[A-Za-z0-9'.,!?-]+", token) for token in tokens):
        return False
    alphabetic_tokens = [token for token in tokens if any(char.isalpha() for char in token)]
    return len(alphabetic_tokens) >= 2


def _normalized_key(kaomoji: str) -> str:
    normalized = unicodedata.normalize("NFKC", kaomoji)
    return "".join(
        char
        for char in normalized
        if not char.isspace() and not unicodedata.category(char).startswith("C")
    )


def _near_duplicate_key(kaomoji: str) -> str | None:
    normalized = _normalized_key(kaomoji)
    if not normalized:
        return None
    return re.sub(r"(.)\1{2,}", r"\1\1", normalized)


def _base_score(record: NormalizedRecord) -> float:
    length_score = _length_score(record.kaomoji)
    structure_score = _structure_score(record.kaomoji)
    semantic_score = _semantic_score(record)
    input_value_score = _input_value_score(record)
    return (
        0.20 * length_score
        + 0.30 * structure_score
        + 0.25 * semantic_score
        + 0.25 * input_value_score
    )


def _build_quality_metrics(
    record: NormalizedRecord,
    dedup_penalty: float,
    final_score: float,
) -> QualityMetrics:
    return QualityMetrics(
        length_score=_length_score(record.kaomoji),
        structure_score=_structure_score(record.kaomoji),
        semantic_score=_semantic_score(record),
        input_value_score=_input_value_score(record),
        dedup_penalty=dedup_penalty,
        final_score=final_score,
    )


def _length_score(kaomoji: str) -> float:
    length = len(kaomoji)
    if 3 <= length <= 24:
        return 1.0
    if 2 <= length <= 40:
        return 0.8
    if 41 <= length <= 80:
        return 0.4
    return 0.0


def _structure_score(kaomoji: str) -> float:
    visible = [char for char in kaomoji if not char.isspace()]
    if not visible:
        return 0.0
    symbol_count = sum(not char.isalnum() for char in visible)
    symbol_ratio = symbol_count / len(visible)
    if symbol_ratio >= 0.45:
        return 1.0
    if symbol_ratio >= 0.25:
        return 0.8
    if symbol_ratio >= 0.10:
        return 0.5
    return 0.2


def _semantic_score(record: NormalizedRecord) -> float:
    canonical_count = len(record.canonical_tags_final)
    if canonical_count >= 2:
        return 1.0
    if canonical_count == 1:
        return 0.85
    if record.normalized_tags:
        return 0.45
    return 0.1


def _input_value_score(record: NormalizedRecord) -> float:
    canonical_count = len(record.canonical_tags_final)
    if canonical_count >= 2:
        return 1.0
    if canonical_count == 1:
        return 0.85
    if len(record.normalized_tags) >= 1:
        return 0.55
    return 0.1


def _select_primary_indexes(
    items: list[dict[str, object]],
    key_name: str,
    counts: dict[str, int],
) -> dict[str, int]:
    primary_indexes: dict[str, int] = {}
    for index, item in enumerate(items):
        if item["hard_filter_reason"] is not None:
            continue
        key = item[key_name]
        if key is None:
            continue
        key = str(key)
        if counts.get(key, 0) <= 1:
            continue
        score = float(item["base_score"])
        current_index = primary_indexes.get(key)
        if current_index is None:
            primary_indexes[key] = index
            continue
        current_score = float(items[current_index]["base_score"])
        if score > current_score:
            primary_indexes[key] = index
    return primary_indexes


def _cluster_role(
    *,
    index: int,
    hard_filter_reason: str | None,
    exact_key: str,
    exact_counts: dict[str, int],
    normalized_key: str,
    normalized_counts: dict[str, int],
    normalized_primary: dict[str, int],
    near_dup_key: str | None,
    near_dup_counts: dict[str, int],
    near_dup_primary: dict[str, int],
) -> str:
    if hard_filter_reason == "exact_duplicate":
        return "duplicate_candidate"
    if exact_counts.get(exact_key, 0) > 1:
        return "primary"
    if normalized_counts.get(normalized_key, 0) > 1:
        if normalized_primary.get(normalized_key) == index:
            return "primary"
        return "duplicate_candidate"
    if near_dup_key is not None and near_dup_counts.get(near_dup_key, 0) > 1:
        if near_dup_primary.get(near_dup_key) == index:
            return "primary"
        return "duplicate_candidate"
    return "unique"


def _build_decision(hard_filter_reason: str | None, final_score: float) -> RecordDecision:
    if hard_filter_reason is not None:
        return RecordDecision(status="drop", keep=False, review=False)
    if final_score >= _KEEP_THRESHOLD:
        return RecordDecision(status="keep", keep=True, review=False)
    if final_score >= _REVIEW_THRESHOLD:
        return RecordDecision(status="review", keep=False, review=True)
    return RecordDecision(status="drop", keep=False, review=False)
