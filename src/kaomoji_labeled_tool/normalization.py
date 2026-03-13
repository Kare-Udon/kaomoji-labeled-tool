from __future__ import annotations

from .models import FilteredRecord, NormalizationConfig, NormalizedRecord


def normalize_record(
    record: FilteredRecord,
    normalization: NormalizationConfig,
    canonical_tags: tuple[str, ...],
) -> NormalizedRecord:
    canonical_set = set(canonical_tags)
    normalized_tags: list[str] = []
    canonical_tags_final: list[str] = []
    unmapped_tags: list[str] = []
    seen_normalized: set[str] = set()
    seen_canonical: set[str] = set()
    seen_unmapped: set[str] = set()

    for tag in record.rule_kept_tags:
        normalized = normalize_tag_value(tag, normalization)
        if normalized in seen_normalized:
            continue
        seen_normalized.add(normalized)
        normalized_tags.append(normalized)

        if normalized in canonical_set:
            if normalized not in seen_canonical:
                seen_canonical.add(normalized)
                canonical_tags_final.append(normalized)
            continue

        if normalized not in seen_unmapped:
            seen_unmapped.add(normalized)
            unmapped_tags.append(normalized)

    return NormalizedRecord(
        kaomoji=record.kaomoji,
        source=record.source,
        merged_tags_raw=list(record.merged_tags_raw),
        rule_removed_tags=list(record.rule_removed_tags),
        rule_kept_tags=list(record.rule_kept_tags),
        normalization_candidates=list(record.normalization_candidates),
        normalized_tags=normalized_tags,
        canonical_tags_final=canonical_tags_final,
        unmapped_tags=unmapped_tags,
    )


def normalize_tag_value(tag: str, normalization: NormalizationConfig) -> str:
    lemma_normalized = normalization.lemma_map.get(tag, tag)
    return normalization.semantic_map.get(lemma_normalized, lemma_normalized)
