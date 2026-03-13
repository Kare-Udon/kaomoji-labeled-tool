from __future__ import annotations

from kaomoji_labeled_tool.models import FilteredRecord, NormalizationConfig, RecordSource


def build_record(*, tags: list[str]) -> FilteredRecord:
    return FilteredRecord(
        kaomoji="(╥﹏╥)",
        source=RecordSource(
            original_tags=["Sad"],
            new_tags=["cry face"],
        ),
        merged_tags_raw=list(tags),
        rule_removed_tags=[],
        rule_kept_tags=list(tags),
        normalization_candidates=[],
    )


def test_normalize_record_applies_lemma_semantic_and_canonical_mapping():
    from kaomoji_labeled_tool.normalization import normalize_record

    record = build_record(tags=["cry", "sad face", "love"])
    normalization = NormalizationConfig(
        lemma_map={"cry": "crying"},
        semantic_map={"sad face": "sad"},
    )
    canonical_tags = ("crying", "sad", "love")

    normalized = normalize_record(record, normalization, canonical_tags)

    assert normalized.normalized_tags == ["crying", "sad", "love"]
    assert normalized.canonical_tags_final == ["crying", "sad", "love"]
    assert normalized.unmapped_tags == []


def test_normalize_record_deduplicates_collapsed_results_in_order():
    from kaomoji_labeled_tool.normalization import normalize_record

    record = build_record(tags=["sobbing", "cry", "tearful", "sad"])
    normalization = NormalizationConfig(
        lemma_map={"sobbing": "crying", "cry": "crying"},
        semantic_map={"tearful": "crying"},
    )
    canonical_tags = ("crying", "sad")

    normalized = normalize_record(record, normalization, canonical_tags)

    assert normalized.normalized_tags == ["crying", "sad"]
    assert normalized.canonical_tags_final == ["crying", "sad"]
    assert normalized.unmapped_tags == []


def test_normalize_record_collects_unmapped_tags_without_entering_canonical():
    from kaomoji_labeled_tool.normalization import normalize_record

    record = build_record(tags=["sparkly", "cute face", "mystery"])
    normalization = NormalizationConfig(
        lemma_map={},
        semantic_map={"cute face": "cute"},
    )
    canonical_tags = ("cute",)

    normalized = normalize_record(record, normalization, canonical_tags)

    assert normalized.normalized_tags == ["sparkly", "cute", "mystery"]
    assert normalized.canonical_tags_final == ["cute"]
    assert normalized.unmapped_tags == ["sparkly", "mystery"]


def test_normalize_record_preserves_traceability_fields():
    from kaomoji_labeled_tool.normalization import normalize_record

    record = FilteredRecord(
        kaomoji="ヽ(・∀・)ﾉ",
        source=RecordSource(
            original_tags=["Happy"],
            new_tags=["featured", "smile"],
        ),
        merged_tags_raw=["featured", "smile"],
        rule_removed_tags=["featured"],
        rule_kept_tags=["smile"],
        normalization_candidates=[],
    )
    normalization = NormalizationConfig(
        lemma_map={"smile": "smiling"},
        semantic_map={},
    )
    canonical_tags = ("smiling",)

    normalized = normalize_record(record, normalization, canonical_tags)

    assert normalized.kaomoji == "ヽ(・∀・)ﾉ"
    assert normalized.source.original_tags == ["Happy"]
    assert normalized.source.new_tags == ["featured", "smile"]
    assert normalized.merged_tags_raw == ["featured", "smile"]
    assert normalized.rule_removed_tags == ["featured"]
    assert normalized.rule_kept_tags == ["smile"]
    assert normalized.canonical_tags_final == ["smiling"]
