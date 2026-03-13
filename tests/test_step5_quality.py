from __future__ import annotations

from kaomoji_labeled_tool.models import NormalizedRecord, RecordSource


def build_record(
    kaomoji: str,
    *,
    canonical_tags: list[str] | None = None,
    normalized_tags: list[str] | None = None,
    unmapped_tags: list[str] | None = None,
) -> NormalizedRecord:
    final_canonical = canonical_tags or []
    final_normalized = normalized_tags or list(final_canonical)
    final_unmapped = unmapped_tags or []
    return NormalizedRecord(
        kaomoji=kaomoji,
        source=RecordSource(original_tags=["sample"], new_tags=["sample"]),
        merged_tags_raw=["sample"],
        rule_removed_tags=[],
        rule_kept_tags=["sample"],
        normalization_candidates=[],
        normalized_tags=final_normalized,
        canonical_tags_final=final_canonical,
        unmapped_tags=final_unmapped,
    )


def test_score_records_hard_filters_invalid_text():
    from kaomoji_labeled_tool.quality import score_records

    records = [
        build_record(" "),
        build_record("a"),
        build_record("a" * 81),
        build_record("hello world"),
        build_record("(^_^)\x00"),
    ]

    scored = score_records(records)

    assert [record.decision.status for record in scored] == [
        "drop",
        "drop",
        "drop",
        "drop",
        "drop",
    ]
    assert [record.hard_filter_reason for record in scored] == [
        "empty_kaomoji",
        "length_out_of_range",
        "length_out_of_range",
        "plain_text",
        "control_character",
    ]


def test_score_records_marks_exact_duplicates_after_first_as_drop():
    from kaomoji_labeled_tool.quality import score_records

    records = [
        build_record("(╥﹏╥)", canonical_tags=["crying"]),
        build_record("(╥﹏╥)", canonical_tags=["crying"]),
    ]

    scored = score_records(records)

    assert scored[0].decision.status == "keep"
    assert scored[0].dedup.cluster_role == "primary"
    assert scored[1].decision.status == "drop"
    assert scored[1].hard_filter_reason == "exact_duplicate"
    assert scored[1].dedup.cluster_role == "duplicate_candidate"


def test_score_records_penalizes_normalized_duplicates_without_hard_drop():
    from kaomoji_labeled_tool.quality import score_records

    records = [
        build_record("(╥﹏╥)", canonical_tags=["crying"]),
        build_record("( ╥﹏╥ )", canonical_tags=["crying"]),
    ]

    scored = score_records(records)

    assert scored[0].dedup.normalized_key == scored[1].dedup.normalized_key
    assert scored[0].decision.status == "keep"
    assert scored[1].hard_filter_reason is None
    assert scored[1].quality.dedup_penalty > 0.0
    assert scored[1].decision.status in {"review", "drop"}


def test_score_records_prefers_structured_tagged_kaomoji_over_plain_text_like_sample():
    from kaomoji_labeled_tool.quality import score_records

    records = [
        build_record("ヽ(・∀・)ﾉ", canonical_tags=["smiling"], normalized_tags=["smiling"]),
        build_record("hello there", normalized_tags=["misc"], unmapped_tags=["misc"]),
    ]

    scored = score_records(records)

    assert scored[0].quality.final_score > scored[1].quality.final_score
    assert scored[0].decision.status == "keep"
    assert scored[1].decision.status == "drop"


def test_score_records_marks_near_duplicates_without_auto_drop():
    from kaomoji_labeled_tool.quality import score_records

    records = [
        build_record("( ͡° ͜ʖ ͡°)", canonical_tags=["smirking"], normalized_tags=["smirking"]),
        build_record("(͡° ͜ʖ ͡°)", canonical_tags=["smirking"], normalized_tags=["smirking"]),
    ]

    scored = score_records(records)

    assert scored[0].dedup.near_dup_key == scored[1].dedup.near_dup_key
    assert scored[0].dedup.cluster_role == "primary"
    assert scored[1].dedup.cluster_role == "duplicate_candidate"
    assert scored[1].hard_filter_reason is None


def test_score_records_does_not_treat_ascii_kaomoji_as_plain_text():
    from kaomoji_labeled_tool.quality import score_records

    scored = score_records([build_record("QwQ", canonical_tags=["crying"])])

    assert scored[0].hard_filter_reason is None
    assert scored[0].decision.status in {"keep", "review"}


def test_score_records_excludes_hard_filtered_samples_from_dedup_primary_selection():
    from kaomoji_labeled_tool.quality import score_records

    records = [
        build_record("(^_^)\x00", canonical_tags=["smiling"]),
        build_record("(^_^)", canonical_tags=["smiling"]),
    ]

    scored = score_records(records)

    assert scored[0].hard_filter_reason == "control_character"
    assert scored[1].hard_filter_reason is None
    assert scored[1].quality.dedup_penalty == 0.0
    assert scored[1].dedup.cluster_role == "unique"
