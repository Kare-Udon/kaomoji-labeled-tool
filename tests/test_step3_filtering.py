from __future__ import annotations

from kaomoji_labeled_tool.models import PreprocessedRecord, RecordSource, RemovalConfig


def build_record(*, tags: list[str]) -> PreprocessedRecord:
    return PreprocessedRecord(
        kaomoji="(╯°□°）╯︵ ┻━┻",
        source=RecordSource(
            original_tags=["Original"],
            new_tags=["New"],
        ),
        merged_tags_raw=tags,
    )


def test_filter_record_removes_exact_and_pattern_matches():
    from kaomoji_labeled_tool.filtering import filter_record

    record = build_record(tags=["popular", "day #4", "angry", "series"])
    removal = RemovalConfig(
        exact=("popular",),
        patterns=(r"day\s*#", r"series"),
    )

    filtered = filter_record(record, removal)

    assert filtered.rule_removed_tags == ["popular", "day #4", "series"]
    assert filtered.rule_kept_tags == ["angry"]
    assert filtered.normalization_candidates == []


def test_filter_record_marks_compound_tags_as_normalization_candidates():
    from kaomoji_labeled_tool.filtering import filter_record

    record = build_record(
        tags=["sad_face", "love/happy", "cry|tears", "super-long-tag", "smiling"]
    )
    removal = RemovalConfig(exact=(), patterns=())

    filtered = filter_record(record, removal)

    assert filtered.rule_removed_tags == []
    assert filtered.rule_kept_tags == [
        "sad_face",
        "love/happy",
        "cry|tears",
        "super-long-tag",
        "smiling",
    ]
    assert filtered.normalization_candidates == [
        "sad_face",
        "love/happy",
        "cry|tears",
        "super-long-tag",
    ]


def test_filter_record_preserves_traceability_fields():
    from kaomoji_labeled_tool.filtering import filter_record

    record = PreprocessedRecord(
        kaomoji="ヽ(・∀・)ﾉ",
        source=RecordSource(
            original_tags=["Happy"],
            new_tags=["featured", "sparkly"],
        ),
        merged_tags_raw=["featured", "sparkly"],
    )
    removal = RemovalConfig(exact=("featured",), patterns=())

    filtered = filter_record(record, removal)

    assert filtered.kaomoji == "ヽ(・∀・)ﾉ"
    assert filtered.source.original_tags == ["Happy"]
    assert filtered.source.new_tags == ["featured", "sparkly"]
    assert filtered.merged_tags_raw == ["featured", "sparkly"]
    assert filtered.rule_removed_tags == ["featured"]
    assert filtered.rule_kept_tags == ["sparkly"]
