from __future__ import annotations

from kaomoji_labeled_tool.models import InputRecord
from kaomoji_labeled_tool.preprocess import normalize_tag, preprocess_record


def test_normalize_tag_applies_nfkc_whitespace_and_lowercase():
    assert normalize_tag("  ＳＡＤ   FACE  ") == "sad face"


def test_preprocess_record_merges_tags_in_order_and_deduplicates():
    record = InputRecord(
        kaomoji="(╥﹏╥)",
        original_tags=[" Sad ", "crying", "SAD"],
        new_tags=["crying", " tears ", "sad"],
    )

    preprocessed = preprocess_record(record)

    assert preprocessed.merged_tags_raw == ["sad", "crying", "tears"]


def test_preprocess_record_discards_tags_empty_after_normalization():
    record = InputRecord(
        kaomoji="(・_・;)",
        original_tags=["   ", "\t", " worried "],
        new_tags=["", "worried"],
    )

    preprocessed = preprocess_record(record)

    assert preprocessed.merged_tags_raw == ["worried"]


def test_preprocess_record_keeps_original_source_for_traceability():
    record = InputRecord(
        kaomoji="ヽ(・∀・)ﾉ",
        original_tags=["Happy"],
        new_tags=["excited"],
    )

    preprocessed = preprocess_record(record)

    assert preprocessed.source.original_tags == ["Happy"]
    assert preprocessed.source.new_tags == ["excited"]
    assert preprocessed.kaomoji == "ヽ(・∀・)ﾉ"
