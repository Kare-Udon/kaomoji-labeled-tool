from __future__ import annotations

from kaomoji_labeled_tool.models import (
    DedupInfo,
    QualityMetrics,
    QualityScoredRecord,
    RecordDecision,
    RecordSource,
)


def build_record(
    kaomoji: str,
    *,
    status: str,
    keep: bool,
    review: bool,
    final_score: float,
    canonical_tags_final: list[str],
    unmapped_tags: list[str],
    structure_score: float = 1.0,
    hard_filter_reason: str | None = None,
    cluster_role: str = "unique",
) -> QualityScoredRecord:
    return QualityScoredRecord(
        kaomoji=kaomoji,
        source=RecordSource(original_tags=["sample"], new_tags=["sample"]),
        merged_tags_raw=["sample"],
        rule_removed_tags=[],
        rule_kept_tags=["sample"],
        normalization_candidates=[],
        normalized_tags=list(canonical_tags_final) or ["sample"],
        canonical_tags_final=canonical_tags_final,
        unmapped_tags=unmapped_tags,
        quality=QualityMetrics(
            length_score=1.0,
            structure_score=structure_score,
            semantic_score=0.7,
            input_value_score=0.7,
            dedup_penalty=0.0,
            final_score=final_score,
        ),
        decision=RecordDecision(status=status, keep=keep, review=review),
        dedup=DedupInfo(
            exact_key=kaomoji,
            normalized_key=kaomoji,
            near_dup_key=kaomoji,
            cluster_role=cluster_role,
        ),
        hard_filter_reason=hard_filter_reason,
    )


def test_select_step6_candidates_prefers_review_then_sparse_tags():
    from kaomoji_labeled_tool.step6_selection import select_step6_candidates

    records = [
        build_record("(╥﹏╥)", status="review", keep=False, review=True, final_score=0.72, canonical_tags_final=["sad"], unmapped_tags=[]),
        build_record("(｡•́︿•̀｡)", status="keep", keep=True, review=False, final_score=0.81, canonical_tags_final=[], unmapped_tags=["mystery", "cry"]),
        build_record("( ˘͈ ᵕ ˘͈♡)", status="keep", keep=True, review=False, final_score=1.0, canonical_tags_final=["love", "smiling"], unmapped_tags=["mystery"]),
        build_record("(^^)", status="drop", keep=False, review=False, final_score=0.1, canonical_tags_final=[], unmapped_tags=[], hard_filter_reason="plain_text"),
        build_record("( ╥﹏╥ )", status="review", keep=False, review=True, final_score=0.70, canonical_tags_final=["sad"], unmapped_tags=[], cluster_role="duplicate_candidate"),
    ]

    candidates = select_step6_candidates(records, limit=10)

    assert [candidate.record.kaomoji for candidate in candidates] == [
        "(╥﹏╥)",
        "(｡•́︿•̀｡)",
        "( ˘͈ ᵕ ˘͈♡)",
    ]
    assert candidates[0].candidate_reason == "review_decision"
    assert candidates[1].candidate_reason == "sparse_or_unmapped_tags"
    assert candidates[2].candidate_reason == "rule_tags_present"


def test_select_step6_candidates_prefers_rule_tag_mix_for_large_batch():
    from kaomoji_labeled_tool.step6_selection import select_step6_candidates

    records = []
    for index in range(35):
        records.append(
            build_record(
                f"(rule-{index:02d})",
                status="keep",
                keep=True,
                review=False,
                final_score=0.65 + index * 0.001,
                canonical_tags_final=["sad"],
                unmapped_tags=[],
            )
        )
    for index in range(25):
        records.append(
            build_record(
                f"(empty-{index:02d})",
                status="review",
                keep=False,
                review=True,
                final_score=0.55 + index * 0.001,
                canonical_tags_final=[],
                unmapped_tags=[],
            )
        )

    candidates = select_step6_candidates(records, limit=50)

    rule_tag_candidates = [candidate for candidate in candidates if candidate.record.canonical_tags_final]
    empty_rule_tag_candidates = [
        candidate for candidate in candidates if not candidate.record.canonical_tags_final
    ]

    assert len(candidates) == 50
    assert len(rule_tag_candidates) == 30
    assert len(empty_rule_tag_candidates) == 20


def test_select_step6_candidates_prefers_higher_structure_within_rule_tag_bucket():
    from kaomoji_labeled_tool.step6_selection import select_step6_candidates

    records = [
        build_record(
            "plain words",
            status="review",
            keep=False,
            review=True,
            final_score=0.55,
            canonical_tags_final=["smiling"],
            unmapped_tags=[],
            structure_score=0.2,
        ),
        build_record(
            "(╥﹏╥)",
            status="review",
            keep=False,
            review=True,
            final_score=0.6,
            canonical_tags_final=["sad"],
            unmapped_tags=[],
            structure_score=1.0,
        ),
    ]

    candidates = select_step6_candidates(records, limit=2)

    assert [candidate.record.kaomoji for candidate in candidates] == [
        "(╥﹏╥)",
        "plain words",
    ]


def test_select_step6_candidates_excludes_obvious_long_text_annotations():
    from kaomoji_labeled_tool.step6_selection import select_step6_candidates

    records = [
        build_record(
            "ℒ☮ṽḙ is the most important thing ever!!!!",
            status="review",
            keep=False,
            review=True,
            final_score=0.55,
            canonical_tags_final=["smiling"],
            unmapped_tags=[],
            structure_score=1.0,
        ),
        build_record(
            "(╥﹏╥)",
            status="review",
            keep=False,
            review=True,
            final_score=0.6,
            canonical_tags_final=["sad"],
            unmapped_tags=[],
            structure_score=1.0,
        ),
    ]

    candidates = select_step6_candidates(records, limit=10)

    assert [candidate.record.kaomoji for candidate in candidates] == ["(╥﹏╥)"]


def test_select_step6_candidates_skips_previously_tested_kaomoji():
    from kaomoji_labeled_tool.step6_selection import select_step6_candidates

    records = [
        build_record(
            "QwQ",
            status="review",
            keep=False,
            review=True,
            final_score=0.6,
            canonical_tags_final=["crying"],
            unmapped_tags=[],
        ),
        build_record(
            "(╥﹏╥)",
            status="review",
            keep=False,
            review=True,
            final_score=0.61,
            canonical_tags_final=["sad"],
            unmapped_tags=[],
        ),
        build_record(
            "OwO",
            status="review",
            keep=False,
            review=True,
            final_score=0.62,
            canonical_tags_final=["cute"],
            unmapped_tags=[],
        ),
    ]

    candidates = select_step6_candidates(
        records,
        limit=2,
        exclude_kaomoji={"QwQ"},
    )

    assert [candidate.record.kaomoji for candidate in candidates] == [
        "OwO",
        "(╥﹏╥)",
    ]
