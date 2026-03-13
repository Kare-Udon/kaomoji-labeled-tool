from __future__ import annotations

from kaomoji_labeled_tool.models import (
    DedupInfo,
    QualityMetrics,
    QualityScoredRecord,
    RecordDecision,
    RecordSource,
    Step6BatchRecord,
    Step6Candidate,
    Step6FewShotExample,
    Step6ModelOutput,
)


def build_quality_record() -> QualityScoredRecord:
    return QualityScoredRecord(
        kaomoji="(╥﹏╥)",
        source=RecordSource(original_tags=["sad"], new_tags=["crying"]),
        merged_tags_raw=["sad", "crying"],
        rule_removed_tags=[],
        rule_kept_tags=["sad", "crying"],
        normalization_candidates=[],
        normalized_tags=["sad", "crying"],
        canonical_tags_final=["sad", "crying"],
        unmapped_tags=[],
        quality=QualityMetrics(
            length_score=1.0,
            structure_score=1.0,
            semantic_score=1.0,
            input_value_score=1.0,
            dedup_penalty=0.0,
            final_score=0.8,
        ),
        decision=RecordDecision(status="review", keep=False, review=True),
        dedup=DedupInfo(
            exact_key="(╥﹏╥)",
            normalized_key="(╥﹏╥)",
            near_dup_key="(╥﹏╥)",
            cluster_role="unique",
        ),
        hard_filter_reason=None,
    )


def test_step6_models_hold_candidate_and_result_payloads():
    candidate = Step6Candidate(
        sample_id="step6-0001",
        record=build_quality_record(),
        candidate_reason="review_decision",
        priority=1,
    )
    example = Step6FewShotExample(
        kaomoji="(╥﹏╥)",
        rule_tags=["sad", "crying"],
        output=Step6ModelOutput(
            primary_tags=["sad", "crying"],
            secondary_tags=["tears"],
            proposed_new_tag=None,
            reject=False,
            reject_reason="",
            confidence=0.97,
            reason_brief="明显哭泣和流泪表情",
        ),
    )
    batch_record = Step6BatchRecord(
        custom_id="step6-0001",
        method="POST",
        url="/v1/responses",
        body={"model": "gpt-5-mini"},
    )

    assert candidate.sample_id == "step6-0001"
    assert example.output.secondary_tags == ["tears"]
    assert batch_record.url == "/v1/responses"
