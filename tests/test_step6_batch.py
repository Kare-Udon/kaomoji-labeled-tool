from __future__ import annotations

import json

from kaomoji_labeled_tool.models import (
    DedupInfo,
    QualityMetrics,
    QualityScoredRecord,
    RecordDecision,
    RecordSource,
    Step6Candidate,
)


def build_candidate() -> Step6Candidate:
    record = QualityScoredRecord(
        kaomoji="(╥﹏╥)",
        source=RecordSource(original_tags=["sad"], new_tags=["crying"]),
        merged_tags_raw=["sad", "crying"],
        rule_removed_tags=[],
        rule_kept_tags=["sad", "crying"],
        normalization_candidates=[],
        normalized_tags=["sad", "crying"],
        canonical_tags_final=["sad", "crying"],
        unmapped_tags=["tears"],
        quality=QualityMetrics(
            length_score=1.0,
            structure_score=1.0,
            semantic_score=1.0,
            input_value_score=1.0,
            dedup_penalty=0.0,
            final_score=0.72,
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
    return Step6Candidate(
        sample_id="step6-0001",
        record=record,
        candidate_reason="review_decision",
        priority=1,
    )


def test_build_batch_record_uses_responses_api_shape():
    from kaomoji_labeled_tool.step6_batch import build_batch_record
    from kaomoji_labeled_tool.step6_prompting import (
        build_default_few_shot_examples,
        build_step6_instructions,
        build_step6_json_schema,
    )

    schema = build_step6_json_schema(["sad", "crying", "tears", "happy"])
    instructions = build_step6_instructions(
        canonical_tags=["sad", "crying", "tears", "happy"],
        few_shot_examples=build_default_few_shot_examples(),
        schema=schema,
    )

    record = build_batch_record(build_candidate(), instructions, schema, model="gpt-5-mini")

    assert record.url == "/v1/responses"
    assert record.body["model"] == "gpt-5-mini"
    assert record.body["reasoning"] == {"effort": "minimal"}
    assert record.body["prompt_cache_key"] == "kaomoji-labeled-tool-step6-batch-final"
    assert record.body["prompt_cache_retention"] == "in_memory"
    assert record.body["text"]["format"]["type"] == "json_schema"


def test_build_batch_record_uses_none_reasoning_for_gpt_5_4():
    from kaomoji_labeled_tool.step6_batch import build_batch_record
    from kaomoji_labeled_tool.step6_prompting import (
        build_default_few_shot_examples,
        build_step6_instructions,
        build_step6_json_schema,
    )

    schema = build_step6_json_schema(["sad", "crying", "tears", "happy"])
    instructions = build_step6_instructions(
        canonical_tags=["sad", "crying", "tears", "happy"],
        few_shot_examples=build_default_few_shot_examples(),
        schema=schema,
    )

    record = build_batch_record(build_candidate(), instructions, schema, model="gpt-5.4")

    assert record.body["reasoning"] == {"effort": "none"}


def test_write_batch_requests_outputs_jsonl(tmp_path):
    from kaomoji_labeled_tool.step6_batch import write_batch_requests
    from kaomoji_labeled_tool.step6_prompting import (
        build_default_few_shot_examples,
        build_step6_instructions,
        build_step6_json_schema,
    )

    output_path = tmp_path / "batch.jsonl"
    schema = build_step6_json_schema(["sad", "crying", "tears", "happy"])
    instructions = build_step6_instructions(
        canonical_tags=["sad", "crying", "tears", "happy"],
        few_shot_examples=build_default_few_shot_examples(),
        schema=schema,
    )

    write_batch_requests([build_candidate()], output_path, instructions, schema, model="gpt-5-mini")

    line = output_path.read_text(encoding="utf-8").strip()
    payload = json.loads(line)
    assert payload["custom_id"] == "step6-0001"


def test_write_few_shot_examples_outputs_user_gold_examples(tmp_path):
    from kaomoji_labeled_tool.step6_batch import write_few_shot_examples
    from kaomoji_labeled_tool.step6_prompting import build_default_few_shot_examples

    output_path = tmp_path / "few_shot.json"
    write_few_shot_examples(build_default_few_shot_examples(), output_path)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(payload) == 10
    assert payload[0]["kaomoji"] == "(*´ω`*)"
    assert payload[-2]["kaomoji"] == "NOWLOADING"
    assert payload[-1]["kaomoji"] == "ʕ•ᴥ•ʔ"


def test_prepare_step6_batch_artifacts_uses_stable_workspace_filenames(tmp_path):
    from kaomoji_labeled_tool.models import PipelineConfig, RemovalConfig, ExportConfig, NormalizationConfig
    from kaomoji_labeled_tool.step6_batch import prepare_step6_batch_artifacts

    result = prepare_step6_batch_artifacts(
        config=PipelineConfig(
            canonical_tags=("sad", "crying", "tears", "happy"),
            normalization=NormalizationConfig(lemma_map={}, semantic_map={}),
            removal=RemovalConfig(exact=(), patterns=()),
            aliases={},
            export=ExportConfig(prefix="km"),
        ),
        scored_records=[build_candidate().record],
        output_dir=tmp_path,
        sample_size=1,
    )

    assert result.few_shot_path == tmp_path / "few_shot_gold.json"
    assert result.testset_path == tmp_path / "testset.jsonl"
    assert result.batch_path == tmp_path / "batch_requests.jsonl"
