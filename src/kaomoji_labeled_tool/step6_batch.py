from __future__ import annotations

import json
from pathlib import Path

from .models import Step6BatchRecord, Step6Candidate, Step6FewShotExample, Step6PreparationResult
from .step6_prompting import (
    build_default_few_shot_examples,
    build_step6_instructions,
    build_step6_json_schema,
)
from .step6_render import render_kaomoji_image_base64
from .step6_selection import select_step6_candidates


def _reasoning_effort_for_model(model: str) -> str:
    if model.startswith("gpt-5.4"):
        return "none"
    return "minimal"


def build_batch_record(
    candidate: Step6Candidate,
    instructions: str,
    schema: dict[str, object],
    *,
    model: str,
) -> Step6BatchRecord:
    record = candidate.record
    body = {
        "model": model,
        "instructions": instructions,
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": json.dumps(
                            {
                                "sample_id": candidate.sample_id,
                                "kaomoji": record.kaomoji,
                                "rule_tags": record.canonical_tags_final,
                                "normalized_tags": record.normalized_tags,
                                "unmapped_tags": record.unmapped_tags,
                                "quality_score": round(record.quality.final_score, 4),
                                "candidate_reason": candidate.candidate_reason,
                            },
                            ensure_ascii=False,
                        ),
                    },
                    {
                        "type": "input_image",
                        "image_url": (
                            "data:image/png;base64,"
                            + render_kaomoji_image_base64(record.kaomoji)
                        ),
                    },
                ],
            }
        ],
        "reasoning": {"effort": _reasoning_effort_for_model(model)},
        "max_output_tokens": 300,
        "prompt_cache_key": "kaomoji-labeled-tool-step6-batch-final",
        "prompt_cache_retention": "in_memory",
        "store": False,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "kaomoji_step6_output",
                "schema": schema,
                "strict": True,
            },
            "verbosity": "low",
        },
    }
    return Step6BatchRecord(
        custom_id=candidate.sample_id,
        method="POST",
        url="/v1/responses",
        body=body,
    )


def write_batch_requests(
    candidates: list[Step6Candidate],
    output_path: str | Path,
    instructions: str,
    schema: dict[str, object],
    *,
    model: str,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(
            build_batch_record(candidate, instructions, schema, model=model).__dict__,
            ensure_ascii=False,
        )
        for candidate in candidates
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_testset(
    candidates: list[Step6Candidate],
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(
            {
                "sample_id": candidate.sample_id,
                "kaomoji": candidate.record.kaomoji,
                "candidate_reason": candidate.candidate_reason,
                "rule_tags": candidate.record.canonical_tags_final,
                "unmapped_tags": candidate.record.unmapped_tags,
                "quality_score": round(candidate.record.quality.final_score, 4),
            },
            ensure_ascii=False,
        )
        for candidate in candidates
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_few_shot_examples(
    few_shot_examples: list[Step6FewShotExample],
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "kaomoji": example.kaomoji,
            "rule_tags": example.rule_tags,
            "output": example.output.__dict__,
        }
        for example in few_shot_examples
    ]
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def prepare_step6_batch_artifacts(
    *,
    config,
    scored_records,
    output_dir: str | Path,
    sample_size: int,
    model: str = "gpt-5-mini",
    exclude_kaomoji: set[str] | None = None,
) -> Step6PreparationResult:
    candidates = select_step6_candidates(
        scored_records,
        limit=sample_size,
        exclude_kaomoji=exclude_kaomoji,
    )
    few_shot_examples: list[Step6FewShotExample] = build_default_few_shot_examples()
    schema = build_step6_json_schema(list(config.canonical_tags))
    instructions = build_step6_instructions(
        canonical_tags=list(config.canonical_tags),
        few_shot_examples=few_shot_examples,
        schema=schema,
    )

    base_dir = Path(output_dir)
    few_shot_path = base_dir / "few_shot_gold.json"
    testset_path = base_dir / "testset.jsonl"
    batch_path = base_dir / "batch_requests.jsonl"
    write_few_shot_examples(few_shot_examples, few_shot_path)
    write_testset(candidates, testset_path)
    write_batch_requests(candidates, batch_path, instructions, schema, model=model)

    return Step6PreparationResult(
        config=config,
        scored_records=list(scored_records),
        candidates=candidates,
        few_shot_examples=few_shot_examples,
        few_shot_path=few_shot_path,
        testset_path=testset_path,
        batch_path=batch_path,
    )
