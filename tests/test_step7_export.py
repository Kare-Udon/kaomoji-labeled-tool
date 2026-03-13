from __future__ import annotations

import json
from pathlib import Path


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _make_output_line(
    *,
    sample_id: str,
    primary_tags: list[str],
    secondary_tags: list[str],
    proposed_new_tag: str | None,
    reject: bool,
    reject_reason: str,
    confidence: float = 0.9,
    reason_brief: str = "ok",
) -> dict:
    return {
        "custom_id": sample_id,
        "response": {
            "body": {
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {
                                "type": "output_text",
                                "text": json.dumps(
                                    {
                                        "primary_tags": primary_tags,
                                        "secondary_tags": secondary_tags,
                                        "proposed_new_tag": proposed_new_tag,
                                        "reject": reject,
                                        "reject_reason": reject_reason,
                                        "confidence": confidence,
                                        "reason_brief": reason_brief,
                                    },
                                    ensure_ascii=False,
                                ),
                            }
                        ],
                    }
                ]
            }
        },
    }


def test_export_step7_results_applies_review_state_and_reject_policy(tmp_path: Path):
    from kaomoji_labeled_tool.step7_export import export_step7_results

    results_dir = tmp_path / "results"
    testset_path = tmp_path / "testset.jsonl"
    review_state_path = tmp_path / "review_state.json"
    output_dir = tmp_path / "exports"

    _write_jsonl(
        results_dir / "part-001_output.jsonl",
        [
            _make_output_line(
                sample_id="s1",
                primary_tags=["smiling"],
                secondary_tags=["happy"],
                proposed_new_tag="bear",
                reject=False,
                reject_reason="",
            ),
            _make_output_line(
                sample_id="s2",
                primary_tags=["smiling"],
                secondary_tags=[],
                proposed_new_tag="computer",
                reject=False,
                reject_reason="",
            ),
            _make_output_line(
                sample_id="s3",
                primary_tags=["angry"],
                secondary_tags=["hands"],
                proposed_new_tag="table_flip",
                reject=True,
                reject_reason="scene ascii",
            ),
        ],
    )
    _write_jsonl(
        testset_path,
        [
            {
                "sample_id": "s1",
                "kaomoji": "ʕ•ᴥ•ʔ",
                "candidate_reason": "rule_tags_present",
                "rule_tags": ["smiling"],
                "quality_score": 0.88,
                "unmapped_tags": [],
            },
            {
                "sample_id": "s2",
                "kaomoji": "( ͡° ͜ʖ ͡°)",
                "candidate_reason": "review_decision",
                "rule_tags": ["smiling"],
                "quality_score": 0.77,
                "unmapped_tags": [],
            },
            {
                "sample_id": "s3",
                "kaomoji": "(╯°□°)╯︵ ┻━┻",
                "candidate_reason": "review_decision",
                "rule_tags": ["angry"],
                "quality_score": 0.61,
                "unmapped_tags": [],
            },
        ],
    )
    review_state_path.write_text(
        json.dumps(
            {
                "bear": {"status": "approved", "note": ""},
                "computer": {"status": "rejected", "note": ""},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    result = export_step7_results(
        results_dir=results_dir,
        testset_path=testset_path,
        review_state_path=review_state_path,
        output_dir=output_dir,
    )

    assert result.summary["total"] == 3
    assert result.summary["kept"] == 2
    assert result.summary["dropped"] == 1
    assert result.summary["approved_proposed_tags_applied"] == 1

    kept_rows = [
        json.loads(line)
        for line in result.kept_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    dropped_rows = [
        json.loads(line)
        for line in result.dropped_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert kept_rows[0]["sample_id"] == "s1"
    assert kept_rows[0]["final_tags"] == ["smiling", "happy", "bear"]
    assert kept_rows[0]["approved_proposed_new_tag"] == "bear"

    assert kept_rows[1]["sample_id"] == "s2"
    assert kept_rows[1]["final_tags"] == ["smiling"]
    assert kept_rows[1]["approved_proposed_new_tag"] is None

    assert dropped_rows == [
        {
            "sample_id": "s3",
            "kaomoji": "(╯°□°)╯︵ ┻━┻",
            "drop_reason": "model_reject",
            "reject_reason": "scene ascii",
            "primary_tags": ["angry"],
            "secondary_tags": ["hands"],
            "proposed_new_tag": "table_flip",
            "candidate_reason": "review_decision",
            "rule_tags": ["angry"],
            "quality_score": 0.61,
            "confidence": 0.9,
        }
    ]


def test_export_step7_results_treats_unknown_proposed_tag_as_pending(tmp_path: Path):
    from kaomoji_labeled_tool.step7_export import export_step7_results

    results_dir = tmp_path / "results"
    testset_path = tmp_path / "testset.jsonl"
    review_state_path = tmp_path / "review_state.json"
    output_dir = tmp_path / "exports"

    _write_jsonl(
        results_dir / "part-001_output.jsonl",
        [
            _make_output_line(
                sample_id="s1",
                primary_tags=["waving"],
                secondary_tags=["happy"],
                proposed_new_tag="greeting_pose",
                reject=False,
                reject_reason="",
            ),
        ],
    )
    _write_jsonl(
        testset_path,
        [
            {
                "sample_id": "s1",
                "kaomoji": "( ´ ▽ ` )ﾉ",
                "candidate_reason": "rule_tags_present",
                "rule_tags": ["waving"],
                "quality_score": 0.91,
                "unmapped_tags": [],
            },
        ],
    )
    review_state_path.write_text("{}\n", encoding="utf-8")

    result = export_step7_results(
        results_dir=results_dir,
        testset_path=testset_path,
        review_state_path=review_state_path,
        output_dir=output_dir,
    )

    kept_rows = [
        json.loads(line)
        for line in result.kept_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert kept_rows == [
        {
            "sample_id": "s1",
            "kaomoji": "( ´ ▽ ` )ﾉ",
            "final_tags": ["waving", "happy"],
            "primary_tags": ["waving"],
            "secondary_tags": ["happy"],
            "approved_proposed_new_tag": None,
            "candidate_reason": "rule_tags_present",
            "rule_tags": ["waving"],
            "quality_score": 0.91,
            "confidence": 0.9,
            "reason_brief": "ok",
        }
    ]
    assert result.summary["pending_or_unknown_proposed_tags"] == 1


def test_export_step7_results_dedupes_approved_proposed_tag_against_existing_tags(tmp_path: Path):
    from kaomoji_labeled_tool.step7_export import export_step7_results

    results_dir = tmp_path / "results"
    testset_path = tmp_path / "testset.jsonl"
    review_state_path = tmp_path / "review_state.json"
    output_dir = tmp_path / "exports"

    _write_jsonl(
        results_dir / "part-001_output.jsonl",
        [
            _make_output_line(
                sample_id="s1",
                primary_tags=["smiling"],
                secondary_tags=["happy"],
                proposed_new_tag="happy",
                reject=False,
                reject_reason="",
            ),
        ],
    )
    _write_jsonl(
        testset_path,
        [
            {
                "sample_id": "s1",
                "kaomoji": "(＾▽＾)",
                "candidate_reason": "rule_tags_present",
                "rule_tags": ["smiling"],
                "quality_score": 0.92,
                "unmapped_tags": [],
            },
        ],
    )
    review_state_path.write_text(
        json.dumps(
            {"happy": {"status": "approved", "note": "与已有标签重合"}},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    result = export_step7_results(
        results_dir=results_dir,
        testset_path=testset_path,
        review_state_path=review_state_path,
        output_dir=output_dir,
    )

    kept_rows = [
        json.loads(line)
        for line in result.kept_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert kept_rows[0]["final_tags"] == ["smiling", "happy"]
    assert result.summary["approved_proposed_tags_applied"] == 1


def test_export_step7_results_merges_approved_proposed_tag_to_canonical_target(tmp_path: Path):
    from kaomoji_labeled_tool.step7_export import export_step7_results

    results_dir = tmp_path / "results"
    testset_path = tmp_path / "testset.jsonl"
    review_state_path = tmp_path / "review_state.json"
    output_dir = tmp_path / "exports"

    _write_jsonl(
        results_dir / "part-001_output.jsonl",
        [
            _make_output_line(
                sample_id="s1",
                primary_tags=["smiling"],
                secondary_tags=["hands"],
                proposed_new_tag="bunny",
                reject=False,
                reject_reason="",
            ),
        ],
    )
    _write_jsonl(
        testset_path,
        [
            {
                "sample_id": "s1",
                "kaomoji": "(•×•U)",
                "candidate_reason": "review_decision",
                "rule_tags": ["smiling"],
                "quality_score": 0.85,
                "unmapped_tags": [],
            },
        ],
    )
    review_state_path.write_text(
        json.dumps(
            {"bunny": {"status": "approved", "note": "可并入 rabbit"}},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    result = export_step7_results(
        results_dir=results_dir,
        testset_path=testset_path,
        review_state_path=review_state_path,
        output_dir=output_dir,
    )

    kept_rows = [
        json.loads(line)
        for line in result.kept_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    approved_tags = json.loads(result.approved_tags_path.read_text(encoding="utf-8"))

    assert kept_rows == [
        {
            "sample_id": "s1",
            "kaomoji": "(•×•U)",
            "final_tags": ["smiling", "hands", "rabbit"],
            "primary_tags": ["smiling"],
            "secondary_tags": ["hands"],
            "approved_proposed_new_tag": "rabbit",
            "candidate_reason": "review_decision",
            "rule_tags": ["smiling"],
            "quality_score": 0.85,
            "confidence": 0.9,
            "reason_brief": "ok",
        }
    ]
    assert approved_tags == {
        "rabbit": {"status": "approved", "note": "可并入 rabbit"}
    }
