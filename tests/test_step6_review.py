from __future__ import annotations

import json


def test_load_review_groups_from_results(tmp_path):
    from kaomoji_labeled_tool.step6_review import load_review_groups

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    testset_path = tmp_path / "testset.jsonl"

    testset_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "sample_id": "step6-0001",
                        "kaomoji": "(＾▽＾)",
                        "candidate_reason": "review",
                        "rule_tags": ["happy"],
                        "unmapped_tags": [],
                        "quality_score": 0.73,
                    }
                ),
                json.dumps(
                    {
                        "sample_id": "step6-0002",
                        "kaomoji": "ʕ•ᴥ•ʔ",
                        "candidate_reason": "review",
                        "rule_tags": ["smiling"],
                        "unmapped_tags": [],
                        "quality_score": 0.81,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    results_dir.joinpath("part-001_output.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "custom_id": "step6-0001",
                        "response": {
                            "body": {
                                "output": [
                                    {
                                        "type": "message",
                                        "content": [
                                            {
                                                "type": "output_text",
                                                "text": json.dumps(
                                                    {
                                                        "primary_tags": ["happy"],
                                                        "secondary_tags": ["smiling"],
                                                        "proposed_new_tag": None,
                                                        "reject": False,
                                                        "reject_reason": "",
                                                        "confidence": 0.91,
                                                        "reason_brief": "happy smile",
                                                    }
                                                ),
                                            }
                                        ],
                                    }
                                ]
                            }
                        },
                    }
                ),
                json.dumps(
                    {
                        "custom_id": "step6-0002",
                        "response": {
                            "body": {
                                "output": [
                                    {
                                        "type": "message",
                                        "content": [
                                            {
                                                "type": "output_text",
                                                "text": json.dumps(
                                                    {
                                                        "primary_tags": ["smiling"],
                                                        "secondary_tags": [],
                                                        "proposed_new_tag": "bear",
                                                        "reject": False,
                                                        "reject_reason": "",
                                                        "confidence": 0.93,
                                                        "reason_brief": "bear face",
                                                    }
                                                ),
                                            }
                                        ],
                                    }
                                ]
                            }
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    groups = load_review_groups(results_dir=results_dir, testset_path=testset_path)

    assert [group.tag for group in groups] == ["bear"]
    assert groups[0].count == 1
    assert groups[0].samples[0].kaomoji == "ʕ•ᴥ•ʔ"
    assert groups[0].samples[0].rule_tags == ["smiling"]


def test_review_state_round_trip(tmp_path):
    from kaomoji_labeled_tool.step6_review import load_review_state, save_review_decision

    state_path = tmp_path / "review_state.json"

    empty = load_review_state(state_path)
    assert empty == {}

    save_review_decision(
        state_path=state_path,
        tag="bear",
        status="approved",
        note="保留为动物类标签",
    )
    state = load_review_state(state_path)
    assert state["bear"]["status"] == "approved"
    assert state["bear"]["note"] == "保留为动物类标签"
    assert "updated_at" in state["bear"]

    save_review_decision(
        state_path=state_path,
        tag="bear",
        status="rejected",
        note="改为并回 smiling",
    )
    updated = load_review_state(state_path)
    assert updated["bear"]["status"] == "rejected"
    assert updated["bear"]["note"] == "改为并回 smiling"
