from __future__ import annotations

import json
from pathlib import Path


def _sample_group():
    from kaomoji_labeled_tool.step6_review import ProposedTagGroup, ReviewSample

    return ProposedTagGroup(
        tag="bear",
        count=2,
        avg_confidence=0.92,
        samples=[
            ReviewSample(
                sample_id="step6-0001",
                kaomoji="ʕ•ᴥ•ʔ",
                primary_tags=["smiling"],
                secondary_tags=[],
                proposed_new_tag="bear",
                reject=False,
                reject_reason="",
                confidence=0.93,
                reason_brief="bear face",
                candidate_reason="review",
                rule_tags=["smiling"],
                quality_score=0.8,
                unmapped_tags=[],
            ),
            ReviewSample(
                sample_id="step6-0002",
                kaomoji="ʕ ´•̥̥̥ ᴥ•̥̥̥`ʔ",
                primary_tags=["sad"],
                secondary_tags=["crying"],
                proposed_new_tag="bear",
                reject=False,
                reject_reason="",
                confidence=0.91,
                reason_brief="sad bear face",
                candidate_reason="review",
                rule_tags=["sad"],
                quality_score=0.78,
                unmapped_tags=[],
            ),
        ],
    )


def test_build_samples_dataframe_rows():
    from kaomoji_labeled_tool.step6_review_app import build_samples_preview_rows

    rows = build_samples_preview_rows(_sample_group(), preview_count=1)

    assert len(rows) == 1
    assert rows[0][0] == "step6-0001"
    assert rows[0][-1] == "bear"
    assert isinstance(rows[0], list)


def test_apply_review_action_persists_state(tmp_path):
    from kaomoji_labeled_tool.step6_review_app import apply_review_action

    state_path = tmp_path / "review_state.json"
    summary = apply_review_action(
        state_path=state_path,
        tag="bear",
        action="approved",
        note="动物脸保留",
    )

    assert "approved" in summary
    assert "bear" in summary
    assert state_path.exists()


def test_cli_review_mode_invokes_launch(monkeypatch, tmp_path):
    from kaomoji_labeled_tool import cli

    called = {}

    def fake_launch_review_app(**kwargs):
        called.update(kwargs)

    monkeypatch.setattr(cli, "launch_review_app", fake_launch_review_app)

    workspace = tmp_path / "workspace"
    results_dir = workspace / "step6" / "results"
    results_dir.mkdir(parents=True)
    testset_path = workspace / "step6" / "testset.jsonl"
    testset_path.write_text("", encoding="utf-8")

    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "review-tags",
            "--workspace",
            str(workspace),
            "--no-launch",
        ]
    )

    cli.run_from_args(args)

    assert called["results_dir"] == Path(results_dir)
    assert called["testset_path"] == Path(testset_path)
    assert called["state_path"] == workspace / "review" / "review_state.json"
    assert called["launch"] is False


def test_launch_review_app_attaches_keyboard_script(tmp_path):
    from kaomoji_labeled_tool.step6_review_app import launch_review_app

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    testset_path = tmp_path / "testset.jsonl"
    testset_path.write_text(
        json.dumps(
            {
                "sample_id": "step6-0001",
                "kaomoji": "ʕ•ᴥ•ʔ",
                "candidate_reason": "review",
                "rule_tags": ["smiling"],
                "unmapped_tags": [],
                "quality_score": 0.8,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    results_dir.joinpath("part-001_output.jsonl").write_text(
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
        )
        + "\n",
        encoding="utf-8",
    )

    app = launch_review_app(
        results_dir=results_dir,
        testset_path=testset_path,
        launch=False,
    )

    assert "review-action-${action}" in app.head
    assert 'bind("j", "next")' in app.head
