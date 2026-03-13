from __future__ import annotations

import json


def test_parse_batch_output_line_and_summarize():
    from kaomoji_labeled_tool.step6_eval import parse_batch_output_line, summarize_step6_results

    line = json.dumps(
        {
            "custom_id": "step6-0001",
            "response": {
                "body": {
                    "output": [
                        {
                            "type": "reasoning",
                            "summary": [],
                            "content": None,
                        },
                        {
                            "type": "message",
                            "role": "assistant",
                            "status": "completed",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": json.dumps(
                                        {
                                            "primary_tags": ["sad", "crying"],
                                            "secondary_tags": ["tears"],
                                            "proposed_new_tag": None,
                                            "reject": False,
                                            "reject_reason": "",
                                            "confidence": 0.97,
                                            "reason_brief": "明显哭泣和流泪表情",
                                        },
                                        ensure_ascii=False,
                                    ),
                                }
                            ],
                        }
                    ]
                }
            },
        },
        ensure_ascii=False,
    )

    result = parse_batch_output_line(line)
    summary = summarize_step6_results([result])

    assert result.custom_id == "step6-0001"
    assert result.output.primary_tags == ["sad", "crying"]
    assert summary["total"] == 1
    assert summary["rejected"] == 0


def test_parse_batch_output_line_raises_on_response_error():
    from kaomoji_labeled_tool.step6_eval import parse_batch_output_line

    line = json.dumps(
        {
            "custom_id": "step6-0002",
            "response": {
                "error": {
                    "message": "invalid request",
                },
                "body": {},
            },
        }
    )

    try:
        parse_batch_output_line(line)
    except ValueError as exc:
        assert "invalid request" in str(exc)
    else:
        raise AssertionError("expected ValueError")
