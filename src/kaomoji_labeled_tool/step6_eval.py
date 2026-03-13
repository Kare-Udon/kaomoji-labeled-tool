from __future__ import annotations

import json

from .models import Step6ModelOutput, Step6ParsedResult


def parse_batch_output_line(line: str) -> Step6ParsedResult:
    payload = json.loads(line)
    custom_id = payload["custom_id"]
    response = payload["response"]
    error = response.get("error")
    if error is not None:
        raise ValueError(f"Batch 响应失败：{error.get('message', 'unknown error')}")

    body = response.get("body", {})
    output_items = body.get("output", [])
    content = _extract_output_text(output_items)
    output = json.loads(content)
    return Step6ParsedResult(
        custom_id=custom_id,
        output=Step6ModelOutput(
            primary_tags=list(output["primary_tags"]),
            secondary_tags=list(output["secondary_tags"]),
            proposed_new_tag=output["proposed_new_tag"],
            reject=bool(output["reject"]),
            reject_reason=str(output["reject_reason"]),
            confidence=float(output["confidence"]),
            reason_brief=str(output["reason_brief"]),
        ),
    )


def _extract_output_text(output_items: list[dict]) -> str:
    for item in output_items:
        if item.get("type") != "message":
            continue
        for content_item in item.get("content", []):
            if content_item.get("type") == "output_text":
                return str(content_item["text"])
    raise ValueError("Batch 响应中缺少可解析的 output_text。")


def summarize_step6_results(results: list[Step6ParsedResult]) -> dict[str, int]:
    rejected = sum(1 for result in results if result.output.reject)
    proposed_new_tags = sum(1 for result in results if result.output.proposed_new_tag is not None)
    return {
        "total": len(results),
        "rejected": rejected,
        "proposed_new_tags": proposed_new_tags,
    }
