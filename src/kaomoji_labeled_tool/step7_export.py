from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .approved_tag_merges import merge_approved_tag
from .step6_eval import parse_batch_output_line
from .step6_review import load_review_state


@dataclass(frozen=True)
class Step7ExportResult:
    kept_path: Path
    dropped_path: Path
    summary_path: Path
    approved_tags_path: Path
    summary: dict[str, int]


def export_step7_results(
    *,
    results_dir: Path,
    testset_path: Path,
    review_state_path: Path,
    output_dir: Path,
) -> Step7ExportResult:
    sample_index = _load_testset_index(testset_path)
    review_state = load_review_state(review_state_path)

    kept_path = output_dir / "kept_records.jsonl"
    dropped_path = output_dir / "dropped_records.jsonl"
    summary_path = output_dir / "export_summary.json"
    approved_tags_path = output_dir / "approved_proposed_tags.json"

    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "total": 0,
        "kept": 0,
        "dropped": 0,
        "approved_proposed_tags_applied": 0,
        "rejected_proposed_tags_ignored": 0,
        "pending_or_unknown_proposed_tags": 0,
    }
    approved_tags: dict[str, dict[str, object]] = {}

    with (
        kept_path.open("w", encoding="utf-8") as kept_handle,
        dropped_path.open("w", encoding="utf-8") as dropped_handle,
    ):
        for output_path in sorted(results_dir.glob("*_output.jsonl")):
            with output_path.open(encoding="utf-8") as handle:
                for raw_line in handle:
                    line = raw_line.strip()
                    if not line:
                        continue
                    parsed = parse_batch_output_line(line)
                    summary["total"] += 1
                    sample = sample_index.get(parsed.custom_id, {})

                    if parsed.output.reject:
                        summary["dropped"] += 1
                        dropped_handle.write(
                            json.dumps(
                                {
                                    "sample_id": parsed.custom_id,
                                    "kaomoji": str(sample.get("kaomoji", "")),
                                    "drop_reason": "model_reject",
                                    "reject_reason": parsed.output.reject_reason,
                                    "primary_tags": parsed.output.primary_tags,
                                    "secondary_tags": parsed.output.secondary_tags,
                                    "proposed_new_tag": parsed.output.proposed_new_tag,
                                    "candidate_reason": str(sample.get("candidate_reason", "")),
                                    "rule_tags": list(sample.get("rule_tags", [])),
                                    "quality_score": float(sample.get("quality_score", 0.0)),
                                    "confidence": parsed.output.confidence,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        continue

                    final_tags = _dedupe_tags(
                        parsed.output.primary_tags
                        + parsed.output.secondary_tags
                        + _approved_proposed_tag(
                            parsed.output.proposed_new_tag,
                            review_state,
                            summary,
                            approved_tags,
                        )
                    )
                    summary["kept"] += 1
                    kept_handle.write(
                        json.dumps(
                            {
                                "sample_id": parsed.custom_id,
                                "kaomoji": str(sample.get("kaomoji", "")),
                                "final_tags": final_tags,
                                "primary_tags": parsed.output.primary_tags,
                                "secondary_tags": parsed.output.secondary_tags,
                                "approved_proposed_new_tag": _approved_tag_value(
                                    parsed.output.proposed_new_tag,
                                    review_state,
                                ),
                                "candidate_reason": str(sample.get("candidate_reason", "")),
                                "rule_tags": list(sample.get("rule_tags", [])),
                                "quality_score": float(sample.get("quality_score", 0.0)),
                                "confidence": parsed.output.confidence,
                                "reason_brief": parsed.output.reason_brief,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    approved_tags_path.write_text(
        json.dumps(approved_tags, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return Step7ExportResult(
        kept_path=kept_path,
        dropped_path=dropped_path,
        summary_path=summary_path,
        approved_tags_path=approved_tags_path,
        summary=summary,
    )


def _approved_proposed_tag(
    proposed_tag: str | None,
    review_state: dict[str, dict[str, str]],
    summary: dict[str, int],
    approved_tags: dict[str, dict[str, object]],
) -> list[str]:
    approved_tag = _approved_tag_value(proposed_tag, review_state)
    if approved_tag is not None:
        summary["approved_proposed_tags_applied"] += 1
        original_note = ""
        if proposed_tag:
            original_note = review_state.get(proposed_tag, {}).get("note", "")
        approved_tags.setdefault(
            approved_tag,
            {
                "status": "approved",
                "note": original_note,
            },
        )
        return [approved_tag]

    if proposed_tag:
        state = review_state.get(proposed_tag, {})
        status = state.get("status", "pending")
        if status == "rejected":
            summary["rejected_proposed_tags_ignored"] += 1
        else:
            summary["pending_or_unknown_proposed_tags"] += 1
    return []


def _approved_tag_value(
    proposed_tag: str | None,
    review_state: dict[str, dict[str, str]],
) -> str | None:
    if not proposed_tag:
        return None
    state = review_state.get(proposed_tag, {})
    if state.get("status", "pending") != "approved":
        return None
    return merge_approved_tag(proposed_tag)


def _dedupe_tags(tags: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for tag in tags:
        if tag in seen:
            continue
        seen.add(tag)
        deduped.append(tag)
    return deduped


def _load_testset_index(testset_path: Path) -> dict[str, dict[str, object]]:
    index: dict[str, dict[str, object]] = {}
    if not testset_path.exists():
        return index
    with testset_path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            index[str(payload["sample_id"])] = payload
    return index
