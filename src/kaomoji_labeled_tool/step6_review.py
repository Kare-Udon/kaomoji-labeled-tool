from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .step6_eval import parse_batch_output_line


@dataclass(frozen=True)
class ReviewSample:
    sample_id: str
    kaomoji: str
    primary_tags: list[str]
    secondary_tags: list[str]
    proposed_new_tag: str
    reject: bool
    reject_reason: str
    confidence: float
    reason_brief: str
    candidate_reason: str
    rule_tags: list[str]
    quality_score: float
    unmapped_tags: list[str]


@dataclass(frozen=True)
class ProposedTagGroup:
    tag: str
    count: int
    avg_confidence: float
    samples: list[ReviewSample]


def load_review_groups(results_dir: Path, testset_path: Path) -> list[ProposedTagGroup]:
    sample_index = _load_testset_index(testset_path)
    groups: dict[str, list[ReviewSample]] = {}

    for output_path in sorted(results_dir.glob("*_output.jsonl")):
        with output_path.open(encoding="utf-8") as handle:
            for line in handle:
                parsed = parse_batch_output_line(line)
                proposed_tag = parsed.output.proposed_new_tag
                if not proposed_tag:
                    continue
                sample = sample_index.get(parsed.custom_id, {})
                groups.setdefault(proposed_tag, []).append(
                    ReviewSample(
                        sample_id=parsed.custom_id,
                        kaomoji=str(sample.get("kaomoji", "")),
                        primary_tags=list(parsed.output.primary_tags),
                        secondary_tags=list(parsed.output.secondary_tags),
                        proposed_new_tag=proposed_tag,
                        reject=parsed.output.reject,
                        reject_reason=parsed.output.reject_reason,
                        confidence=parsed.output.confidence,
                        reason_brief=parsed.output.reason_brief,
                        candidate_reason=str(sample.get("candidate_reason", "")),
                        rule_tags=list(sample.get("rule_tags", [])),
                        quality_score=float(sample.get("quality_score", 0.0)),
                        unmapped_tags=list(sample.get("unmapped_tags", [])),
                    )
                )

    review_groups = [
        ProposedTagGroup(
            tag=tag,
            count=len(samples),
            avg_confidence=round(
                sum(sample.confidence for sample in samples) / len(samples), 4
            ),
            samples=sorted(samples, key=lambda item: (-item.confidence, item.sample_id)),
        )
        for tag, samples in groups.items()
    ]
    return sorted(review_groups, key=lambda group: (-group.count, group.tag))


def load_review_state(state_path: Path) -> dict[str, dict[str, str]]:
    if not state_path.exists():
        return {}
    with state_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("审阅状态文件格式错误，顶层必须是对象。")
    return payload


def save_review_decision(state_path: Path, tag: str, status: str, note: str) -> None:
    state = load_review_state(state_path)
    state[tag] = {
        "status": status,
        "note": note,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def summarize_group(group: ProposedTagGroup, state: dict[str, dict[str, str]]) -> str:
    decision = state.get(group.tag, {})
    status = decision.get("status", "pending")
    note = decision.get("note", "")
    note_suffix = f"；备注：{note}" if note else ""
    return (
        f"tag={group.tag}｜样本数={group.count}｜平均置信度={group.avg_confidence:.2f}"
        f"｜审批状态={status}{note_suffix}"
    )


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
