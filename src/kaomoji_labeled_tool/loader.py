from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import InputRecord


def load_dataset(path: str | Path) -> list[InputRecord]:
    dataset_path = Path(path)
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("数据集根节点必须是对象，键为 kaomoji。")

    records: list[InputRecord] = []
    for kaomoji, raw_record in payload.items():
        records.append(_build_record(kaomoji=kaomoji, raw_record=raw_record))
    return records


def _build_record(*, kaomoji: Any, raw_record: Any) -> InputRecord:
    if not isinstance(kaomoji, str) or not kaomoji:
        raise ValueError("每条记录都必须包含非空 kaomoji。")
    if not isinstance(raw_record, dict):
        raise TypeError(f"{kaomoji!r} 对应的记录必须是对象。")

    original_tags = _require_tag_list(raw_record, "original_tags", kaomoji)
    new_tags = _require_tag_list(raw_record, "new_tags", kaomoji)
    return InputRecord(
        kaomoji=kaomoji,
        original_tags=original_tags,
        new_tags=new_tags,
    )


def _require_tag_list(raw_record: dict[str, Any], field_name: str, kaomoji: str) -> list[str]:
    if field_name not in raw_record:
        raise ValueError(f"{kaomoji!r} 缺少必填字段 {field_name!r}。")

    value = raw_record[field_name]
    if not isinstance(value, list):
        raise TypeError(f"{kaomoji!r} 的 {field_name!r} 必须是列表。")
    if any(not isinstance(item, str) for item in value):
        raise TypeError(f"{kaomoji!r} 的 {field_name!r} 只能包含字符串。")
    return value
