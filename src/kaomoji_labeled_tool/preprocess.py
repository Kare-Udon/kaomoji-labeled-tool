from __future__ import annotations

import re
import unicodedata

from .models import InputRecord, PreprocessedRecord, RecordSource

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_tag(tag: str) -> str:
    normalized = unicodedata.normalize("NFKC", tag)
    normalized = _WHITESPACE_RE.sub(" ", normalized).strip().lower()
    return normalized


def merge_raw_tags(record: InputRecord) -> list[str]:
    merged_tags: list[str] = []
    seen: set[str] = set()
    for raw_tag in [*record.original_tags, *record.new_tags]:
        normalized = normalize_tag(raw_tag)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        merged_tags.append(normalized)
    return merged_tags


def preprocess_record(record: InputRecord) -> PreprocessedRecord:
    return PreprocessedRecord(
        kaomoji=record.kaomoji,
        source=RecordSource(
            original_tags=list(record.original_tags),
            new_tags=list(record.new_tags),
        ),
        merged_tags_raw=merge_raw_tags(record),
    )
