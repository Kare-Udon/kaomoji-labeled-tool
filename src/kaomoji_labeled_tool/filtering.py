from __future__ import annotations

import re

from .models import FilteredRecord, PreprocessedRecord, RemovalConfig


def filter_record(record: PreprocessedRecord, removal: RemovalConfig) -> FilteredRecord:
    removed_tags: list[str] = []
    kept_tags: list[str] = []
    normalization_candidates: list[str] = []

    for tag in record.merged_tags_raw:
        if should_remove_tag(tag, removal):
            removed_tags.append(tag)
            continue

        kept_tags.append(tag)
        if is_normalization_candidate(tag):
            normalization_candidates.append(tag)

    return FilteredRecord(
        kaomoji=record.kaomoji,
        source=record.source,
        merged_tags_raw=list(record.merged_tags_raw),
        rule_removed_tags=removed_tags,
        rule_kept_tags=kept_tags,
        normalization_candidates=normalization_candidates,
    )


def should_remove_tag(tag: str, removal: RemovalConfig) -> bool:
    if tag in removal.exact:
        return True
    return any(re.search(pattern, tag) for pattern in removal.patterns)


def is_normalization_candidate(tag: str) -> bool:
    if "_" in tag or "/" in tag or "|" in tag:
        return True
    return tag.count("-") > 1
