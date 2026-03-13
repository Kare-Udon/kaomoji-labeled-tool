from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class InputRecord:
    kaomoji: str
    original_tags: list[str]
    new_tags: list[str]


@dataclass(frozen=True)
class RecordSource:
    original_tags: list[str]
    new_tags: list[str]


@dataclass(frozen=True)
class PreprocessedRecord:
    kaomoji: str
    source: RecordSource
    merged_tags_raw: list[str]


@dataclass(frozen=True)
class FilteredRecord:
    kaomoji: str
    source: RecordSource
    merged_tags_raw: list[str]
    rule_removed_tags: list[str]
    rule_kept_tags: list[str]
    normalization_candidates: list[str]


@dataclass(frozen=True)
class NormalizedRecord:
    kaomoji: str
    source: RecordSource
    merged_tags_raw: list[str]
    rule_removed_tags: list[str]
    rule_kept_tags: list[str]
    normalization_candidates: list[str]
    normalized_tags: list[str]
    canonical_tags_final: list[str]
    unmapped_tags: list[str]


@dataclass(frozen=True)
class PipelineRunResult:
    config: "PipelineConfig"
    records: list[NormalizedRecord]


@dataclass(frozen=True)
class QualityMetrics:
    length_score: float
    structure_score: float
    semantic_score: float
    input_value_score: float
    dedup_penalty: float
    final_score: float


@dataclass(frozen=True)
class RecordDecision:
    status: str
    keep: bool
    review: bool


@dataclass(frozen=True)
class DedupInfo:
    exact_key: str
    normalized_key: str
    near_dup_key: str | None
    cluster_role: str


@dataclass(frozen=True)
class QualityScoredRecord:
    kaomoji: str
    source: RecordSource
    merged_tags_raw: list[str]
    rule_removed_tags: list[str]
    rule_kept_tags: list[str]
    normalization_candidates: list[str]
    normalized_tags: list[str]
    canonical_tags_final: list[str]
    unmapped_tags: list[str]
    quality: QualityMetrics
    decision: RecordDecision
    dedup: DedupInfo
    hard_filter_reason: str | None


@dataclass(frozen=True)
class QualityPipelineRunResult:
    config: "PipelineConfig"
    records: list[QualityScoredRecord]


@dataclass(frozen=True)
class Step6ModelOutput:
    primary_tags: list[str]
    secondary_tags: list[str]
    proposed_new_tag: str | None
    reject: bool
    reject_reason: str
    confidence: float
    reason_brief: str


@dataclass(frozen=True)
class Step6FewShotExample:
    kaomoji: str
    rule_tags: list[str]
    output: Step6ModelOutput


@dataclass(frozen=True)
class Step6Candidate:
    sample_id: str
    record: QualityScoredRecord
    candidate_reason: str
    priority: int


@dataclass(frozen=True)
class Step6BatchRecord:
    custom_id: str
    method: str
    url: str
    body: dict[str, Any]


@dataclass(frozen=True)
class Step6ParsedResult:
    custom_id: str
    output: Step6ModelOutput


@dataclass(frozen=True)
class Step6PreparationResult:
    config: "PipelineConfig"
    scored_records: list[QualityScoredRecord]
    candidates: list[Step6Candidate]
    few_shot_examples: list[Step6FewShotExample]
    few_shot_path: Path
    testset_path: Path
    batch_path: Path


@dataclass(frozen=True)
class NormalizationConfig:
    lemma_map: dict[str, str]
    semantic_map: dict[str, str]


@dataclass(frozen=True)
class RemovalConfig:
    exact: tuple[str, ...]
    patterns: tuple[str, ...]


@dataclass(frozen=True)
class ExportConfig:
    prefix: str


@dataclass(frozen=True)
class PipelineConfig:
    canonical_tags: tuple[str, ...]
    normalization: NormalizationConfig
    removal: RemovalConfig
    aliases: dict[str, dict[str, tuple[str, ...]]]
    export: ExportConfig
