from __future__ import annotations

from pathlib import Path

from .config import load_pipeline_config
from .filtering import filter_record
from .loader import load_dataset
from .models import PipelineRunResult, QualityPipelineRunResult, Step6PreparationResult
from .normalization import normalize_record
from .preprocess import preprocess_record
from .quality import score_records
from .step6_batch import prepare_step6_batch_artifacts


def run_pipeline_steps_1_to_4(
    dataset_path: str | Path,
    config_dir: str | Path,
) -> PipelineRunResult:
    config, normalized_records = _run_pipeline_steps_1_to_4_records(dataset_path, config_dir)
    return PipelineRunResult(
        config=config,
        records=normalized_records,
    )


def run_pipeline_steps_1_to_5(
    dataset_path: str | Path,
    config_dir: str | Path,
) -> QualityPipelineRunResult:
    config, normalized_records = _run_pipeline_steps_1_to_4_records(dataset_path, config_dir)
    quality_records = score_records(normalized_records)
    return QualityPipelineRunResult(
        config=config,
        records=quality_records,
    )


def run_pipeline_steps_1_to_6_prep(
    *,
    dataset_path: str | Path,
    config_dir: str | Path,
    output_dir: str | Path,
    sample_size: int = 50,
    exclude_kaomoji: set[str] | None = None,
) -> Step6PreparationResult:
    result = run_pipeline_steps_1_to_5(dataset_path, config_dir)
    return prepare_step6_batch_artifacts(
        config=result.config,
        scored_records=result.records,
        output_dir=output_dir,
        sample_size=sample_size,
        exclude_kaomoji=exclude_kaomoji,
    )


def _run_pipeline_steps_1_to_4_records(
    dataset_path: str | Path,
    config_dir: str | Path,
):
    config = load_pipeline_config(config_dir)
    input_records = load_dataset(dataset_path)
    normalized_records = []

    for input_record in input_records:
        preprocessed = preprocess_record(input_record)
        filtered = filter_record(preprocessed, config.removal)
        normalized = normalize_record(filtered, config.normalization, config.canonical_tags)
        normalized_records.append(normalized)

    return config, normalized_records
