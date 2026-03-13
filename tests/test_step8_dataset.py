from __future__ import annotations

import json
from pathlib import Path


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_export_hf_dataset_writes_expected_fields_and_files(tmp_path: Path):
    from kaomoji_labeled_tool.step8_dataset import export_hf_dataset

    kept_path = tmp_path / "kept_records.jsonl"
    output_dir = tmp_path / "hf_dataset"

    _write_jsonl(
        kept_path,
        [
            {
                "sample_id": "s1",
                "kaomoji": "ʕ•ᴥ•ʔ",
                "final_tags": ["smiling", "happy", "bear"],
                "primary_tags": ["smiling"],
                "secondary_tags": ["happy"],
                "approved_proposed_new_tag": "bear",
                "quality_score": 0.88,
                "confidence": 0.93,
            },
            {
                "sample_id": "s2",
                "kaomoji": "( ´ ▽ ` )ﾉ",
                "final_tags": ["waving", "happy"],
                "primary_tags": ["waving"],
                "secondary_tags": ["happy", "happy"],
                "approved_proposed_new_tag": None,
                "quality_score": 0.77,
                "confidence": 0.91,
            },
        ],
    )

    result = export_hf_dataset(
        kept_records_path=kept_path,
        output_dir=output_dir,
    )

    assert result.summary["total"] == 2
    assert result.parquet_path.exists()
    assert result.jsonl_path.exists()
    assert result.schema_path.exists()
    assert result.readme_path.exists()

    rows = [
        json.loads(line)
        for line in result.jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows == [
        {
            "id": "s1",
            "kaomoji": "ʕ•ᴥ•ʔ",
            "tags": ["smiling", "happy", "bear"],
            "primary_tags": ["smiling"],
            "secondary_tags": ["happy", "bear"],
            "source": "emoticon_dict+llm+human_review",
            "quality_score": 0.88,
            "llm_confidence": 0.93,
        },
        {
            "id": "s2",
            "kaomoji": "( ´ ▽ ` )ﾉ",
            "tags": ["waving", "happy"],
            "primary_tags": ["waving"],
            "secondary_tags": ["happy"],
            "source": "emoticon_dict+llm+human_review",
            "quality_score": 0.77,
            "llm_confidence": 0.91,
        },
    ]

    schema = json.loads(result.schema_path.read_text(encoding="utf-8"))
    assert list(schema["fields"].keys()) == [
        "id",
        "kaomoji",
        "tags",
        "primary_tags",
        "secondary_tags",
        "source",
        "quality_score",
        "llm_confidence",
    ]
    readme = result.readme_path.read_text(encoding="utf-8")
    assert "Kaomoji Labeled Dataset" in readme
    assert "https://github.com/ekohrt/emoticon_kaomoji_dataset?tab=readme-ov-file" in readme
    assert "How the data was cleaned" in readme
    assert "Project repository placeholder" in readme


def test_export_hf_dataset_parquet_is_readable(tmp_path: Path):
    import numpy as np
    import pandas as pd

    from kaomoji_labeled_tool.step8_dataset import export_hf_dataset

    kept_path = tmp_path / "kept_records.jsonl"
    output_dir = tmp_path / "hf_dataset"

    _write_jsonl(
        kept_path,
        [
            {
                "sample_id": "s1",
                "kaomoji": "qwq",
                "final_tags": ["sad", "crying"],
                "primary_tags": ["sad"],
                "secondary_tags": ["crying"],
                "approved_proposed_new_tag": None,
                "quality_score": 0.66,
                "confidence": 0.87,
            }
        ],
    )

    result = export_hf_dataset(
        kept_records_path=kept_path,
        output_dir=output_dir,
    )

    frame = pd.read_parquet(result.parquet_path)
    records = frame.to_dict(orient="records")
    for record in records:
        for field in ("tags", "primary_tags", "secondary_tags"):
            value = record[field]
            if isinstance(value, np.ndarray):
                record[field] = value.tolist()

    assert records == [
        {
            "id": "s1",
            "kaomoji": "qwq",
            "tags": ["sad", "crying"],
            "primary_tags": ["sad"],
            "secondary_tags": ["crying"],
            "source": "emoticon_dict+llm+human_review",
            "quality_score": 0.66,
            "llm_confidence": 0.87,
        }
    ]


def test_export_hf_dataset_merges_approved_tag_before_secondary_tags(tmp_path: Path):
    from kaomoji_labeled_tool.step8_dataset import export_hf_dataset

    kept_path = tmp_path / "kept_records.jsonl"
    output_dir = tmp_path / "hf_dataset"

    _write_jsonl(
        kept_path,
        [
            {
                "sample_id": "s1",
                "kaomoji": "(•×•U)",
                "final_tags": ["smiling", "hands", "rabbit"],
                "primary_tags": ["smiling"],
                "secondary_tags": ["hands"],
                "approved_proposed_new_tag": "bunny",
                "quality_score": 0.85,
                "confidence": 0.9,
            }
        ],
    )

    result = export_hf_dataset(
        kept_records_path=kept_path,
        output_dir=output_dir,
    )

    rows = [
        json.loads(line)
        for line in result.jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows == [
        {
            "id": "s1",
            "kaomoji": "(•×•U)",
            "tags": ["smiling", "hands", "rabbit"],
            "primary_tags": ["smiling"],
            "secondary_tags": ["hands", "rabbit"],
            "source": "emoticon_dict+llm+human_review",
            "quality_score": 0.85,
            "llm_confidence": 0.9,
        }
    ]
