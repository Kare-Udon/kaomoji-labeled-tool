from __future__ import annotations

import json

import pytest

from kaomoji_labeled_tool.config import load_pipeline_config
from kaomoji_labeled_tool.loader import load_dataset


def test_load_dataset_returns_normalized_records(tmp_path):
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(
        json.dumps(
            {
                "(╥﹏╥)": {
                    "original_tags": ["sad"],
                    "new_tags": ["crying"],
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    records = load_dataset(dataset_path)

    assert len(records) == 1
    assert records[0].kaomoji == "(╥﹏╥)"
    assert records[0].original_tags == ["sad"]
    assert records[0].new_tags == ["crying"]


def test_load_dataset_rejects_missing_required_fields(tmp_path):
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(
        json.dumps({"(╥﹏╥)": {"original_tags": ["sad"]}}, ensure_ascii=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="new_tags"):
        load_dataset(dataset_path)


def test_load_dataset_rejects_non_list_tags(tmp_path):
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(
        json.dumps(
            {
                "(╥﹏╥)": {
                    "original_tags": "sad",
                    "new_tags": ["crying"],
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(TypeError, match="original_tags"):
        load_dataset(dataset_path)


def test_load_pipeline_config_reads_required_files(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "canonical_tags.yaml").write_text(
        "canonical_tags:\n  - happy\n  - sad\n",
        encoding="utf-8",
    )
    (config_dir / "tag_normalization.yaml").write_text(
        "lemma_map:\n  cry: crying\nsemantic_map:\n  sad face: sad\n",
        encoding="utf-8",
    )
    (config_dir / "tag_removal.yaml").write_text(
        "exact:\n  - popular\npatterns:\n  - '#\\\\d+'\n",
        encoding="utf-8",
    )
    (config_dir / "tag_aliases.yaml").write_text(
        "happy:\n  zh: [开心]\n  ja: [嬉しい]\n  en: [happy]\n",
        encoding="utf-8",
    )
    (config_dir / "export.yaml").write_text(
        "prefix: km\n",
        encoding="utf-8",
    )

    config = load_pipeline_config(config_dir)

    assert config.canonical_tags == ("happy", "sad")
    assert config.normalization.lemma_map["cry"] == "crying"
    assert config.removal.exact == ("popular",)
    assert config.aliases["happy"]["zh"] == ("开心",)
    assert config.export.prefix == "km"


def test_load_pipeline_config_requires_all_files(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "canonical_tags.yaml").write_text("canonical_tags: []\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="tag_normalization.yaml"):
        load_pipeline_config(config_dir)
