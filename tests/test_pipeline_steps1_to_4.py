from __future__ import annotations

import json


def write_config(config_dir) -> None:
    (config_dir / "canonical_tags.yaml").write_text(
        "canonical_tags:\n  - crying\n  - sad\n  - love\n",
        encoding="utf-8",
    )
    (config_dir / "tag_normalization.yaml").write_text(
        "\n".join(
            [
                "lemma_map:",
                "  cry: crying",
                "semantic_map:",
                "  sad face: sad",
                "  in love: love",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (config_dir / "tag_removal.yaml").write_text(
        "\n".join(
            [
                "exact:",
                "  - featured",
                "patterns:",
                "  - 'day\\s*#'",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (config_dir / "tag_aliases.yaml").write_text("{}", encoding="utf-8")
    (config_dir / "export.yaml").write_text("prefix: km\n", encoding="utf-8")


def test_run_pipeline_steps_1_to_4_returns_normalized_records(tmp_path):
    from kaomoji_labeled_tool.pipeline import run_pipeline_steps_1_to_4

    dataset_path = tmp_path / "emoticon_dict.json"
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    write_config(config_dir)
    dataset_path.write_text(
        json.dumps(
            {
                "(╥﹏╥)": {
                    "original_tags": [" Cry ", "sad face", "featured"],
                    "new_tags": ["cry", "in love", "day #4", "mystery"],
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = run_pipeline_steps_1_to_4(dataset_path, config_dir)

    assert len(result.records) == 1
    record = result.records[0]
    assert record.kaomoji == "(╥﹏╥)"
    assert record.merged_tags_raw == ["cry", "sad face", "featured", "in love", "day #4", "mystery"]
    assert record.rule_removed_tags == ["featured", "day #4"]
    assert record.rule_kept_tags == ["cry", "sad face", "in love", "mystery"]
    assert record.normalized_tags == ["crying", "sad", "love", "mystery"]
    assert record.canonical_tags_final == ["crying", "sad", "love"]
    assert record.unmapped_tags == ["mystery"]
    assert result.config.export.prefix == "km"
