from __future__ import annotations

import json


def write_config(config_dir) -> None:
    (config_dir / "canonical_tags.yaml").write_text(
        "\n".join(
            [
                "canonical_tags:",
                "  - crying",
                "  - sad",
                "  - love",
                "  - smiling",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (config_dir / "tag_normalization.yaml").write_text(
        "\n".join(
            [
                "lemma_map:",
                "  cry: crying",
                "  smile: smiling",
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


def test_run_pipeline_steps_1_to_5_returns_quality_scored_records(tmp_path):
    from kaomoji_labeled_tool.pipeline import run_pipeline_steps_1_to_5

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
                },
                "( ╥﹏╥ )": {
                    "original_tags": ["cry"],
                    "new_tags": [],
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = run_pipeline_steps_1_to_5(dataset_path, config_dir)

    assert len(result.records) == 2

    primary = result.records[0]
    assert primary.kaomoji == "(╥﹏╥)"
    assert primary.normalized_tags == ["crying", "sad", "love", "mystery"]
    assert primary.canonical_tags_final == ["crying", "sad", "love"]
    assert primary.decision.status == "keep"
    assert primary.dedup.cluster_role == "primary"
    assert primary.hard_filter_reason is None

    duplicate_candidate = result.records[1]
    assert duplicate_candidate.kaomoji == "( ╥﹏╥ )"
    assert duplicate_candidate.dedup.normalized_key == primary.dedup.normalized_key
    assert duplicate_candidate.quality.dedup_penalty > 0.0
    assert duplicate_candidate.hard_filter_reason is None
    assert duplicate_candidate.decision.status in {"review", "drop"}
