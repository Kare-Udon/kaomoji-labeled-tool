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
                "  - heart",
                "  - cute",
                "  - soft",
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


def test_run_pipeline_steps_1_to_6_prep_writes_testset_and_batch(tmp_path):
    from kaomoji_labeled_tool.pipeline import run_pipeline_steps_1_to_6_prep

    dataset_path = tmp_path / "emoticon_dict.json"
    config_dir = tmp_path / "config"
    output_dir = tmp_path / "artifacts"
    config_dir.mkdir()
    write_config(config_dir)
    dataset_path.write_text(
        json.dumps(
                {
                    "(╥﹏╥)": {
                        "original_tags": [" Cry ", "sad face"],
                        "new_tags": ["cry", "mystery"],
                    },
                    "QwQ": {
                        "original_tags": ["cry"],
                        "new_tags": ["mystery", "void"],
                    },
                    "( ˘͈ ᵕ ˘͈♡)": {
                        "original_tags": ["love"],
                        "new_tags": ["heart", "smile"],
                    },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = run_pipeline_steps_1_to_6_prep(
        dataset_path=dataset_path,
        config_dir=config_dir,
        output_dir=output_dir,
        sample_size=2,
    )

    assert len(result.candidates) == 2
    assert [candidate.record.kaomoji for candidate in result.candidates] == [
        "QwQ",
        "( ˘͈ ᵕ ˘͈♡)",
    ]
    assert result.candidates[0].record.canonical_tags_final == ["crying"]
    assert result.candidates[1].record.canonical_tags_final == ["love", "heart", "smiling"]
    assert result.few_shot_path.exists()
    assert result.testset_path.exists()
    assert result.batch_path.exists()
    few_shot_payload = json.loads(result.few_shot_path.read_text(encoding="utf-8"))
    assert len(few_shot_payload) == 10
    assert few_shot_payload[0]["kaomoji"] == "(*´ω`*)"
    assert few_shot_payload[-2]["output"]["reject"] is True
    assert few_shot_payload[-1]["output"]["proposed_new_tag"] == "bear"


def test_run_pipeline_steps_1_to_6_prep_can_exclude_previous_kaomoji(tmp_path):
    from kaomoji_labeled_tool.pipeline import run_pipeline_steps_1_to_6_prep

    dataset_path = tmp_path / "emoticon_dict.json"
    config_dir = tmp_path / "config"
    output_dir = tmp_path / "artifacts"
    config_dir.mkdir()
    write_config(config_dir)
    dataset_path.write_text(
        json.dumps(
            {
                "QwQ": {
                    "original_tags": ["cry"],
                    "new_tags": ["mystery"],
                },
                "(╥﹏╥)": {
                    "original_tags": [" Cry ", "sad face"],
                    "new_tags": ["cry", "mystery"],
                },
                "( ˘͈ ᵕ ˘͈♡)": {
                    "original_tags": ["love"],
                    "new_tags": ["heart", "smile"],
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = run_pipeline_steps_1_to_6_prep(
        dataset_path=dataset_path,
        config_dir=config_dir,
        output_dir=output_dir,
        sample_size=2,
        exclude_kaomoji={"QwQ"},
    )

    assert [candidate.record.kaomoji for candidate in result.candidates] == [
        "( ˘͈ ᵕ ˘͈♡)",
        "(╥﹏╥)",
    ]
