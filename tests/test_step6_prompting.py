from __future__ import annotations


def test_build_default_few_shot_examples_returns_user_focused_gold_set():
    from kaomoji_labeled_tool.step6_prompting import build_default_few_shot_examples

    examples = build_default_few_shot_examples()

    assert len(examples) == 10
    assert [example.kaomoji for example in examples] == [
        "(*´ω`*)",
        "^ᴗ.ᴗ^♡",
        "( ߹꒳߹ )",
        "٩( ´ω` )و",
        "(  ¯꒳¯ )ᐝ",
        "( ᐢ. ̫ .ᐢ )",
        "(*ˊᵕˋ*)੭ ﾉ",
        "ヾ(´｡•｡`)ﾉ(><)",
        "NOWLOADING",
        "ʕ•ᴥ•ʔ",
    ]
    assert examples[8].output.reject is True
    assert examples[8].output.primary_tags == []
    assert examples[9].output.proposed_new_tag == "bear"


def test_build_step6_instructions_contains_canonical_tags_and_few_shot_examples():
    from kaomoji_labeled_tool.step6_prompting import (
        build_default_few_shot_examples,
        build_step6_instructions,
        build_step6_json_schema,
    )

    examples = build_default_few_shot_examples()
    schema = build_step6_json_schema(["sad", "crying", "tears", "happy"])
    instructions = build_step6_instructions(
        canonical_tags=["sad", "crying", "tears", "happy"],
        few_shot_examples=examples,
        schema=schema,
    )

    assert "You are a kaomoji tagging assistant." in instructions
    assert "canonical tags" in instructions
    assert "sad, crying, tears, happy" in instructions
    assert "(*´ω`*)" in instructions
    assert "^ᴗ.ᴗ^♡" in instructions
    assert "style_tags" not in instructions
    assert "If reject=true, primary_tags and secondary_tags must be empty arrays" in instructions
    assert "Do not output broad style words like cute, dramatic, soft, deadpan, or tiny." in instructions
    assert "\"kaomoji\": \"NOWLOADING\"" in instructions
    assert "\"reject\": true" in instructions
    assert "\"proposed_new_tag\": \"bear\"" in instructions


def test_build_step6_json_schema_limits_tag_cardinality():
    from kaomoji_labeled_tool.step6_prompting import build_step6_json_schema

    schema = build_step6_json_schema(["sad", "crying", "happy", "smiling"])

    assert schema["properties"]["primary_tags"]["maxItems"] == 2
    assert schema["properties"]["secondary_tags"]["maxItems"] == 3
    assert schema["properties"]["proposed_new_tag"]["anyOf"][0]["pattern"] == "^[a-z0-9_]+$"


def test_build_step6_json_schema_excludes_style_tags_field():
    from kaomoji_labeled_tool.step6_prompting import build_step6_json_schema

    schema = build_step6_json_schema(["sad", "crying", "happy", "smiling"])

    primary_enum = schema["properties"]["primary_tags"]["items"]["enum"]
    secondary_enum = schema["properties"]["secondary_tags"]["items"]["enum"]

    assert "sad" in primary_enum
    assert "sad" in secondary_enum
    assert "style_tags" not in schema["properties"]


def test_build_step6_json_schema_avoids_unsupported_uniqueitems_keyword():
    from kaomoji_labeled_tool.step6_prompting import build_step6_json_schema

    schema = build_step6_json_schema(["sad", "crying", "happy", "smiling"])

    assert "uniqueItems" not in schema["properties"]["primary_tags"]
    assert "uniqueItems" not in schema["properties"]["secondary_tags"]
