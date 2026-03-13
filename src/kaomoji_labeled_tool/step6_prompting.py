from __future__ import annotations

import json

from .models import Step6FewShotExample, Step6ModelOutput

STYLE_TAGS: tuple[str, ...] = ("cute", "soft", "tiny", "dramatic", "deadpan")


def build_default_few_shot_examples() -> list[Step6FewShotExample]:
    return [
        Step6FewShotExample(
            kaomoji="(*´ω`*)",
            rule_tags=["happy"],
            output=Step6ModelOutput(
                primary_tags=["happy"],
                secondary_tags=["smiling"],
                proposed_new_tag=None,
                reject=False,
                reject_reason="",
                confidence=0.95,
                reason_brief="relaxed happy smile",
            ),
        ),
        Step6FewShotExample(
            kaomoji="^ᴗ.ᴗ^♡",
            rule_tags=["love"],
            output=Step6ModelOutput(
                primary_tags=["love"],
                secondary_tags=["heart"],
                proposed_new_tag=None,
                reject=False,
                reject_reason="",
                confidence=0.93,
                reason_brief="soft affectionate face",
            ),
        ),
        Step6FewShotExample(
            kaomoji="( ߹꒳߹ )",
            rule_tags=["sad"],
            output=Step6ModelOutput(
                primary_tags=["sad", "crying"],
                secondary_tags=["tears"],
                proposed_new_tag=None,
                reject=False,
                reject_reason="",
                confidence=0.97,
                reason_brief="clear sad crying face",
            ),
        ),
        Step6FewShotExample(
            kaomoji="٩( ´ω` )و",
            rule_tags=["cheering"],
            output=Step6ModelOutput(
                primary_tags=["cheering", "excited"],
                secondary_tags=["hands"],
                proposed_new_tag=None,
                reject=False,
                reject_reason="",
                confidence=0.94,
                reason_brief="raised hands cheering motion",
            ),
        ),
        Step6FewShotExample(
            kaomoji="(  ¯꒳¯ )ᐝ",
            rule_tags=["sleeping"],
            output=Step6ModelOutput(
                primary_tags=["sleepy"],
                secondary_tags=["calm"],
                proposed_new_tag=None,
                reject=False,
                reject_reason="",
                confidence=0.9,
                reason_brief="sleepy calm expression",
            ),
        ),
        Step6FewShotExample(
            kaomoji="( ᐢ. ̫ .ᐢ )",
            rule_tags=["shy"],
            output=Step6ModelOutput(
                primary_tags=["shy"],
                secondary_tags=[],
                proposed_new_tag=None,
                reject=False,
                reject_reason="",
                confidence=0.92,
                reason_brief="shy restrained face",
            ),
        ),
        Step6FewShotExample(
            kaomoji="(*ˊᵕˋ*)੭ ﾉ",
            rule_tags=["waving"],
            output=Step6ModelOutput(
                primary_tags=["waving"],
                secondary_tags=["happy"],
                proposed_new_tag=None,
                reject=False,
                reject_reason="",
                confidence=0.93,
                reason_brief="friendly waving face",
            ),
        ),
        Step6FewShotExample(
            kaomoji="ヾ(´｡•｡`)ﾉ(><)",
            rule_tags=["worried"],
            output=Step6ModelOutput(
                primary_tags=["worried"],
                secondary_tags=["hands"],
                proposed_new_tag=None,
                reject=False,
                reject_reason="",
                confidence=0.89,
                reason_brief="worried face with hand motion",
            ),
        ),
        Step6FewShotExample(
            kaomoji="NOWLOADING",
            rule_tags=[],
            output=Step6ModelOutput(
                primary_tags=[],
                secondary_tags=[],
                proposed_new_tag=None,
                reject=True,
                reject_reason="plain loading text, not a kaomoji",
                confidence=0.99,
                reason_brief="loading text only",
            ),
        ),
        Step6FewShotExample(
            kaomoji="ʕ•ᴥ•ʔ",
            rule_tags=["smiling"],
            output=Step6ModelOutput(
                primary_tags=["smiling"],
                secondary_tags=[],
                proposed_new_tag="bear",
                reject=False,
                reject_reason="",
                confidence=0.93,
                reason_brief="bear face with a friendly smile",
            ),
        ),
    ]


def build_step6_json_schema(canonical_tags: list[str]) -> dict[str, object]:
    semantic_tags = [tag for tag in canonical_tags if tag not in STYLE_TAGS]
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "primary_tags": {
                "type": "array",
                "items": {"type": "string", "enum": semantic_tags},
                "maxItems": 2,
            },
            "secondary_tags": {
                "type": "array",
                "items": {"type": "string", "enum": semantic_tags},
                "maxItems": 3,
            },
            "proposed_new_tag": {
                "anyOf": [
                    {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 32,
                        "pattern": "^[a-z0-9_]+$",
                    },
                    {"type": "null"},
                ]
            },
            "reject": {"type": "boolean"},
            "reject_reason": {"type": "string", "maxLength": 120},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "reason_brief": {"type": "string", "maxLength": 80},
        },
        "required": [
            "primary_tags",
            "secondary_tags",
            "proposed_new_tag",
            "reject",
            "reject_reason",
            "confidence",
            "reason_brief",
        ],
    }


def build_step6_instructions(
    *,
    canonical_tags: list[str],
    few_shot_examples: list[Step6FewShotExample],
    schema: dict[str, object],
) -> str:
    sections = [
        "You are a kaomoji tagging assistant.",
        "Use the text form, rule_tags, and the attached image to assign controlled tags.",
        "rule_tags are strong prior signals. If they fit the text form and image, keep them in primary_tags or secondary_tags.",
        "Do not output broad style words like cute, dramatic, soft, deadpan, or tiny. Do not use them as proposed_new_tag either.",
        "Rules:",
        "1. primary_tags: at most 2 items. Use only core emotion or action.",
        "2. secondary_tags: at most 3 items. Use only secondary semantics.",
        "3. Prefer canonical tags whenever they are sufficient.",
        "4. Use proposed_new_tag only when canonical tags miss a stable semantic category. Otherwise use null.",
        "5. proposed_new_tag must be one stable concept, not a temporary object, layout detail, or meme-specific detail. Use lowercase snake_case.",
        "6. If the sample is not a keepable kaomoji, set reject=true and give a short reject_reason.",
        "7. If reject=true, primary_tags and secondary_tags must be empty arrays, and proposed_new_tag must be null.",
        "8. Reject only plain text, lone symbols, broken fragments, loading text, or pure scene ASCII art.",
        "9. Keep reason_brief short and concrete.",
        "",
        f"canonical tags: {', '.join(canonical_tags)}",
        "",
        "Gold examples:",
    ]

    for index, example in enumerate(few_shot_examples, start=1):
        sections.extend(
            [
                f"Example {index}",
                (
                    "Input: "
                    + json.dumps(
                        {"kaomoji": example.kaomoji, "rule_tags": example.rule_tags},
                        ensure_ascii=False,
                    )
                ),
                "Output: " + json.dumps(example.output.__dict__, ensure_ascii=False),
                "",
            ]
        )

    sections.extend(
        [
            "Output must follow this JSON schema:",
            json.dumps(schema, ensure_ascii=False, sort_keys=True),
            "Return JSON only.",
        ]
    )
    return "\n".join(sections)
