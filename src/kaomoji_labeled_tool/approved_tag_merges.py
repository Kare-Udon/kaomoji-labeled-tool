from __future__ import annotations


APPROVED_TAG_MERGE_MAP: dict[str, str] = {
    "bunny": "rabbit",
    "middle_finger_gesture": "middle_finger",
    "thanks_message": "thanks",
    "greeting_message": "greeting",
    "goodbye_message": "goodbye",
    "serving": "offering",
    "serving_drink": "offering_drink",
    "offering_gift": "gift_giving",
    "muscular": "flexing",
    "running_pose": "running",
    "fighting_stance": "fighting_pose",
    "boxing_pose": "boxing",
    "table_flip_recovery": "table_upright",
    "table_flip_reversal": "table_upright",
    "magic_wand": "magic",
    "wand": "magic",
    "wand_magic": "magic",
}


def merge_approved_tag(tag: str | None) -> str | None:
    if not tag:
        return None
    return APPROVED_TAG_MERGE_MAP.get(tag, tag)
