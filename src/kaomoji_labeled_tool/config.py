from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .models import ExportConfig, NormalizationConfig, PipelineConfig, RemovalConfig


def load_pipeline_config(config_dir: str | Path) -> PipelineConfig:
    base_dir = Path(config_dir)
    canonical_payload = _read_yaml(base_dir / "canonical_tags.yaml")
    normalization_payload = _read_yaml(base_dir / "tag_normalization.yaml")
    removal_payload = _read_yaml(base_dir / "tag_removal.yaml")
    aliases_payload = _read_yaml(base_dir / "tag_aliases.yaml")
    export_payload = _read_yaml(base_dir / "export.yaml")

    canonical_tags = tuple(_require_string_list(canonical_payload, "canonical_tags"))
    normalization = NormalizationConfig(
        lemma_map=_require_string_mapping(normalization_payload, "lemma_map"),
        semantic_map=_require_string_mapping(normalization_payload, "semantic_map"),
    )
    removal = RemovalConfig(
        exact=tuple(_require_string_list(removal_payload, "exact")),
        patterns=tuple(_require_string_list(removal_payload, "patterns")),
    )
    aliases = _parse_aliases(aliases_payload)
    export = ExportConfig(prefix=_require_string(export_payload, "prefix"))

    return PipelineConfig(
        canonical_tags=canonical_tags,
        normalization=normalization,
        removal=removal,
        aliases=aliases,
        export=export,
    )


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"缺少配置文件：{path.name}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise TypeError(f"配置文件 {path.name} 的根节点必须是对象。")
    return payload


def _require_string_list(payload: dict[str, Any], key: str) -> list[str]:
    if key not in payload:
        raise ValueError(f"配置缺少字段：{key}")
    value = payload[key]
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise TypeError(f"配置字段 {key} 必须是字符串列表。")
    return value


def _require_string_mapping(payload: dict[str, Any], key: str) -> dict[str, str]:
    if key not in payload:
        raise ValueError(f"配置缺少字段：{key}")
    value = payload[key]
    if not isinstance(value, dict):
        raise TypeError(f"配置字段 {key} 必须是字符串映射。")
    if any(not isinstance(k, str) or not isinstance(v, str) for k, v in value.items()):
        raise TypeError(f"配置字段 {key} 只能包含字符串键值。")
    return value


def _require_string(payload: dict[str, Any], key: str) -> str:
    if key not in payload:
        raise ValueError(f"配置缺少字段：{key}")
    value = payload[key]
    if not isinstance(value, str) or not value:
        raise TypeError(f"配置字段 {key} 必须是非空字符串。")
    return value


def _parse_aliases(payload: dict[str, Any]) -> dict[str, dict[str, tuple[str, ...]]]:
    aliases: dict[str, dict[str, tuple[str, ...]]] = {}
    for canonical_tag, lang_map in payload.items():
        if not isinstance(canonical_tag, str):
            raise TypeError("别名表的标签键必须是字符串。")
        if not isinstance(lang_map, dict):
            raise TypeError(f"别名表中 {canonical_tag!r} 的值必须是对象。")
        parsed_lang_map: dict[str, tuple[str, ...]] = {}
        for lang, values in lang_map.items():
            if not isinstance(lang, str):
                raise TypeError(f"别名表中 {canonical_tag!r} 的语言键必须是字符串。")
            if not isinstance(values, list) or any(not isinstance(item, str) for item in values):
                raise TypeError(f"别名表中 {canonical_tag!r}.{lang!r} 必须是字符串列表。")
            parsed_lang_map[lang] = tuple(values)
        aliases[canonical_tag] = parsed_lang_map
    return aliases
