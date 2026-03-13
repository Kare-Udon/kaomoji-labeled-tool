from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .approved_tag_merges import merge_approved_tag


@dataclass(frozen=True)
class Step8DatasetResult:
    parquet_path: Path
    jsonl_path: Path
    schema_path: Path
    readme_path: Path
    summary: dict[str, int]


def export_hf_dataset(
    *,
    kept_records_path: Path,
    output_dir: Path,
) -> Step8DatasetResult:
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = output_dir / "data.parquet"
    jsonl_path = output_dir / "data.jsonl"
    schema_path = output_dir / "schema.json"
    readme_path = output_dir / "README.md"

    rows = [_to_dataset_row(payload) for payload in _load_jsonl(kept_records_path)]

    frame = pd.DataFrame(rows)
    frame.to_parquet(parquet_path, index=False)
    jsonl_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    schema_path.write_text(
        json.dumps(
            {
                "fields": {
                    "id": "string",
                    "kaomoji": "string",
                    "tags": "list[string]",
                    "primary_tags": "list[string]",
                    "secondary_tags": "list[string]",
                    "source": "string",
                    "quality_score": "float64",
                    "llm_confidence": "float64",
                }
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    readme_path.write_text(_build_readme(len(rows)), encoding="utf-8")

    return Step8DatasetResult(
        parquet_path=parquet_path,
        jsonl_path=jsonl_path,
        schema_path=schema_path,
        readme_path=readme_path,
        summary={"total": len(rows)},
    )


def _to_dataset_row(payload: dict[str, object]) -> dict[str, object]:
    primary_tags = _string_list(payload.get("primary_tags", []))
    secondary_tags = _dedupe_tags(_string_list(payload.get("secondary_tags", [])))
    approved_tag = merge_approved_tag(_string_or_none(payload.get("approved_proposed_new_tag")))
    if approved_tag:
        secondary_tags = _dedupe_tags(secondary_tags + [approved_tag])

    return {
        "id": str(payload.get("sample_id", "")),
        "kaomoji": str(payload.get("kaomoji", "")),
        "tags": _dedupe_tags(primary_tags + secondary_tags),
        "primary_tags": primary_tags,
        "secondary_tags": secondary_tags,
        "source": "emoticon_dict+llm+human_review",
        "quality_score": float(payload.get("quality_score", 0.0)),
        "llm_confidence": float(payload.get("confidence", 0.0)),
    }


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item]


def _string_or_none(value: object) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    return value


def _dedupe_tags(tags: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for tag in tags:
        if tag in seen:
            continue
        seen.add(tag)
        deduped.append(tag)
    return deduped


def _build_readme(total: int) -> str:
    return f"""# Kaomoji Labeled Dataset

This repository folder contains the final release package prepared for Hugging Face.

At a glance:

- Rows: {total:,}
- Format: `parquet` and `jsonl`
- Main unit: one kaomoji per row
- Core fields: `id`, `kaomoji`, `tags`, `primary_tags`, `secondary_tags`, `source`, `quality_score`, `llm_confidence`
- No train/validation/test split is included. This dataset was assembled as a reusable resource, not as a benchmark.

## What this dataset is

This is a cleaned, tagged kaomoji dataset built on top of the public GitHub repository:

- Source repository: <https://github.com/ekohrt/emoticon_kaomoji_dataset?tab=readme-ov-file>

The original repository provides a large kaomoji collection with two tag fields per item: `original_tags` and `new_tags`. This project takes that raw material, runs it through a rule-based cleaning pipeline, uses an LLM to fill in missing or weak labels, reviews newly proposed tags by hand, and then exports a stable dataset that other people can browse, filter, and reuse.

The end result is not a verbatim mirror of the source repository. It is a derived dataset with normalized tags, deduplicated samples, low-quality or broken records removed, and approved expanded tags merged into the public output.

## Files in this folder

- `data.parquet`: the main release file and the one you will usually want on Hugging Face
- `data.jsonl`: the same rows in line-delimited JSON
- `schema.json`: field-level schema
- `README.md`: this dataset card

## Data schema

Each row has the following fields.

| Field | Type | Meaning |
| --- | --- | --- |
| `id` | `string` | Stable sample identifier in this export |
| `kaomoji` | `string` | The kaomoji itself |
| `tags` | `list[string]` | Final deduplicated tag list |
| `primary_tags` | `list[string]` | Main semantic tags predicted for this kaomoji |
| `secondary_tags` | `list[string]` | Supporting tags; approved proposed tags are merged here |
| `source` | `string` | Provenance marker for this release |
| `quality_score` | `float64` | Rule-based quality score from the cleaning pipeline |
| `llm_confidence` | `float64` | Confidence score returned by the LLM annotator |

Notes:

- `tags` is built from `primary_tags + secondary_tags`, with duplicates removed while preserving order.
- `secondary_tags` already includes any approved `proposed_new_tag`.
- This public export does not expose `style_tags` as a separate field.

## Dataset composition

The released dataset has {total:,} retained rows.

The broader processing run looked like this:

- 62,149 records were loaded from the original source dataset at the start of the pipeline.
- 48,323 records entered the final strict LLM annotation stage.
- 1,012 records were later dropped because the model marked them as `reject=true`.
- {total:,} records were kept in the final public export.

The annotation process also used a small manually labeled supplement of high-frequency personal-use kaomoji. Those samples were passed through the same later stages so they would not become a second-class side dataset.

## Tag system

This project uses two layers of tags.

### 1. Project-defined base tags

The rule pipeline starts from a fixed canonical inventory of 58 tags. These are the tags the project explicitly set out to support.

They roughly fall into these groups:

- emotions and states
- facial actions and expressions
- visual or body-part cues
- a small style-oriented subset that existed in the project inventory

Examples include `happy`, `sad`, `angry`, `love`, `crying`, `smiling`, `waving`, `blushing`, `heart`, `tears`, `hands`, and `eyes`.

### 2. Approved expanded tags

During the LLM stage, the model was allowed to suggest one `proposed_new_tag` when the base inventory was too narrow. Those proposed tags were not accepted automatically. They went through manual review first.

Examples of approved expanded tags include `bear`, `cat`, `dog`, `rabbit`, `running`, `pointing`, `thumbs_up`, `table_flip`, `flower`, `glasses`, and `gun`.

In practice:

- base tags are the project's original controlled vocabulary
- expanded tags are the hand-approved additions that came out of the LLM review loop

## How the data was cleaned

This section is the short version of the full pipeline.

### Step 1. Load and validate the source records

Rules applied:

- the root JSON node must be an object keyed by kaomoji strings
- every record must have a non-empty `kaomoji`
- every record must include both `original_tags` and `new_tags`
- both tag fields must be `list[str]`

### Step 2. Normalize and merge raw tags

Rules applied:

- Unicode normalization with `NFKC`
- collapse repeated whitespace to a single space
- trim surrounding whitespace
- lowercase tags
- drop empty tags
- preserve first occurrence while removing duplicates

### Step 3. Remove obvious noise from tags

Rules applied:

- exact-match removal for non-semantic labels such as `featured` or `popular`
- regex-based removal for numbering and sequence-like labels such as `day #4`
- mark tags containing `_`, `/`, `|`, or multiple hyphens as normalization candidates

### Step 4. Normalize tags into the base vocabulary

Rules applied:

- map variants through `lemma_map`
- map semantic equivalents through `semantic_map`
- split the result into canonical tags and unmapped tags

### Step 5. Score record quality and detect duplicates

Hard filters:

- empty kaomoji
- length shorter than 2 or longer than 80
- control characters
- plain multi-token text strings

Duplicate handling:

- exact duplicate detection
- normalized duplicate detection
- near-duplicate detection with repeated-character compression

Quality thresholds:

- `>= 0.75`: keep
- `0.55 to < 0.75`: review
- `< 0.55`: drop

### Step 6. Strict candidate selection for LLM annotation

The final LLM pass focused on rows with sparse rule tags, many unmapped tags, or structurally valid kaomoji that still looked under-described.

### Step 7. LLM annotation

The annotation stage used OpenAI Batch with `gpt-5.4`. Each request included the kaomoji string, rule tags, normalized and unmapped tag context, the rule-based quality score, and a low-resolution rendered image of the kaomoji.

The model returned:

- `primary_tags`
- `secondary_tags`
- optional `proposed_new_tag`
- `reject`
- `reject_reason`
- `confidence`

### Step 8. Manual review of proposed tags

Rejecting a proposed tag did not mean rejecting the kaomoji itself. It only meant "do not add this new label".

### Step 9. Final export

Rows with `reject=true` are excluded from the public dataset. For the rows that remain, approved proposed tags are merged into `secondary_tags`, and `tags` is built from the union of `primary_tags` and `secondary_tags`.

## Source field

The public `source` field is fixed to:

```text
emoticon_dict+llm+human_review
```

## What the scores mean

`quality_score` is a local rule-based score. It reflects how plausible and useful the sample looked before the LLM stage.

`llm_confidence` is the model's own confidence estimate for the annotation it returned.

These two numbers should not be treated as the same thing.

## Known limitations

- Some tags are easier to apply consistently than others.
- Scene-like ASCII constructions are fuzzy at the boundary between "kaomoji" and "ASCII art".
- The source repository is heterogeneous. Some rows are neat single-face kaomoji; others are long action strings or memes.
- `quality_score` and `llm_confidence` are operational signals, not gold labels.

## Recommended uses

Good fits:

- browsing and searching kaomoji by emotion, gesture, or motif
- building a taggable kaomoji picker
- data exploration and clustering
- retrieval systems

Less ideal fits:

- benchmark-grade evaluation without extra curation
- tasks that require perfectly stable fine-grained labels

## Reproducibility and code links

The release was produced by a local pipeline in this repository. The public code links are placeholders for now and can be replaced once the code is uploaded.

- Source repository used as raw input: <https://github.com/ekohrt/emoticon_kaomoji_dataset?tab=readme-ov-file>
- Project repository placeholder: `https://github.com/<your-org>/<your-repo>`
- Pipeline entrypoint placeholder: `https://github.com/<your-org>/<your-repo>/blob/main/src/kaomoji_labeled_tool/cli.py`
- Dataset export logic placeholder: `https://github.com/<your-org>/<your-repo>/blob/main/src/kaomoji_labeled_tool/step8_dataset.py`
- Step 7 merge/export placeholder: `https://github.com/<your-org>/<your-repo>/blob/main/src/kaomoji_labeled_tool/step7_export.py`
- Step 6 prompting placeholder: `https://github.com/<your-org>/<your-repo>/blob/main/src/kaomoji_labeled_tool/step6_prompting.py`

## Citation

If you use this dataset, please cite both the original source repository and this derived dataset release.
"""
