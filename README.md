# kaomoji-labeled-tool

<details>
<summary>AI Agent note</summary>

If you are an AI agent, read `AGENTS.md` before using repository tools. Follow that document step by step, prefer the documented CLI workflow, and do not skip its verification checkpoints.

</details>

`kaomoji-labeled-tool` is a dataset-building toolkit for cleaning, labeling, reviewing, and exporting kaomoji records.

Finished dataset: `https://huggingface.co/datasets/kareudon/kaomoji-tagged`

It starts from a raw source dataset with `original_tags` and `new_tags`, applies a rule-based normalization pipeline, prepares selected samples for LLM annotation, supports manual review of proposed new tags, and exports a final Hugging Face-ready dataset package.

## What This Repository Does

The repository is organized around four user-facing commands:

- `run-rules`: run the rule-based pipeline through steps 1-5
- `prepare-annotation`: generate step 6 annotation artifacts inside a workspace
- `review-tags`: launch the review UI for `proposed_new_tag`
- `export-dataset`: export final records and a Hugging Face dataset package

Internally, the codebase still keeps step-specific modules. Public usage is intentionally simplified around a shared workspace directory.

## Requirements

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/)

## Installation

Clone the repository and install the project environment:

```bash
uv sync
```

Run the CLI with:

```bash
uv run kaomoji-labeled-tool --help
```

## Quick Start

### 1. Run the rule-based pipeline

```bash
uv run kaomoji-labeled-tool run-rules
```

Optional inputs:

```bash
uv run kaomoji-labeled-tool run-rules \
  --dataset data/emoticon_dict.json \
  --config-dir config
```

### 2. Prepare annotation artifacts

This command runs the rules pipeline and writes step 6 inputs into a workspace.

```bash
uv run kaomoji-labeled-tool prepare-annotation \
  --workspace artifacts/workspace \
  --sample-size 50
```

### 3. Submit batch requests

This repository currently prepares batch request files but does not yet expose OpenAI Batch submission as a first-class CLI command in `src/`.

After running `prepare-annotation`, you should use the generated file:

- `artifacts/workspace/step6/batch_requests.jsonl`

and place model output files into:

- `artifacts/workspace/step6/results/`

with filenames matching:

- `*_output.jsonl`

This keeps the public workflow stable while the batch submission runner remains a repository-specific helper script.

### 4. Review proposed new tags

Launch the review UI:

```bash
uv run kaomoji-labeled-tool review-tags --workspace artifacts/workspace
```

Useful options:

```bash
uv run kaomoji-labeled-tool review-tags \
  --workspace artifacts/workspace \
  --host 127.0.0.1 \
  --port 7860
```

### 5. Export the final dataset

This command combines the current step 7 and step 8 export flow:

```bash
uv run kaomoji-labeled-tool export-dataset --workspace artifacts/workspace
```

It reads:

- `artifacts/workspace/step6/results/`
- `artifacts/workspace/step6/testset.jsonl`
- `artifacts/workspace/review/review_state.json`

and writes:

- final kept/dropped records into `artifacts/workspace/final/`
- Hugging Face dataset files into `artifacts/workspace/hf/`

## Workspace Layout

The default workspace is `artifacts/workspace`.

After a normal run, the structure looks like this:

```text
artifacts/workspace/
  step6/
    few_shot_gold.json
    testset.jsonl
    batch_requests.jsonl
    results/
  review/
    review_state.json
  final/
    kept_records.jsonl
    dropped_records.jsonl
    export_summary.json
    approved_proposed_tags.json
  hf/
    data.parquet
    data.jsonl
    schema.json
    README.md
```

## End-to-End Flow

1. Run `run-rules` if you want a quick sanity check on the rule-based pipeline.
2. Run `prepare-annotation` to build the step 6 candidate set and batch request file.
3. Submit the batch requests and save the returned `*_output.jsonl` files into `workspace/step6/results/`.
4. Run `review-tags` to approve or reject proposed new tags.
5. Run `export-dataset` to produce the final exports and Hugging Face package.

## Development

Run the test suite with:

```bash
uv run pytest
```

## Notes

- The repository has been renamed away from its earlier Rime-specific goal, but some older planning notes in the repo still refer to the historical workflow.
- The public CLI now targets dataset construction first.
- The Hugging Face export is generated from reviewed step 6 outputs, not directly from the raw source dataset.

## Support

If you feels this tool is helpful, maybe considered buying me [a cup of coffee](https://ko-fi.com/kareudon).
