# Repository Agent Guide

## Priority

1. Read this file before using any repository tool.
2. Then read `README.md` for the human-facing overview.
3. Prefer the documented CLI workflow over ad hoc scripts.

## Goal

Use this repository to build a labeled kaomoji dataset through a fixed workspace-based flow:

1. Run the rule pipeline.
2. Prepare step 6 annotation artifacts.
3. Place model batch outputs into the workspace.
4. Review proposed new tags.
5. Export final records and the Hugging Face package.

## Environment

1. Use Python 3.12+.
2. Use `uv` for environment management and command execution.
3. From the repository root, run:

```bash
uv sync
uv run kaomoji-labeled-tool --help
```

If the CLI help does not work, stop and fix the environment before changing code or data.

## Standard Operating Sequence

### Step 1: Run the rule-based pipeline

Use the default dataset and config unless the user explicitly provides others.

```bash
uv run kaomoji-labeled-tool run-rules
```

Optional explicit inputs:

```bash
uv run kaomoji-labeled-tool run-rules \
  --dataset data/emoticon_dict.json \
  --config-dir config
```

Checkpoint:
- Confirm the command prints counts for `keep`, `review`, `drop`, and canonical tags.

### Step 2: Prepare annotation artifacts

Write step 6 artifacts into the shared workspace. Prefer the default workspace unless the user asks otherwise.

```bash
uv run kaomoji-labeled-tool prepare-annotation \
  --workspace artifacts/workspace \
  --sample-size 50
```

Expected outputs:
- `artifacts/workspace/step6/few_shot_gold.json`
- `artifacts/workspace/step6/testset.jsonl`
- `artifacts/workspace/step6/batch_requests.jsonl`
- `artifacts/workspace/step6/results/`

Checkpoint:
- Confirm the candidate count and output file paths printed by the command.

### Step 3: Handle model batch outputs

This repository prepares batch request files but does not expose a first-class CLI submission command for the external batch job.

Required input to external tooling:
- `artifacts/workspace/step6/batch_requests.jsonl`

Required output returned into this repository:
- place one or more `*_output.jsonl` files in `artifacts/workspace/step6/results/`

Do not continue to review or export until the `results` directory contains the returned model outputs.

### Step 4: Review proposed new tags

Launch the review UI:

```bash
uv run kaomoji-labeled-tool review-tags --workspace artifacts/workspace
```

Useful server options:

```bash
uv run kaomoji-labeled-tool review-tags \
  --workspace artifacts/workspace \
  --host 127.0.0.1 \
  --port 7860
```

Checkpoint:
- Ensure the review state is written to `artifacts/workspace/review/review_state.json`.

### Step 5: Export the final dataset

After review is complete, export final records and the Hugging Face package:

```bash
uv run kaomoji-labeled-tool export-dataset --workspace artifacts/workspace
```

Expected outputs:
- `artifacts/workspace/final/kept_records.jsonl`
- `artifacts/workspace/final/dropped_records.jsonl`
- `artifacts/workspace/final/export_summary.json`
- `artifacts/workspace/final/approved_proposed_tags.json`
- `artifacts/workspace/hf/data.parquet`
- `artifacts/workspace/hf/data.jsonl`
- `artifacts/workspace/hf/schema.json`
- `artifacts/workspace/hf/README.md`

Checkpoint:
- Confirm the command prints `total`, `kept`, `dropped`, and `hf_total`.

## When You Need Code Changes

1. Read the relevant source and tests first.
2. Follow existing patterns in `src/kaomoji_labeled_tool/` and `tests/`.
3. Prefer small explicit changes over broad rewrites.
4. Run targeted tests first, then run the broader relevant test suite.
5. Use `uv run pytest` for test execution.
6. Update `README.md` only if user-facing behavior or workflow changed.

## When You Need Verification

Before claiming completion:

1. Re-run the exact commands you changed or documented.
2. Run relevant tests if code changed.
3. Check that any referenced paths and filenames actually match the repository outputs.
4. Report blockers clearly if an external batch step or manual review prevents full end-to-end verification.

## Guardrails

1. Do not invent new workflow steps when the existing CLI already covers the task.
2. Do not bypass the workspace layout without a user request.
3. Do not assume external model output exists; verify `artifacts/workspace/step6/results/`.
4. Do not edit unrelated files.
5. If requirements are ambiguous or risky, ask the user before changing behavior.
