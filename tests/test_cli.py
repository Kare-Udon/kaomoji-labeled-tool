from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace


def test_build_parser_parses_run_rules_subcommand():
    from kaomoji_labeled_tool.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["run-rules"])

    assert args.command == "run-rules"
    assert args.dataset == Path("data/emoticon_dict.json")


def test_build_parser_prepare_annotation_uses_workspace_default():
    from kaomoji_labeled_tool.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["prepare-annotation"])

    assert args.command == "prepare-annotation"
    assert args.workspace == Path("artifacts/workspace")


def test_build_parser_parses_export_dataset_subcommand():
    from kaomoji_labeled_tool.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["export-dataset"])

    assert args.command == "export-dataset"
    assert args.workspace == Path("artifacts/workspace")


def test_run_from_args_export_dataset_uses_workspace_paths(monkeypatch, tmp_path, capsys):
    from kaomoji_labeled_tool.cli import run_from_args

    calls: list[tuple[str, object, object]] = []

    def fake_export_step7_results(*, results_dir, testset_path, review_state_path, output_dir):
        calls.append(("step7", results_dir, output_dir))
        output_dir.mkdir(parents=True, exist_ok=True)
        kept_path = output_dir / "kept_records.jsonl"
        kept_path.write_text("", encoding="utf-8")
        return SimpleNamespace(summary={"total": 3, "kept": 2, "dropped": 1})

    def fake_export_hf_dataset(*, kept_records_path, output_dir):
        calls.append(("step8", kept_records_path, output_dir))
        return SimpleNamespace(summary={"total": 2})

    monkeypatch.setattr("kaomoji_labeled_tool.cli.export_step7_results", fake_export_step7_results)
    monkeypatch.setattr("kaomoji_labeled_tool.cli.export_hf_dataset", fake_export_hf_dataset)

    workspace = tmp_path / "workspace"
    args = SimpleNamespace(command="export-dataset", workspace=workspace)

    run_from_args(args)
    captured = capsys.readouterr()

    assert calls == [
        ("step7", workspace / "step6" / "results", workspace / "final"),
        ("step8", workspace / "final" / "kept_records.jsonl", workspace / "hf"),
    ]
    assert "导出完成" in captured.out
