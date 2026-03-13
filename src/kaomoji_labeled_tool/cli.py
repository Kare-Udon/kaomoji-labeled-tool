from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from .pipeline import run_pipeline_steps_1_to_5, run_pipeline_steps_1_to_6_prep
from .step6_review_app import launch_review_app
from .step7_export import export_step7_results
from .step8_dataset import export_hf_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Kaomoji 标注与数据集构建工具。"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_rules = subparsers.add_parser(
        "run-rules",
        help="执行规则阶段（步骤 1-5）。",
    )
    _add_dataset_args(run_rules)

    prepare_annotation = subparsers.add_parser(
        "prepare-annotation",
        help="准备 LLM 标注阶段所需的 step6 工件。",
    )
    _add_dataset_args(prepare_annotation)
    _add_workspace_arg(prepare_annotation)
    prepare_annotation.add_argument(
        "--sample-size",
        type=int,
        default=50,
        help="步骤 6 候选样本数量。",
    )

    review_tags = subparsers.add_parser(
        "review-tags",
        help="启动 proposed_new_tag 审阅工具。",
    )
    _add_workspace_arg(review_tags)
    review_tags.add_argument(
        "--host",
        default="127.0.0.1",
        help="Gradio 监听地址。",
    )
    review_tags.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Gradio 监听端口。",
    )
    review_tags.add_argument(
        "--no-launch",
        action="store_true",
        help="仅构建审阅工具，不实际启动服务。",
    )

    export_dataset = subparsers.add_parser(
        "export-dataset",
        help="导出最终记录与 Hugging Face 数据集包。",
    )
    _add_workspace_arg(export_dataset)

    return parser


def _add_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/emoticon_dict.json"),
        help="原始数据集路径。",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("config"),
        help="配置目录路径。",
    )


def _add_workspace_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("artifacts/workspace"),
        help="统一工作目录路径。",
    )


def _workspace_paths(workspace: Path) -> dict[str, Path]:
    step6_dir = workspace / "step6"
    review_dir = workspace / "review"
    final_dir = workspace / "final"
    hf_dir = workspace / "hf"
    return {
        "step6_dir": step6_dir,
        "step6_results_dir": step6_dir / "results",
        "step6_testset_path": step6_dir / "testset.jsonl",
        "review_state_path": review_dir / "review_state.json",
        "final_dir": final_dir,
        "final_kept_records_path": final_dir / "kept_records.jsonl",
        "hf_dir": hf_dir,
    }


def run_from_args(args: argparse.Namespace) -> None:
    if args.command == "run-rules":
        _run_rules_command(args)
        return
    if args.command == "prepare-annotation":
        _prepare_annotation_command(args)
        return
    if args.command == "review-tags":
        _review_tags_command(args)
        return
    if args.command == "export-dataset":
        _export_dataset_command(args)
        return
    raise ValueError(f"未知命令：{args.command}")


def _run_rules_command(args: argparse.Namespace) -> None:
    result = run_pipeline_steps_1_to_5(args.dataset, args.config_dir)
    decision_counts = Counter(record.decision.status for record in result.records)
    print(
        "规则阶段完成："
        f" 已加载 {len(result.records)} 条记录，"
        f" keep={decision_counts.get('keep', 0)}，"
        f" review={decision_counts.get('review', 0)}，"
        f" drop={decision_counts.get('drop', 0)}，"
        f" {len(result.config.canonical_tags)} 个 canonical tags。"
    )


def _prepare_annotation_command(args: argparse.Namespace) -> None:
    paths = _workspace_paths(args.workspace)
    result = run_pipeline_steps_1_to_6_prep(
        dataset_path=args.dataset,
        config_dir=args.config_dir,
        output_dir=paths["step6_dir"],
        sample_size=args.sample_size,
    )
    print(
        "标注准备完成："
        f" 候选样本 {len(result.candidates)} 条，"
        f" 金标样本文件 {result.few_shot_path}，"
        f" 测试集文件 {result.testset_path}，"
        f" batch 文件 {result.batch_path}。"
    )


def _review_tags_command(args: argparse.Namespace) -> None:
    paths = _workspace_paths(args.workspace)
    launch_review_app(
        results_dir=paths["step6_results_dir"],
        testset_path=paths["step6_testset_path"],
        state_path=paths["review_state_path"],
        host=args.host,
        port=args.port,
        launch=not args.no_launch,
    )
    if args.no_launch:
        print("proposed_new_tag 审阅工具已完成初始化。")


def _export_dataset_command(args: argparse.Namespace) -> None:
    paths = _workspace_paths(args.workspace)
    step7_result = export_step7_results(
        results_dir=paths["step6_results_dir"],
        testset_path=paths["step6_testset_path"],
        review_state_path=paths["review_state_path"],
        output_dir=paths["final_dir"],
    )
    step8_result = export_hf_dataset(
        kept_records_path=paths["final_kept_records_path"],
        output_dir=paths["hf_dir"],
    )
    print(
        "导出完成："
        f" total={step7_result.summary['total']}，"
        f" kept={step7_result.summary['kept']}，"
        f" dropped={step7_result.summary['dropped']}，"
        f" hf_total={step8_result.summary['total']}，"
        f" 工作目录 {args.workspace}。"
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_from_args(args)
