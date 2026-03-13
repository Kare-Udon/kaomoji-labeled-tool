from __future__ import annotations

from pathlib import Path

from .step6_review import (
    ProposedTagGroup,
    load_review_groups,
    load_review_state,
    save_review_decision,
    summarize_group,
)

KEYBOARD_SHORTCUTS_HEAD = """
<script>
(() => {
  const bind = (key, action) => {
    document.addEventListener("keydown", (event) => {
      if (event.target && ["INPUT", "TEXTAREA"].includes(event.target.tagName)) {
        return;
      }
      if (event.key === key) {
        const button = document.getElementById(`review-action-${action}`);
        if (button) {
          button.click();
        }
      }
    });
  };
  bind("j", "next");
  bind("k", "prev");
  bind("a", "approve");
  bind("r", "reject");
  bind("s", "skip");
})();
</script>
"""


def build_samples_preview_rows(
    group: ProposedTagGroup, preview_count: int
) -> list[list[object]]:
    rows: list[list[object]] = []
    for sample in group.samples[:preview_count]:
        rows.append(
            [
                sample.sample_id,
                sample.kaomoji,
                ", ".join(sample.primary_tags),
                ", ".join(sample.secondary_tags),
                ", ".join(sample.rule_tags),
                sample.quality_score,
                sample.confidence,
                sample.candidate_reason,
                sample.reason_brief,
                sample.proposed_new_tag,
            ]
        )
    return rows


def apply_review_action(state_path: Path, tag: str, action: str, note: str) -> str:
    save_review_decision(state_path=state_path, tag=tag, status=action, note=note)
    state = load_review_state(state_path)
    decision = state[tag]
    return f"{tag} 已标记为 {decision['status']}。备注：{decision['note']}"


def launch_review_app(
    *,
    results_dir: Path,
    testset_path: Path,
    state_path: Path | None = None,
    host: str = "127.0.0.1",
    port: int = 7860,
    launch: bool = True,
):
    import gradio as gr

    groups = load_review_groups(results_dir=results_dir, testset_path=testset_path)
    if not groups:
        raise ValueError("未找到任何 proposed_new_tag 样本，无法启动审阅工具。")

    state_file = state_path or results_dir.parent / "review_state.json"
    initial_state = load_review_state(state_file)
    group_index = {group.tag: group for group in groups}
    tag_choices = [group.tag for group in groups]

    def render(tag: str, preview_count: int):
        group = group_index[tag]
        state = load_review_state(state_file)
        return (
            summarize_group(group, state),
            build_samples_preview_rows(group, preview_count),
            state.get(tag, {}).get("note", ""),
        )

    def move(current_tag: str, step: int, preview_count: int):
        idx = tag_choices.index(current_tag)
        next_tag = tag_choices[(idx + step) % len(tag_choices)]
        summary, rows, note = render(next_tag, preview_count)
        return next_tag, summary, rows, note

    def act(current_tag: str, action: str, note: str, preview_count: int):
        message = apply_review_action(
            state_path=state_file,
            tag=current_tag,
            action=action,
            note=note,
        )
        summary, rows, saved_note = render(current_tag, preview_count)
        return summary, rows, saved_note, message

    with gr.Blocks(title="Proposed New Tag 审阅工具") as app:
        gr.Markdown(
            "## Proposed New Tag 审阅工具\n"
            f"- results: `{results_dir}`\n"
            f"- testset: `{testset_path}`\n"
            f"- state: `{state_file}`"
        )
        with gr.Row():
            tag_dropdown = gr.Dropdown(
                choices=tag_choices,
                value=tag_choices[0],
                label="当前 proposed tag",
            )
            preview_count = gr.Dropdown(
                choices=[10, 20, 50, 100],
                value=20,
                label="预览样本数",
            )
        summary_box = gr.Textbox(
            label="当前 tag 摘要",
            value=summarize_group(groups[0], initial_state),
            interactive=False,
        )
        preview_table = gr.Dataframe(
            headers=[
                "sample_id",
                "kaomoji",
                "primary_tags",
                "secondary_tags",
                "rule_tags",
                "quality_score",
                "confidence",
                "candidate_reason",
                "reason_brief",
                "proposed_new_tag",
            ],
            value=build_samples_preview_rows(groups[0], 20),
            label="样本预览",
            interactive=False,
        )
        note_box = gr.Textbox(
            label="备注",
            value=initial_state.get(tag_choices[0], {}).get("note", ""),
            lines=2,
        )
        status_box = gr.Textbox(label="操作状态", interactive=False)

        with gr.Row():
            prev_button = gr.Button("上一个 (K)", elem_id="review-action-prev")
            next_button = gr.Button("下一个 (J)", elem_id="review-action-next")
            approve_button = gr.Button("批准 (A)", elem_id="review-action-approve")
            reject_button = gr.Button("拒绝 (R)", elem_id="review-action-reject")
            skip_button = gr.Button("待定 (S)", elem_id="review-action-skip")

        tag_dropdown.change(
            render,
            inputs=[tag_dropdown, preview_count],
            outputs=[summary_box, preview_table, note_box],
        )
        preview_count.change(
            render,
            inputs=[tag_dropdown, preview_count],
            outputs=[summary_box, preview_table, note_box],
        )
        prev_button.click(
            move,
            inputs=[tag_dropdown, gr.State(-1), preview_count],
            outputs=[tag_dropdown, summary_box, preview_table, note_box],
        )
        next_button.click(
            move,
            inputs=[tag_dropdown, gr.State(1), preview_count],
            outputs=[tag_dropdown, summary_box, preview_table, note_box],
        )
        approve_button.click(
            lambda tag, note, count: act(tag, "approved", note, count),
            inputs=[tag_dropdown, note_box, preview_count],
            outputs=[summary_box, preview_table, note_box, status_box],
        )
        reject_button.click(
            lambda tag, note, count: act(tag, "rejected", note, count),
            inputs=[tag_dropdown, note_box, preview_count],
            outputs=[summary_box, preview_table, note_box, status_box],
        )
        skip_button.click(
            lambda tag, note, count: act(tag, "pending", note, count),
            inputs=[tag_dropdown, note_box, preview_count],
            outputs=[summary_box, preview_table, note_box, status_box],
        )
    app.head = KEYBOARD_SHORTCUTS_HEAD

    if launch:
        app.launch(server_name=host, server_port=port, head=app.head)
    return app
