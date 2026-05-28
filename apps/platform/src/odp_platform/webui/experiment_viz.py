from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from odp_platform.common.paths import RUNS_DIR


def _list_experiments() -> list[str]:
    exp_dir = RUNS_DIR / "experiments"
    if not exp_dir.exists():
        return []
    return sorted(
        [d.name for d in exp_dir.iterdir() if d.is_dir()],
        reverse=True,
    )


def _load_results_csv(experiment: str) -> list[dict] | None:
    csv_path = RUNS_DIR / "experiments" / experiment / "results.csv"
    if not csv_path.exists():
        return None
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _load_train_summary(experiment: str) -> dict:
    summary_path = RUNS_DIR / "experiments" / experiment / "train_summary.json"
    if summary_path.exists():
        try:
            with open(summary_path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _load_config_report(experiment: str) -> dict:
    report_path = RUNS_DIR / "experiments" / experiment / "config_report.json"
    if report_path.exists():
        try:
            with open(report_path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _get_pregen_image(experiment: str, filename: str) -> Image.Image | None:
    img_path = RUNS_DIR / "experiments" / experiment / filename
    if img_path.exists():
        try:
            return Image.open(img_path)
        except Exception:
            pass
    return None


def _get_weights_info(experiment: str) -> dict:
    weights_dir = RUNS_DIR / "experiments" / experiment / "weights"
    result = {"best": None, "last": None}
    if weights_dir.exists():
        for pt in weights_dir.iterdir():
            if pt.suffix == ".pt":
                size_mb = pt.stat().st_size / (1024 * 1024)
                key = pt.stem
                result[key] = f"{pt.name} ({size_mb:.1f} MB)"
    return result


def _plot_metric_curves_from_csv(experiment: str) -> Image.Image | None:
    rows = _load_results_csv(experiment)
    if not rows:
        return None

    epochs = [int(r.get("epoch", 0)) for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    for key, label, color in [
        ("train/box_loss", "Box Loss", "#e74c3c"),
        ("train/cls_loss", "Cls Loss", "#3498db"),
        ("train/dfl_loss", "DFL Loss", "#2ecc71"),
    ]:
        vals = [float(r.get(key, 0)) for r in rows if r.get(key, "")]
        if vals and any(v != 0 for v in vals):
            ax1.plot(epochs[:len(vals)], vals, label=label, color=color, linewidth=1.5)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    for key, label, color in [
        ("metrics/mAP50(B)", "mAP50", "#e67e22"),
        ("metrics/mAP50-95(B)", "mAP50-95", "#9b59b6"),
        ("metrics/precision(B)", "Precision", "#2ecc71"),
        ("metrics/recall(B)", "Recall", "#e74c3c"),
    ]:
        vals = [float(r.get(key, 0)) for r in rows if r.get(key, "")]
        if vals and any(v != 0 for v in vals):
            ax2.plot(epochs[:len(vals)], vals, label=label, color=color, linewidth=1.5)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Value")
    ax2.set_title("Validation Metrics")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return Image.open(buf)


def _plot_best_metrics_bar(experiment: str) -> Image.Image | None:
    rows = _load_results_csv(experiment)
    if not rows:
        return None

    last = rows[-1]
    metrics_map = [
        ("mAP50", float(last.get("metrics/mAP50(B)", 0))),
        ("mAP50-95", float(last.get("metrics/mAP50-95(B)", 0))),
        ("Precision", float(last.get("metrics/precision(B)", 0))),
        ("Recall", float(last.get("metrics/recall(B)", 0))),
    ]
    names = [m[0] for m in metrics_map]
    values = [m[1] for m in metrics_map]

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#e67e22", "#9b59b6", "#2ecc71", "#e74c3c"]
    bars = ax.bar(names, values, color=colors, width=0.5, edgecolor="white")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(values) * 1.25 if max(values) > 0 else 1)
    ax.set_ylabel("Value")
    ax.set_title(f"Best Metrics — {experiment}")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return Image.open(buf)


def _summarize_experiment(experiment: str) -> str:
    if not experiment:
        return "请选择实验"
    rows = _load_results_csv(experiment)
    if not rows:
        return f"实验 {experiment} 无 results.csv 或训练未完成"

    last = rows[-1]
    total_epochs = len(rows)
    best_map50 = max(float(r.get("metrics/mAP50(B)", 0)) for r in rows)
    best_map50_95 = max(float(r.get("metrics/mAP50-95(B)", 0)) for r in rows)

    weights_info = _get_weights_info(experiment)
    summary_data = _load_train_summary(experiment)

    lines = [
        f"实验: {experiment}",
        f"总轮数: {total_epochs}",
        f"最佳 mAP50: {best_map50:.4f}",
        f"最佳 mAP50-95: {best_map50_95:.4f}",
        f"最终 mAP50: {float(last.get('metrics/mAP50(B)', 0)):.4f}",
        f"最终 Precision: {float(last.get('metrics/precision(B)', 0)):.4f}",
        f"最终 Recall: {float(last.get('metrics/recall(B)', 0)):.4f}",
        f"模型权重: {weights_info.get('best', '无')}",
    ]
    if summary_data:
        total_imgs = summary_data.get("total_images", 0)
        total_epochs_actual = summary_data.get("epochs", 0)
        if total_imgs:
            lines.append(f"训练图片数: {total_imgs}")
    return "\n".join(lines)


def _load_experiment_charts(experiment: str) -> tuple:
    if not experiment:
        blanks = []
        for _ in range(7):
            blanks.append(Image.new("RGB", (400, 300), (240, 240, 240)))
        return tuple(blanks) + ("请选择实验",)

    results_png = _get_pregen_image(experiment, "results.png")
    confusion = _get_pregen_image(experiment, "confusion_matrix.png")
    confusion_norm = _get_pregen_image(experiment, "confusion_matrix_normalized.png")
    labels_img = _get_pregen_image(experiment, "labels.jpg")
    pr_curve = _get_pregen_image(experiment, "BoxPR_curve.png")
    f1_curve = _get_pregen_image(experiment, "BoxF1_curve.png")

    dynamic_curves = _plot_metric_curves_from_csv(experiment)
    bar_chart = _plot_best_metrics_bar(experiment)

    fallback = Image.new("RGB", (400, 300), (240, 240, 240))
    return (
        results_png or dynamic_curves or fallback,
        confusion or fallback,
        confusion_norm or fallback,
        labels_img or fallback,
        pr_curve or fallback,
        f1_curve or fallback,
        bar_chart or fallback,
        _summarize_experiment(experiment),
    )


def _refresh_experiments() -> gr.update:
    return gr.update(choices=_list_experiments())


def create_experiment_viz_ui() -> None:
    experiments = _list_experiments()
    initial = experiments[0] if experiments else None

    blank = Image.new("RGB", (400, 300), (240, 240, 240))

    with gr.Row(elem_classes=["odp-row", "odp-row-two"]):
        exp_dd = gr.Dropdown(
            label="选择实验",
            choices=experiments,
            value=initial,
            filterable=True,
            interactive=True,
            scale=3,
        )
        refresh_btn = gr.Button("刷新", scale=1)

    summary_box = gr.Textbox(label="实验摘要", lines=6, interactive=False)

    with gr.Tabs():
        with gr.TabItem("训练曲线"):
            results_plot = gr.Image(value=blank, label="Loss + 验证指标", container=True, height=400)
            bar_plot = gr.Image(value=blank, label="最佳指标柱状图", container=True, height=350)

        with gr.TabItem("评估矩阵"):
            with gr.Row(elem_classes=["odp-row", "odp-row-two"]):
                confusion_plot = gr.Image(value=blank, label="混淆矩阵", container=True, height=400)
                confusion_norm_plot = gr.Image(value=blank, label="归一化混淆矩阵", container=True, height=400)
            with gr.Row(elem_classes=["odp-row", "odp-row-two"]):
                pr_plot = gr.Image(value=blank, label="PR 曲线", container=True, height=350)
                f1_plot = gr.Image(value=blank, label="F1 曲线", container=True, height=350)

        with gr.TabItem("类别分布"):
            labels_plot = gr.Image(value=blank, label="类别分布", container=True, height=450)

    refresh_btn.click(
        fn=_refresh_experiments,
        outputs=[exp_dd],
    )

    exp_dd.change(
        fn=_load_experiment_charts,
        inputs=[exp_dd],
        outputs=[
            results_plot, confusion_plot, confusion_norm_plot,
            labels_plot, pr_plot, f1_plot, bar_plot, summary_box,
        ],
    )
