from __future__ import annotations

import csv
import io
import logging
import re
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from odp_platform.common.paths import ROOT_DIR, RUNS_DIR
from odp_platform.webui.utils import list_dataset_names, platform_env

logger = logging.getLogger(__name__)


_train_process: subprocess.Popen | None = None
_train_lock = threading.Lock()
_stop_event = threading.Event()


def _query_gpu_memory() -> str:
    if not torch.cuda.is_available():
        return "GPU 不可用"
    lines = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        total = props.total_memory / 1024**3
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        free = total - reserved
        pct = reserved / total * 100
        name = props.name
        lines.append(
            f"GPU{i} [{name}]: 已用 {allocated:.1f}G / 预占 {reserved:.1f}G "
            f"/ 空闲 {free:.1f}G / 总计 {total:.1f}G ({pct:.0f}%)"
        )
    return "\n".join(lines)


def _query_gpu_processes() -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
        return "无运行中进程"
    except Exception:
        return "nvidia-smi 不可用"


def _auto_batch_hint(model: str, imgsz: int) -> str:
    if not torch.cuda.is_available():
        return ""
    try:
        total = torch.cuda.get_device_properties(0).total_memory
        total_gb = total / 1024**3
        if total_gb < 4:
            suggested = 2
        elif total_gb < 6:
            suggested = 4
        elif total_gb < 8:
            suggested = 8
        elif total_gb < 12:
            suggested = 16
        else:
            suggested = 32
        if imgsz > 640:
            suggested = max(2, suggested // 2)
        return (
            f"检测到 GPU 显存 {total_gb:.0f} GiB，建议 batch ≤ {suggested}。"
            f"如遇 OOM 请降低 batch/imgsz，或开启 AMP。"
        )
    except Exception:
        return ""


def _maybe_oom(line: str) -> str | None:
    low = line.lower()
    if "out of memory" in low or "cuda out of memory" in low:
        return "检测到 CUDA OOM！请尝试：① 降低 batch ② 降低 imgsz ③ 开启 AMP（混合精度）"
    if "cuda error" in low:
        return "检测到 CUDA 错误，可能是显存不足或驱动问题"
    return None


def _parse_progress(line: str) -> dict:
    info = {}
    m = re.search(r"(\d+)/(\d+)\s+\[{1,2}\d+:\d+<{1,2}([\d:]+)", line)
    if m:
        info["epoch"] = int(m.group(1))
        info["total_epochs"] = int(m.group(2))
        info["eta"] = m.group(3)
    m = re.search(r"box_loss[:\s]*([\d.]+)", line)
    if m:
        info["box_loss"] = float(m.group(1))
    m = re.search(r"cls_loss[:\s]*([\d.]+)", line)
    if m:
        info["cls_loss"] = float(m.group(1))
    m = re.search(r"dfl_loss[:\s]*([\d.]+)", line)
    if m:
        info["dfl_loss"] = float(m.group(1))
    m = re.search(r"mAP50[\(B\)]*[:\s]*([\d.]+)", line)
    if m:
        info["map50"] = float(m.group(1))
    m = re.search(r"mAP50-95[:\s]*([\d.]+)", line)
    if m:
        info["map50_95"] = float(m.group(1))
    m = re.search(r"fitness[:\s]*([\d.]+)", line)
    if m:
        info["fitness"] = float(m.group(1))
    return info


def _run_training_impl(
    dataset: str,
    dataset_path: str,
    experiment_name: str,
    model: str,
    epochs: int,
    batch: int,
    imgsz: int,
    lr0: float,
    device: str,
    workers: int,
    no_validate: str,
    dry_run: str,
) -> str:
    global _train_process

    _stop_event.clear()
    torch.cuda.empty_cache()

    dataset_actual = dataset_path.strip() or dataset
    if not dataset_actual:
        yield "请选择数据集或填入数据集路径"
        return

    name = experiment_name.strip() or f"webui_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    args = [
        sys.executable, "-m", "odp_platform.cli.train",
        "--dataset", dataset_actual,
        "--name", name,
        "--model", model.strip() or "yolo11n.pt",
        "--epochs", str(int(epochs)),
        "--batch", str(int(batch)),
        "--imgsz", str(int(imgsz)),
        "--lr0", str(float(lr0)),
        "--workers", str(int(workers)),
    ]
    if device.strip():
        args.extend(["--device", device.strip()])
    if no_validate == "是":
        args.append("--no-validate")
    if dry_run == "是":
        args.append("--dry-run")

    with _train_lock:
        if _train_process is not None:
            yield "已有训练任务正在运行，请先停止"
            return
        _train_process = subprocess.Popen(
            args,
            cwd=ROOT_DIR,
            env=platform_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

    output_lines = []
    gpu_check_counter = 0

    try:
        for line in iter(_train_process.stdout.readline, ""):
            if _stop_event.is_set():
                _train_process.terminate()
                output_lines.append("\n[训练已停止]")
                torch.cuda.empty_cache()
                break

            line = line.rstrip()

            oom_msg = _maybe_oom(line)
            if oom_msg:
                output_lines.append(f"\n⚠️ {oom_msg}")
                _train_process.terminate()
                output_lines.append("\n[训练已终止]")
                break

            info = _parse_progress(line)
            if info:
                parts = []
                if "epoch" in info:
                    parts.append(f"Epoch {info['epoch']}/{info['total_epochs']}")
                if "eta" in info:
                    parts.append(f"ETA {info['eta']}")
                if "box_loss" in info:
                    parts.append(f"box_loss {info['box_loss']:.4f}")
                if "map50" in info:
                    parts.append(f"mAP50 {info['map50']:.4f}")
                if "map50_95" in info:
                    parts.append(f"mAP50-95 {info['map50_95']:.4f}")
                gpu = _query_gpu_memory()
                if gpu:
                    output_lines.append(f"[{' | '.join(parts)}]")
                    output_lines.append(f"  {gpu}")
                else:
                    output_lines.append(line)
            else:
                output_lines.append(line)

            gpu_check_counter += 1
            if gpu_check_counter % 20 == 0:
                gpu_line = _query_gpu_memory()
                if gpu_line:
                    output_lines.append(f"[GPU] {gpu_line}")

            yield "\n".join(output_lines[-200:])

        ret = _train_process.wait()
        output_lines.append(f"\n退出码: {ret}")
        gpu_final = _query_gpu_memory()
        if gpu_final:
            output_lines.append(f"[GPU] {gpu_final}")
        torch.cuda.empty_cache()
        yield "\n".join(output_lines[-200:])
    except Exception as exc:
        output_lines.append(f"\n训练异常: {exc}")
        yield "\n".join(output_lines[-200:])
    finally:
        with _train_lock:
            if _train_process and _train_process.stdout:
                _train_process.stdout.close()
            _train_process = None
        torch.cuda.empty_cache()


def _stop_training() -> str:
    global _train_process
    _stop_event.set()
    with _train_lock:
        if _train_process is None:
            return "没有正在运行的训练任务"
        _train_process.terminate()
        try:
            _train_process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            _train_process.kill()
            _train_process.wait()
        _train_process = None
    torch.cuda.empty_cache()
    gpu = _query_gpu_memory()
    return f"训练已终止\n{gpu}" if gpu else "训练已终止"


def _refresh_datasets():
    datasets = list_dataset_names()
    return gr.update(choices=datasets, value=datasets[0] if datasets else None, interactive=True)


def _gpu_status():
    return _query_gpu_memory()


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


def _plot_training_curves(experiment: str) -> tuple:
    if not experiment:
        return None, None, None

    rows = _load_results_csv(experiment)
    if not rows:
        return None, None, None

    matplotlib.rcParams["font.size"] = 10
    epochs = [int(r.get("epoch", 0)) for r in rows]

    # Loss curves
    fig_loss, ax_loss = plt.subplots(figsize=(10, 5))
    for key, label, color in [
        ("train/box_loss", "Box Loss", "#e74c3c"),
        ("train/cls_loss", "Cls Loss", "#3498db"),
        ("train/dfl_loss", "DFL Loss", "#2ecc71"),
    ]:
        vals = [float(r.get(key, 0)) for r in rows if r.get(key, "")]
        if vals and any(v != 0 for v in vals):
            ax_loss.plot(epochs[:len(vals)], vals, label=label, color=color, linewidth=1.5)
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title(f"Training Loss — {experiment}")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)
    fig_loss.tight_layout()

    # Metric curves
    fig_metric, ax_metric = plt.subplots(figsize=(10, 5))
    for key, label, color in [
        ("metrics/mAP50(B)", "mAP50", "#e67e22"),
        ("metrics/mAP50-95(B)", "mAP50-95", "#9b59b6"),
        ("metrics/precision(B)", "Precision", "#2ecc71"),
        ("metrics/recall(B)", "Recall", "#e74c3c"),
    ]:
        vals = [float(r.get(key, 0)) for r in rows if r.get(key, "")]
        if vals and any(v != 0 for v in vals):
            ax_metric.plot(epochs[:len(vals)], vals, label=label, color=color, linewidth=1.5)
    ax_metric.set_xlabel("Epoch")
    ax_metric.set_ylabel("Value")
    ax_metric.set_title(f"Validation Metrics — {experiment}")
    ax_metric.legend()
    ax_metric.grid(True, alpha=0.3)
    fig_metric.tight_layout()

    # Bar chart for best metrics
    fig_bar, ax_bar = plt.subplots(figsize=(8, 4))
    if rows:
        last = rows[-1]
        metrics_map = [
            ("mAP50", float(last.get("metrics/mAP50(B)", 0))),
            ("mAP50-95", float(last.get("metrics/mAP50-95(B)", 0))),
            ("Precision", float(last.get("metrics/precision(B)", 0))),
            ("Recall", float(last.get("metrics/recall(B)", 0))),
        ]
        names = [m[0] for m in metrics_map]
        values = [m[1] for m in metrics_map]
        colors_bar = ["#e67e22", "#9b59b6", "#2ecc71", "#e74c3c"]
        bars = ax_bar.bar(names, values, color=colors_bar, width=0.5, edgecolor="white")
        for bar, val in zip(bars, values):
            ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{val:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax_bar.set_ylim(0, max(values) * 1.25 if max(values) > 0 else 1)
        ax_bar.set_ylabel("Value")
        ax_bar.set_title(f"Best Metrics — {experiment}")
        ax_bar.grid(True, alpha=0.3, axis="y")
    fig_bar.tight_layout()

    buf_loss = io.BytesIO()
    fig_loss.savefig(buf_loss, format="png", dpi=120, bbox_inches="tight")
    buf_loss.seek(0)
    plt.close(fig_loss)

    buf_metric = io.BytesIO()
    fig_metric.savefig(buf_metric, format="png", dpi=120, bbox_inches="tight")
    buf_metric.seek(0)
    plt.close(fig_metric)

    buf_bar = io.BytesIO()
    fig_bar.savefig(buf_bar, format="png", dpi=120, bbox_inches="tight")
    buf_bar.seek(0)
    plt.close(fig_bar)

    return buf_loss, buf_metric, buf_bar


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

    exp_dir = RUNS_DIR / "experiments" / experiment
    weights_dir = exp_dir / "weights"
    has_best = (weights_dir / "best.pt").exists()

    lines = [
        f"实验: {experiment}",
        f"总轮数: {total_epochs}",
        f"最佳 mAP50: {best_map50:.4f}",
        f"最佳 mAP50-95: {best_map50_95:.4f}",
        f"最终 mAP50: {float(last.get('metrics/mAP50(B)', 0)):.4f}",
        f"最终 Precision: {float(last.get('metrics/precision(B)', 0)):.4f}",
        f"最终 Recall: {float(last.get('metrics/recall(B)', 0)):.4f}",
        f"模型权重: {'已保存' if has_best else '无'}",
    ]
    return "\n".join(lines)


def _get_pregen_image_safe(experiment: str, filename: str) -> Image.Image:
    img_path = RUNS_DIR / "experiments" / experiment / filename
    if img_path.exists():
        try:
            return Image.open(img_path)
        except Exception:
            pass
    return Image.new("RGB", (400, 300), (240, 240, 240))


def _load_admin_experiment_data(experiment: str) -> tuple:
    if not experiment:
        bl = Image.new("RGB", (400, 300), (240, 240, 240))
        return (bl, bl, bl, bl, bl, bl, bl, "请选择实验")

    rp = _get_pregen_image_safe(experiment, "results.png")
    cm = _get_pregen_image_safe(experiment, "confusion_matrix.png")
    cn = _get_pregen_image_safe(experiment, "confusion_matrix_normalized.png")
    lb = _get_pregen_image_safe(experiment, "labels.jpg")
    pr = _get_pregen_image_safe(experiment, "BoxPR_curve.png")
    f1 = _get_pregen_image_safe(experiment, "BoxF1_curve.png")

    rows = _load_results_csv(experiment)
    bc = Image.new("RGB", (400, 300), (240, 240, 240))
    if rows:
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
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)
        plt.close(fig)
        bc = Image.open(buf)

    return (rp, cm, cn, lb, pr, f1, bc, _summarize_experiment(experiment))


def create_training_ui() -> None:
    datasets = list_dataset_names()
    batch_hint = _auto_batch_hint("yolo11n.pt", 640)

    with gr.Row(elem_classes=["odp-row", "odp-row-five"]):
        refresh_btn = gr.Button("刷新")
        dataset_dd = gr.Dropdown(
            label="数据集",
            choices=datasets,
            value=datasets[0] if datasets else None,
            filterable=True,
            interactive=True,
        )
        dataset_path = gr.Textbox(
            label="数据集路径（可替代下拉选择）",
            placeholder="eg. configs/datasets/rsod.yaml",
            max_lines=1,
        )
        experiment_name = gr.Textbox(
            label="实验名",
            placeholder="webui_rsod_001",
            max_lines=1,
        )
        model = gr.Textbox(
            label="模型",
            value="yolo11n.pt",
            max_lines=1,
        )
    with gr.Row(elem_classes=["odp-row", "odp-row-four"]):
        epochs = gr.Number(label="Epochs", value=1, precision=0, minimum=1)
        batch = gr.Number(label="Batch", value=1, precision=0, minimum=1)
        imgsz = gr.Number(label="Image Size", value=640, precision=0, minimum=32)
        lr0 = gr.Number(label="LR0", value=0.01, minimum=0.000001)
    with gr.Row(elem_classes=["odp-row", "odp-row-two"]):
        device = gr.Textbox(
            label="Device（空=auto）",
            value="",
            max_lines=1,
        )
        workers = gr.Number(label="Workers", value=2, precision=0, minimum=0)
    with gr.Row(elem_classes=["odp-row", "odp-row-two"]):
        no_validate = gr.Dropdown(label="跳过质检", choices=["否", "是"], value="否", filterable=False)
        dry_run = gr.Dropdown(label="仅生成配置（不训练）", choices=["否", "是"], value="否", filterable=False)
    gpu_status_box = gr.Textbox(
        label="GPU 状态",
        value=_query_gpu_memory(),
        interactive=False,
        max_lines=3,
    )
    if batch_hint:
        gr.Info(batch_hint)
    with gr.Row(elem_classes=["odp-row", "odp-row-two"]):
        run_btn = gr.Button("开始训练", variant="primary", size="lg")
        stop_btn = gr.Button("停止训练", variant="stop", size="lg")
    with gr.Row(elem_classes=["odp-row"]):
        refresh_gpu_btn = gr.Button("刷新 GPU 信息", size="sm", scale=0)
    output = gr.Code(label="输出", language="shell", lines=20)

    refresh_btn.click(fn=_refresh_datasets, outputs=[dataset_dd])
    run_btn.click(
        fn=_run_training_impl,
        inputs=[
            dataset_dd,
            dataset_path,
            experiment_name,
            model,
            epochs,
            batch,
            imgsz,
            lr0,
            device,
            workers,
            no_validate,
            dry_run,
        ],
        outputs=[output],
    )
    stop_btn.click(
        fn=_stop_training,
        outputs=[output],
    )
    refresh_gpu_btn.click(
        fn=_gpu_status,
        outputs=[gpu_status_box],
    )

    gr.Markdown("---")
    gr.Markdown("### 训练结果可视化")
    experiments = _list_experiments()
    blank = Image.new("RGB", (400, 300), (240, 240, 240))

    with gr.Row(elem_classes=["odp-row", "odp-row-two"]):
        exp_dd = gr.Dropdown(
            label="选择实验",
            choices=experiments,
            value=experiments[0] if experiments else None,
            filterable=True,
            interactive=True,
        )
        refresh_exp_btn = gr.Button("刷新实验列表")
    with gr.Row():
        exp_summary = gr.Textbox(label="实验摘要", lines=6, interactive=False)

    with gr.Tabs():
        with gr.TabItem("训练曲线"):
            with gr.Row():
                exp_results = gr.Image(value=blank, label="Loss + 验证指标", container=True, height=380)
            with gr.Row():
                exp_bar = gr.Image(value=blank, label="最佳指标柱状图", container=True, height=300)
        with gr.TabItem("评估矩阵"):
            with gr.Row():
                exp_confusion = gr.Image(value=blank, label="混淆矩阵", container=True, height=340)
            with gr.Row():
                exp_confusion_norm = gr.Image(value=blank, label="归一化混淆矩阵", container=True, height=340)
            with gr.Row():
                exp_pr = gr.Image(value=blank, label="PR 曲线", container=True, height=300)
            with gr.Row():
                exp_f1 = gr.Image(value=blank, label="F1 曲线", container=True, height=300)
        with gr.TabItem("类别分布"):
            with gr.Row():
                exp_labels = gr.Image(value=blank, label="类别分布", container=True, height=400)

    refresh_exp_btn.click(
        fn=lambda: gr.update(choices=_list_experiments()),
        outputs=[exp_dd],
    )
    exp_dd.change(
        fn=_load_admin_experiment_data,
        inputs=[exp_dd],
        outputs=[exp_results, exp_confusion, exp_confusion_norm,
                 exp_labels, exp_pr, exp_f1, exp_bar, exp_summary],
    )
