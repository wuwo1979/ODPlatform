from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from ultralytics import YOLO

from odp_platform.common.logging_utils import get_logger
from odp_platform.common.paths import (
    APP_DIR,
    CONFIGS_DIR,
    CONFIGS_DATASETS_DIR,
    LOGGING_DIR,
    ROOT_DIR,
    RUNS_DIR,
    dataset_yaml_path,
)
from odp_platform.run_config import (
    ConfigSnapshot,
    build_config,
    generate_template_to_file,
    save_snapshot_to_file,
)

logger = logging.getLogger("odp-val")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="odp-val",
        description=(
            "ODPlatform 模型评估命令 —— 使用 Ultralytics YOLO 验证模型精度。\n"
            "  1) 配置加载（支持 YAML/CLI 覆盖）\n"
            "  2) 模型推理验证\n"
            "  3) 结果保存到 runs/val/ 目录\n"
            "日志自动保存到 logging/val/ 目录，便于与评估结果对应。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--model", "-m", required=True,
                        help="模型路径（.pt 文件）或模型名称（如 yolo11n.pt）")
    parser.add_argument("--dataset", "-d", required=True,
                        help="数据集名（= configs/datasets/<name>.yaml）")
    parser.add_argument("--task", "-t", default="detect",
                        choices=("detect", "segment", "classify"),
                        help="算法任务类型 (默认 detect)")

    parser.add_argument("--config", "-c", default=None,
                        help="验证配置 YAML（不指定则自动生成）")
    parser.add_argument("--conf", type=float, default=None,
                        help="置信度阈值（覆盖配置中的值）")
    parser.add_argument("--iou", type=float, default=None,
                        help="NMS IoU 阈值（覆盖配置中的值）")
    parser.add_argument("--device", default=None,
                        help="验证设备（覆盖配置中的值，默认 cpu）")
    parser.add_argument("--batch", type=int, default=None,
                        help="验证批次大小（覆盖配置中的值）")
    parser.add_argument("--imgsz", type=int, default=None,
                        help="输入图像尺寸（覆盖配置中的值）")
    parser.add_argument("--half", type=bool, default=None,
                        help="半精度推理 FP16（覆盖配置中的值）")
    parser.add_argument("--max-det", type=int, default=None,
                        help="单张图像最大检测框数（覆盖配置中的值）")
    parser.add_argument("--name", default=None,
                        help="实验名称（默认自动生成）")
    parser.add_argument("--split", default="val",
                        choices=("train", "val", "test"),
                        help="验证数据集划分 (默认 val)")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅生成配置和参数预览，不实际验证")
    parser.add_argument("--save-json", action="store_true",
                        help="保存详细验证结果为 JSON")

    return parser


def _collect_cli_overrides(args) -> dict:
    overrides = {}
    for key in ("conf", "iou", "device", "batch", "imgsz", "max_det"):
        val = getattr(args, key, None)
        if val is not None:
            overrides[key] = val
    if args.half is not None:
        overrides["half"] = args.half
    return overrides


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    get_logger(base_path=LOGGING_DIR, log_type="val", log_level=logging.INFO,
               logger_name="odp-val")

    experiment_name = args.name or f"val_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    dataset = args.dataset
    model_src = args.model

    logger.info("=" * 60)
    logger.info(f"ODPlatform 模型评估 — {experiment_name}")
    logger.info(f"模型: {model_src}  数据集: {dataset}  划分: {args.split}")
    logger.info("=" * 60)

    # ── Step 1: 解析模型路径 ──
    model_path = Path(model_src)
    if not model_path.exists():
        search_dirs = [
            Path.cwd(),
            RUNS_DIR / "experiments",
            ROOT_DIR / "models",
            APP_DIR.parent.parent / "models",
        ]
        found = False
        for search_dir in search_dirs:
            candidate = search_dir / model_src
            if candidate.exists():
                model_path = candidate
                found = True
                break
            for pt in sorted(search_dir.glob(f"**/{model_src}")):
                model_path = pt
                found = True
                break
        if not found:
            logger.error(f"模型文件不存在: {model_src}")
            logger.info(f"搜索路径: {[str(d) for d in search_dirs]}")
            return 2

    model_path = model_path.resolve()
    logger.info(f"[1/3] 模型路径: {model_path} ({model_path.stat().st_size / 1024**2:.1f} MB)")

    # ── Step 2: 运行配置 ──
    logger.info("[2/3] 运行配置")

    config_path = args.config
    cli_overrides = _collect_cli_overrides(args)
    cli_overrides["model"] = str(model_path)

    if config_path:
        config_file = Path(config_path)
        if not config_file.is_absolute():
            config_file = APP_DIR / config_file
        logger.info(f"      加载配置: {config_file}")
        bundle = build_config(
            task="val",
            yaml_path=config_file,
            cli_args=cli_overrides if cli_overrides else None,
        )
    else:
        logger.info("      自动生成验证配置")
        auto_config = CONFIGS_DIR / f"{experiment_name}_config.yaml"
        generate_template_to_file(task="val", output_path=auto_config, force=True)
        bundle = build_config(
            task="val",
            yaml_path=auto_config,
            cli_args=cli_overrides if cli_overrides else None,
        )

    if not bundle.valid:
        logger.error("      配置验证失败:")
        for err in bundle.errors:
            logger.error(f"        [ERROR] {err.field}: {err.message}")
        return 2

    snapshot = ConfigSnapshot.from_bundle(bundle)
    snapshot_run_dir = RUNS_DIR / "run_config" / experiment_name
    snapshot_run_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_run_dir / "config_snapshot.json"
    save_snapshot_to_file(snapshot, snapshot_path)

    ultralytics_args = bundle.to_ultralytics_args()
    logger.info(f"      Ultralytics 参数: {json.dumps(ultralytics_args, ensure_ascii=False)}")

    # ── Step 3: 执行验证 ──
    data_yaml = dataset_yaml_path(dataset)
    if not data_yaml.exists():
        logger.error(f"      数据集 YAML 不存在: {data_yaml}")
        logger.info(f"      可用数据集: {list(CONFIGS_DATASETS_DIR.glob('*.yaml'))}")
        return 2

    val_project_dir = RUNS_DIR / "val"
    val_project_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        logger.info("[3/3] 跳过验证（--dry-run）")
        logger.info(f"\n实验 '{experiment_name}' 的完整配置已保存:")
        logger.info(f"  配置快照: {snapshot_path}")
        logger.info("\n直接验证命令:")
        cmd_parts = [
            "yolo",
            f"task={args.task}",
            f"mode=val",
            f"model={model_path}",
            f"data={data_yaml}",
        ]
        for k, v in ultralytics_args.items():
            if k in ("model",):
                continue
            cmd_parts.append(f"{k}={v}")
        logger.info(" ".join(cmd_parts))
        return 0

    logger.info(f"[3/3] 启动验证 — 模型: {model_path.name}, 数据: {data_yaml.name}")

    try:
        model = YOLO(str(model_path))
        results = model.val(
            data=str(data_yaml),
            project=str(val_project_dir),
            name=experiment_name,
            split=args.split,
            save_json=args.save_json,
            **ultralytics_args,
        )

        metrics = {
            "map50": float(getattr(results, "map50", 0)),
            "map50_95": float(getattr(results, "map50_95", 0)),
            "precision": float(getattr(results, "p", 0)),
            "recall": float(getattr(results, "r", 0)),
            "fitness": float(getattr(results, "fitness", 0)),
        }
        val_dir = val_project_dir / experiment_name

        metrics_path = val_dir / "metrics.json"
        metrics_path.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        logger.info(f"      ✓ 评估完成")
        logger.info(f"      输出目录: {val_dir}")
        logger.info(f"      mAP50:    {metrics['map50']:.4f}")
        logger.info(f"      mAP50-95: {metrics['map50_95']:.4f}")
        logger.info(f"      Precision: {metrics['precision']:.4f}")
        logger.info(f"      Recall:    {metrics['recall']:.4f}")

        summary = {
            "experiment": experiment_name,
            "model": str(model_path),
            "dataset": dataset,
            "split": args.split,
            "task": args.task,
            "metrics": metrics,
            "output_dir": str(val_dir),
            "timestamp": datetime.now().isoformat(),
            "success": True,
        }
        summary_path = snapshot_run_dir / "val_summary.json"
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(f"      评估摘要: {summary_path}")

        return 0

    except KeyboardInterrupt:
        logger.warning("评估被用户中断")
        return 130
    except Exception as e:
        logger.exception(f"评估失败: {e}")
        summary = {
            "experiment": experiment_name,
            "model": str(model_path),
            "dataset": dataset,
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "error": str(e),
        }
        summary_path = snapshot_run_dir / "val_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return 2


if __name__ == "__main__":
    sys.exit(main())
