from __future__ import annotations

import csv
import json
import logging
import re
from pathlib import Path
from typing import Callable

import requests

from odp_platform.common.paths import CONFIGS_DATASETS_DIR, RUNS_DIR
from odp_platform.webui.utils import list_model_files

logger = logging.getLogger(__name__)


# =========================================================
# Tool Implementations  (all return ASCII-safe text)
# =========================================================

def tool_list_models() -> str:
    models = list_model_files()
    if not models:
        return "[INFO] 未找到任何 .pt 模型文件"
    lines = [f"[INFO] 共 {len(models)} 个模型:"]
    for m in models:
        p = Path(m)
        size_mb = p.stat().st_size / 1024 / 1024 if p.exists() else 0
        lines.append(f"  - {m} ({size_mb:.1f}MB)")
    return "\n".join(lines)


def tool_list_datasets() -> str:
    yamls = sorted(CONFIGS_DATASETS_DIR.glob("*.yaml"))
    if not yamls:
        return "[INFO] 未找到任何数据集配置文件"
    lines = [f"[INFO] 共 {len(yamls)} 个数据集:"]
    for y in yamls:
        lines.append(f"  - {y.stem}")
    return "\n".join(lines)


def tool_list_experiments() -> str:
    exp_dir = RUNS_DIR / "experiments"
    if not exp_dir.exists():
        return "[INFO] 暂无训练实验"
    exps = sorted([d.name for d in exp_dir.iterdir() if d.is_dir()], reverse=True)
    if not exps:
        return "[INFO] 暂无训练实验"
    lines = [f"[INFO] 共 {len(exps)} 个实验:"]
    for name in exps:
        csv_path = exp_dir / name / "results.csv"
        if csv_path.exists():
            try:
                with open(csv_path) as f:
                    rows = list(csv.DictReader(f))
                if rows:
                    best = max(rows, key=lambda r: float(r.get("metrics/mAP50(B)", 0)))
                    lines.append(
                        f"  - {name}  (best mAP50={float(best.get('metrics/mAP50(B)', 0)):.4f})"
                    )
                    continue
            except Exception:
                pass
        lines.append(f"  - {name}")
    return "\n".join(lines)


def tool_get_experiment(name: str) -> str:
    exp_dir = RUNS_DIR / "experiments" / name
    if not exp_dir.exists():
        return f"[ERROR] 实验不存在: {name}"
    csv_path = exp_dir / "results.csv"
    if not csv_path.exists():
        return f"[ERROR] 实验 {name} 无 results.csv"
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return f"[ERROR] 实验 {name} 的 results.csv 为空"
    last = rows[-1]
    best = max(rows, key=lambda r: float(r.get("metrics/mAP50(B)", 0)))
    info = {
        "实验名": name,
        "总轮数": len(rows),
        "最终指标": {
            "mAP50": last.get("metrics/mAP50(B)", "N/A"),
            "mAP50-95": last.get("metrics/mAP50-95(B)", "N/A"),
            "precision": last.get("metrics/precision(B)", "N/A"),
            "recall": last.get("metrics/recall(B)", "N/A"),
        },
        "最佳轮次": {
            "epoch": best.get("epoch", "N/A"),
            "mAP50": best.get("metrics/mAP50(B)", "N/A"),
        },
    }
    config_path = exp_dir / "config_snapshot.json"
    if config_path.exists():
        try:
            info["配置"] = json.loads(config_path.read_text())
        except Exception:
            pass
    return json.dumps(info, ensure_ascii=False, indent=2)


def tool_run_inference(model_path: str, image_path: str, conf: float = 0.25, iou: float = 0.45) -> str:
    model_file = Path(model_path)
    image_file = Path(image_path)
    if not model_file.exists():
        return f"[ERROR] 模型文件不存在: {model_path}"
    if not image_file.exists():
        return f"[ERROR] 图片文件不存在: {image_path}"
    try:
        from odp_platform.inference.engine import Detector
        import cv2
        detector = Detector(str(model_file))
        detector.warmup()
        img = cv2.imread(str(image_file))
        if img is None:
            return f"[ERROR] 无法读取图片: {image_path}"
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = detector.detect(img_rgb)
        lines = [
            f"[OK] 推理完成",
            f"  模型: {model_file.name}",
            f"  图片: {image_file.name}",
            f"  检测目标: {len(result.detections)} 个",
            f"  推理耗时: {result.inference_ms:.1f}ms",
        ]
        for d in result.detections[:10]:
            lines.append(
                f"  - {d.class_name} "
                f"(置信度: {d.confidence:.3f}, "
                f"框: [{d.bbox[0]:.3f}, {d.bbox[1]:.3f}, "
                f"{d.bbox[2]:.3f}, {d.bbox[3]:.3f}])"
            )
        if len(result.detections) > 10:
            lines.append(f"  ... 还有 {len(result.detections) - 10} 个目标")
        return "\n".join(lines)
    except ImportError as e:
        return f"[ERROR] 推理模块未就绪: {e}"
    except Exception as e:
        return f"[ERROR] 推理失败: {e}"


def tool_get_gpu_info() -> str:
    try:
        import torch
    except ImportError:
        return "[INFO] PyTorch 未安装"
    if not torch.cuda.is_available():
        return "[INFO] GPU 不可用 (CUDA)"
    lines = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        total = props.total_memory / 1024**3
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        free = total - reserved
        lines.append(
            f"  GPU{i} [{props.name}]: "
            f"已用 {allocated:.1f}G / 空闲 {free:.1f}G / 总计 {total:.1f}G"
        )
    return "[INFO] GPU 状态:\n" + "\n".join(lines)


# =========================================================
# Intent Router — 关键词匹配，不依赖 LLM function calling
# =========================================================

_TOOL_MAP: list[tuple[re.Pattern, str, Callable, list[str]]] = []


def _route(pattern: str, tool_name: str, fn: Callable, params: list[str] | None = None):
    _TOOL_MAP.append((re.compile(pattern, re.IGNORECASE), tool_name, fn, params or []))


_route(r"模型|model|\.pt|权重|checkpoint", "list_models", tool_list_models)
_route(r"数据集|dataset|数据", "list_datasets", tool_list_datasets)
_route(r"实验|训练|exp|train.*结果|训练结果", "list_experiments", tool_list_experiments)
_route(r"GPU|显存|显卡|cuda|gpu", "tool_get_gpu_info", tool_get_gpu_info)


def _resolve_image_path(text: str) -> str | None:
    candidates = re.findall(r'[a-zA-Z]:\\(?:[^\\\s]+\\)*[^\\\s]+\.(?:jpg|jpeg|png|bmp)', text)
    if not candidates:
        candidates = re.findall(r'(?:/[^\s/]+)+\.(?:jpg|jpeg|png|bmp)', text)
    return candidates[0] if candidates else None


def _resolve_model_path(text: str) -> str | None:
    candidates = re.findall(r'[a-zA-Z]:\\(?:[^\\\s]+\\)*[^\\\s]+\.pt', text)
    if not candidates:
        candidates = re.findall(r'(?:/[^\s/]+)+\.pt', text)
    if candidates:
        return candidates[0]
    models = list_model_files()
    if models:
        for m in models:
            name = Path(m).stem.lower()
            if name in text.lower():
                return m
    return None


def run_agent(
    user_message: str,
    history: list[dict[str, str]] | None,
    api_key: str,
    api_base: str,
    model_name: str,
) -> tuple[list[dict[str, str]], str]:
    """执行 agent：关键词匹配 -> 本地执行工具 -> LLM 美化输出。

    Returns:
        (updated_history, empty_string_for_textbox)
    """
    text = (user_message or "").strip()
    history = list(history or [])
    if not text:
        return history, ""

    history.append({"role": "user", "content": text})

    tool_result: str | None = None

    # 检测/推理（需要路径参数）
    if re.search(r"推理|检测|识别|infer|detect", text, re.IGNORECASE):
        model_path = _resolve_model_path(text)
        image_path = _resolve_image_path(text)
        if model_path and image_path:
            tool_result = tool_run_inference(model_path, image_path)
        elif model_path and not image_path:
            tool_result = (
                f"[WARN] 检测到模型 {model_path}，但未找到图片路径。\n"
                f"请提供图片路径，例如: C:\\path\\to\\image.jpg"
            )
        elif not model_path and image_path:
            tool_result = (
                f"[WARN] 检测到图片 {image_path}，但未找到模型路径。\n"
                f"请提供 .pt 模型路径"
            )
        else:
            models = list_model_files()
            if models:
                tool_result = "[INFO] 请指定模型和图片路径。可用模型:\n" + \
                    "\n".join(f"  - {m}" for m in models)
            else:
                tool_result = "[ERROR] 没有可用模型，请先上传 .pt 文件"

    # 关键词匹配通用工具
    if tool_result is None:
        for pattern, name, fn, _ in _TOOL_MAP:
            if pattern.search(text):
                tool_result = fn()
                break

    # 匹配到工具：用 LLM 美化回复
    if tool_result is not None:
        formatted = _format_with_llm(text, tool_result, api_key, api_base, model_name)
        history.append({"role": "assistant", "content": formatted})
        return history, ""

    # 没有匹配到工具，走普通 LLM 对话
    return _simple_chat_fallback(history, api_key, api_base, model_name)


def _format_with_llm(
    user_text: str,
    tool_result: str,
    api_key: str,
    api_base: str,
    model_name: str,
) -> str:
    """把工具执行结果发给 LLM 写成自然语言回复。"""
    system_prompt = (
        "你是 ODPlatform 的智能助手（基于 DeepSeek，不是 OpenAI）。\n"
        "下面是一个工具的执行结果，请你用自然语言、有条理地向用户介绍这些信息。\n"
        "不要编造不存在的信息，直接基于结果数据回答。\n"
        "如果结果是JSON，用简洁的格式展示关键信息。"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"用户问题: {user_text}\n\n工具返回结果:\n{tool_result}\n\n请用中文自然回答。"},
    ]
    return _call_llm(messages, api_key, api_base, model_name)


def _simple_chat_fallback(
    history: list[dict[str, str]],
    api_key: str,
    api_base: str,
    model_name: str,
) -> tuple[list[dict[str, str]], str]:
    """没有匹配工具时，直接 LLM 对话。"""
    system_prompt = (
        "你是 DeepSeek 模型（不是 OpenAI），是 ODPlatform 的智能助手。"
        "你可以帮助用户查询模型、数据集、实验、GPU状态和执行推理检测。"
    )
    messages = [{"role": "system", "content": system_prompt}]
    for msg in history:
        messages.append({"role": msg["role"], "content": str(msg.get("content", ""))})
    content = _call_llm(messages, api_key, api_base, model_name)
    history.append({"role": "assistant", "content": content})
    return history, ""


def _call_llm(
    messages: list[dict[str, str]],
    api_key: str,
    api_base: str,
    model_name: str,
) -> str:
    """调用 LLM API 获取回复。使用 requests 库避免 urllib 编码问题。"""
    url = f"{api_base.rstrip('/')}/chat/completions"
    payload = {
        "model": model_name,
        "messages": messages,
        "stream": False,
        "max_tokens": 4096,
    }
    try:
        resp = requests.post(
            url,
            json=payload,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except requests.exceptions.HTTPError as exc:
        body = exc.response.text[:300] if exc.response is not None else ""
        return f"API 请求失败 ({exc.response.status_code if exc.response else '?'}): {body}"
    except requests.exceptions.RequestException as exc:
        return f"请求异常: {exc}"
