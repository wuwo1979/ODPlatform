from __future__ import annotations

import csv
import io
import json
import logging
import tempfile
import threading
import time as _time
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from odp_platform.common.paths import RUNS_DIR
from odp_platform.webui.utils import list_images, list_model_files

logger = logging.getLogger(__name__)

_detector_cache: dict[str, Any] = {}
_cache_lock = threading.Lock()
_server_cam_stop = threading.Event()
_server_cap_ref = [None]
_server_cap_lock = threading.Lock()


def _release_detector_cache():
    with _cache_lock:
        for model_path, detector in list(_detector_cache.items()):
            try:
                detector.release()
            except Exception:
                pass
        _detector_cache.clear()
    torch.cuda.empty_cache()


def _release_server_camera():
    with _server_cap_lock:
        cap = _server_cap_ref[0]
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
            _server_cap_ref[0] = None
    _server_cam_stop.set()
    torch.cuda.empty_cache()


def _gpu_info() -> str:
    import torch
    if not torch.cuda.is_available():
        return ""
    parts = []
    for i in range(torch.cuda.device_count()):
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        cached = torch.cuda.memory_reserved(i) / 1024**3
        parts.append(f"GPU{i}: {allocated:.1f}/{cached:.1f}/{total:.1f} GiB")
    return " | ".join(parts)


def _model_choices() -> list[str]:
    return list_model_files()


def _refresh_models() -> gr.update:
    models = _model_choices()
    return gr.update(
        choices=models,
        value=models[0] if models else None,
    )


def _get_or_create_detector(model_path: str, conf: float, iou: float) -> Any | None:
    if not model_path:
        return None
    try:
        from odp_platform.inference.engine import Detector
    except ImportError as exc:
        logger.error("推理模块未就绪: %s", exc)
        return None
    with _cache_lock:
        detector = _detector_cache.get(model_path)
        if detector is None:
            logger.info("首次加载模型: %s", model_path)
            try:
                detector = Detector(model_path)
            except Exception as exc:
                logger.error("模型加载失败 %s: %s", model_path, exc)
                return None
            detector._model_path = model_path
            _detector_cache[model_path] = detector
        detector.conf = float(conf)
        detector.iou = float(iou)
    return detector


def _run_single_detection(
    image: Any,
    model_path: str,
    conf: float,
    iou: float,
    detector_state: Any,
) -> tuple[Any | None, list[dict[str, Any]], str, Any]:
    if image is None:
        return None, [], "未选择图片", detector_state
    if not model_path:
        return None, [], "未选择模型", detector_state

    try:
        import numpy as np

        from odp_platform.inference.visualizer import draw_detections
    except ImportError as exc:
        return None, [], f"推理依赖未就绪: {exc}", detector_state

    try:
        if isinstance(image, np.ndarray):
            image_np = image
        else:
            image_np = np.array(image)
        if image_np.size == 0:
            return None, [], "图片数据为空", detector_state
        if image_np.ndim not in {2, 3}:
            return None, [], f"图片维度异常: {image_np.ndim}", detector_state

        is_cached = (
            detector_state is not None
            and hasattr(detector_state, '_model_path')
            and detector_state._model_path == model_path
        )
        if is_cached:
            detector = detector_state
            detector.conf = float(conf)
            detector.iou = float(iou)
        else:
            from odp_platform.inference.engine import Detector
            detector = Detector(model_path)
            detector.conf = float(conf)
            detector.iou = float(iou)
            detector._model_path = model_path

        result = detector.detect(image_np)
        rendered = draw_detections(image_np, result.detections)
        rows = [
            {
                "class": detection.class_name,
                "conf": round(detection.confidence, 4),
                "bbox": list(detection.bbox),
            }
            for detection in result.detections
        ]
        status = f"{Path(model_path).name} | 检测数: {len(rows)} | {result.inference_ms:.1f} ms"
        return rendered, rows, status, detector
    except Exception as exc:
        logger.exception("单图检测失败")
        import numpy as np
        blank = np.full((100, 300, 3), 200, dtype=np.uint8)
        return blank, [], f"检测失败: {exc}", detector_state


def _run_folder_detection(
    folder_path: str,
    model_path: str,
    conf: float,
    iou: float,
    detector_state: Any,
) -> tuple[list[Any], str, Any]:
    if not folder_path:
        return [], "请输入文件夹路径", detector_state
    if not model_path:
        return [], "未选择模型", detector_state

    try:
        import numpy as np
        from PIL import Image

        from odp_platform.inference.visualizer import draw_detections
    except ImportError as exc:
        return [], f"依赖未就绪: {exc}", detector_state

    folder = Path(folder_path)
    if not folder.is_dir():
        return [], f"路径不存在或不是文件夹: {folder_path}", detector_state

    image_paths = list_images(folder)
    if not image_paths:
        return [], f"文件夹中无图片: {folder_path}", detector_state

    is_cached = (
        detector_state is not None
        and hasattr(detector_state, '_model_path')
        and detector_state._model_path == model_path
    )
    if is_cached:
        detector = detector_state
    else:
        try:
            from odp_platform.inference.engine import Detector
            detector = Detector(model_path)
        except Exception as exc:
            return [], f"模型加载失败: {exc}", detector_state
        detector._model_path = model_path
    detector.conf = float(conf)
    detector.iou = float(iou)

    results = []
    for img_path in image_paths[:100]:
        try:
            image = np.array(Image.open(img_path))
            result = detector.detect(image)
            rendered = draw_detections(image, result.detections)
            results.append(rendered)
        except Exception as exc:
            logger.warning("跳过图片 %s: %s", img_path, exc)
            continue

    status = f"处理完成: {len(results)}/{len(image_paths)} 张 | 模型: {Path(model_path).name}"
    return results, status, detector


def _run_folder_detection_wrapped(
    folder_path: str,
    output_dir: str,
    model_path: str,
    conf: float,
    iou: float,
    detector_state: Any,
) -> tuple:
    if not folder_path:
        return [], "请输入文件夹路径", detector_state, {}, [], []
    if not model_path:
        return [], "未选择模型", detector_state, {}, [], []

    try:
        import cv2
        import numpy as np
        from PIL import Image

        from odp_platform.inference.visualizer import draw_detections
    except ImportError as exc:
        return [], f"依赖未就绪: {exc}", detector_state, {}, [], []

    folder = Path(folder_path)
    if not folder.is_dir():
        return [], f"路径不存在: {folder_path}", detector_state, {}, [], []

    image_paths = list_images(folder)
    if not image_paths:
        return [], f"文件夹中无图片: {folder_path}", detector_state, {}, [], []

    is_cached = (
        detector_state is not None
        and hasattr(detector_state, '_model_path')
        and detector_state._model_path == model_path
    )
    if is_cached:
        detector = detector_state
    else:
        try:
            from odp_platform.inference.engine import Detector
            detector = Detector(model_path)
        except Exception as exc:
            return [], f"模型加载失败: {exc}", detector_state, {}, [], []
        detector._model_path = model_path
    detector.conf = float(conf)
    detector.iou = float(iou)

    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
    else:
        out_path = None

    results = []
    detail_rows = []
    class_totals: dict[str, int] = {}
    total_ms = 0.0
    total_objs = 0

    for img_path in image_paths[:100]:
        try:
            image = np.array(Image.open(img_path))
            t0 = __import__("time").time()
            result = detector.detect(image)
            elapsed = (__import__("time").time() - t0) * 1000
            rendered = draw_detections(image, result.detections)
            results.append(rendered)

            if out_path:
                rendered_bgr = cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(out_path / f"{img_path.stem}_result.jpg"), rendered_bgr)

            classes_in_img = {}
            for d in result.detections:
                classes_in_img[d.class_name] = classes_in_img.get(d.class_name, 0) + 1
                class_totals[d.class_name] = class_totals.get(d.class_name, 0) + 1
            total_objs += len(result.detections)
            total_ms += elapsed

            class_str = ", ".join(f"{k}:{v}" for k, v in classes_in_img.items()) if classes_in_img else "-"
            detail_rows.append([
                img_path.name,
                len(result.detections),
                class_str,
                round(sum(d.confidence for d in result.detections) / max(len(result.detections), 1), 4),
                round(elapsed, 1),
            ])
        except Exception as exc:
            logger.warning("跳过图片 %s: %s", img_path, exc)
            detail_rows.append([img_path.name, 0, "error", 0, 0])
            continue

    summary = {
        "总图片": len(image_paths),
        "成功": len(results),
        "总目标数": total_objs,
        "总耗时(ms)": round(total_ms, 1),
        "平均每张(ms)": round(total_ms / max(len(results), 1), 1),
        "类别分布": class_totals,
    }
    status = f"处理完成: {len(results)}/{len(image_paths)} 张 | {total_objs} 个目标 | 模型: {Path(model_path).name}"
    return results, status, detector, summary, detail_rows, detail_rows


def _run_server_camera(
    cam_id: int,
    model_path: str,
    conf: float,
    iou: float,
    cam_res: str = "640x480",
) -> Any:
    import os
    import warnings

    import cv2
    import numpy as np

    from odp_platform.inference.visualizer import draw_detections, draw_info_panel

    _release_server_camera()
    _server_cam_stop.clear()

    os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
    os.environ["OBSENSOR_DEBUG"] = "0"
    warnings.filterwarnings("ignore", message=".*obsensor.*")
    warnings.filterwarnings("ignore", message=".*FFMPEG.*")

    backends_to_try = [cv2.CAP_DSHOW]
    if hasattr(cv2, "CAP_MSMF"):
        backends_to_try.insert(0, cv2.CAP_MSMF)

    cap = None
    for backend in backends_to_try:
        try:
            candidate = cv2.VideoCapture(cam_id, backend)
            if candidate.isOpened():
                cap = candidate
                break
            candidate.release()
        except Exception:
            continue

    if cap is None:
        try:
            cap = cv2.VideoCapture(cam_id)
            if not cap.isOpened():
                yield None
                return
        except Exception:
            yield None
            return

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    try:
        res_parts = [int(x) for x in cam_res.split("x")]
        if len(res_parts) == 2:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, res_parts[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res_parts[1])
    except Exception:
        pass

    with _server_cap_lock:
        _server_cap_ref[0] = cap

    import time as _time
    frame_count = 0
    fps_timer = _time.time()
    fps_samples = []
    last_model_path = None
    last_detector = None

    try:
        while not _server_cam_stop.is_set():
            ret, frame = cap.read()
            if not ret:
                _time.sleep(0.01)
                continue

            now = _time.time()
            frame_dt = now - fps_timer
            fps_timer = now
            fps_samples.append(frame_dt)
            if len(fps_samples) > 15:
                fps_samples.pop(0)
            loop_fps = len(fps_samples) / sum(fps_samples) if sum(fps_samples) > 0 else 0

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame_rgb.shape[:2]

            num_detections = 0
            infer_ms = 0.0

            if model_path:
                try:
                    if last_model_path != model_path:
                        torch.cuda.empty_cache()
                        last_detector = _get_or_create_detector(model_path, conf, iou)
                        last_model_path = model_path

                    detector = last_detector
                    if detector is not None:
                        detector.conf = float(conf)
                        detector.iou = float(iou)
                        t0 = _time.time()
                        result = detector.detect(frame_rgb)
                        infer_ms = (_time.time() - t0) * 1000
                        num_detections = len(result.detections)
                        frame_rgb = draw_detections(frame_rgb, result.detections)
                except Exception as exc:
                    logger.warning("服务器摄像头推理失败: %s", exc)

            frame_count += 1

            info_frame = draw_info_panel(
                frame_rgb,
                fps=loop_fps,
                infer_ms=infer_ms,
                frame_index=frame_count,
                num_detections=num_detections,
                resolution=(w, h),
            )
            if num_detections == 0:
                cv2.putText(
                    info_frame, "No detections - adjust Conf or model",
                    (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2,
                )
            yield info_frame

    except GeneratorExit:
        pass
    finally:
        _release_server_camera()


def _select_model(model_path: str) -> str:
    if not model_path:
        return "未选择模型"
    return f"当前模型: {Path(model_path).name}"


def _chat(
    message: str,
    history: list[dict[str, str]] | None,
    api_key: str,
    api_base: str,
    model_name: str,
    enable_tools: bool,
) -> tuple[list[dict[str, str]], str]:
    text = (message or "").strip()
    history = list(history or [])
    if not text:
        return history, ""
    if not api_key:
        history.append({"role": "assistant", "content": "请先填写 API Key"})
        return history, ""

    if enable_tools:
        try:
            from odp_platform.webui.llm_agent import run_agent
            return run_agent(text, history, api_key, api_base, model_name)
        except ImportError as exc:
            history.append({"role": "assistant", "content": f"Agent 模块未就绪: {exc}"})
            return history, ""
        except Exception as exc:
            history.append({"role": "assistant", "content": f"Agent 执行异常: {exc}"})
            return history, ""

    history.append({"role": "user", "content": text})
    content = _simple_chat(history, api_key, api_base, model_name)
    history.append({"role": "assistant", "content": content})
    return history, ""


def _simple_chat(
    history: list[dict[str, str]],
    api_key: str,
    api_base: str,
    model_name: str,
) -> str:
    import json
    import urllib.error
    import urllib.request

    base = api_base.rstrip("/")
    url = f"{base}/chat/completions"

    messages = [{"role": "system", "content": "你是 DeepSeek 模型，不是 OpenAI。你是一个有用的AI助手。"}]
    for msg in history:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        messages.append({"role": role, "content": content})

    payload = json.dumps({
        "model": model_name,
        "messages": messages,
        "stream": False,
        "max_tokens": 4096,
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return f"API 请求失败 ({exc.code}): {body[:300]}"
    except Exception as exc:
        return f"请求异常: {exc}"


def _clear_chat() -> tuple[list[dict[str, str]], str]:
    return [], ""


def _run_video_detection(
    video_path: str,
    model_path: str,
    conf: float,
    iou: float,
    frame_interval: int,
    max_frames: int,
) -> tuple:
    if not video_path or not model_path:
        return [], "请上传视频并选择模型", [], {}, ""

    try:
        from odp_platform.inference.visualizer import draw_detections
    except ImportError as exc:
        return [], f"依赖未就绪: {exc}", [], {}, ""

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return [], f"无法打开视频: {video_path}", [], {}, ""

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        if max_frames > 0:
            total_frames = min(total_frames, max_frames)

        detector = _get_or_create_detector(model_path, conf, iou)
        if detector is None:
            return [], "模型加载失败", [], {}, ""

        out_temp = Path(tempfile.mkdtemp(prefix="odp_video_"))
        out_video_path = str(out_temp / "output.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_fps = max(1.0, fps / max(frame_interval, 1))
        video_writer = cv2.VideoWriter(out_video_path, fourcc, out_fps, (w, h))

        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        processed_count = 0
        total_objs = 0
        total_infer_ms = 0.0
        sample_frames = []
        class_totals: dict[str, int] = {}
        frame_details = []
        t_start = _time.time()

        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            if max_frames > 0 and frame_idx >= max_frames:
                break

            if frame_idx % frame_interval == 0:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                t0 = _time.time()
                result = detector.detect(frame_rgb)
                elapsed_ms = (_time.time() - t0) * 1000

                rendered = draw_detections(frame_rgb, result.detections)
                processed_count += 1
                total_infer_ms += elapsed_ms

                classes_in_frame = {}
                for d in result.detections:
                    classes_in_frame[d.class_name] = classes_in_frame.get(d.class_name, 0) + 1
                    class_totals[d.class_name] = class_totals.get(d.class_name, 0) + 1
                total_objs += len(result.detections)

                if len(sample_frames) < 500:
                    sample_frames.append(rendered)

                frame_details.append({
                    "帧": frame_idx,
                    "目标数": len(result.detections),
                    "耗时(ms)": round(elapsed_ms, 1),
                    "类别": ", ".join(f"{k}:{v}" for k, v in classes_in_frame.items()) if classes_in_frame else "-",
                })

                rendered_bgr = cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR)
                video_writer.write(rendered_bgr)

            frame_idx += 1

        cap.release()
        video_writer.release()
        elapsed_total = _time.time() - t_start
        avg_ms = total_infer_ms / max(processed_count, 1)

        summary = {
            "总帧数": frame_idx,
            "处理帧数": processed_count,
            "跳帧间隔": frame_interval,
            "总目标数": total_objs,
            "推理总耗时(ms)": round(total_infer_ms, 1),
            "平均每帧(ms)": round(avg_ms, 1),
            "实际耗时(s)": round(elapsed_total, 1),
            "等效FPS": round(processed_count / max(elapsed_total, 0.001), 1),
            "类别分布": class_totals,
        }
        status = (f"完成: {processed_count}/{frame_idx} 帧 | {total_objs} 个目标 | "
                  f"平均 {avg_ms:.0f}ms/帧 | 等效FPS: {processed_count/max(elapsed_total,0.001):.1f}")
        return sample_frames, status, frame_details, summary, out_video_path

    except Exception as exc:
        logger.exception("视频检测失败")
        return [], f"视频检测失败: {exc}", [], {}, ""


def create_single_detection_ui() -> None:
    models = _model_choices()
    detector_state = gr.State(None)

    with gr.Row(elem_classes=["odp-row", "odp-param-row"]):
        model_dd = gr.Dropdown(
            choices=models,
            value=models[0] if models else None,
            label="模型",
            filterable=True,
            interactive=True,
        )
        refresh_btn = gr.Button("刷新")
        conf_slider = gr.Slider(
            0.01, 0.99, 0.25, step=0.01, precision=2, label="Confidence",
            min_width=360, buttons=[], elem_classes=["odp-threshold-slider"],
        )
        iou_slider = gr.Slider(
            0.01, 0.99, 0.45, step=0.01, precision=2, label="IoU",
            min_width=360, buttons=[], elem_classes=["odp-threshold-slider"],
        )

    with gr.Row(elem_classes=["odp-row", "odp-row-two"]):
        with gr.Column(scale=1):
            image_in = gr.Image(type="numpy", label="上传图片", sources=["upload"])
            detect_btn = gr.Button("开始检测", variant="primary", size="lg")
        with gr.Column(scale=1):
            image_out = gr.Image(type="numpy", label="检测结果", container=True)
    status = gr.Textbox(label="状态", value="等待检测", interactive=False, max_lines=1)
    result_json = gr.JSON(label="检测列表")

    refresh_btn.click(fn=_refresh_models, outputs=[model_dd])
    detect_btn.click(
        fn=_run_single_detection,
        inputs=[image_in, model_dd, conf_slider, iou_slider, detector_state],
        outputs=[image_out, result_json, status, detector_state],
    )


def create_folder_detection_ui() -> None:
    models = _model_choices()
    detector_state = gr.State(None)
    folder_results_state = gr.State([])

    with gr.Row(elem_classes=["odp-row", "odp-param-row"]):
        model_dd = gr.Dropdown(
            choices=models,
            value=models[0] if models else None,
            label="模型",
            filterable=True,
            interactive=True,
        )
        refresh_btn = gr.Button("刷新")
        conf_slider = gr.Slider(
            0.01, 0.99, 0.25, step=0.01, precision=2, label="Confidence",
            min_width=360, buttons=[], elem_classes=["odp-threshold-slider"],
        )
        iou_slider = gr.Slider(
            0.01, 0.99, 0.45, step=0.01, precision=2, label="IoU",
            min_width=360, buttons=[], elem_classes=["odp-threshold-slider"],
        )

    with gr.Row(elem_classes=["odp-row", "odp-row-two"]):
        with gr.Column(scale=1):
            folder_in = gr.Textbox(
                label="图片文件夹路径",
                placeholder="eg. F:/datasets/test_images",
                max_lines=1,
            )
            output_dir = gr.Textbox(
                label="输出目录（可选）",
                placeholder="留空默认不保存",
                max_lines=1,
            )
            folder_detect_btn = gr.Button("处理文件夹", variant="primary", size="lg")
            folder_status = gr.Textbox(
                label="状态", value="等待处理", interactive=False, max_lines=1
            )
        with gr.Column(scale=2):
            folder_gallery = gr.Gallery(
                label="文件夹检测结果", columns=3, height=480,
                object_fit="contain",
            )
    with gr.Row(elem_classes=["odp-row", "odp-result-row"]):
        folder_summary = gr.JSON(label="统计摘要", value={})
        folder_detail = gr.Dataframe(
            label="检测明细",
            headers=["文件名", "检测数", "类别", "平均置信度", "耗时(ms)"],
            value=[],
            interactive=False,
        )

    refresh_btn.click(fn=_refresh_models, outputs=[model_dd])
    folder_detect_btn.click(
        fn=_run_folder_detection_wrapped,
        inputs=[folder_in, output_dir, model_dd, conf_slider, iou_slider, detector_state],
        outputs=[folder_gallery, folder_status, detector_state, folder_summary, folder_detail, folder_results_state],
    )


def create_video_detection_ui() -> None:
    models = _model_choices()

    with gr.Row(elem_classes=["odp-row", "odp-param-row"]):
        model_dd = gr.Dropdown(
            choices=models,
            value=models[0] if models else None,
            label="模型",
            filterable=True,
            interactive=True,
        )
        refresh_btn = gr.Button("刷新")
        conf_slider = gr.Slider(
            0.01, 0.99, 0.25, step=0.01, precision=2, label="Confidence",
            min_width=360, buttons=[], elem_classes=["odp-threshold-slider"],
        )
        iou_slider = gr.Slider(
            0.01, 0.99, 0.45, step=0.01, precision=2, label="IoU",
            min_width=360, buttons=[], elem_classes=["odp-threshold-slider"],
        )

    with gr.Row(elem_classes=["odp-row", "odp-row-two"]):
        with gr.Column(scale=1):
            video_in = gr.Video(label="上传视频文件", sources=["upload"])
            with gr.Row():
                frame_interval = gr.Slider(1, 30, 5, step=1, label="跳帧间隔（每N帧检测一次）")
                max_frames = gr.Slider(100, 5000, 1000, step=100, label="最大处理帧数")
            detect_btn = gr.Button("开始检测", variant="primary", size="lg")
            video_status = gr.Textbox(
                label="状态", value="等待视频", interactive=False, max_lines=2
            )
            video_download = gr.File(label="下载结果视频")
        with gr.Column(scale=2):
            video_gallery = gr.Gallery(
                label="检测结果预览", columns=4, height=480,
                object_fit="contain",
            )
    with gr.Row(elem_classes=["odp-row", "odp-row-two"]):
        video_summary = gr.JSON(label="统计摘要", value={})
        video_detail = gr.Dataframe(
            label="逐帧明细",
            headers=["帧", "目标数", "耗时(ms)", "类别"],
            value=[],
            interactive=False,
        )

    refresh_btn.click(fn=_refresh_models, outputs=[model_dd])
    detect_btn.click(
        fn=_run_video_detection,
        inputs=[video_in, model_dd, conf_slider, iou_slider, frame_interval, max_frames],
        outputs=[video_gallery, video_status, video_detail, video_summary, video_download],
    )


def create_live_camera_ui() -> None:
    models = _model_choices()

    with gr.Row(elem_classes=["odp-row", "odp-param-row"]):
        model_dd = gr.Dropdown(
            choices=models,
            value=models[0] if models else None,
            label="模型",
            filterable=True,
            interactive=True,
        )
        refresh_btn = gr.Button("刷新")
        conf_slider = gr.Slider(
            0.01, 0.99, 0.25, step=0.01, precision=2, label="Confidence",
            min_width=360, buttons=[], elem_classes=["odp-threshold-slider"],
        )
        iou_slider = gr.Slider(
            0.01, 0.99, 0.45, step=0.01, precision=2, label="IoU",
            min_width=360, buttons=[], elem_classes=["odp-threshold-slider"],
        )
    with gr.Row(elem_classes=["odp-row", "odp-row-two"]):
        cam_id = gr.Number(label="摄像头 ID", value=0, precision=0, minimum=0, maximum=10, scale=1)
        cam_res = gr.Dropdown(
            label="分辨率",
            choices=["640x480", "1280x720", "1920x1080"],
            value="640x480",
            scale=1,
        )
        server_cam_status = gr.Textbox(
            label="状态", value="未启动", interactive=False, max_lines=1, scale=2
        )
    with gr.Row(elem_classes=["odp-row", "odp-row-three"]):
        start_server_cam_btn = gr.Button("启动", variant="primary")
        stop_server_cam_btn = gr.Button("停止", variant="stop")
        refresh_server_cam_btn = gr.Button("释放并刷新", variant="secondary")
    gpu_info_box = gr.Textbox(
        label="GPU 显存", value="", interactive=False, max_lines=2
    )
    server_cam_out = gr.Image(streaming=True, label="实时检测结果", container=True)

    refresh_btn.click(fn=_refresh_models, outputs=[model_dd])
    start_server_cam_btn.click(
        fn=_run_server_camera,
        inputs=[cam_id, model_dd, conf_slider, iou_slider, cam_res],
        outputs=[server_cam_out],
    )
    stop_server_cam_btn.click(
        fn=lambda: (_release_server_camera(), "已停止")[1],
        outputs=[server_cam_status],
    )
    refresh_server_cam_btn.click(
        fn=lambda: (_release_server_camera(), _gpu_info())[1],
        outputs=[gpu_info_box],
    ).then(
        fn=lambda: "已释放，可重新启动",
        outputs=[server_cam_status],
    )


def create_detection_results_ui() -> None:
    with gr.Row(elem_classes=["odp-row", "odp-row-action"]):
        refresh_btn = gr.Button("刷新")
        status = gr.Textbox(label="状态", value="暂无检测结果", interactive=False, max_lines=1)
    results = gr.Dataframe(
        label="检测历史",
        headers=["ID", "模型", "状态", "目标数", "时间"],
        value=[],
        interactive=False,
        wrap=True,
    )
    details = gr.JSON(label="结果详情", value=[])

    refresh_btn.click(
        fn=lambda: ("暂无检测结果", [], []),
        outputs=[status, results, details],
    )


# ── 实验训练结果 辅助函数 ──────────────────────

@lru_cache(maxsize=1)
def _list_experiments_cached() -> list[str]:
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
        if total_imgs:
            lines.append(f"训练图片数: {total_imgs}")
    return "\n".join(lines)


def _load_experiment_charts(experiment: str) -> tuple:
    if not experiment:
        bl = Image.new("RGB", (400, 300), (240, 240, 240))
        return (bl, bl, bl, bl, bl, bl, bl, "请选择实验")
    rp = _get_pregen_image(experiment, "results.png")
    cm = _get_pregen_image(experiment, "confusion_matrix.png")
    cn = _get_pregen_image(experiment, "confusion_matrix_normalized.png")
    lb = _get_pregen_image(experiment, "labels.jpg")
    pr = _get_pregen_image(experiment, "BoxPR_curve.png")
    f1 = _get_pregen_image(experiment, "BoxF1_curve.png")
    dc = _plot_metric_curves_from_csv(experiment)
    bc = _plot_best_metrics_bar(experiment)
    fb = Image.new("RGB", (400, 300), (240, 240, 240))
    return (
        rp or dc or fb,
        cm or fb,
        cn or fb,
        lb or fb,
        pr or fb,
        f1 or fb,
        bc or fb,
        _summarize_experiment(experiment),
    )


def create_model_selection_ui() -> None:
    models = _model_choices()
    with gr.Row(elem_classes=["odp-row", "odp-row-action"]):
        refresh_btn = gr.Button("刷新")
        model_dd = gr.Dropdown(
            choices=models,
            value=models[0] if models else None,
            label="可用模型",
            filterable=True,
            interactive=True,
            scale=3,
        )
    with gr.Row(elem_classes=["odp-row", "odp-row-action"]):
        model_upload = gr.File(
            label="上传 .pt 模型文件",
            file_types=[".pt"],
            file_count="single",
            scale=1,
        )
        model_path_input = gr.Textbox(
            label="或手动输入模型路径",
            placeholder="eg. F:/models/my_model.pt",
            max_lines=1,
            scale=2,
        )
    with gr.Row(elem_classes=["odp-row", "odp-row-action"]):
        select_btn = gr.Button("设为当前模型", variant="primary")
        status = gr.Textbox(
            label="状态",
            value=_select_model(models[0]) if models else "未发现模型",
            interactive=False,
            max_lines=1,
            scale=3,
        )

    def _upload_model(file) -> gr.update:
        if file is None:
            return gr.update()
        import shutil
        from odp_platform.common.paths import CHECKPOINTS_DIR
        src_name = getattr(file, 'orig_name', None) or Path(file.name).name
        dst = CHECKPOINTS_DIR / src_name
        shutil.copy2(file.name, dst)
        new_models = _model_choices()
        return gr.update(choices=new_models, value=str(dst))

    def _use_custom_path(path: str) -> gr.update:
        if not path or not Path(path).exists():
            return gr.update()
        new_models = _model_choices()
        if path not in new_models:
            new_models.insert(0, path)
        return gr.update(choices=new_models, value=path)

    refresh_btn.click(fn=_refresh_models, outputs=[model_dd])
    model_upload.upload(fn=_upload_model, inputs=[model_upload], outputs=[model_dd])
    model_path_input.submit(fn=_use_custom_path, inputs=[model_path_input], outputs=[model_dd])
    select_btn.click(fn=_select_model, inputs=[model_dd], outputs=[status])

    # ── 实验训练结果（折叠） ─────────────────────────
    gr.Markdown("---")
    with gr.Accordion("📊 实验训练结果", open=False):
        _list_experiments_cached.cache_clear()
        exps = _list_experiments_cached()
        exp_initial = exps[0] if exps else None
        blank = Image.new("RGB", (400, 300), (240, 240, 240))

        with gr.Row(elem_classes=["odp-row", "odp-row-two"]):
            exp_dd = gr.Dropdown(
                label="选择实验", choices=exps, value=exp_initial,
                filterable=True, interactive=True, scale=3,
            )
            exp_refresh_btn = gr.Button("刷新实验", scale=1)
        with gr.Row():
            exp_summary = gr.Textbox(label="实验摘要", lines=6, interactive=False)

        with gr.Tabs():
            with gr.TabItem("训练曲线"):
                with gr.Row():
                    exp_results = gr.Image(value=blank, label="Loss + 验证指标", container=True, height=320)
                with gr.Row():
                    exp_bar = gr.Image(value=blank, label="最佳指标柱状图", container=True, height=260)
            with gr.TabItem("评估矩阵"):
                with gr.Row():
                    exp_confusion = gr.Image(value=blank, label="混淆矩阵", container=True, height=280)
                with gr.Row():
                    exp_confusion_norm = gr.Image(value=blank, label="归一化混淆矩阵", container=True, height=280)
                with gr.Row():
                    exp_pr = gr.Image(value=blank, label="PR 曲线", container=True, height=260)
                with gr.Row():
                    exp_f1 = gr.Image(value=blank, label="F1 曲线", container=True, height=260)
            with gr.TabItem("类别分布"):
                with gr.Row():
                    exp_labels = gr.Image(value=blank, label="类别分布", container=True, height=350)

        exp_refresh_btn.click(
            fn=lambda: gr.update(choices=_list_experiments_cached()),
            outputs=[exp_dd],
        )
        exp_dd.change(
            fn=_load_experiment_charts,
            inputs=[exp_dd],
            outputs=[exp_results, exp_confusion, exp_confusion_norm,
                     exp_labels, exp_pr, exp_f1, exp_bar, exp_summary],
        )


def create_llm_chat_ui() -> None:
    with gr.Row(elem_classes=["odp-row"]):
        with gr.Column(scale=1):
            api_key = gr.Textbox(
                label="API Key",
                type="password",
                placeholder="sk-... 必填",
                scale=1,
            )
        with gr.Column(scale=1):
            api_base = gr.Textbox(
                label="API Base URL",
                value="https://api.deepseek.com",
                scale=1,
            )
        with gr.Column(scale=1):
            model_name = gr.Textbox(
                label="模型名称",
                value="deepseek-v4-flash",
                placeholder="如 deepseek-v4-flash、deepseek-v4-pro、gpt-4o",
                scale=1,
            )
    with gr.Row(elem_classes=["odp-agent-toggle"]):
        enable_tools = gr.Checkbox(
            label="✅ Agent 工具已启用（模型/实验/推理/GPU）",
            value=True,
            scale=1,
        )
    chatbot = gr.Chatbot(label="对话", height=400)
    message = gr.Textbox(label="输入", placeholder="输入问题，按Enter发送", max_lines=3)
    with gr.Row(elem_classes=["odp-row", "odp-row-two"]):
        send_btn = gr.Button("发送", variant="primary", scale=1)
        clear_btn = gr.Button("清空对话", scale=1)

    inputs = [message, chatbot, api_key, api_base, model_name, enable_tools]
    outputs = [chatbot, message]

    send_btn.click(fn=_chat, inputs=inputs, outputs=outputs)
    message.submit(fn=_chat, inputs=inputs, outputs=outputs)
    clear_btn.click(fn=_clear_chat, outputs=[chatbot, message])


def create_user_info_ui() -> None:
    with gr.Row(elem_classes=["odp-row", "odp-row-three"]):
        gr.Textbox(label="用户名", value="guest", interactive=False, max_lines=1)
        gr.Textbox(label="角色", value="user", interactive=False, max_lines=1)
        gr.Textbox(label="状态", value="未登录", interactive=False, max_lines=1)
    gr.JSON(
        label="检测概览",
        value={
            "检测任务": 0,
            "已完成": 0,
            "最近模型": "",
        },
    )
