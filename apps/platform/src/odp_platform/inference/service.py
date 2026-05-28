from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import cv2

from odp_platform.inference.engine import Detector
from odp_platform.inference.visualizer import draw_detections, draw_info_panel
from odp_platform.inference.sources import ImageSource, VideoWriter
from odp_platform.inference.frame_source import (
    SourceType,
    CameraConfig,
    create_frame_source,
    create_threaded_source,
)
from odp_platform.inference.pipeline_config import PipelineConfig, load_pipeline_config
from odp_platform.common.paths import RUNS_DIR, LOGGING_DIR
from odp_platform.common.logging_utils import get_logger

logger = logging.getLogger(__name__)


# ============================================================================
# Data structures
# ============================================================================

@dataclass
class InferStats:
    frames: int = 0
    detections: int = 0
    per_class: dict[str, int] = field(default_factory=dict)
    infer_time_sec: float = 0.0
    capture_fps: float = 0.0
    infer_fps: float = 0.0
    render_fps: float = 0.0
    loop_fps: float = 0.0

    @property
    def avg_fps(self) -> float:
        return self.frames / self.infer_time_sec if self.infer_time_sec > 0 else 0.0

    @property
    def avg_latency_ms(self) -> float:
        return (self.infer_time_sec / self.frames * 1000.0) if self.frames else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "frames": self.frames,
            "detections": self.detections,
            "per_class": dict(self.per_class),
            "infer_time_sec": round(self.infer_time_sec, 4),
            "avg_fps": round(self.avg_fps, 2),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "fps": {
                "capture": self.capture_fps,
                "infer": self.infer_fps,
                "render": self.render_fps,
                "loop": self.loop_fps,
            },
        }


@dataclass
class InferResult:
    success: bool
    output_dir: Path
    stats: dict[str, Any]
    infer_time: float = 0.0
    saved: bool = True
    error: str = ""
    audit_path: Optional[Path] = None
    log_path: Optional[str] = None


# ============================================================================
# Service
# ============================================================================

class InferService:
    def predict(
        self,
        yaml_path: Optional[str] = None,
        pipeline_yaml: Optional[str] = None,
        cli_args: Optional[dict[str, Any]] = None,
        *,
        beautify: bool = True,
        rename_log: bool = True,
        threaded: bool = False,
        warmup_frames: int = 0,
        window_name: str = "odp-infer",
        show_info: bool = True,
    ) -> InferResult:
        start = datetime.now()
        log = logging.getLogger("odp_platform.inference.service")
        output_dir: Optional[Path] = None

        try:
            args = cli_args or {}
            source_str = args.get("source", "0")
            model_path = args.get("model", "")
            conf = args.get("conf", 0.25)
            iou = args.get("iou", 0.45)
            imgsz = args.get("imgsz", None)
            max_det = args.get("max_det", None)
            classes = args.get("classes", None)
            want_show = args.get("show", False)
            want_save = args.get("save", True)
            experiment_name = args.get("experiment_name", None)
            device = args.get("device", None)

            pipe_cfg = load_pipeline_config(pipeline_yaml)

            log.info(f"loading model: {model_path}")
            predict_kwargs = {"conf": conf, "iou": iou}
            if imgsz is not None:
                predict_kwargs["imgsz"] = imgsz
            if max_det is not None:
                predict_kwargs["max_det"] = max_det
            if classes is not None:
                predict_kwargs["classes"] = classes
            if device is not None:
                predict_kwargs["device"] = device

            detector = Detector(model_path, conf=conf, iou=iou)
            detector.warmup()

            if experiment_name is None:
                timestamp = start.strftime("%Y%m%d-%H%M%S")
                model_stem = Path(model_path).stem
                experiment_name = f"infer_{timestamp}_{model_stem}"

            output_dir = RUNS_DIR / "infer" / experiment_name
            output_dir.mkdir(parents=True, exist_ok=True)

            if rename_log:
                log_path = LOGGING_DIR / f"infer_{experiment_name}.log"
            else:
                log_path = None

            is_camera = (
                str(source_str).isdigit()
                or str(source_str).startswith("rtsp://")
                or str(source_str).startswith("rtmp://")
            )

            if is_camera:
                result = self._run_camera_pipeline(
                    source_str=source_str,
                    detector=detector,
                    pipe_cfg=pipe_cfg,
                    output_dir=output_dir,
                    want_show=want_show,
                    want_save=want_save,
                    show_info=show_info,
                    warmup_frames=warmup_frames,
                    window_name=window_name,
                    threaded=threaded,
                )
            else:
                result = self._run_file_pipeline(
                    source_str=source_str,
                    detector=detector,
                    pipe_cfg=pipe_cfg,
                    output_dir=output_dir,
                    want_show=want_show,
                    want_save=want_save,
                    show_info=show_info,
                    threaded=threaded,
                )

            infer_time = (datetime.now() - start).total_seconds()

            if result.frames > 0:
                log.info("=" * 60)
                log.info(f"推理总耗时: {infer_time:.2f} 秒")
                log.info(f"输出目录:   {output_dir}")
                if want_save:
                    log.info("结果已保存到上面的目录.")
                if log_path:
                    log.info(f"本次日志:   {log_path}")
                log.info("=" * 60)

            return InferResult(
                success=True,
                output_dir=output_dir,
                stats=result.to_dict(),
                infer_time=infer_time,
                saved=want_save,
                log_path=str(log_path) if log_path else None,
            )

        except Exception as e:
            log.error(f"推理失败: {e}", exc_info=True)
            infer_time = (datetime.now() - start).total_seconds()
            return InferResult(
                success=False,
                output_dir=output_dir or Path("unknown"),
                stats={},
                infer_time=infer_time,
                error=str(e),
            )

    def _run_file_pipeline(
        self,
        source_str: str,
        detector: Detector,
        pipe_cfg: PipelineConfig,
        output_dir: Path,
        want_show: bool,
        want_save: bool,
        show_info: bool,
        threaded: bool,
    ) -> InferStats:
        stats = InferStats()
        source = ImageSource(source_str)
        video_writer = None

        cap = source.cap
        is_stream = cap is not None

        if is_stream and want_save:
            fps = source.get_fps() or 30.0
            video_writer = VideoWriter(str(output_dir / "output.mp4"), fps=fps)

        for image in source:
            result = detector.detect(image)
            stats.frames += 1
            stats.detections += len(result.detections)
            stats.infer_time_sec += result.inference_ms / 1000.0

            for det in result.detections:
                stats.per_class[det.class_name] = stats.per_class.get(det.class_name, 0) + 1

            annotated = draw_detections(image, result.detections)

            if show_info:
                annotated = draw_info_panel(
                    annotated,
                    fps=1.0 / (result.inference_ms / 1000.0) if result.inference_ms > 0 else 0,
                    infer_ms=result.inference_ms,
                    frame_index=stats.frames - 1,
                    num_detections=len(result.detections),
                )

            if video_writer:
                video_writer.write(annotated)
            elif want_save:
                cv2.imwrite(str(output_dir / f"frame_{stats.frames-1:06d}.jpg"),
                            cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

            if want_show:
                cv2.imshow("odp-infer", annotated)
                if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                    break

        source.release()
        if video_writer:
            video_writer.release()
        if want_show:
            cv2.destroyAllWindows()

        return stats

    def _run_camera_pipeline(
        self,
        source_str: str,
        detector: Detector,
        pipe_cfg: PipelineConfig,
        output_dir: Path,
        want_show: bool,
        want_save: bool,
        show_info: bool,
        warmup_frames: int,
        window_name: str,
        threaded: bool,
    ) -> InferStats:
        stats = InferStats()
        camera_config = pipe_cfg.build_camera_config()

        if threaded:
            source = create_threaded_source(
                source=source_str,
                maxlen=2,
                camera_config=camera_config,
                buffer_strategy="latest",
            )
        else:
            source = create_frame_source(
                int(source_str) if source_str.isdigit() else source_str,
                camera_config=camera_config,
            )

        source.open()
        video_writer = None
        if want_save:
            video_writer = VideoWriter(str(output_dir / "output.mp4"), fps=camera_config.fps if camera_config else 30)

        frame_count = 0
        last_time = time.time()

        try:
            while True:
                frame_obj = source.read()
                if frame_obj is None:
                    continue

                image = frame_obj.image
                frame_count += 1

                if frame_count <= warmup_frames:
                    continue

                t0 = time.time()
                result = detector.detect(image)
                infer_ms = (time.time() - t0) * 1000

                stats.frames += 1
                stats.detections += len(result.detections)
                stats.infer_time_sec += infer_ms / 1000.0

                for det in result.detections:
                    stats.per_class[det.class_name] = stats.per_class.get(det.class_name, 0) + 1

                annotated = draw_detections(image, result.detections)

                if show_info:
                    now = time.time()
                    loop_fps = 1.0 / (now - last_time) if (now - last_time) > 0 else 0
                    last_time = now
                    annotated = draw_info_panel(
                        annotated,
                        fps=loop_fps,
                        infer_ms=infer_ms,
                        frame_index=frame_count,
                        num_detections=len(result.detections),
                    )

                if video_writer:
                    video_writer.write(annotated)

                if want_show:
                    cv2.imshow(window_name, annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord('q'), 27):
                        break

        except KeyboardInterrupt:
            pass
        finally:
            source.close()
            if video_writer:
                video_writer.release()
            cv2.destroyAllWindows()

        return stats


def infer_yolo(
    yaml_path: Optional[str] = None,
    pipeline_yaml: Optional[str] = None,
    cli_args: Optional[dict[str, Any]] = None,
    *,
    beautify: bool = True,
    rename_log: bool = True,
    threaded: bool = False,
    warmup_frames: int = 0,
    window_name: str = "odp-infer",
    show_info: bool = True,
) -> InferResult:
    service = InferService()
    return service.predict(
        yaml_path=yaml_path,
        pipeline_yaml=pipeline_yaml,
        cli_args=cli_args,
        beautify=beautify,
        rename_log=rename_log,
        threaded=threaded,
        warmup_frames=warmup_frames,
        window_name=window_name,
        show_info=show_info,
    )