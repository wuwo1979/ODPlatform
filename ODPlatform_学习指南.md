# ODPlatform - 全流程学习指南

> 作者: MuU | 目标: 从零掌握目标检测全栈开发技术路径  
> 版本: v3.0 — 基于五层架构 + 答辩视角  
> **命令速查**：[ODPlatform_命令速查.md](ODPlatform_命令速查.md)  
> **答辩演练**：[ODPlatform_答辩演练问题集.md](docs/ODPlatform_答辩演练问题集.md)

---

## 第零章：一句话定义

> **"一个让用户从原始标注数据到目标检测结果，全流程可视化的工程化平台。"**
> —— 不出现 YOLO、Gradio、FastAPI 等技术名词，适合答辩开场白

---

## 插曲：30 秒快速入门

想在 30 秒内跑通一个检测任务？

```bash
pip install -e ./apps/platform          # 安装
odp-init                                 # 初始化目录
odp-webui                                # 启动 WebUI (http://127.0.0.1:7860)
```

打开浏览器 → "单图检测" Tab → 选模型 → 上传图片 → 点检测。完成。

> 详细命令参考 [ODPlatform_命令速查.md](ODPlatform_命令速查.md)

---

## 第一章：五层架构总览

ODPlatform 的核心架构是 **CLI → Service → Core → Config → Common** 五层。

```
                    ┌──────────────────────────────────────────────┐
                    │              WebUI (Gradio)                   │
                    │         app.py + 6 用户 Tab + 5 管理 Tab       │
                    │            (不属于五层，在顶层调用)              │
                    └──────────────────────┬───────────────────────┘
                                           │
                    ┌──────────────────────▼───────────────────────┐
    第1层 CLI       │  cli/ 命令行入口                              │
                    │  ├── train.py     解析 --dataset/--model      │
                    │  ├── infer.py     解析 --source/--conf        │
                    │  ├── transform_data.py  解析 --format         │
                    │  ├── validate_data.py   解析 --dataset        │
                    │  ├── val.py       解析 --model/--dataset      │
                    │  ├── config_cli.py    配置生成/溯源/快照       │
                    │  ├── init_project.py   创建运行时目录          │
                    │  └── reset_project.py  清理运行时数据          │
                    └──────────────────────┬───────────────────────┘
                                           │ 调用
                    ┌──────────────────────▼───────────────────────┐
    第2层 Service   │  业务流程编排层                               │
                    │                                               │
                    │  data_pipeline/orchestrator.py                │
                    │    DatasetPipeline: raw → YOLO → split → YAML │
                    │                                               │
                    │  data_validation/service.py                   │
                    │    validate_dataset(): 执行全部 check → 报告    │
                    │                                               │
                    │  training/experiment.py                       │
                    │    run_experiment(): YOLO.train() + hooks      │
                    │                                               │
                    │  inference/service.py                         │
                    │    InferService.predict(): 帧源→检测→输出      │
                    │                                               │
                    │  evaluation/service.py                        │
                    │    ValService.validate(): YOLO.val()           │
                    │                                               │
                    │  run_config/service.py                        │
                    │    build_config(): 三源合并→校验→溯源          │
                    └──────────────────────┬───────────────────────┘
                                           │ 编排
                    ┌──────────────────────▼───────────────────────┐
    第3层 Core      │  业务逻辑实现层                               │
                    │                                               │
                    │  data_pipeline/core/                          │
                    │    coco.py / pascal_voc.py / yolo.py          │
                    │    ← @register 注册表，一个文件一个格式         │
                    │                                               │
                    │  data_validation/checks/                      │
                    │    yaml_schema.py / pair_existence.py         │
                    │    label_format.py / split_uniqueness.py      │
                    │    ← @check 注册表，一个文件一项检查            │
                    │                                               │
                    │  training/  ← experiment 兼 Service+Core      │
                    │    tracker.py / recipe.py / callbacks.py      │
                    │                                               │
                    │  inference/                                   │
                    │    engine.py (Detector) / visualizer.py       │
                    │    frame_source/ (Camera/Video/Image Source)  │
                    │                                               │
                    │  evaluation/ ← 直接封装 YOLO.val()            │
                    └──────────────────────┬───────────────────────┘
                                           │ 读取
                    ┌──────────────────────▼───────────────────────┐
    第4层 Config    │  run_config/ 配置管理子系统                    │
                    │                                               │
                    │  registry.py  ← @config_generator 注册        │
                    │  loader.py    ← YAML 加载 + CLI 参数解析      │
                    │  merger.py    ← 默认值→YAML→CLI 三源合并      │
                    │  validator.py ← 字段类型/范围/必填校验        │
                    │  schema.py    ← ConfigBundle/TraceRecord      │
                    │  service.py   ← build_config / restore        │
                    │  fields/      ← 各任务字段定义                │
                    │    train.py / val.py / predict.py             │
                    └──────────────────────┬───────────────────────┘
                                           │ 使用
                    ┌──────────────────────▼───────────────────────┐
    第5层 Common    │  common/ 基础设施层                           │
                    │                                               │
                    │  paths.py         ← .odp-workspace marker     │
                    │  logging_utils.py ← 根 Logger 装配            │
                    │  constants.py     ← AnnotationFormat/Task     │
                    │  performance_utils.py ← @time_it / timer      │
                    │  string_utils.py  ← format_table              │
                    │  system_utils.py  ← log_device_info           │
                    └──────────────────────────────────────────────┘
```

### 各层职责

| 层 | 职责 | 核心原则 |
|----|------|---------|
| **CLI** | 参数解析、流程控制、调用 Service | 不做业务逻辑，只做参数→函数映射 |
| **Service** | 业务流程编排、协调子模块、错误处理 | 一个函数完成一个完整用例 |
| **Core** | 业务逻辑具体实现 | 单一职责，每个文件只做一件事 |
| **Config** | 配置管理（三源合并、溯源、快照） | 字段为中心，独立于业务链 |
| **Common** | 基础设施 | 不依赖任何业务模块 |

### 两个外部系统

| 系统 | 位置 | 与五层的关系 |
|------|------|-------------|
| **Gradio WebUI** | `webui/` | 在五层之上，通过 Service/Core 层调用推理和训练 |
| **FastAPI 后端** | `apps/web-backend/` | 独立服务，可选。训练 hooks 自动同步数据到后端 |

---

## 第二章：各层详解

### 2.1 CLI 层（第 1 层）

**路径**：`apps/platform/src/odp_platform/cli/`

10 个命令，全部注册在 `pyproject.toml` 的 `[project.scripts]`：

| 命令 | 文件 | 做了什么 |
|------|------|---------|
| `odp-init` | `init_project.py` | 调用 `paths.get_dirs_to_initialize()` 创建全部运行时目录 |
| `odp-reset` | `reset_project.py` | 清理运行时数据（日志/运行产物） |
| `odp-transform` | `transform_data.py` | argc: --dataset --format --task → 调用 `DatasetPipeline` |
| `odp-validate` | `validate_data.py` | argc: --dataset / --yaml → 调用 `validate_dataset()` |
| `odp-config` | `config_cli.py` | argc: generate / trace / snapshot → 调用 `run_config.service` |
| `odp-train` | `train.py` | argc: --dataset --model --epochs → 调用 `run_experiment()` |
| `odp-val` | `val.py` | argc: --model --dataset → 调用 `ValService.validate()` |
| `odp-infer` | `infer.py` | argc: --source --model --conf → 调用 `InferService.predict()` |
| `odp-webui` | `webui/app.py:main()` | 启动 Gradio 服务 `gr.Blocks().launch()` |
| `odp-backend` | `backend.py` | 启动 FastAPI `subprocess(["uvicorn", "main:app"])` |

**CLI 层设计模式**：每个命令文件 = 一个 `_build_parser()` + 一个 `main()` 函数。

> 源码参考：[cli/train.py](apps/platform/src/odp_platform/cli/train.py) — 典型的 CLI → Service 调用模式

### 2.2 Service 层（第 2 层）

> 源码参考：[inference/service.py](apps/platform/src/odp_platform/inference/service.py) — InferService 编排 | [training/experiment.py](apps/platform/src/odp_platform/training/experiment.py) — 训练编排

服务层编排完整业务流程，一个函数完成一个完整用例。

#### data_pipeline — 数据转换服务

```python
# orchestrator.py — 数据集端到端编排
pipe = DatasetPipeline(
    dataset_name="rsod",
    annotation_format="pascal_voc",  # → 注册表查找 converter
)
pipe.pipeline()  # raw → YOLO → split → yaml
```

调用链：

```
odp-transform --dataset rsod --format pascal_voc
  → DatasetPipeline.pipeline()
      → _check_raw()              验证 data/raw/rsod/{images,annotations}/
      → service.converter_data_to_yolo()  Core 层查注册表
          → core/pascal_voc.py    逐 XML 解析，写 YOLO txt
      → split_pairs()             随机划分 train/val/test
      → materialize()             落盘到 data/{train,val,test}/
      → write_dataset_yaml()      写 configs/datasets/rsod.yaml
```

#### data_validation — 数据校验服务

```python
# service.py — 端到端质检
report = validate_dataset(
    yaml_path=Path("configs/datasets/rsod.yaml"),
    task_type="detect",
)
# report.exit_code: 0=通过, 1=警告, 2=错误
```

调用链：

```
odp-validate --dataset rsod
  → validate_dataset(yaml_path)
      → build_snapshot()          扫描目录，构建不可变快照
      → run_all_checks()          执行全部 @check 注册的检查项
          → checks/yaml_schema.py             验证 YAML 字段
          → checks/pair_existence.py          验证图片-标签配对
          → checks/label_format.py            验证每行标签格式
          → checks/split_uniqueness.py        验证无跨集重复
      → ValidationReport          生成报告 + 退出码
```

#### training — 训练服务

```python
# experiment.py — 训练入口
config = ExperimentConfig(
    name="rsod_baseline",
    dataset="rsod",
    model="yolo11n.pt",
    epochs=100,
)
result = run_experiment(config)
```

调用链：

```
odp-train --dataset rsod --model yolo11n.pt
  → build_config(task="train")            Config 层：三源合并
  → run_experiment(config)                Service 层：编排训练
      → TrainingHooks.on_train_start()    通知后端（可选）
      → YOLO.train(**train_kwargs)        Ultralytics 实际训练
      → TrainingHooks.on_epoch_end()      逐轮同步到后端（可选）
      → _sync_to_backend()                最终结果同步
      → 复制 best.pt → checkpoints/       模型归档
```

#### inference — 推理服务

```python
# service.py — 推理编排
service = InferService()
result = service.predict(
    cli_args={"source": "test.jpg", "model": "best.pt"}
)
```

调用链：

```
odp-infer --source test.jpg --model best.pt
  → InferService.predict(cli_args)
      → create_frame_source(source_str)   帧源识别
          ├── str.isdigit() → CameraSource
          ├── .mp4/.avi    → VideoSource
          ├── .jpg/.png    → ImageSource
          └── 目录路径      → ImageFolderSource
      → Detector(model_path)              模型加载
      → 循环逐帧:
          frame = source.read()
          result = detector.detect(frame)
          rendered = draw_detections(frame, result.detections)
          → 输出: 图片/视频/终端显示
      → InferResult(stats, output_dir)
```

### 2.3 Core 层（第 3 层）

> 源码参考：[inference/engine.py](apps/platform/src/odp_platform/inference/engine.py) — Detector | [data_pipeline/core/](apps/platform/src/odp_platform/data_pipeline/core/) — 格式转换器

核心层实现单一业务逻辑，每个文件只做一件事。

#### 注册表模式 — 数据转换器

`data_pipeline/core/` 下的转换器通过 `@register` 装饰器自动注册：

```python
# core/pascal_voc.py
@register(AnnotationFormat.PASCAL_VOC, supported_tasks=(Task.DETECT,))
def convert_pascal_voc(input_dir, output_labels_dir, options):
    """Pascal VOC XML → YOLO txt。一个函数完成一个格式的转换。"""
    for xml_file in input_dir.glob("*.xml"):
        tree = ET.parse(xml_file)
        # 提取 bbox → 归一化 → 写 .txt
    return class_names
```

新增格式 = 新建 `core/new_format.py` + 加 `@register("format_name")`，不改任何旧代码。

#### 注册表模式 — 数据校验器

`data_validation/checks/` 下的检查项通过 `@check` 装饰器注册：

```python
# checks/yaml_schema.py
@check("yaml_schema")
def validate_yaml_schema(ctx: CheckContext) -> CheckResult:
    """验证 YAML 的 nc/path/names/train/val/test 字段完整性。"""
    ...
```

新增检查 = 新建 `checks/check_name.py` + 加 `@check("name")`，不改旧代码。

#### Detector — 推理引擎

```python
# inference/engine.py
class Detector:
    def __init__(self, model_path, conf=0.25, iou=0.45):
        self._model = YOLO(model_path)

    def detect(self, image: np.ndarray) -> InferenceResult:
        # YOLO 前向传播 → NMS → Detection[]
        return InferenceResult(detections=[...], inference_ms=12.3, ...)

    def warmup(self):
        """GPU JIT 预热：纯黑图跑一次推理，消除首次 CUDA 编译延迟"""
        if torch.cuda.is_available():
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            self.detect(dummy)
```

#### frame_source — 帧源抽象

```python
# frame_source/core/base.py
class FrameSource(ABC):
    @abstractmethod
    def open(self) -> bool
    @abstractmethod
    def read(self) -> Optional[Frame]
    @abstractmethod
    def close(self) -> None
    def seek(self, frame=None, time_sec=None) -> bool  # 可选支持

# frame_source/factory.py — 注册表模式
@register_source("camera")(CameraSource)
@register_source("video", extensions=["mp4","avi"])(VideoSource)
@register_source("image", extensions=["jpg","png"])(ImageSource)
@register_source("folder")(ImageFolderSource)
```

新增输入源 = 新建 Source 类 + `@register_source("name")`，factory 不用改。

### 2.4 Config 层（第 4 层）

> 源码参考：[run_config/](apps/platform/src/odp_platform/run_config/) — 配置管理子系统，核心文件 merger.py（三源合并）、schema.py（TraceRecord）

配置管理子系统 `run_config/` 独立于业务链：

```
run_config/
├── registry.py        @config_generator 装饰器 + list_fields()
├── service.py         build_config / restore_from_snapshot / save_snapshot
├── loader.py          load_yaml + parse_cli_args + resolve_yaml_path
├── merger.py          三源合并（默认值→YAML→CLI）+ TraceReport
├── schema.py          ConfigBundle / ConfigSnapshot / TraceRecord
├── validator.py       字段类型/范围/必填校验
├── template.py        配置模板生成
└── fields/            各任务字段定义
    ├── train.py       训练字段（epochs, batch, lr0, optimizer, amp...）
    ├── val.py         评估字段（conf, iou, half, max_det...）
    └── predict.py     推理字段（source, save_txt, save_conf...）
```

**三源合并**：

```python
# 配置来源优先级（高→低）：
# 1. CLI 参数  --epochs 200
# 2. YAML 文件 train.yaml  → epochs: 100
# 3. 代码默认值             → ConfigField(default=50)
#
# 最终值: epochs = 200（CLI 胜出）

bundle = build_config(
    task="train",
    yaml_path=Path("train.yaml"),
    cli_args={"epochs": 200},
)
bundle.config   # 合并后的完整配置
bundle.trace    # 每个字段的来源链（从哪来→最终值）
```

**配置快照**：

```python
# 每次训练自动保存
snapshot_path = exp_dir / "config_snapshot.json"
snapshot_path.write_text(config.to_json())

# 可恢复历史配置
restored = restore_from_snapshot(loaded_snapshot)
```

### 2.5 Common 层（第 5 层）

> 源码参考：[common/paths.py](apps/platform/src/odp_platform/common/paths.py) — marker 路径探测 | [common/logging_utils.py](apps/platform/src/odp_platform/common/logging_utils.py) — 日志装配

基础设施层，不依赖任何业务模块。

| 文件 | 功能 | 关键实现 |
|------|------|---------|
| `paths.py` | 路径探测 + 路径常量 | `.odp-workspace` marker 向上遍历 + `@cache` |
| `logging_utils.py` | 日志装配 | 幂等保护 + propagate=False + colorlog 彩色输出 |
| `constants.py` | 共享枚举 | `AnnotationFormat`、`Task`、`RunTask` class |
| `performance_utils.py` | 性能工具 | `@time_it`、`timer()`、`MetricTracker` |
| `string_utils.py` | 格式化工具 | `format_table_row()` |
| `system_utils.py` | 系统信息 | `log_device_info()` |

**Marker 路径探测**：

```python
# common/paths.py
@cache
def _find_workspace_root(start: Path) -> Path:
    """从当前文件向上找 .odp-workspace 标记文件。"""
    current = start.resolve()
    for parent in [current, *current.parents]:
        if (parent / WORKSPACE_MARKER).exists():
            return parent
    raise FileNotFoundError(f"未找到 {WORKSPACE_MARKER}")

ROOT_DIR = _find_workspace_root(Path(__file__))
DATA_DIR = ROOT_DIR / "data"
CHECKPOINTS_DIR = DATA_DIR / "models" / "checkpoints"
```

**日志系统**：

```python
# 根 Logger（入口调一次）
get_logger(base_path=LOGGING_DIR, log_type="train")
# → 自动创建 FileHandler → logging/train/*.log
# → 自动创建 StreamHandler → 彩色控制台

# 业务模块只需一行
logger = logging.getLogger(__name__)
logger.info("训练开始")  # 自动冒泡到根 logger
```

---

## 第三章：端到端全流程

### 3.1 训练全链路（带层标注）

```
用户输入：
  odp-train --dataset rsod --model yolo11n.pt --epochs 100
    │
    ▼
┌── CLI 层 ─────────────────────────────────────────────────┐
│  cli/train.py                                              │
│  ├── argparse 解析：dataset=rsod, model=yolo11n.pt,        │
│  │               epochs=100, task=detect, batch=16...      │
│  └── build_config(task="train", cli_args=overrides)        │
└────────────────────────────────────────────────────────────┘
    │
    ▼
┌── Config 层 ───────────────────────────────────────────────┐
│  run_config/service.py                                      │
│  ├── loader.load_yaml()       ← 读取 YAML 配置             │
│  ├── merger.merge()           ← 三源合并 + 溯源            │
│  ├── validator.validate()     ← 校验字段                   │
│  └── → ConfigBundle(config, trace, errors)                 │
└────────────────────────────────────────────────────────────┘
    │
    ▼
┌── Service 层 ──────────────────────────────────────────────┐
│  training/experiment.py                                     │
│  ├── 创建 runs/experiments/rsod_baseline/                   │
│  ├── 保存 config_snapshot.json                              │
│  ├── TrainingHooks.on_train_start()                         │
│  │   └── POST /api/experiments（静默降级）                  │
│  ├── YOLO.train(**train_kwargs)                             │
│  │   └── hooks.on_epoch_end() → POST epochs                 │
│  ├── 复制 best.pt → checkpoints/best_rsod_baseline.pt       │
│  └── → ExperimentResult(map50, map50_95, precision, recall) │
└────────────────────────────────────────────────────────────┘
    │
    ▼
  输出：
  runs/experiments/rsod_baseline/
    ├── config_snapshot.json   配置快照
    ├── results.csv            逐轮指标
    ├── weights/best.pt        最佳权重
    ├── results.png            训练曲线
    ├── BoxPR_curve.png        PR 曲线
    ├── confusion_matrix.png   混淆矩阵
    └── labels.jpg             类别分布
```

### 3.2 推理全链路（带层标注）

```
用户输入：
  odp-infer --source test.jpg --model best.pt --conf 0.25
    │
    ▼
┌── CLI 层 ─────────────────────────────────────────────────┐
│  cli/infer.py                                              │
│  ├── argparse 解析：source, model, conf, iou, ...          │
│  └── InferService.predict(cli_args={...})                  │
└────────────────────────────────────────────────────────────┘
    │
    ▼
┌── Service 层 ──────────────────────────────────────────────┐
│  inference/service.py                                       │
│  ├── create_frame_source("test.jpg")                       │
│  │   → factory 识别 .jpg 后缀 → ImageSource               │
│  ├── Detector("best.pt")                                   │
│  │   → YOLO 加载 + warmup()                               │
│  ├── 逐帧循环:                                             │
│  │   ├── frame = source.read()                             │
│  │   ├── result = detector.detect(frame.image)             │
│  │   ├── rendered = draw_detections(frame, detections)     │
│  │   └── 保存/显示                                         │
│  └── → InferResult(stats, output_dir)                      │
│                                                            │
│  摄像头模式（--source 0 --threaded）:                       │
│  ├── create_threaded_source("0")                           │
│  │   → CameraSource + ThreadedSource（采集消费分离）       │
│  ├── 同上逐帧循环                                          │
│  └── 生成器 yield rendered → 实时流                        │
└────────────────────────────────────────────────────────────┘
    │
    ▼
┌── Core 层 ────────────────────────────────────────────────┐
│  inference/engine.py                                       │
│  Detector.detect():                                        │
│  ├── YOLO(image, conf=0.25, iou=0.45, verbose=False)      │
│  ├── 解析 results.boxes → Detection[]                      │
│  │   └── Detection(class_id, class_name, confidence, bbox) │
│  └── → InferenceResult(detections, inference_ms, shape)    │
│                                                            │
│  inference/visualizer.py                                   │
│  draw_detections():                                        │
│  └── cv2.rectangle + cv2.putText → 标注图片                │
└────────────────────────────────────────────────────────────┘
```

### 3.3 WebUI 全链路

```
用户浏览器 (http://127.0.0.1:7860)
    │
    ▼
┌── Gradio 应用 (webui/app.py) ──────────────────────────────┐
│  用户模式 6 Tab + 管理员模式 +5 Tab                         │
│                                                            │
│  [单图检测]: 用户上传图片 → 点击检测                         │
│    → _run_single_detection(image, model_path, conf, iou)   │
│      → _get_or_create_detector()  ← Core 层 Detector       │
│      → detector.detect(image_np)                           │
│      → draw_detections(image_np, detections)               │
│      → 返回 (标注图片, JSON 明细, 状态) 到 Gradio           │
│                                                            │
│  [实时摄像头]: 用户点击启动                                  │
│    → _run_server_camera(camera_id, model_path, ...)        │
│      → cv2.VideoCapture(int(camera_id))                    │
│      → 循环 yield rendered → gr.Image 流式更新             │
│      → _server_cam_stop Event 控制停止                     │
│                                                            │
│  [LLM对话]: 用户输入"有什么模型"                             │
│    → run_agent()  ← llm_agent.py                           │
│      → 关键词匹配"模型" → tool_list_models()               │
│      → _format_with_llm() → LLM 美化回复                    │
│                                                            │
│  [管理员→训练]: 填写参数 → 启动                             │
│    → run_experiment(config)  ← Service 层                  │
│    → 实时日志流 + 训练曲线                                  │
└────────────────────────────────────────────────────────────┘
```

### 3.4 WebUI 性能优化要点

```python
# 1. 模型缓存（避免切换 Tab 重复加载 YOLO）
_detector_cache: dict[str, Detector] = {}
_cache_lock = threading.Lock()

# 2. 模型文件扫描缓存（5 秒 TTL，避免频繁扫描磁盘）
@lru_cache(maxsize=1)
def list_model_files(ttl: int = 5) -> list[str]: ...

# 3. 摄像头单例管理（确保只有一个实例）
_server_cap_ref = [None]
_server_cap_lock = threading.Lock()
```

---

## 第四章：模块依赖与调用规则

```
                    common/  ← 第5层：被所有模块依赖
                        │
        ┌───────────────┼────────────────┐
        │               │                │
   data_pipeline   data_validation   run_config  ← 第4层
        │               │            (独立模块)
        └───────┬───────┘                │
                ▼                        │
           training ◄────────────────────┘  ← 第2+3层
                │
                ▼
           evaluation
                │
                ▼
           inference ◄─── webui (外部)
                │
                ▼
          cli/ (第1层) ──── 编排全部模块
```

**调用规则**：
1. **CLI 层**是唯一入口，编排全部业务模块
2. **Service 层**编排 Core 层，一个函数完成一个完整用例
3. **Core 层**实现单一业务逻辑，不跨模块调用
4. **Config 层**独立于业务链，只被 Service 层调用
5. **Common 层**被所有模块依赖，不依赖任何业务模块
6. **禁止**：反向依赖、循环依赖、`sys.path.append`

**特殊说明**：
- `training/` 的 Service 和 Core 合并在 `experiment.py`（薄 Service + 厚 Core）
- `webui/` 不在五层之内，是 Gradio 前端，通过 import 调用 inference 和 training
- `web-backend/` 是独立 FastAPI 服务，训练 hooks 通过 HTTP 同步数据

---

## 第五章：答辩 FAQ（按层组织）

### CLI 层

**Q: CLI 命令是怎么注册的？**
`pyproject.toml` 的 `[project.scripts]` 段映射命令到函数。`pip install -e .` 后 pip 自动创建可执行脚本。

```toml
odp-train = "odp_platform.cli.train:main"
```

**Q: 为什么 entry-point (`odp-train`) 和 `python -m odp_platform.cli.train` 两种方式？**
entry-point 生成的是独立的可执行脚本（加入 PATH），`python -m` 是标准模块调用方式。前者更短的命令，后者用于调试。

**Q: CLI 参数怎么和 YAML 配置合并？**
CLI 层只解析 argparse，不合并。把参数传给 `build_config(cli_args=...)`，由 Config 层的 `merger.py` 完成三源合并。

### Service 层

**Q: 训练和推理的 Service 层设计差异？**
- 训练：`experiment.py` 的 `run_experiment()` 是厚 Service（含 Core 逻辑）
- 推理：`inference/service.py` 的 `InferService` 是薄 Service（编排 Core 层 Detector + frame_source）
- 评估：`evaluation/service.py` 的 `ValService` 是极薄 Service（直接包装 YOLO.val()）

**Q: WebUI 调 Service 层还是 Core 层？**
WebUI 直接调 Core 层（`_get_or_create_detector` → `Detector.detect`），没有中间 Service。这是 WebUI 的实际情况，技术上可以再加一层 WebUI Service，但当前 5 人团队直接调 Core 更高效。

### Core 层

**Q: 注册表模式解决了什么问题？**
消除 if/elif 分支。新增格式只需新建文件 + 加装饰器，不改旧代码。

```python
# 不加注册表前——每新增格式要改 factory
def convert(format_name, ...):
    if format_name == "pascal_voc": ...
    elif format_name == "coco": ...
    elif format_name == "labelme": ...  # 改这行

# 加注册表后——新增格式不碰现有代码
@register("labelme")
def convert_labelme(...): ...
```

**Q: data_pipeline 和 data_validation 的注册表有什么区别？**
- data_pipeline 是 **互斥分发**：按格式选 1 个 converter
- data_validation 是 **聚合执行**：跑全部 check，一个 ERROR 不影响其他 check 继续

**Q: frame_source 的职责边界？**
只做"帧输入"（open/read/close），不做推理、不做可视化。新增 RTSP 源只需 `@register_source("rtsp")` + 实现 FrameSource 抽象类。

### Config 层

**Q: 三源合并的覆盖顺序？**
CLI 参数（最高）> YAML 文件 > 代码默认值（最低）。每个字段的溯源链由 `TraceRecord` 记录。

**Q: 为什么不用 Pydantic 而用 ConfigField 注册表？**
Pydantic 模型是"任务为中心"，同名字段在不同任务有不同默认值时，继承体系复杂。ConfigField 是"字段为中心"，每个字段独立定义，同名字段首次注册胜出。

**Q: 配置快照的用途？**
1. 实验复现：`restore_from_snapshot()` 恢复历史配置
2. 来源追溯：`odp-config trace` 查字段来源
3. 回滚保护：错误配置可快速回滚

### Common 层

**Q: marker 文件定位为什么优于硬编码？**
项目可放在任意路径，移动后不需改代码。硬编码绝对路径不可移植，`os.getcwd()` 依赖运行目录。

**Q: logging 的幂等保护怎么实现的？**
```python
if logger.handlers:
    return logger  # 重复调用不重复添加 handler
```
防止多次初始化导致日志重复输出。

### 跨模块

**Q: CLI 和 WebUI 调的是同一套代码吗？**
是。`odp-infer` CLI 和 WebUI 单图检测都调 `inference/engine.py` 的 `Detector.detect()`。`odp-train` CLI 和管理员训练 Tab 都调 `training/experiment.py` 的 `run_experiment()`。

**Q: 后端不可达会怎样？**
训练不受影响。所有 HTTP 调用有 3 秒超时 + try/except + 静默降级：
```python
try:
    requests.post(url, timeout=3)
except RequestException:
    logger.warning("后端不可达，实验仅保存在本地")
```

**Q: CUDA OOM 怎么办？**
1. 降 batch（最有效）2. 降 imgsz 3. 开 AMP（默认已开）4. `device="cpu"` 回退

**Q: 最大的架构亮点和隐患？**
亮点：五层架构清晰，注册表模式保证扩展性，运行时不依赖后端。隐患：training 模块 Service+Core 混合，WebUI 直调 Core 层略过 Service，长期迭代需重构。

---

## 第六章：常见问题排查（新增）

### 摄像头打不开

```
[ WARN:0@9.428] global cap.cpp:480 ... backend is generally available but can't be used
[ERROR:0@11.313] global obsensor_uvc_stream_channel.cpp:163 Camera index out of range
```

**可能原因**（按概率排序）：
1. 摄像头被其他程序占用（相机/浏览器/会议软件）→ 关闭占用程序
2. 摄像头硬件开关被关闭（部分笔记本有物理开关）→ 检查
3. OpenCV 后端枚举失败（MSMF/DSHOW）→ 代码中有降级逻辑，会自动尝试

**代码降级路径**：MSMF → DSHOW → 无后端 `VideoCapture(cam_id)`。全部失败则显示"无摄像头"占位图。

### CUDA Out of Memory

```
torch.cuda.OutOfMemoryError: CUDA out of memory.
```

**解决方案**（按推荐顺序）：
1. 降低 `batch` 参数
2. 降低 `imgsz` 参数
3. 开启 AMP（混合精度训练，默认已开）
4. 回退到 CPU：`device="cpu"`

### 模型加载失败

```
Error: No model found at path ...
```

**排查步骤**：
1. 确认 `.pt` 文件在 `data/models/checkpoints/` 目录下
2. 确认文件未损坏：`python -c "from ultralytics import YOLO; YOLO('path/to/model.pt')"`
3. 确认文件名不包含中文或特殊字符
4. WebUI 中可手动输入绝对路径绕过文件扫描

### Windows 中文乱码

```
??? ?? ???
```

修复方式：日志系统已自动调用 `sys.stdout.reconfigure(encoding="utf-8")`。如果仍乱码，检查终端是否支持 UTF-8（Windows Terminal 推荐）。

---

## 第七章：学习路线图

### 7.1 按层学习顺序

| 优先级 | 层 | 核心文件 | 预计时间 | 理由 |
|:------:|----|---------|:--------:|------|
| ⭐⭐⭐ | Common | paths.py, logging_utils.py | 2h | 一切的基础 |
| ⭐⭐⭐ | Config | run_config/service.py, merger.py | 2h | 理解三源合并 |
| ⭐⭐⭐ | CLI | cli/ 全部 10 个命令 | 2h | 理解入口 |
| ⭐⭐⭐ | Service+Core | data_pipeline/ | 3h | 注册表模式范例 |
| ⭐⭐ | Service+Core | training/experiment.py, callbacks.py | 2h | 训练核心 |
| ⭐⭐ | Service+Core | inference/engine.py, service.py, frame_source/ | 3h | 推理体系 |
| ⭐ | Service | data_validation/service.py, checks/ | 1h | 质量保障 |
| ⭐ | Service | evaluation/service.py | 0.5h | 模型评估 |

### 7.2 答辩重点准备

参考 [docs/ODPlatform_答辩演练问题集.md](docs/ODPlatform_答辩演练问题集.md)，高频考点 Top 5：

1. **五层架构数据流**（Q1-1）：CLI→Service→Core→Config→Common，用 `odp-train` 举例
2. **三源合并 + 溯源**（Q5-1/Q5-2）：默认值→YAML→CLI，TraceRecord 记录来源
3. **全流程数据追踪**（Q8-1）：从前端到后端完整走一遍
4. **Monorepo 优劣**（Q1-2）：统一版本、原子提交、共享基础设施
5. **注册表调度模式**（Q4-4）：data_pipeline 互斥分发 vs data_validation 聚合执行

### 7.3 推荐阅读路径

```
① common/paths.py           ← marker 路径探测
② common/logging_utils.py   ← 日志体系
③ run_config/service.py     ← 三源合并配置
④ data_pipeline/registry.py + orchestrator.py  ← 注册表模式范例
⑤ training/experiment.py + callbacks.py         ← 训练核心
⑥ inference/engine.py + frame_source/           ← 推理体系
⑦ cli/train.py + infer.py                       ← CLI 入口
⑧ webui/app.py + user_tabs.py                   ← 前端集成
⑨ data_validation/service.py + checks/          ← 质量保障
```
