# ODPlatform - 全流程学习指南

> 作者: MuU | 目标: 从零掌握目标检测全栈开发技术路径
> **命令速查请见**：[ODPlatform_命令速查.md](ODPlatform_命令速查.md)

---

## 1. 项目全景概览

### 1.1 这是什么项目？

ODPlatform 是一个**基于 YOLO 的多格式目标检测开发平台**，覆盖从数据处理到模型推理的完整流程。

```
数据输入 → 格式转换 → 数据校验 → 模型训练 → 模型评估 → 模型推理
  VOC/COCO   YOLO格式    验证清洗    YOLO训练   mAP计算    单图/批量
  LabelMe    统一管理      可视化     调参       结果分析    流水线
```

### 1.2 架构演进三阶段

| 阶段 | 内容 | 状态 |
|------|------|:----:|
| **Stage 0**（D1 前） | 加 `.odp-workspace` marker，paths.py 改 marker 模式 | ✅ 已完成 |
| **Stage 1**（当前） | 创建 `apps/platform/`，`src/` 布局，pyproject.toml | ⚠ 当前状态 |
| **Stage 2**（V1.1） | 加 `web-backend/` `web-frontend/`，配 workspace | 🔮 未来 |

> 设计理念：**"今天动 0.5 步，明天能省 100 步"**——渐进演进，而非一步到位。

### 1.3 项目目录结构

```
ODPlatform/
├── apps/platform/           ← 核心引擎（学习的重点）
│   └── src/odp_platform/
│       ├── _version.py          版本号（单一数据源）
│       ├── common/              基础工具（路径/日志/性能）
│       ├── config/              配置管理（Pydantic v2）
│       ├── data_pipeline/       数据管道（格式转换+划分）
│       ├── data_validation/     数据校验
│       ├── training/            模型训练
│       ├── evaluation/          模型评估
│       ├── inference/           模型推理
│       ├── webui/               Gradio 可视化前端（用户+管理员双模式）
│       └── cli/                 命令行入口
├── pyproject.toml           ← 顶层 workspace 配置
├── data/                    ← 数据集（已 gitignore）
├── docs/                    ← 文档 + ADR 架构记录
└── scripts/                 ← 运维脚本
```

### 1.4 支持的数据格式

| 输入格式 | 输出格式 | 命令 |
|---------|---------|------|
| Pascal VOC (XML) | YOLO (txt) | `odp-trans voc` / `odp-transform` |
| COCO (JSON) | YOLO (txt) | `odp-trans coco` |
| LabelMe (JSON) | YOLO (txt) | `odp-trans labelme` |
| YOLO (txt) | COCO (JSON) | `odp-trans yolo2coco` |

---

## 2. 全栈技术路径

### 2.1 技术栈全景

| 层级 | 技术 | 用途 |
|------|------|------|
| 编程语言 | Python 3.10+ | 主力开发语言 |
| 包管理 | pip + hatchling | 构建与发布 |
| 配置管理 | Pydantic v2 | 配置校验与类型安全 |
| 数据格式 | XML / JSON / YAML | 数据集标注格式 |
| 目标检测 | Ultralytics YOLO | 模型训练与推理 |
| 数值计算 | NumPy / Pandas | 数据处理 |
| 图像处理 | OpenCV | 图像读写与预处理 |
| 日志 | logging + colorlog | 彩色控制台 + 文件日志 |
| 测试/类型/风格 | pytest / mypy / ruff | 代码质量 |
| 构建后端 | hatchling | 现代 Python 打包 |

### 2.2 学习路线

```
第一阶段：Python 基础与工程化
  Python 语法 → 类型注解 → logging → argparse → 包管理

第二阶段：数据处理
  Pascal VOC (XML解析) → COCO (JSON) → YOLO (txt) → LabelMe
  → OpenCV → NumPy → Pandas

第三阶段：目标检测
  YOLO 原理 → Ultralytics API → 训练调参 → mAP 评估 → 推理部署

第四阶段：工程化与运维
  pyproject.toml → pip install -e . → CLI 注册 → 日志系统
  → GitHub 协作 → ADR 架构记录

第五阶段：生产化
  Docker → ONNX/TensorRT导出 → FastAPI 服务 → Web 前端
```

---

## 3. 项目架构详解

### 3.1 分层架构

```
CLI 层 (cli/)            ← 用户交互入口
   │
服务层 (service.py)      ← 业务流程编排
   │
核心层 (core/)           ← 业务逻辑实现
   │
基础层 (common/)         ← 基础设施（路径、日志、性能）
```

### 3.2 模块依赖关系

```
common/ (基础) → config/ (配置) → data_pipeline/ (数据)
                                            ↓
                                   data_validation/ (校验)
                                            ↓
                                   training/ (训练)
                                            ↓
                                   evaluation/ → inference/
```

- `common/` 被所有模块依赖（路径、日志、性能工具）
- `data_pipeline/` 输出划分后的数据集供训练使用
- `data_validation/` 校验数据质量，产出清洗报告
- `cli/` 是入口，调用各模块服务层

### 3.3 路径中心化设计

路径统一在 [common/paths.py](file:///f:/python_projects/class/ODPlatform/apps/platform/src/odp_platform/common/paths.py) 管理，通过 `.odp-workspace` marker 自动定位根目录：

```python
from odp_platform.common.paths import ROOT_DIR, DATA_DIR, LOGGING_DIR
# 任何模块只需导入，不需要自己计算路径
dataset_path = DATA_DIR / "RSOD" / "images"
```

---

## 4. 日志系统设计

### 4.1 设计哲学

采用**"根 Logger 装配 + 业务模块冒泡"**模式：

```
根 Logger "odp_platform"
  ├── FileHandler → logging/train/*.log
  └── StreamHandler → 彩色控制台输出
  
子 Logger（自动冒泡到根）
  odp_platform.training.service
  odp_platform.data_pipeline.core.coco
  ...
```

### 4.2 使用方式

**入口初始化一次**（CLI 入口已做，通常无需手动调用）：

```python
from odp_platform.common.logging_utils import get_logger
from odp_platform.common.paths import LOGGING_DIR
get_logger(base_path=LOGGING_DIR, log_type="train")
```

**业务模块只需一行**：

```python
import logging
logger = logging.getLogger(__name__)
logger.info("开始训练 epoch 1/100")   # 自动冒泡到根 logger
```

根 logger 已配好 handler 和格式，子 logger 无需重复配置。

### 4.3 命名空间分离

| 目录 | 用途 | 是否被 reset 清理 |
|------|------|:----:|
| `apps/platform/logging/` | 业务日志（训练/推理/校验） | ✅ |
| `.odp-meta/logs/` | 工具审计日志（reset 操作） | ❌ 受保护 |

### 4.4 关键实现要点（logging_utils.py）

```python
# 1. 幂等保护：防止重复配置
if logger.handlers:
    return logger

# 2. 关闭冒泡（避免日志重复到父级）
logger.propagate = False

# 3. 控制台彩色输出（colorlog 库）
if _HAS_COLORLOG:
    console_handler.setFormatter(ColoredFormatter(...))
```

### 4.5 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| 没有日志输出 | 入口未调 `get_logger()` | CLI 入口调一次 |
| DEBUG 信息不显示 | 默认 INFO 级别 | `$env:ODP_LOG_LEVEL="DEBUG"` |
| 日志重复输出 | `propagate` 未关闭 | logging_utils 已处理 |
| 日志文件名含义 | 命名规则 `{type}_{日期}-{时间}.log` | 如 `train_20260519-103000.log` |

---

## 5. pyproject.toml 与命令打包

### 5.1 为什么用 pyproject.toml？

PEP 621 标准化后，一个 pyproject.toml 替代了 setup.py + setup.cfg + requirements.txt + MANIFEST.in，且不包含可执行代码（安全）。

### 5.2 两层配置体系

**顶层 pyproject.toml**（根目录）— 开发工具配置，不被 pip install：

```toml
[tool.ruff]
line-length = 100

[tool.mypy]
python_version = "3.10"

[tool.pytest.ini_options]
testpaths = ["apps/platform/tests"]
```

**子项目 pyproject.toml**（apps/platform/）— 可发布的包配置：

```toml
[build-system]
requires = ["hatchling>=1.18.0"]
build-backend = "hatchling.build"

[project]
name = "odp-platform"
dynamic = ["version"]
requires-python = ">=3.10"

[project.scripts]
odp-init     = "odp_platform.cli.init_project:initialize_project"
odp-trans    = "odp_platform.cli.trans:main"
# ... 其他命令
```

### 5.3 [project.scripts] 原理

`[project.scripts]` 将 Python 函数注册为系统级命令：

```toml
odp-init = "odp_platform.cli.init_project:initialize_project"
#          └──────包路径──────┘ └──────函数名──────┘
```

`pip install -e .` 后，pip 自动创建脚本，在终端直接运行 `odp-init` 等价于调用该函数。

### 5.4 版本号单一数据源

```toml
# pyproject.toml
dynamic = ["version"]
[tool.hatch.version]
path = "src/odp_platform/_version.py"
```

```python
# _version.py — 只需改这一个文件
__version__ = "0.1.0"
```

---

## 6. 性能记录工具

### 6.1 @time_it 装饰器（函数计时）

```python
from odp_platform.common.performance_utils import time_it

@time_it(name="数据加载", iterations=3)   # 多次执行取平均
def load_data():
    ...
```

### 6.2 timer 上下文管理器（代码块计时）

```python
from odp_platform.common.performance_utils import timer

with timer("训练一个 epoch"):
    model.train()
```

### 6.3 MetricTracker 指标追踪器

```python
from odp_platform.common.performance_utils import MetricTracker

tracker = MetricTracker("loss")
for step in range(100):
    tracker.record(step, compute_loss())

print(tracker.summary())    # latest / best / average / count
tracker.to_csv(Path("loss_history.csv"))  # 导出 CSV
```

### 6.4 PerformanceTracker 全流程追踪

```python
from odp_platform.common.performance_utils import PerformanceTracker

pt = PerformanceTracker()
pt.start("全流程")
# ... 处理步骤 ...
pt.mark("数据加载完成")
# ... 更多步骤 ...
pt.stop("全流程")
print(pt.summary())
```

### 6.5 TrainingHistory 训练历史归档

```python
from odp_platform.common.performance_utils import TrainingHistory

history = TrainingHistory(Path("data/runs"))
record = history.new_run("exp001", "yolo11n", {"epochs": 100})
record.finish({"map50": 0.85})
history.save(record)    # 保存为 JSON
history.list_runs()     # 列出所有历史
```

---

## 7. ADR 架构决策记录

### 7.1 什么是 ADR？

ADR（Architecture Decision Record）记录关键架构决策的背景、选项和结论，保存在 `docs/architecture/`：

```
docs/architecture/
├── ADR-001-monorepo-layout.md
├── ADR-002-pyproject-toml.md
├── ADR-003-paths-by-marker.md
├── ADR-004-logging-architecture.md
└── ADR-005-data-pipeline-design.md
```

### 7.2 ADR 结构模板

```markdown
# ADR-NNN: 标题

## 状态
[提议 / 接受 / 已弃用]

## 背景
为什么要做这个决策？

## 选项
- A 方案：...
- B 方案：...

## 决策
选择了哪个方案，为什么？

## 后果
这个决策带来了什么影响？
```

### 7.3 何时应该写 ADR？

| 场景 | 示例 |
|------|------|
| 引入新技术栈 | "用 Pydantic v2 还是 dataclass？" |
| 架构变更 | "单项目还是 Monorepo？" |
| 接口设计 | "CLI 用 argparse 还是 click？" |
| 数据流设计 | "数据管道如何分层？" |

> ADR 的价值不在于写了多少，而在于**关键决策有据可查**。

---

## 8. WebUI 可视化前端

### 8.1 双模式设计

ODPlatform 的 Gradio Web 界面分为**用户模式**和**管理员模式**：

- **用户模式**：面向日常检测使用，包含 6 个 Tab
  - 单图检测 / 文件夹检测 / 视频检测 / 实时摄像头 / 模型选择 / LLM对话
- **管理员模式**：点击齿轮图标输入密码进入，额外包含 Dashboard / 模型演示 / 数据集浏览 / 训练 / 数据校验 / 配置管理

### 8.2 模型选择与实验可视化

模型选择 Tab 支持：
- 从 `data/models/checkpoints/` 自动扫描 `.pt` 文件
- 上传自定义 `.pt` 模型
- 手动输入模型路径
- **实验训练结果**：折叠面板展开后，展示训练曲线、混淆矩阵、PR/F1 曲线、类别分布

### 8.3 核心源文件

| 文件 | 职责 |
|------|------|
| `webui/app.py` | Gradio 应用入口，组装配件 + 启动后端 |
| `webui/user_tabs.py` | 用户模式所有 Tab（检测/摄像头/模型/LLM）+ 实验可视化 |
| `webui/training_tab.py` | 管理员训练 Tab |
| `webui/dashboard.py` | 管理员 Dashboard |
| `webui/config_tab.py` | 配置管理 Tab |
| `webui/validation_tab.py` | 数据校验 Tab |
| `webui/model_demo.py` | 模型演示 Tab |
| `webui/dataset_browser.py` | 数据集浏览 Tab |
| `webui/utils.py` | 工具函数（模型扫描、图片列表等） |

### 8.4 启动方式

```bash
# 完整 WebUI（含后端）
odp-webui

# 仅后端
odp-backend
```

访问 http://localhost:7860

---

## 9. 训练全流程详解

### 9.1 训练入口

ODPlatform 提供两种训练入口：

| 入口 | 方式 | 适用场景 |
|------|------|---------|
| CLI 命令行 | `odp-train --dataset rsod --model yolo11n.pt` | 服务器/脚本批量训练 |
| WebUI 管理员 | 训练 Tab 图形化配置 | 快速实验/调参 |
| Python API | `run_experiment(config)` | 自定义训练流程 |

### 9.2 训练流程图

```
用户输入
  │
  ├── 数据集名 (--dataset)
  ├── 模型名   (--model)
  └── 超参数   (--epochs, --batch, --lr0 ...)
  │
  ▼
┌─────────────────────────────┐
│  Step 1: 数据转换           │   odp-transform (可选)
│   VOC/COCO/LabelMe → YOLO   │
└─────────────────────────────┘
  │
  ▼
┌─────────────────────────────┐
│  Step 2: 数据质检           │   odp-validate
│   YAML校验/标签检查/泄露检测 │
└─────────────────────────────┘
  │
  ▼
┌─────────────────────────────┐
│  Step 3: 配置生成           │   odp-config
│   自动生成 + 快照保存       │
└─────────────────────────────┘
  │
  ▼
┌─────────────────────────────┐
│  Step 4: 模型训练           │   YOLO.train()
│   Ultralytics 引擎          │
└─────────────────────────────┘
  │
  ▼
┌─────────────────────────────┐
│  输出                       │
│  ├─ results.csv (逐轮指标)   │
│  ├─ weights/best.pt (权重)   │
│  ├─ results.png (损失曲线)   │
│  ├─ confusion_matrix.png    │
│  ├─ BoxPR_curve.png         │
│  └─ labels.jpg (类别分布)    │
└─────────────────────────────┘
  │
  ▼
┌─────────────────────────────┐
│  后处理                      │
│  ├─ checkpoint → checkpoints/│
│  ├─ Backend hooks → 数据库   │
│  └─ POST /api/experiments    │
└─────────────────────────────┘
```

### 9.3 ExperimentConfig 配置对象

```python
@dataclass
class ExperimentConfig:
    name: str           # 实验名称
    dataset: str        # 数据集名（对应 configs/datasets/<name>.yaml）
    model: str          # 模型名或 .pt 路径
    task: str           # 任务类型 (detect/segment/classify)
    epochs: int         # 训练轮数
    batch: int          # 批次大小
    imgsz: int          # 输入图片尺寸
    lr0: float          # 初始学习率
    device: str         # 训练设备 (""=auto, "0"=GPU0, "cpu")
    workers: int        # 数据加载线程
    optimizer: str      # 优化器 (auto/SGD/Adam/AdamW)
    amp: bool           # 混合精度训练
    patience: int       # EarlyStopping 耐心值
    seed: int           # 随机种子
```

### 9.4 训练输出目录结构

```
data/runs/experiments/<experiment_name>/
├── config_snapshot.json     # 配置快照
├── results.csv              # 逐轮指标 CSV
├── results.png              # 训练损失 + 验证指标曲线
├── confusion_matrix.png     # 混淆矩阵
├── confusion_matrix_normalized.png  # 归一化混淆矩阵
├── BoxPR_curve.png          # PR 曲线
├── BoxF1_curve.png          # F1 曲线
├── labels.jpg               # 类别分布
├── weights/
│   ├── best.pt              # 最佳权重
│   └── last.pt              # 最后 epoch 权重
└── train_summary.json       # 训练摘要
```

训练完成后，best.pt 自动复制到 `data/models/checkpoints/best_<实验名>.pt`。

### 9.5 CSV 指标适配

Ultralytics 的 results.csv 列名在不同版本中可能变化。项目通过 `_COLUMN_ALIASES` 映射表做兼容：

| CSV 原始列名 | 规范化名 | 含义 |
|---|---|---|
| `metrics/mAP50(B)` | `map50` | 平均精度 @IoU=0.5 |
| `metrics/mAP50-95(B)` | `map50_95` | 平均精度 @IoU=0.5:0.95 |
| `metrics/precision(B)` | `precision` | 精确率 |
| `metrics/recall(B)` | `recall` | 召回率 |
| `train/box_loss` | `box_loss` | 边界框损失 |
| `lr/pg0` | `lr` | 学习率 |

YOLO 升级改列名时，只需在 `_COLUMN_ALIASES` 中加一行映射，不会导致图表白屏。

### 9.6 Backend Hooks 机制

训练过程中，`TrainingHooks` 类自动将指标同步到后端数据库：

```
on_train_start()  → POST /api/experiments         ← 注册实验
on_epoch_end()    → POST /api/experiments/{id}/epochs  ← 写入逐轮指标
on_train_end()    → PATCH /api/experiments/{id}/status ← 更新完成状态
on_train_failed() → PATCH /api/experiments/{id}/status ← 标记失败
```

后端不可达时，静默降级（warning 日志），不阻塞训练。

---

## 10. 推理全流程详解

### 10.1 推理入口

| 入口 | 命令/方式 | 适用场景 |
|------|----------|---------|
| CLI | `odp-infer detect --model best.pt --input test.jpg` | 批量/脚本 |
| WebUI 用户 | 单图/文件夹/视频 Tab | 日常检测 |
| Python API | `Detector().detect(image)` | 集成到其他系统 |
| Agent 对话 | 在 LLM Tab 说"帮我检测这张图" | 快速演示 |

### 10.2 推理流程图

```
输入源识别 (create_frame_source)
  │
  ├── 单张图片 → ImageSource
  ├── 文件夹   → ImageFolderSource
  ├── 视频文件 → VideoSource
  ├── 摄像头   → CameraSource (OpenCV)
  └── 数字 ID  → CameraSource (设备号)
  │
  ▼
模型加载 (Detector)
  │
  ├── 加载 .pt 权重 → YOLO(model_path)
  ├── GPU JIT 预热 → warmup()
  └── 缓存到 _detector_cache (复用)
  │
  ▼
逐帧推理 (detect(image))
  │
  ├── YOLO 前向传播
  ├── NMS 非极大值抑制
  └── 返回 InferenceResult
  │
  ▼
结果可视化 (draw_detections)
  │
  ├── 绘制检测框 (cv2.rectangle)
  ├── 标签+置信度 (cv2.putText)
  └── 信息面板 (draw_info_panel)
  │
  ▼
结果输出
  ├── 标注图片
  ├── 检测明细表 (类别/置信度/坐标)
  ├── CSV 导出（文件夹/视频）
  └── 实时流（摄像头）
```

### 10.3 Detector 核心类

```python
class Detector:
    def __init__(self, model_path: str, device: str = "auto",
                 conf: float = 0.25, iou: float = 0.45):
        # 加载 YOLO 模型
        # 自动选择设备 (CUDA > MPS > CPU)

    def warmup(self, image_size=(640, 640)):
        # 用纯黑图跑一次推理，消除首次 CUDA kernel 编译延迟
        # 仅在 CUDA 下生效

    def detect(self, image: np.ndarray) -> InferenceResult:
        # YOLO 推理 → NMS → 返回检测结果

    def release(self):
        # 释放 GPU 显存

    @property
    def model_path(self) -> str:
    @property
    def class_names(self) -> list[str]:
```

### 10.4 InferenceResult 数据结构

```python
@dataclass
class InferenceResult:
    detections: list[Detection]   # 检测结果列表
    inference_ms: float           # 推理耗时 (ms)
    image_size: tuple[int, int]   # (宽, 高)

@dataclass
class Detection:
    class_id: int                 # 类别 ID
    class_name: str               # 类别名称
    confidence: float             # 置信度 [0, 1]
    bbox: tuple[float, ...]       # 归一化边界框 [x1, y1, x2, y2]
```

### 10.5 帧源架构 (frame_source)

`inference/frame_source/` 是独立的帧输入抽象层，可整包拷贝到其他项目：

```
frame_source/
├── core/
│   ├── base.py      FrameSource 抽象基类
│   ├── config.py    CameraConfig (Pydantic)
│   └── types.py     类型常量
├── sources/
│   ├── camera.py    CameraSource (OpenCV)
│   ├── video.py     VideoSource
│   └── image.py     ImageSource / ImageFolderSource
├── wrappers/
│   ├── threaded.py  ThreadedSource (实时推理首选)
│   └── aio.py       AsyncSource (异步接口)
├── factory.py       create_frame_source() + 注册表模式
└── overlay.py       HUD 叠加层
```

**注册表模式**：新增帧源类型只需 `@register_source("rtsp")`，无需改 factory 的 if-else 链。

```python
@register_source("rtsp", extensions=[])
class RTSPSource(FrameSource):
    ...
```

**线程化包装**（实时推理首选）：采集放后台线程，消费端从缓冲拿最新帧，解决"消费慢拖累采集"问题。

```python
with create_threaded_source("0", warmup_frames=30) as src:
    for frame in src:
        results = model(frame.image)
```

---

## 11. Agent 智能助手系统

### 11.1 架构设计

Agent 系统采用**关键词路由 + 本地工具执行 + LLM 排版**的三层架构：

```
用户消息
  │
  ▼
┌─────────────────────┐
│  Intent Router      │  关键词正则匹配（不依赖 LLM）
│  模型→list_models   │
│  数据集→list_datasets│
│  实验→list_experiments│
│  推理/检测→run_inference│
│  GPU/显存→get_gpu_info│
└─────────────────────┘
  │
  ├── 匹配 → 本地执行工具 → 原始数据
  │                       │
  │                       ▼
  │               ┌──────────────────┐
  │               │  LLM 排版美化    │  → 自然语言回复
  │               └──────────────────┘
  │
  └── 未匹配 → 普通 LLM 对话
```

### 11.2 为什么不用 LLM function calling？

`deepseek-v4-flash` 对 OpenAI 格式的 function calling 支持不稳定，经常忽略 tools 参数直接当成普通聊天。改用关键词路由后：

- **100% 可靠**：工具执行完全不依赖 LLM
- **即时响应**：本地执行，无需等 LLM 返回工具调用
- **低消耗**：不浪费 token 在 function calling 上
- LLM 只负责把结构化数据写成自然语言（一次调用）

### 11.3 可用工具

| 关键词触发 | 函数 | 功能 |
|---|---|---|
| 模型/model/.pt/权重 | `tool_list_models()` | 列出所有 .pt 模型+文件大小 |
| 数据集/dataset | `tool_list_datasets()` | 列出所有数据集 YAML |
| 实验/训练/exp | `tool_list_experiments()` | 列出实验 + best mAP50 |
| 推理/检测+路径 | `tool_run_inference()` | 对图片执行 YOLO 推理 |
| GPU/显存/cuda | `tool_get_gpu_info()` | GPU 显存使用状态 |

### 11.4 路径自动解析

推理检测时，Agent 自动从用户消息中提取路径：

```
用户说："用data/models/checkpoints/best_rsod.pt检测这张图C:\test\image.jpg"
                                      └──── model_path ────┘ └─── image_path ──┘
```

支持 Windows 路径 (`C:\...`) 和 Linux 路径 (`/...`)，以及模糊匹配（说模型名自动查找）。

---

## 12. 配置管理详解

### 12.1 配置层级

ODPlatform 支持四层配置覆盖（优先级从高到低）：

```
1. CLI 参数         --epochs 200 --batch 32      ← 最高优先级
2. 配置文件 YAML    configs/train_config.yaml
3. 环境变量         ODP_EPOCHS=200
4. 默认值           ExperimentConfig 字段默认值
```

### 12.2 配置快照

每次训练自动保存配置快照：

```
data/runs/experiments/<实验名>/config_snapshot.json
data/runs/run_config/<实验名>/config_snapshot.json
data/runs/run_config/<实验名>/config_report.json
```

可以随时恢复历史快照：

```bash
odp-config snapshot restore --snapshot runs/config_snapshots/<name>.yaml
```

### 12.3 配置溯源 (trace)

查看每个配置字段的来源层级：

```bash
odp-config trace --config configs/train_config.yaml
# 输出示例: classes → CLI > env var > default
```

### 12.4 配置管理架构

```
config_manager/
├── registry.py       @config_generator 注册装饰器
├── service.py        核心调度 (generate/validate/trace/snapshot)
├── snapshot.py       snapshot 导出与恢复
├── generator.py      配置生成器基类
├── validator.py      配置校验器
├── tracer.py         配置溯源
└── generators/
    └── train.py      训练配置生成器
```

---

## 13. 架构设计原则与演进

### 13.1 核心设计原则

| 原则 | 体现 |
|------|------|
| **单一数据源** | 版本号在 `_version.py`，路径在 `paths.py`，常量在 `constants.py` |
| **渐进演进** | 先 monorepo 后拆服务，避免过度设计 |
| **幂等保护** | 日志初始化、路径探测、模型加载均可重复调用 |
| **静默降级** | 后端不可达、文件不存在、GPU 不可用时告警不崩溃 |
| **fail-fast** | 数据格式不匹配、路径不存在时立即报错，不静默吞错 |
| **开放-封闭** | factory 注册表模式、config 生成器注册、hook 回调 |

### 13.2 三阶段演进

```
Stage 0 (D1前)     Stage 1 (当前)         Stage 2 (V1.1)
┌──────────┐      ┌──────────────┐      ┌──────────────────┐
│ marker   │  →   │ apps/platform│  →   │ web-backend/     │
│ paths.py │      │ CLI + WebUI  │      │ web-frontend/    │
│          │      │ 全功能单服务  │      │ 前后端分离       │
└──────────┘      └──────────────┘      └──────────────────┘
```

当前 Stage 1：`apps/platform/` 是单体，CLI 和 WebUI 共用同一套核心库。

### 13.3 跨模块调用规范

```
common/ (基础层) ← 所有模块依赖，不依赖任何业务模块
  ├── paths.py        路径探测（.odp-workspace marker）
  ├── logging_utils.py 日志装配
  ├── constants.py     常量和枚举
  └── performance_utils.py 性能工具

业务层依赖链（单向）：
  data_pipeline → data_validation → training → evaluation → inference
       ↓               ↓               ↓           ↓            ↓
      CLI           CLI              CLI         CLI          CLI / WebUI
```

**禁止**：业务层反向依赖、循环依赖、`sys.path.append` hack（安装模式正常时不需要）。

---

## 14. 答辩 FAQ / 常见架构问题

### Q1: 为什么要用 Monorepo 而不是多仓库？

**A**: Monorepo 的优势：
- 统一版本管理（一个 pyproject.toml）
- 跨模块修改原子化（一个 commit 改多个模块）
- 共享基础设施（common/ 路径/日志/性能工具）
- 降低 CI/CD 复杂度（一次构建）
- 适合小团队（5 人以下）协作

后期如果团队扩大，可以拆为独立仓库，当前架构已预留了拆分的接口边界。

### Q2: 路径探测的 `.odp-workspace` marker 机制怎么工作？

**A**: 从当前文件目录向上遍历，寻找包含 `.odp-workspace` 标记文件的目录作为根目录。这样项目可以放在任意路径下，不需要硬编码绝对路径，支持软链接和符号链接。

```python
def _find_workspace_root(marker=".odp-workspace") -> Path:
    """从当前文件向上找 marker 文件。"""
    current = Path(__file__).resolve().parent
    for parent in current.parents:
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(f"未找到 {marker} 文件")
```

### Q3: 为什么 WebUI 支持双模式（用户/管理员）？

**A**: 满足不同角色的使用场景：
- **用户模式**：面向日常检测操作人员，界面简洁，只有检测/摄像头/模型/对话功能
- **管理员模式**：面向开发/运维人员，额外提供训练/数据校验/配置管理/模型演示等工具
- 通过密码切换，无需两套部署

### Q4: Agent 对话中的工具调用和传统 RAG 有什么区别？

**A**: 传统 RAG（检索增强生成）是把问题发给知识库搜索文档，然后把文档片段发给 LLM 生成回答。Agent 的工具调用是直接执行代码获取实时数据：

| | RAG | ODP Agent |
|---|---|---|
| 数据源 | 静态文档 | 实时系统状态 |
| 执行方式 | 向量检索 | 代码执行 |
| 示例 | "YOLO是什么？" → 搜文档 | "有哪些模型？" → 扫描目录 |
| 刷新率 | 取决于索引更新 | 每次实时 |
| 可靠性 | 可能检索到旧信息 | 100% 当前状态 |

### Q5: 训练性能优化做了哪些？

**A**:
1. **GPU JIT 预热**：`warmup()` 用纯黑图跑一次推理，消除首次 CUDA kernel 编译延迟
2. **模型缓存**：`_detector_cache` 缓存已加载的模型，切换 Tab 不重新加载
3. **EarlyStopping**：验证集 mAP 连续 patience 轮不提升时自动停止，节省时间
4. **AMP 混合精度**：默认开启，减少显存占用 ~40%
5. **模型扫描缓存**：`list_model_files()` 有 5 秒 TTL，避免频繁扫描磁盘

### Q6: CSV 列名适配器解决什么问题？

**A**: Ultralytics YOLO 在不同版本中可能修改 results.csv 的列名（如 `mAP50(B)` → `mAP50`）。如果代码直接按列名读取，升级 YOLO 会导致图表白屏。项目通过 `_COLUMN_ALIASES` 映射表做兼容：

```python
_COLUMN_ALIASES = {
    "metrics/mAP50(B)": "map50",      # YOLOv8 格式
    "mAP50": "map50",                  # YOLOv11 可能格式
}
```

升级 YOLO 只需在映射表中加一行，不需要改业务代码。

### Q7: 摄像头流处理怎么保证实时性？

**A**: 
- 采用 `create_threaded_source()` 包装，采集线程和消费线程分离
- 缓冲策略为 `"latest"`（只保留最新帧），不堆积旧帧
- 支持分辨率切换（640x480 / 1280x720 / 1920x1080）
- 多后端支持（MSMF / DSHOW），Windows 下自动选择最优后端
- 资源泄漏防护：`_release_server_camera()` 确保标签页关闭时释放摄像头

### Q8: 配置快照有什么实际用途？

**A**:
1. **实验复现**：通过快照恢复历史训练配置，精确复现实验结果
2. **来源追溯**：`odp-config trace` 查看每个字段来自哪个层级
3. **对比分析**：对比多次实验的配置差异，找出影响指标的关键参数
4. **回滚保护**：错误配置导致训练失败时，可快速回滚到上次有效配置

### Q9: 后端不可达时系统会崩溃吗？

**A**: 不会。所有后端通信都有重试 + 超时 + 静默降级机制：

```python
try:
    requests.post(url, json=payload, timeout=3)
except requests.RequestException:
    logger.warning("后端不可达，实验仅保存在本地")
```

- 训练继续执行，不受后端状态影响
- 指标保存在本地 CSV 中，后端恢复后可通过脚本同步
- Dashboard 显示"后端不可用"提示但不阻塞其他功能

### Q10: 出现"CUDA out of memory"怎么办？

**A**: 按优先级尝试：
1. 降低 `batch` 大小（最有效）
2. 降低 `imgsz` 图片尺寸
3. 开启 `--amp` 混合精度（默认已开启）
4. 使用 `device="cpu"` 回退到 CPU 训练
5. 调用 `torch.cuda.empty_cache()` 清理缓存

WebUI 训练时会自动检测 GPU 显存并给出建议 batch 值：

```
检测到 GPU 显存 6 GiB，建议 batch ≤ 4
```

---

## 15. 学习路线图 (更新版)

### 15.1 按模块优先级学习

| 优先级 | 模块 | 学习目标 | 预计时间 |
|:------:|------|---------|:--------:|
| ⭐⭐⭐ | common/ | 路径/日志/性能工具 | 2 小时 |
| ⭐⭐⭐ | data_pipeline/ | 格式转换+划分流程 | 3 小时 |
| ⭐⭐⭐ | cli/ | 命令注册+参数解析 | 2 小时 |
| ⭐⭐ | training/ | 训练配置+启动 | 2 小时 |
| ⭐⭐ | evaluation/ | 评估指标+报表 | 1 小时 |
| ⭐⭐ | inference/ | 推理+结果输出 | 1 小时 |
| ⭐⭐ | webui/ | Gradio 前端界面 | 2 小时 |
| ⭐ | data_validation/ | 校验+清洗 | 1 小时 |
| ⭐ | config/ | Pydantic 配置 | 1 小时 |

### 15.2 推荐学习步骤

**第 1 步：理解基础工具层（common/）**
- 阅读 [common/paths.py](file:///f:/python_projects/class/ODPlatform/apps/platform/src/odp_platform/common/paths.py) — 路径探测机制
- 阅读 [common/logging_utils.py](file:///f:/python_projects/class/ODPlatform/apps/platform/src/odp_platform/common/logging_utils.py) — 日志装配
- 阅读 [common/performance_utils.py](file:///f:/python_projects/class/ODPlatform/apps/platform/src/odp_platform/common/performance_utils.py) — 性能工具

**第 2 步：理解数据管道（data_pipeline/）**
- 阅读 registry.py → service.py → core/ 各转换器
- 理解格式能力矩阵和覆盖率 fail-fast

**第 3 步：理解 CLI 入口（cli/）**
- 阅读各命令文件，理解 `argparse` 用法
- 对比 pyproject.toml 中的 `[project.scripts]` 注册

**第 4 步：运行完整流程**
- 从 `odp-init` → `odp-transform` → `odp-train` → `odp-val` → `odp-infer`

### 15.3 核心代码阅读路径

```
1. apps/platform/src/odp_platform/common/paths.py       ← 路径是基础
2. apps/platform/src/odp_platform/common/logging_utils.py ← 日志很重要
3. apps/platform/src/odp_platform/data_pipeline/        ← 数据处理核心
4. apps/platform/src/odp_platform/cli/                  ← 入口命令
5. apps/platform/src/odp_platform/training/             ← 训练流程
6. apps/platform/tests/                                 ← 测试用例
```

### 15.4 自学延伸方向

- **模型优化**：YOLO 模型压缩（剪枝/量化/TensorRT）
- **Web 服务**：用 FastAPI 封装推理 API
- **多任务学习**：检测 + 分割 + 分类联合训练
- **数据增强**：Mosaic / MixUp / Albumentations
- **MLOps**：实验追踪（MLflow/W&B）、模型版本管理

> **记住**：不要试图一次学完——今天读懂一个模块，明天跑通一条命令，后天改好一个 Bug，每周进步一点点。