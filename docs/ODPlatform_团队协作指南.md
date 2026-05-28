# ODPlatform 统一架构与协同开发方案

> 面向 5 位工程师（训练 / 算法 / 前端 / 后端 / 推理）。架构师已定义统一的参数、目录、日志和接口。各角色在统一框架内发挥。

---

## 零、仓库现状与开发环境

### 0.1 仓库现有模块（clone 后可直接看到）

| 模块 | 路径 | 状态 |
|---|---|---|
| common | `odp_platform/common/` | **已有** — paths、constants、logging_utils、system_utils |
| data_pipeline | `odp_platform/data_pipeline/` | **已有** — 数据转换管线 |
| data_validation | `odp_platform/data_validation/` | **已有** — 质检 service、registry、checks |
| run_config | `odp_platform/run_config/` | **已有** — 运行配置 schema、loader、merger |
| cli（部分） | `odp_platform/cli/` | **已有** — `odp-train`/`odp-init`/`odp-validate`/`odp-config`/`odp-transform`/`odp-reset` |
| training | `odp_platform/training/` | **待创建** |
| inference | `odp_platform/inference/` | **待创建** |
| webui | `odp_platform/webui/` | **待创建** |
| web-backend | `apps/web-backend/` | **仅有 README.md** |

**数据集**：`data/yolo/visdrone/` 已有 VisDrone 数据。RSOD 未纳入 git（`data/` 在 `.gitignore` 中），手动放置到 `data/yolo/rsod/`。

### 0.2 环境要求

```
Python >= 3.10
Git Bash / PowerShell
建议有 CUDA 12.x + PyTorch 2.x（非必须，CPU 也能开发调试）
```

### 0.3 安装（所有人第一步执行）

```bash
git clone https://github.com/wuwo1979/ODPlatform.git
cd ODPlatform
pip install -e ./apps/platform --no-build-isolation --no-deps
```

### 0.4 各模块所需的依赖（各角色自己添加到 pyproject.toml）

当前 `apps/platform/pyproject.toml` 中 `dependencies` 仅有：`colorlog`、`ultralytics`、`scikit-learn`、`pillow`、`pyyaml`。以下依赖**各角色在首次提交时自行添加**，一行一个追加到 `dependencies` 列表中：

```toml
# [训练] 已有 ultralytics，无需新增

# [推理] 新增
"numpy",
"opencv-python",

# [前端] 新增
"gradio>=5.0",
"matplotlib",

# [后端] 新增 — 注意后端是一个独立服务，放到 apps/web-backend/requirements.txt
# （不在 pyproject.toml 中声明，防止污染 platform 端依赖）
```

`apps/web-backend/requirements.txt`（后端工程师创建）：
```
fastapi>=0.115.0
uvicorn[standard]>=0.30.0
pydantic>=2.0
requests>=2.28
```

**规则**：谁新增依赖谁修改对应文件，PR 中在描述里标注原因。架构师审查时确认.

### 0.5 验证环境（每人克隆后执行，确认可正常 import）

```bash
# 验证现有基础设施
python -c "from odp_platform.common.paths import ROOT_DIR, LOGGING_DIR, CHECKPOINTS_DIR; print('paths OK')"
python -c "from odp_platform.common.logging_utils import get_logger; print('logging OK')"
python -c "from odp_platform.common.constants import Task, AnnotationFormat; print('constants OK')"

# 验证 CLI
odp-train --help
odp-validate --help
```

---

## 一、统一架构总览

```
                    ┌──────────────────────────────────────┐
                    │          UI (Gradio 5.x)              │
                    │        webui/app.py + 6 Tab           │
                    └────────┬──────────────┬──────────────┘
                             │ gr.Image     │ HTTP API
                             │ gr.JSON      │
                             ▼              ▼
              ┌──────────────────┐  ┌──────────────────────┐
              │  inference/      │  │  web-backend/         │
              │  Detector.detect │  │  FastAPI + SQLite     │
              │  visualizer.py   │  │  experiments/models   │
              └───────┬──────────┘  └──────────┬───────────┘
                      │                        │
                      │ best.pt                │ POST epoch
                      ▼                        ▼
              ┌──────────────────────────────────────────────┐
              │              training/                        │
              │  ExperimentConfig → run_experiment()          │
              │     → ExperimentResult + best.pt              │
              │  recipe.py    callbacks.py    tracker.py      │
              └──────┬───────────────────────────────────────┘
                     │
                     │ ExperimentConfig
                     ▼
              ┌──────────────────────────────────────────────┐
              │       training/experiments/                  │
              │  算法工程师的实验（small_object/dense_opt）    │
              └──────────────────────────────────────────────┘
```

**数据流向**：`ExperimentConfig → YOLO.train() → best.pt → Detector → UI`；同时 `→ POST /api/experiments → SQLite → GET /api → UI`

---

## 二、统一参数体系

### 2.1 核心参数（训练工程师定义，所有人 import）

```python
# training/experiment.py —— 训练工程师维护

from dataclasses import dataclass, asdict

@dataclass
class ExperimentConfig:
    name: str                          # 唯一标识 "rsod_yolo11n_640_001"
    dataset: str                       # "rsod" | "visdrone"
    model: str = "yolo11n.pt"
    task: str = "detect"
    epochs: int = 100
    batch: int = 16
    imgsz: int = 640
    lr0: float = 0.01
    device: str = ""                   # ""=auto, "0"=GPU0, "cpu"
    workers: int = 2
    optimizer: str = "auto"
    amp: bool = True
    patience: int = 50
    seed: int = 42
    note: str = ""

    def to_json(self) -> str:
        import json
        return json.dumps(asdict(self), ensure_ascii=False)

    def to_cli_args(self) -> dict:
        return {
            "dataset": self.dataset, "model": self.model, "task": self.task,
            "epochs": self.epochs, "batch": self.batch, "imgsz": self.imgsz,
            "lr0": self.lr0, "device": self.device, "workers": self.workers,
            "amp": self.amp, "name": self.name,
        }


@dataclass
class ExperimentResult:
    name: str
    dataset: str
    model: str
    imgsz: int
    epochs_run: int
    best_epoch: int
    map50: float
    map50_95: float
    precision: float
    recall: float
    train_duration_sec: float
    model_path: str
    config_snapshot_path: str

    def to_dict(self) -> dict:
        return asdict(self)
```

### 2.2 推理参数（推理工程师定义）

```python
# inference/engine.py —— 推理工程师维护

from dataclasses import dataclass

@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]  # (x1,y1,x2,y2) 归一化 0~1

@dataclass
class InferenceResult:
    detections: list[Detection]
    inference_ms: float
    input_shape: tuple[int, int]  # (h, w)
```

### 2.3 后端 API 参数（后端工程师定义，字段必须对齐上面）

```python
# web-backend/schemas.py —— 后端工程师维护

from pydantic import BaseModel
from typing import Optional

class ExperimentCreate(BaseModel):
    name: str               # 对齐 ExperimentConfig.name
    dataset: str            # 对齐 ExperimentConfig.dataset
    model: str
    task: str = "detect"
    config_json: str        # ExperimentConfig.to_json()

class ExperimentUpdate(BaseModel):
    status: Optional[str] = None
    best_map50: Optional[float] = None      # 对齐 ExperimentResult.map50
    best_map50_95: Optional[float] = None
    best_epoch: Optional[int] = None
    model_path: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None

class EpochData(BaseModel):
    epoch: int
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    map50: Optional[float] = None
    map50_95: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    lr: Optional[float] = None
```

**对齐原则**：`ExperimentConfig` 是参数源头，`ExperimentCreate` 的字段名和语义必须一致。新增字段时训练工程师先改 → 通知后端同步 → 前端同步 DataFrame 列。

---

## 三、统一目录结构

### 3.1 目录树（★ = 本次新建，无标记 = 已存在只读）

```
ODPlatform/
├── apps/
│   ├── platform/
│   │   ├── src/odp_platform/
│   │   │   ├── common/              # [已有] paths / constants / logging_utils
│   │   │   ├── data_pipeline/       # [已有]
│   │   │   ├── data_validation/     # [已有]
│   │   │   ├── run_config/          # [已有]
│   │   │   ├── cli/                 # [已有] + ★ infer.py
│   │   │   ├── training/            # ★ [训练]
│   │   │   │   ├── __init__.py
│   │   │   │   ├── experiment.py    # ExperimentConfig + run_experiment()
│   │   │   │   ├── recipe.py        # 预设配方
│   │   │   │   ├── tracker.py       # 指标采集
│   │   │   │   ├── callbacks.py     # 训练回调
│   │   │   │   └── experiments/     # ★ [算法] 实验代码
│   │   │   │       ├── small_object/
│   │   │   │       └── dense_optimization/
│   │   │   ├── inference/           # ★ [推理]
│   │   │   │   ├── __init__.py
│   │   │   │   ├── engine.py        # Detection / Detector
│   │   │   │   ├── visualizer.py
│   │   │   │   ├── sources.py
│   │   │   │   ├── benchmark.py
│   │   │   │   └── utils.py
│   │   │   └── webui/               # ★ [前端]
│   │   │       ├── __init__.py
│   │   │       ├── app.py           # 主入口 + Gr.Launch
│   │   │       ├── user_tabs.py     # 用户 Tab：检测/摄像头/模型/LLM + 实验可视化
│   │   │       ├── dashboard.py
│   │   │       ├── dataset_browser.py
│   │   │       ├── dataset_analysis.py
│   │   │       ├── training_tab.py
│   │   │       ├── model_demo.py
│   │   │       ├── validation_tab.py
│   │   │       ├── config_tab.py
│   │   │       └── utils.py
│   │   └── pyproject.toml           # [已有] 各角色加 dependency
│   │
│   └── web-backend/                 # ★ [后端]
│       ├── main.py
│       ├── requirements.txt
│       ├── db/
│       │   ├── database.py
│       │   ├── models.py
│       │   └── init_db.py
│       ├── api/
│       │   ├── experiments.py
│       │   └── models.py
│       ├── hooks.py
│       └── schemas.py
│
├── data/
│   ├── models/
│   │   ├── pretrained/              # [已有] 预训练权重
│   │   └── checkpoints/             # [已有] 训练产出 best.pt
│   └── runs/
│       ├── experiments/             # ★ [训练] 实验输出
│       ├── data_validation/         # [已有]
│       └── run_config/              # [已有]
│
├── docs/
│   ├── experiments/                 # ★ [算法]
│   └── results/                     # ★ [训练]
│
└── .github/workflows/               # [架构师]
```

### 3.2 写权限矩阵

| 路径 | 训练 | 算法 | 前端 | 后端 | 推理 |
|---|---|---|---|---|---|
| `training/experiment.py` | **写** | 读 | — | 读 | — |
| `training/recipe.py` | **写** | 读 | — | — | — |
| `training/tracker.py` | **写** | 读 | 读 | — | 读 |
| `training/callbacks.py` | **写** | — | — | — | — |
| `training/experiments/` | — | **写** | — | — | — |
| `inference/` | — | — | 读 | — | **写** |
| `webui/` | — | — | **写** | — | — |
| `cli/infer.py` | — | — | — | — | **写** |
| `apps/web-backend/` | — | — | — | **写** | — |
| `docs/experiments/` | — | **写** | — | — | — |
| `docs/results/` | **写** | 读 | 读 | — | — |
| `data/models/` | **写** | 读 | — | — | 读 |
| `data/runs/experiments/` | **写** | 读 | 读 | — | — |

> **写** = 创建/修改，**读** = 只能 import 或读取，**—** = 无关。跨越边界需 PR 中说明。

---

## 四、统一日志规范

所有模块使用 `odp_platform.common.logging_utils.get_logger()`，不直接用 `print()` 或自己配 handler。

```python
# 每个模块顶部统一写法（已存在的 common/logging_utils.py 提供此函数）
from odp_platform.common.logging_utils import get_logger
from odp_platform.common.paths import LOGGING_DIR
import logging

logger = get_logger(
    base_path=LOGGING_DIR,           # → apps/platform/logging/
    log_type="train",                # 决定文件名 train_YYYYMMDD.log
    log_level=logging.INFO,
    logger_name="odp-train",
)
```

日志文件统一落在 `apps/platform/logging/`，命名格式 `<log_type>_<YYYYMMDD>.log`。

| 模块 | log_type | logger_name |
|---|---|---|
| 训练/实验 | `train` | `odp-train` |
| 推理 | `infer` | `odp-infer` |
| 后端 API | `backend` | `odp-backend` |
| 前端 UI | `webui` | `odp-webui` |

**算法工程师**不单独建日志体系，实验结果通过 `callbacks.py` 的回调输出到训练日志中。

---

## 五、统一接口契约

### 5.1 训练 → 推理

```
训练产出: data/models/checkpoints/best_<exp_name>.pt
推理消费: Detector(ExperimentResult.model_path)

约定:
  - ExperimentResult.model_path 存绝对路径，推理直接读
  - 推理只读 models/ 目录，不写入
```

### 5.2 训练 → 后端

```
训练 callbacks.py 调用 hooks.py:
  on_training_start(exp_name, config_json) → POST /api/experiments → 返回 exp_id
  on_epoch_end(exp_id, epoch, metrics)     → POST /api/experiments/{id}/epochs
  on_training_end(exp_id, map50, model_path) → PATCH /api/experiments/{id}/status

metrics 字典格式（o=overall）:
  {"train_loss": 2.3, "val_loss": 1.8, "map50": 0.872, "map50_95": 0.651,
   "precision": 0.91, "recall": 0.85, "lr": 0.008}
```

### 5.3 后端 → 前端

```
GET /api/experiments?dataset=rsod&limit=20 → gr.DataFrame
GET /api/experiments/{id}/epochs → gr.LinePlot

前端不直接读 data/runs/，全部走 API。
Dashboard 初始化时可临时扫描 data/runs/experiments/ 做快速概览。
```

### 5.4 推理 → 前端

```python
from odp_platform.inference.engine import Detector
from odp_platform.inference.visualizer import draw_detections

detector = Detector(model_path, conf=0.25, iou=0.45)
result = detector.detect(image_np)           # → InferenceResult

# gr.JSON: [{"class":"car","conf":0.92,"bbox":[0.1,0.2,0.3,0.4]}, ...]
# gr.Image: draw_detections(image_np, result.detections)
```

### 5.5 算法 → 训练

算法工程师不直接调 `ultralytics`，通过训练模块统一入口：

```python
from odp_platform.training.experiment import ExperimentConfig, run_experiment

config = ExperimentConfig(
    name="rsod_cbam_001", dataset="rsod",
    model="training/experiments/small_object/yolo11n_p2.yaml",
    epochs=100, note="CBAM 消融实验",
)
result = run_experiment(config)
print(f"mAP50={result.map50:.3f}  model={result.model_path}")
```

---

## 六、各角色详细任务

---

### 6.1 模型训练工程师

**任务**：搭建实验工厂——输入 `ExperimentConfig`，输出 `ExperimentResult` + `best.pt`，支持单次训练和批量训练。

**交付文件**：

| 文件 | 职责 |
|---|---|
| `training/__init__.py` | 空或 re-export 关键符号 |
| `training/experiment.py` | `ExperimentConfig` / `ExperimentResult` + `run_experiment()` + `run_batch()` |
| `training/recipe.py` | `RSOD_BASELINE` / `VISDRONE_BASELINE` / `LR_SWEEP` 等预设列表 |
| `training/tracker.py` | `collect_results(dataset)` 扫描 `experiments/` 生成 `docs/results/comparison_*.csv` |
| `training/callbacks.py` | `TrainingHooks` 类：`on_train_start` / `on_epoch_end` / `on_train_end` |

**骨架代码 — `training/experiment.py` 核心实现思路**：

```python
# training/experiment.py
from dataclasses import dataclass, asdict
import json, subprocess, time, csv
from pathlib import Path
from odp_platform.common.paths import CHECKPOINTS_DIR, RUNS_DIR, DOCS_DIR
from odp_platform.common.logging_utils import get_logger
from odp_platform.common.paths import LOGGING_DIR

logger = get_logger(base_path=LOGGING_DIR, log_type="train", logger_name="odp-train")

@dataclass
class ExperimentConfig:
    # ... 字段定义见第二章 2.1 ...

def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    """核心入口：接收配置，执行训练，返回结果。"""

    # 1. 构造 odptrain 命令行（利用已有的 CLI）
    exp_dir = RUNS_DIR / "experiments" / config.name
    exp_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    cmd = [
        "odp-train", "-d", config.dataset,
        "--epochs", str(config.epochs),
        "--batch", str(config.batch),
        "--imgsz", str(config.imgsz),
        "--model", config.model,
        "--device", config.device or "auto",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        logger.error(f"训练失败 (exit={proc.returncode}):\n{proc.stderr}")
        raise RuntimeError(f"odp-train 返回非零退出码 {proc.returncode}")

    # 2. 从 results.csv 解析最终指标
    results_csv = exp_dir / "results.csv"
    metrics = _parse_metrics(results_csv) if results_csv.exists() else {}

    duration = time.time() - t0

    # 3. 复制 best.pt 到 checkpoints/（用 config.name 保证不同实验不互相覆盖）
    best_pt = exp_dir / "weights" / "best.pt"
    checkpoint_name = f"best_{config.name}.pt"
    checkpoint = CHECKPOINTS_DIR / checkpoint_name
    if best_pt.exists():
        import shutil; shutil.copy(best_pt, checkpoint)
    else:
        logger.warning(f"best.pt 未找到: {best_pt}，跳过复制")

    # 4. 保存配置快照
    snapshot_path = exp_dir / "config_snapshot.json"
    snapshot_path.write_text(config.to_json(), encoding="utf-8")

    return ExperimentResult(
        name=config.name, dataset=config.dataset, model=config.model,
        imgsz=config.imgsz, epochs_run=metrics.get("epoch", config.epochs),
        best_epoch=metrics.get("best_epoch", 0),
        map50=metrics.get("map50", 0), map50_95=metrics.get("map50_95", 0),
        precision=metrics.get("precision", 0), recall=metrics.get("recall", 0),
        train_duration_sec=duration,
        model_path=str(checkpoint),
        config_snapshot_path=str(snapshot_path),
    )


def _parse_metrics(csv_path: Path) -> dict:
    """解析 ultralytics 输出的 results.csv 最后一行。"""
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return {}
    last = rows[-1]
    return {k.strip(): float(v.strip()) for k, v in last.items()}
```

**`training/callbacks.py` 调用后端 hooks**：

```python
# training/callbacks.py
import time
from typing import Optional
import requests
from odp_platform.common.logging_utils import get_logger
from odp_platform.common.paths import LOGGING_DIR

logger = get_logger(base_path=LOGGING_DIR, log_type="train", logger_name="odp-train")

BACKEND_URL = "http://127.0.0.1:8000"
MAX_RETRIES = 3
RETRY_DELAY = 1.0  # 秒


def _post_with_retry(url: str, json_data: dict, label: str = "") -> Optional[dict]:
    """带重试的 POST，后端暂不可用时记录告警。"""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.post(url, json=json_data, timeout=5)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.RequestException as e:
            logger.warning(
                f"[{label}] 后端通信失败 (尝试 {attempt}/{MAX_RETRIES}): {e}"
            )
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
    logger.error(f"[{label}] 重试耗尽，放弃本次同步")
    return None


class TrainingHooks:
    """训练回调，负责与后端通信。后端未启动时记录告警但不中断训练。"""

    def __init__(self, experiment_name: str, config_json: str):
        self.exp_name = experiment_name
        self.config_json = config_json
        self.exp_id: Optional[int] = None

    def on_train_start(self, dataset: str, model: str):
        result = _post_with_retry(
            f"{BACKEND_URL}/api/experiments",
            {"name": self.exp_name, "config_json": self.config_json,
             "dataset": dataset, "model": model},
            label="train_start",
        )
        if result:
            self.exp_id = result["id"]
            logger.info(f"实验注册成功: id={self.exp_id}")

    def on_epoch_end(self, epoch: int, metrics: dict):
        if self.exp_id is None:
            return
        _post_with_retry(
            f"{BACKEND_URL}/api/experiments/{self.exp_id}/epochs",
            {"epoch": epoch, **metrics},
            label=f"epoch_{epoch}",
        )

    def on_train_end(self, map50: float, model_path: str):
        if self.exp_id is None:
            return
        result = _post_with_retry(
            f"{BACKEND_URL}/api/experiments/{self.exp_id}/status",
            {"status": "completed", "best_map50": map50, "model_path": model_path},
            label="train_end",
        )
        if result:
            logger.info(f"实验完成同步: id={self.exp_id}")
```

**集成到 `run_experiment()` 中**：

```python
# 在 run_experiment() 的训练循环中：
hooks = TrainingHooks(config.name, config.to_json())
hooks.on_train_start()
for epoch in range(config.epochs):
    # YOLO.train()...
    hooks.on_epoch_end(epoch, metrics)
hooks.on_train_end(result.map50, str(checkpoint))
```

**验收**：

```bash
python -c "
from odp_platform.training.experiment import ExperimentConfig, run_experiment
config = ExperimentConfig(name='test_nano', dataset='rsod',
    model='yolo11n.pt', epochs=1, batch=1)
result = run_experiment(config)
print(f'OK: {result.to_dict()}')
"
```

**提交范围**：`apps/platform/src/odp_platform/training/` 全部（**除** `experiments/` 子目录）

---

### 6.2 算法优化工程师

**任务**：利用 `ExperimentConfig` + `run_experiment()` 在两个数据集上做单变量改进实验，产出消融报告和最优配置。

#### RSOD —— 小目标检测

遥感场景核心矛盾：目标像素极少（~20×20），默认 P3/64 检测头感受野过大。

| 实验 | 改动 | Config |
|---|---|---|
| A-baseline | 不改 | `model="yolo11n.pt", imgsz=640` |
| A+P2 | 加 P2 检测头（4× 下采样），搭配 P2 专用 anchor | `model="experiments/small_object/yolo11n_p2.yaml"` |
| A+HR | 高分辨率 1280 | `imgsz=1280, batch=8` |
| A+CBAM | Neck 中嵌入 CBAM 通道-空间注意力 | 传自定义模型 cfg |
| A+P2+HR+CBAM | 最优组合 | 组合以上参数 |

**`yolo11n_p2.yaml` 骨架**：

```yaml
# training/experiments/small_object/yolo11n_p2.yaml
# 在 YOLO11n 基础上添加 P2 检测头
nc: 4              # RSOD 类别数，从 data.yaml 读取
scales:
  n: [0.33, 0.25, 1024]

head:
  - [-1, 1, Conv, [64, 3, 2]]                    # P2/4
  - [[-1, 6], 1, Concat, [1]]                     # 融合 P2 + 原 P3
  - [-1, 3, C2f, [256]]                           # P2 处理

# 其余沿用 yolo11n.yaml 结构，在 detect 段加 [15, 19, 23, 27] 四层输出
```

#### VisDrone —— 密集 + 长尾

无人机俯拍：单图百目标，NMS 过度抑制；pedestrian:ignored_region ≈ 80:2，极端长尾。

| 实验 | 改动 | Config |
|---|---|---|
| B-baseline | 不改 | `model="yolo11s.pt", patience=50` |
| B+SoftNMS | 高斯衰减替代硬删除，IoU 阈值 0.5 | 替换 `ultralytics/utils/ops.py` 中 NMS |
| B+FocalLoss | Focal Loss γ=2 缓解类别不平衡 | 修改 `loss.py` 中分类损失项 |
| B+SIoU | SIoU 损失替换 CIoU | 修改 `bbox_iou()` |
| B+Best | SoftNMS + Focal + SIoU 组合 | 组合以上 |

**实验规范**：

1. 每个实验 `seed=42`，单变量改动
2. 通过 `run_experiment()` 执行，不直接调 ultralytics
3. 结果写入统一格式：

```markdown
| 配置 | mAP50 | mAP50-95 | mAP_small (rsod) / mAP50_ped (visdrone) | params(M) | FPS | model_path |
|---|---|---|---|---|---|---|
| baseline | 0.872 | 0.651 | 0.423 | 2.6 | 120 | checkpoints/best_rsod_yolo11n_640.pt |
| +CBAM   | 0.891 | 0.672 | 0.451 | 2.9 | 115 | checkpoints/best_rsod_cbam_640.pt |
| 提升     | +0.019 | +0.021 | +0.028 | +0.3M | -5 | — |
```

**最终交付**：`docs/experiments/ablation_report.md` + `docs/experiments/best_config.yaml`（可直接 `odp-train -c best_config.yaml`）

**提交范围**：`training/experiments/` 下 `.py` / `.yaml` + `docs/experiments/` 下 `.md` / `.yaml`。**不碰** `experiment.py` / `recipe.py` / `tracker.py`。

---

### 6.3 UI & 前端工程师

**任务**：Gradio 5.x 搭建 6 Tab，调用训练/推理模块和后端 API，把命令行操作图形化。

**骨架代码 — `webui/app.py`**：

```python
# webui/app.py
import gradio as gr
from odp_platform.webui.user_tabs import (
    create_single_detection_ui, create_folder_detection_ui,
    create_video_detection_ui, create_live_camera_ui,
    create_model_selection_ui, create_llm_chat_ui,
)
from odp_platform.webui.dashboard import create_dashboard_ui
from odp_platform.webui.dataset_browser import create_dataset_browser_ui
from odp_platform.webui.training_tab import create_training_ui
from odp_platform.webui.model_demo import create_model_demo_ui
from odp_platform.webui.validation_tab import create_validation_ui
from odp_platform.webui.config_tab import create_config_ui
```

**双模式 Tab 架构**：

用户模式（`_create_user_tabs()`）：

| Tab | 文件 | 关键调用 | 依赖方 |
|---|---|---|---|
| 单图检测 | `user_tabs.py` | `Detector`, `draw_detections` | 推理模块 |
| 文件夹检测 | `user_tabs.py` | `list_images`, `Detector` | 推理模块 |
| 视频检测 | `user_tabs.py` | OpenCV 逐帧推理 | 推理模块 |
| 实时摄像头 | `user_tabs.py` | OpenCV `VideoCapture`, 多后端 MSMF/DSHOW | 推理模块 |
| 模型选择 | `user_tabs.py` | `list_model_files()` 模型扫描 + 上传 + 实验可视化 | 无 |
| LLM对话 | `user_tabs.py` | DeepSeek API（urllib） | 无 |

管理员模式额外 Tab：

| Tab | 文件 | 关键调用 | 依赖方 |
|---|---|---|---|
| Dashboard | `dashboard.py` | 后端 API | 后端服务 |
| 模型演示 | `model_demo.py` | `Detector`, `draw_detections` | 推理模块 |
| 数据集浏览 | `dataset_browser.py` | 读 YAML + YOLO labels | 无 |
| 训练 | `training_tab.py` | `subprocess: odp-train` | CLI |
| 数据校验 | `validation_tab.py` | `subprocess: odp-validate` | CLI |
| 配置管理 | `config_tab.py` | `subprocess: odp-config` | CLI |

**`model_demo.py` 核心交互**：

```python
# webui/model_demo.py
import gradio as gr
import numpy as np
from PIL import Image
from pathlib import Path
from odp_platform.common.paths import CHECKPOINTS_DIR
from odp_platform.inference.engine import Detector
from odp_platform.inference.visualizer import draw_detections

def _scan_models() -> list[str]:
    return [str(p) for p in CHECKPOINTS_DIR.glob("*.pt")]

def create_model_demo_ui():
    detector_state = gr.State(None)

    def load_model(path):
        detector_state.value = Detector(path)
        return f"已加载: {Path(path).name}"

    def run_inference(image, conf, iou):
        det = detector_state.value
        if det is None:
            return None, "请先选择模型", gr.update()
        det.conf = conf; det.iou = iou
        result = det.detect(np.array(image))
        vis = draw_detections(np.array(image), result.detections)
        return vis, [{"class": d.class_name, "conf": round(d.confidence, 3),
                      "bbox": d.bbox} for d in result.detections]

    with gr.Column():
        model_dd = gr.Dropdown(label="模型", choices=_scan_models(), interactive=True)
        status = gr.Textbox(label="状态", value="未加载", interactive=False)
        conf_slider = gr.Slider(0.01, 0.99, 0.25, label="Confidence")
        iou_slider = gr.Slider(0.01, 0.99, 0.45, label="IoU")
        image_in = gr.Image(type="pil", label="上传图片")
        btn = gr.Button("推理", variant="primary", interactive=False)
        image_out = gr.Image(label="检测结果")
        json_out = gr.JSON(label="检测列表")

    # 仅当模型已选择且有效时启用推理按钮
    model_dd.change(
        fn=load_model, inputs=[model_dd], outputs=[status]
    ).then(
        fn=lambda: gr.update(interactive=True), outputs=[btn]
    )

    btn.click(run_inference, [image_in, conf_slider, iou_slider],
              [image_out, json_out, status])
```

**数据集浏览 — YOLO 标注绘制**：

```python
# webui/dataset_browser.py 中标签叠加逻辑
def draw_yolo_boxes(image: np.ndarray, label_path: str, names: dict[int, str],
                    colors: dict[int, tuple]) -> np.ndarray:
    """读 YOLO label 文件，在图片上绘制检测框。
    注意：需处理标签文件缺失、格式错误、越界值等情况。"""
    from pathlib import Path
    h, w = image.shape[:2]
    lp = Path(label_path)
    if not lp.exists():
        return image  # 标签文件缺失，返回原图

    with open(lp) as f:
        for line_no, line in enumerate(f, 1):
            parts = line.strip().split()
            if len(parts) < 5:
                continue  # 跳过格式错误的行
            try:
                cls_id, cx, cy, bw, bh = map(float, parts[:5])
            except ValueError:
                continue  # 跳过无法解析的行

            # 裁剪越界值到 [0, 1] 范围
            cx, cy = max(0, min(1, cx)), max(0, min(1, cy))
            bw, bh = max(0, min(1, bw)), max(0, min(1, bh))

            x1 = int((cx - bw / 2) * w); y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w); y2 = int((cy + bh / 2) * h)
            color = colors.get(int(cls_id), (0, 255, 0))
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, names.get(int(cls_id), "?"), (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return image
```

**验收**：

```bash
python apps/platform/src/odp_platform/webui/app.py
# 浏览器打开 http://127.0.0.1:7860，能看到 6 个 Tab
```

**提交范围**：`apps/platform/src/odp_platform/webui/` 全部 + `pyproject.toml` 加 `gradio>=5.0` 和 `matplotlib`

---

### 6.4 数据库 & 后端接口工程师

**任务**：FastAPI + SQLite 搭建实验数据存储与查询服务，提供训练回调钩子。

**数据库表（4 张）**：

```sql
-- db/init_db.py 中执行

CREATE TABLE experiments (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT NOT NULL UNIQUE,
    dataset         TEXT NOT NULL,       -- 对齐 ExperimentConfig.dataset
    model           TEXT NOT NULL,
    task            TEXT DEFAULT 'detect',
    config_json     TEXT NOT NULL,       -- ExperimentConfig.to_json()
    status          TEXT DEFAULT 'pending',
    best_map50      REAL,
    best_map50_95   REAL,
    best_epoch      INTEGER,
    start_time      TEXT,
    end_time        TEXT,
    model_path      TEXT,
    created_at      TEXT DEFAULT (datetime('now'))
);

CREATE TABLE training_epochs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id   INTEGER REFERENCES experiments(id),
    epoch           INTEGER NOT NULL,
    train_loss      REAL, val_loss      REAL,
    map50           REAL, map50_95      REAL,
    precision       REAL, recall        REAL,
    lr              REAL,
    UNIQUE(experiment_id, epoch)
);

CREATE TABLE models (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id   INTEGER REFERENCES experiments(id),
    filename        TEXT NOT NULL,
    format          TEXT DEFAULT 'pt',
    map50           REAL, map50_95      REAL,
    file_size_mb    REAL,
    created_at      TEXT DEFAULT (datetime('now'))
);

CREATE TABLE validation_reports (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset         TEXT NOT NULL,
    run_id          TEXT NOT NULL,
    passed          INTEGER DEFAULT 0,
    warnings        INTEGER DEFAULT 0,
    errors          INTEGER DEFAULT 0,
    report_json     TEXT,
    created_at      TEXT DEFAULT (datetime('now'))
);
```

**骨架代码 — `main.py`**：

```python
# web-backend/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .db.database import init_db
from .api import experiments, models

app = FastAPI(title="ODPlatform API", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(experiments.router, prefix="/api")
app.include_router(models.router, prefix="/api")

@app.on_event("startup")
def startup():
    init_db()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
```

**API 端点**：

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/experiments` | 创建实验 → 返回 `{"id": 1}` |
| PATCH | `/api/experiments/{id}/status` | 更新 running/completed/failed + 最终指标 |
| POST | `/api/experiments/{id}/epochs` | 写入单 epoch 数据 |
| GET | `/api/experiments?dataset=&status=&limit=20` | 查询列表 |
| GET | `/api/experiments/{id}/epochs` | 查询训练曲线 |
| GET | `/api/validation/reports?dataset=&limit=20` | 查询质检历史 |

**`api/experiments.py` POST 示例**：

```python
# web-backend/api/experiments.py
from fastapi import APIRouter, HTTPException
from ..schemas import ExperimentCreate, ExperimentUpdate, EpochData
from ..db.database import get_db
import sqlite3

router = APIRouter()

@router.post("/experiments")
def create_experiment(data: ExperimentCreate):
    db = get_db()
    try:
        cursor = db.execute(
            "INSERT INTO experiments (name, dataset, model, task, config_json) VALUES (?,?,?,?,?)",
            (data.name, data.dataset, data.model, data.task, data.config_json)
        )
        db.commit()
        return {"id": cursor.lastrowid, "name": data.name}
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail=f"实验 '{data.name}' 已存在")
```

**额外的 API 端点（models 和 validation_reports 表对应的接口）**：

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/models` | 注册模型 → `{"experiment_id": 1, "filename": "best_xxx.pt", "map50": 0.891}` |
| GET | `/api/models?experiment_id=1` | 查询某实验的所有模型 |
| GET | `/api/validation/reports?dataset=&limit=20` | 查询质检报告历史 |
| POST | `/api/validation/reports` | 写入新的质检报告 |

**`hooks.py` — 训练侧的客户端 SDK**（后端维护，训练工程师只是 `from hooks import TrainingHooks`）：

```python
# web-backend/hooks.py —— 后端工程师维护
# 放在 web-backend/ 下供训练模块通过 PYTHONPATH 引用

import requests

BASE = "http://127.0.0.1:8000"

def on_training_start(name: str, config_json: str, dataset: str, model: str) -> int:
    r = requests.post(f"{BASE}/api/experiments", json={
        "name": name, "config_json": config_json,
        "dataset": dataset, "model": model,
    })
    r.raise_for_status()
    return r.json()["id"]

def on_epoch_end(experiment_id: int, epoch: int, metrics: dict):
    requests.post(f"{BASE}/api/experiments/{experiment_id}/epochs", json={
        "epoch": epoch, **metrics
    })

def on_training_end(experiment_id: int, map50: float, model_path: str):
    requests.patch(f"{BASE}/api/experiments/{experiment_id}/status", json={
        "status": "completed", "best_map50": map50, "model_path": model_path
    })
```

**与训练的协同**：训练工程师在 `callbacks.py` 中 `from hooks import TrainingHooks` 封装。双方约定 `hooks.py` 的函数签名不变，后端负责 HTTP 细节，训练侧不关心 API 路径。

**验收**：

```bash
cd apps/web-backend
pip install -r requirements.txt
python main.py &
# 另一个终端:
curl -X POST http://127.0.0.1:8000/api/experiments \
  -H "Content-Type: application/json" \
  -d '{"name":"test","dataset":"rsod","model":"yolo11n.pt","config_json":"{}"}'
# 返回 {"id":1,"name":"test"}
```

**提交范围**：`apps/web-backend/` 全部（**不碰** `apps/platform/` 下任何文件）

---

### 6.5 推理与可视化工程师

**任务**：封装统一的 `Detector` 推理类 + 可视化渲染，供前端和 CLI 共用。

**骨架代码 — `inference/engine.py`**：

```python
# inference/engine.py
from dataclasses import dataclass
import numpy as np
from ultralytics import YOLO

@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]

@dataclass
class InferenceResult:
    detections: list[Detection]
    inference_ms: float
    input_shape: tuple[int, int]

class Detector:
    """统一推理入口。前端/CLI/Benchmark 共用一个类。"""

    def __init__(self, model_path: str, device: str = "auto",
                 conf: float = 0.25, iou: float = 0.45):
        from pathlib import Path
        p = Path(model_path)
        if not p.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        if not p.suffix == ".pt":
            raise ValueError(f"仅支持 .pt 格式，收到: {p.suffix}")
        self.model_path = str(p.resolve())
        self.model = YOLO(self.model_path)
        self.device = device
        self.conf = conf
        self.iou = iou

    def detect(self, image: np.ndarray) -> InferenceResult:
        import time
        t0 = time.perf_counter()
        results = self.model(image, device=self.device,
                             conf=self.conf, iou=self.iou, verbose=False)
        elapsed = (time.perf_counter() - t0) * 1000

        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for cls_id, conf_val, xyxy in zip(
                boxes.cls.cpu().numpy(),
                boxes.conf.cpu().numpy(),
                boxes.xyxyn.cpu().numpy(),
            ):
                detections.append(Detection(
                    class_id=int(cls_id),
                    class_name=r.names[int(cls_id)],
                    confidence=float(conf_val),
                    bbox=tuple(xyxy.tolist()),
                ))

        return InferenceResult(
            detections=detections,
            inference_ms=elapsed,
            input_shape=image.shape[:2],
        )

    def warmup(self, image_size=(640, 640)):
        """GPU JIT 预热，消除首次推理的 CUDA kernel 编译延迟。
        注意：仅在 CUDA 设备上有效，CPU 推理跳过。"""
        dummy = np.zeros((*image_size, 3), dtype=np.uint8)
        result = self.detect(dummy)
        if not result.detections:
            logger.debug("warmup 完成，无检测结果（预期行为）")
        elif len(result.detections) > 0:
            logger.warning(f"warmup 在纯黑图上检测到 {len(result.detections)} 个目标，请检查 conf 阈值")
```

**骨架代码 — `inference/visualizer.py`**：

```python
# inference/visualizer.py
import numpy as np
import cv2
from .engine import Detection

CLASS_COLORS: dict[int, tuple[int, int, int]] = {
    0: (0, 255, 0),   1: (255, 0, 0),   2: (0, 0, 255),
    3: (255, 255, 0), 4: (255, 0, 255), 5: (0, 255, 255),
    6: (128, 128, 0), 7: (128, 0, 128), 8: (0, 128, 128),
    9: (128, 128, 128),
}

def draw_detections(
    image: np.ndarray,
    detections: list[Detection],
    line_width: int = 2,
    font_scale: float = 0.6,
) -> np.ndarray:
    """在图片上绘制检测框和置信度标签。"""
    vis = image.copy()
    h, w = image.shape[:2]
    for d in detections:
        color = CLASS_COLORS.get(d.class_id, (0, 255, 0))
        x1, y1, x2, y2 = [int(v * w) if i % 2 == 0 else int(v * h)
                           for i, v in enumerate(d.bbox)]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, line_width)
        label = f"{d.class_name} {d.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(vis, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(vis, label, (x1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)
    return vis
```

**CLI — `cli/infer.py`**：

```python
# cli/infer.py — 注册为 odp-infer 命令
import argparse
from pathlib import Path
from odp_platform.inference.engine import Detector
from odp_platform.inference.visualizer import draw_detections
from odp_platform.inference.sources import infer_image
import cv2

def main():
    parser = argparse.ArgumentParser(prog="odp-infer")
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", default="output.jpg")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    detector = Detector(args.model, device=args.device, conf=args.conf, iou=args.iou)
    detector.warmup()

    image = cv2.imread(args.source)
    result = detector.detect(image)
    vis = draw_detections(image, result.detections)
    cv2.imwrite(args.output, vis)
    print(f"检测完成: {len(result.detections)} 个目标, {result.inference_ms:.1f}ms → {args.output}")

if __name__ == "__main__":
    main()
```

**注册到 pyproject.toml**：推理工程师在 `[project.scripts]` 段新增一行：

```toml
odp-infer = "odp_platform.cli.infer:main"
```

**验收**：

```bash
python -c "
from odp_platform.inference.engine import Detector, Detection, InferenceResult
print('import OK')
"
odp-infer --model data/models/checkpoints/best_rsod.pt --source test.jpg --output result.jpg
```

**提交范围**：`odp_platform/inference/` 全部 + `odp_platform/cli/infer.py` + `pyproject.toml` 加依赖和 entry-point

---

### 6.6 跨模块容错规范（全员遵守）

跨模块调用是协同出错的最高发区。以下规则所有人必须遵守：

**A. 调用外部模块前，先校验入参合法性**

```python
# 反例：信任外部传入的路径
model = YOLO(user_supplied_path)  # 崩溃

# 正例：入口处集中校验
def __init__(self, model_path: str, ...):
    p = Path(model_path)
    if not p.exists():
        raise FileNotFoundError(...)
    if not p.suffix == ".pt":
        raise ValueError(...)
```

**B. 网络/进程调用必须有重试和日志**

```python
# 反例：静默吞错
try: requests.post(...)
except: pass

# 正例：重试 + 告警
for attempt in range(MAX_RETRIES):
    try:
        r = requests.post(url, timeout=5)
        r.raise_for_status()
        break
    except RequestException as e:
        logger.warning(f"第{attempt}次失败: {e}")
else:
    logger.error("重试耗尽")
```

**C. 子进程必须检查退出码**

```python
proc = subprocess.run(cmd, capture_output=True, text=True)
if proc.returncode != 0:
    logger.error(f"命令失败 (exit={proc.returncode}):\n{proc.stderr}")
    raise RuntimeError(...)
```

**D. 文件 I/O 必须处理缺失和脏数据**

- 读外部文件（labels、configs）时：检查存在性、格式合法性、值域裁剪
- 写文件时：避免硬编码文件名导致覆盖（用 `config.name` 等唯一标识）

**E. 数据库写操作必须处理约束冲突**

- SQLite `UNIQUE` 冲突 → 返回 409
- INSERT/UPDATE 异常 → log 错误 + 返回友好信息，不裸抛堆栈

**F. UI 按钮/控件必须有状态校验**

- 依赖外部模块的控件（如模型加载后方可推理）——用 `gr.update(interactive=...)` 前置校验
- 操作失败时输出可读文字而非空白/崩溃

---

## 七、协同检查清单

架构师审查 PR 时对照：

| # | 检查项 | 涉及角色 |
|---|---|---|
| 1 | `ExperimentConfig` 新增字段后，后端 `schemas.py`、前端 `gr.DataFrame` 列同步了吗？ | 训练→后端→前端 |
| 2 | `Detection` 字段变更后，前端 `gr.JSON` 展示格式同步了吗？ | 推理→前端 |
| 3 | 新代码是否 import 自 `common/paths.py` 而非手拼路径？ | 全员 |
| 4 | 新代码是否 import 自 `common/constants.py` 而非硬编码字面量？ | 全员 |
| 5 | 新模块日志是否用 `get_logger()` 而非 `print()`？ | 全员 |
| 6 | 是否修改了 `common/` `data_pipeline/` `data_validation/` `run_config/` 下的文件？ | 全员禁止 |
| 7 | 越界写入：是否在他人写权限目录下新增文件？ | 全员按矩阵 |
| 8 | 新增依赖是否在 `pyproject.toml` 或 `requirements.txt` 声明？ | 后端/推理/前端 |
| 9 | 算法实验代码是否通过 `run_experiment()` 而非直接调 ultralytics？ | 算法 |
| 10 | `hooks.py` 的函数签名变更后，训练侧 `callbacks.py` 调参是否同步？ | 后端→训练 |
| 11 | 跨模块 HTTP/子进程调用：是否有重试 + `logger.warning` 而非静默 `pass`？ | 训练→后端 |
| 12 | 外部输入（模型路径、标签文件、数据库记录）是否做了存在性/合法性校验？ | 全员 |
| 13 | 文件写入是否使用唯一标识（`config.name`）作为路径避免覆盖？ | 训练/推理 |

---

## 八、Git 分支与阶段

```
main（保护，架构师审查合并）
  ├── feat/training     ← [训练]
  ├── feat/algorithm    ← [算法]
  ├── feat/frontend     ← [前端]
  ├── feat/backend      ← [后端]
  └── feat/inference    ← [推理]
```

**提交规范**：

```bash
# 每次提交限定在本人写权限范围内（见 3.2 矩阵）
git add apps/platform/src/odp_platform/training/experiment.py   # 只加自己的文件
git commit -m "feat(training): add ExperimentConfig + run_experiment"
```

**禁止 `git add -A`**，防止误提交他人文件。

| 阶段 | 目标 | 并行工作 |
|---|---|---|
| W1 | 骨架跑通 | 训练 `ExperimentConfig` + `run_experiment` → 推理 `Detector` → 后端建表+API → 前端 Gradio 6 Tab 骨架 |
| W2 | RSOD 全链路 | 训练跑 RSOD baseline → 推理测试 best.pt → 前端面板联调 → 后端 epoch API 联调 |
| W3 | VisDrone + 算法 | 训练跑 VisDrone → 算法做改进实验 → 前端完善 dataset_browser/model_demo → 后端整表联调 |
| W4 | 联调集成 | 全链路：配置→训练→DB→API→UI，端到端跑通 |
| W5 | 收尾 | 最终测试、best_config.yaml 发布、v1.0 tag |