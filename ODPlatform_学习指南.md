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

## 9. 学习路线图

### 9.1 按模块优先级学习

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

### 9.2 推荐学习步骤

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

### 9.3 核心代码阅读路径

```
1. apps/platform/src/odp_platform/common/paths.py       ← 路径是基础
2. apps/platform/src/odp_platform/common/logging_utils.py ← 日志很重要
3. apps/platform/src/odp_platform/data_pipeline/        ← 数据处理核心
4. apps/platform/src/odp_platform/cli/                  ← 入口命令
5. apps/platform/src/odp_platform/training/             ← 训练流程
6. apps/platform/tests/                                 ← 测试用例
```

### 9.4 自学延伸方向

- **模型优化**：YOLO 模型压缩（剪枝/量化/TensorRT）
- **Web 服务**：用 FastAPI 封装推理 API
- **多任务学习**：检测 + 分割 + 分类联合训练
- **数据增强**：Mosaic / MixUp / Albumentations
- **MLOps**：实验追踪（MLflow/W&B）、模型版本管理

> **记住**：不要试图一次学完——今天读懂一个模块，明天跑通一条命令，后天改好一个 Bug，每周进步一点点。