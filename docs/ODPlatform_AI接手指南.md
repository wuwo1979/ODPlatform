# ODPlatform AI 接手指南

> **目标**：让新启动的 AI 在 5 分钟内了解项目全貌，避免每次重复询问上下文。

---

## 一、项目定位

ODPlatform 是一个**通用目标检测开发平台**。它不是一个训练好的模型，而是一个**工程平台**——覆盖从原始标注数据 → 格式转换 → 数据校验 → 配置管理 → 训练/推理的全链路。

应用方向包括但不限于：遥感图像检测、安全帽检测、车辆检测、工业质检等。

核心技术栈：
- **深度学习框架**：Ultralytics YOLOv**（作为底层训练/推理引擎）
- **前端/UI**：Gradio（Python Web UI）
- **数据格式**：Pascal VOC / COCO JSON → YOLO txt
- **配置管理**：自研 field 注册表 + YAML 配置链

---

## 二、环境配置（关键！）

### Conda 环境

```bash
# 环境名：odp-gpu
# Python 版本：3.10+
# 位置：E:\Anaconda\envs\odp-gpu\
conda activate odp-gpu
```

### 安装项目包（首次 / 依赖变更后）

```bash
# 必须 cd 到 apps/platform 目录安装
cd F:\python_projects\class\ODPlatform\apps\platform
pip install -e .
```

如需重装（entry-point 变更时）：

```bash
pip install -e . --force-reinstall --no-deps
```

### 项目根目录

```
F:\python_projects\class\ODPlatform
```

**所有命令都在项目根目录下执行**，不要 cd 到 `apps/platform/src`。

---

## 三、目录结构速览

```
ODPlatform/
├── apps/
│   └── platform/                        # 核心平台（唯一活跃端）
│       ├── src/odp_platform/
│       │   ├── cli/                     # CLI 命令入口（odp-*）
│       │   │   ├── init_project.py      #   odp-init
│       │   │   ├── reset_project.py     #   odp-reset
│       │   │   ├── transform_data.py    #   odp-transform（D3 数据管线）
│       │   │   ├── validate_data.py     #   odp-validate（D4 数据校验）
│       │   │   ├── config_cli.py        #   odp-config（D5 配置管理）
│       │   │   └── train.py            #   odp-train（集成入口）
│       │   ├── common/                  # 公共模块
│       │   │   ├── paths.py             #   路径中心化管理
│       │   │   ├── constants.py         #   共享常量/枚举
│       │   │   ├── logging_utils.py     #   日志工具
│       │   │   └── string_utils.py      #   字符串工具
│       │   ├── data_pipeline/           # D3：数据格式转换+划分
│       │   │   ├── registry.py          #   @register 装饰器
│       │   │   ├── service.py           #   调度层
│       │   │   ├── orchestrator.py      #   端到端管线
│       │   │   ├── core/{coco,pascal_voc,yolo}.py
│       │   │   └── split/{manifest,splitter,materializer,yaml_writer}.py
│       │   ├── data_validation/         # D4：数据质检
│       │   │   ├── registry.py          #   @check 装饰器
│       │   │   ├── service.py           #   调度层
│       │   │   ├── snapshot.py          #   数据集快照
│       │   │   ├── report.py / render.py
│       │   │   └── checks/              #   4 个检查器
│       │   ├── run_config/              # D5：运行配置管理
│       │   │   ├── registry.py          #   @config_field 装饰器
│       │   │   ├── service.py           #   调度层
│       │   │   ├── schema.py / validator.py
│       │   │   ├── template.py / merger.py / loader.py
│       │   │   └── fields/{train,val,predict}.py
│       │   ├── webui/                   # Gradio 前端
│       │   │   ├── app.py               #   create_app() 入口
│       │   │   ├── dashboard.py         #   Dashboard Tab
│       │   │   ├── dataset_browser.py   #   数据集浏览 Tab
│       │   │   ├── training_tab.py      #   训练 Tab
│       │   │   ├── model_demo.py        #   模型演示 Tab
│       │   │   ├── validation_tab.py    #   数据校验 Tab
│       │   │   ├── config_tab.py        #   配置管理 Tab
│       │   │   └── utils.py             #   UI 工具函数
│       │   └── _version.py              # 版本号：0.1.0
│       ├── configs/
│       │   ├── datasets/                # 数据集 YAML 配置
│       │   ├── train_config.yaml        # 训练配置
│       │   └── *.example.yaml           # 参考配置
│       ├── logging/                     # 运行时日志
│       └── pyproject.toml              # 包定义 + entry-points
├── data/
│   ├── raw/<dataset>/                   # 原始数据（images/ + annotations/）
│   ├── {train,val,test}/{images,labels}/ # 转换后 YOLO 格式
│   ├── models/{pretrained,checkpoints}/
│   └── runs/{data_validation,run_config}/
├── docs/
│   ├── architecture/                    # ADR 决策记录
│   ├── srs/                             # 需求规格说明书
│   └── teaching/                        # 教学讲义
├── scripts/                             # 工具脚本
└── pyproject.toml                       # 顶层工具配置（ruff/mypy/pytest）
```

### 关键路径速查

| 用途 | 路径 |
|------|------|
| 包源码 | `apps/platform/src/odp_platform/` |
| CLI 入口 | `apps/platform/src/odp_platform/cli/` |
| WebUI 入口 | `apps/platform/src/odp_platform/webui/app.py:create_app()` |
| 数据集配置 | `apps/platform/configs/datasets/*.yaml` |
| 日志目录 | `apps/platform/logging/` |
| 原始数据 | `data/raw/<dataset_name>/` |
| YOLO 数据 | `data/{train,val,test}/{images,labels}/` |
| 文档 | `docs/` |

---

## 四、可用数据集

| 数据集名 | 格式 | 类别数 | 类别 | 样本数 |
|----------|------|--------|------|--------|
| `rsod` | Pascal VOC | 4 | aircraft, oiltank, overpass, playground | 936 |
| `voc` | Pascal VOC | 4 | cat, dog, bird, fish | 10 |
| `coco_demo` | COCO | 3 | person, bicycle, car | 5 |
| `safety_helmet` | Pascal VOC | 3 | head, helmet, person | 8 |

数据量小的（voc/coco_demo/safety_helmet）是测试数据集，`rsod` 是真实遥感数据集。

---

## 五、CLI 命令大全

安装 `pip install -e ./apps/platform` 后获得以下命令：

| 命令 | 功能 | 对应模块 |
|------|------|----------|
| `odp-init` | 创建运行时目录结构 | `cli/init_project.py` |
| `odp-reset` | 安全清除运行时产物 | `cli/reset_project.py` |
| `odp-transform` | 数据格式转换（VOC/COCO → YOLO）+ 划分 | `cli/transform_data.py` |
| `odp-validate` | 数据质量校验（4 项检查） | `cli/validate_data.py` |
| `odp-config` | 配置生成/验证/溯源/快照 | `cli/config_cli.py` |
| `odp-train` | 端到端训练（含前置校验） | `cli/train.py` |
| `odp-webui` | 启动 Gradio 前端 | `webui/app.py:main()` |

### 常用命令示例

```bash
# 初始化项目目录
odp-init

# 转换数据集
odp-transform --dataset rsod --format pascal_voc

# 数据校验
odp-validate --dataset rsod

# 生成训练配置
odp-config generate --task train

# 启动 WebUI（需先安装 gradio）
odp-webui
# 或：
python -c "import sys; sys.path.insert(0, 'apps/platform/src'); from odp_platform.webui import create_app; create_app().launch(server_name='0.0.0.0', server_port=7860)"

# 运行测试
cd apps\platform\src
python -m pytest tests -v
```

---

## 六、Gradio 前端（WebUI）

入口文件：`apps/platform/src/odp_platform/webui/app.py`

核心函数：`create_app()` → 返回 `gr.Blocks` 对象

6 个 Tab：
1. **Dashboard** — 项目概览、实验状态
2. **数据集浏览** — 查看图片 + 标注
3. **训练** — 配置参数 + 启动训练（调用 `odp-train`）
4. **模型演示** — 加载模型 + 推理 + 结果可视化（调用 `model_demo.py`）
5. **数据校验** — 运行质检 + 查看报告
6. **配置管理** — 生成/验证/追踪配置

### 启动方式

```bash
conda activate odp-gpu
cd F:\python_projects\class\ODPlatform
python -c "import sys; sys.path.insert(0, 'apps/platform/src'); from odp_platform.webui import create_app; app = create_app(); app.launch(server_name='0.0.0.0', server_port=7860)"
```

注意 Gradio 版本兼容：当前 `pyproject.toml` 写的是 `gradio>=5.0,<6.0`，实际环境安装了 6.14.0。Gradio 6.x 的兼容差异：
- `theme` / `css` 参数在 `gr.Blocks()` 中仍可用（只报 `UserWarning`）
- 如果要消除 warning，可以把 `theme` 和 `css` 移到 `launch()` 方法中

前端的 CSS 液态玻璃效果作者为 UI 工程师，不要随意修改。

---

## 七、Git 工作流

### 远程仓库

```
origin  https://github.com/wuwo1979/ODPlatform.git
```

### 分支策略

- `main` — 主分支，稳定的可发布版本
- `pr/<number>` — PR 测试分支（`git fetch origin pull/<id>/head:pr/<id>`）
- 功能开发在 fork 仓库中完成，通过 PR 提交

### 拉取 PR 测试

```bash
git fetch origin pull/<PR编号>/head:pr/<PR编号>
git checkout pr/<PR编号>
# 测试完成后切回 main
git checkout main
```

### 一致性验证（本地 vs GitHub）

每次操作前建议确认本地与远程一致：

```bash
git fetch origin main         # 拉取最新远程状态
git diff HEAD origin/main     # 对比本地和远程差异
# 无输出 = 完全一致
git status                    # 确认 working tree clean
```

如果发现不一致，先 `git pull origin main` 同步，再继续操作。

### 提交规范

用中文写 commit message，清晰描述改动内容。不要用英文/Chinglish。

### 已知历史问题：2026-05-24 本地 .git 重建事件

> **背景**：首次 PR #1 合并后，AI 执行 `git rebase origin/main` 失败，导致本地 `.git` 目录丢失。
>
> **恢复方式**：`git init` → `git remote add origin` → `git fetch origin main` → `git branch main origin/main` → `git checkout -f main`
>
> **验证**：`git diff HEAD origin/main` 输出为空（完全一致）。本地 `main` 与远程 `origin/main` 完全同步，149 个文件全量跟踪。`.gitignore` 正常生效。
>
> **影响**：本地 `pr/<编号>` 分支丢失（PR 已合并则不需要）。未来 PR 测试时重新 `git fetch origin pull/<id>/head:pr/<id>` 即可。

---

## 八、测试

```bash
# 运行所有测试
cd apps\platform\src
python -m pytest tests -v

# 运行单元测试
python -m pytest tests -v -m unit

# 运行集成测试
python -m pytest tests -v -m integration
```

测试框架：pytest
测试目录：
- `apps/platform/src/tests/` — 主测试目录（按子系统分）
- `tests/` — 顶层集成测试

---

## 九、代码质量工具

```bash
# Lint（ruff）
ruff check apps/platform/src/

# 类型检查（mypy）
mypy apps/platform/src/

# 格式化
ruff format apps/platform/src/
```

配置在顶层 `pyproject.toml` 的 `[tool.ruff]` 和 `[tool.mypy]` 中。

---

## 十、关键架构决策（ADR）

所有 ADR 记录在 `docs/architecture/`：

| ADR | 内容 |
|-----|------|
| ADR-001 | Monorepo 结构 |
| ADR-002 | 数据管线设计 |
| ADR-002-paths | 路径策略 |
| ADR-003 | 命名规范 |
| ADR-004 | 数据校验子系统 |
| ADR-013 | 运行配置子系统（field 注册表） |

核心设计理念：
- **路径中心化**：所有路径统一在 `common/paths.py` 管理
- **注册表模式**：数据格式（`@register`）、校验检查器（`@check`）、配置字段（`@config_field`）都通过装饰器注册
- **不可变快照**：数据校验时先拍快照，保证分析一致性

---

## 十一、AI 首次启动 Checklist

当新 AI 启动来接手此项目时，请按以下顺序操作：

1. 读此文档（`docs/ODPlatform_AI接手指南.md`）
2. 激活 conda 环境：`conda activate odp-gpu`
3. 查看当前分支：`git branch`
4. 如果需要 PR 测试：`git fetch origin pull/<id>/head:pr/<id> && git checkout pr/<id>`
5. 如有依赖变更：`pip install -e ./apps/platform --force-reinstall --no-deps`
6. 查看项目根目录结构：`ls F:\python_projects\class\ODPlatform`
7. 所有 CLI 命令在项目根目录执行

---

## 十二、常见注意事项

1. **路径问题**：所有命令在项目根目录执行，不是 `apps/platform/src`，不是 `apps/platform`
2. **虚拟环境**：任何时候都使用 `odp-gpu`，不要用全局 Python
3. **不要修改别人的代码**：PR 代码保持原样，兼容问题先确认再改
4. **数据集路径**：自动由 `paths.py` 管理，手动配置用 `odp_meta.dataset` 字段
5. **日志**：运行时日志在 `apps/platform/logging/webui/` 下，按时间命名
6. **版本号**：在 `_version.py` 中维护（`__version__ = "0.1.0"`）