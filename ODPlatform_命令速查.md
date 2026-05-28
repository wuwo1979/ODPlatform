# ODPlatform 命令速查

> 严格对齐 D3 / D4 讲义与需求规格说明书
> 注意：所有命令在项目根目录执行，不要在 `apps/platform/src` 下执行

---

## 一、环境搭建

```bash
# 激活环境（不同机器按实际情况）
conda activate odp-gpu

# 安装项目核心包（安装后才有 odp-xxx 命令）
pip install -e ./apps/platform

# 重装（修改 pyproject.toml 或 entry-point 后必须执行）
pip install -e ./apps/platform --force-reinstall --no-deps
```

---

## 二、D2 — 项目初始化与重置

### odp-init

```bash
# 创建所有运行时目录
odp-init
```

创建目录清单（D2 + D3 增量）：

```
data/raw/              # 数据集存放根
data/train/{images,labels}
data/val/{images,labels}
data/test/{images,labels}
data/runs/
outputs/
models/pretrained/
models/trained/
configs/datasets/      # D3 新增
```

### odp-reset

安全撤销 `odp-init` 创建的运行时产物，**只删白名单目录，不动 `data/raw/`**。

```bash
odp-reset                  # dry-run，只打印不删除
odp-reset --yes            # 真删除 + 交互确认
odp-reset --yes --force    # 跳过交互，直接删除
odp-reset --dry-run        # 显式 dry-run
```

---

## 三、D3 — 数据格式转换与划分

### 数据集目录约定

```
data/raw/<dataset_name>/
├── images/         # 原始图像 (jpg/png/...)
└── annotations/    # 原始标注 (xml/json/txt, 按格式而定)
```

### 查看支持的能力矩阵

```bash
odp-transform --list
```

输出示例：

```
格式能力矩阵:
  pascal_voc       -> ['detect']
  coco             -> ['detect', 'segment']
  yolo             -> ['detect']
```

### 端到端转换（原始数据集 → YOLO → 划分 → yaml）

```bash
# VOC → YOLO（推荐方式）
odp-transform --dataset voc --format pascal_voc

# 自定义划分比例 + 随机种子
odp-transform --dataset voc --format pascal_voc --train-rate 0.7 --val-rate 0.2

# 指定任务类型
odp-transform --dataset coco_demo --format coco --task segment

# COCO 91→80 类别映射
odp-transform --dataset coco_demo --format coco --coco-cls91to80

# 指定类别白名单
odp-transform --dataset voc --format pascal_voc --classes cat dog bird
```

### 输出产物

```
data/
├── train/images/*.jpg
├── train/labels/*.txt
├── val/images/*.jpg
├── val/labels/*.txt
└── test/images/*.jpg
└── test/labels/*.txt

configs/datasets/<dataset_name>.yaml    # ultralytics 训练 yaml
```

### 覆盖率 fail-fast

标注覆盖率低于 50% 时立即阻断：

```bash
odp-transform --dataset broken_voc --format pascal_voc
# ❌ 图像-标注覆盖率过低: 5.0% (硬阈值 50%)
```

### od-pipeline 架构说明

```
data_pipeline/
├── registry.py       # 注册表 + @register 装饰器
├── service.py        # 调度层（converter_data_to_yolo）
├── orchestrator.py   # 端到端 DatasetPipeline
├── core/
│   ├── pascal_voc.py # VOC XML → YOLO txt
│   ├── coco.py       # COCO JSON → YOLO txt
│   └── yolo.py       # YOLO 直通
└── split/
    ├── manifest.py     # 划分清单
    ├── splitter.py     # 随机划分
    ├── materializer.py # 落盘（copy/hardlink）
    └── yaml_writer.py  # 生成 ultralytics yaml
```

---

## 四、D4 — 数据校验

### 快速质检（推荐方式）

```bash
# 按数据集名质检（自动查找 configs/datasets/<name>.yaml）
odp-validate --dataset voc

# 指定任务类型
odp-validate --dataset voc --task detect

# 调试方式：直接指定 yaml 路径
odp-validate --yaml configs\datasets\voc.yaml

# 详细模式（显示 DEBUG 日志）
odp-validate --dataset voc --verbose

# 不写 JSON 报告
odp-validate --dataset voc --no-report
```

### 验收场景

#### 7.1 健康数据集

```bash
odp-validate --dataset voc --task detect
echo "exit: $?"
```

期望：4 PASS，退出码 0，生成 `runs/data_validation/<run_id>/report.json`

#### 7.2 数据泄露检测

```bash
# 将一张训练集图片复制到验证集
cp data/voc/train/images/xxx.jpg data/voc/val/images/
odp-validate --dataset voc --task detect
# 期望：split_uniqueness ERROR，其他 3 个 check 仍跑完
# 恢复
rm data/voc/val/images/xxx.jpg
```

#### 7.3 坏 yaml 诊断

构造 `nc=3` 但 `names=['a','b']`（长度不匹配），yaml_schema 输出 ERROR。

#### 7.5 退出码语义

| 场景 | 退出码 |
|---|---|
| 全 PASS | 0 |
| 仅 INFO，无 WARNING/ERROR | 0 |
| 含 WARNING，无 ERROR | 1 |
| 含 ERROR | 2 |
| Ctrl-C | 3 |

### 数据校验架构说明

```
data_validation/
├── registry.py       # @check 装饰器 + 自动 import
├── snapshot.py       # DatasetSnapshot + build_snapshot
├── service.py        # run_all_checks + validate_dataset
├── report.py         # ValidationReport（纯数据）
├── render.py         # render_to_logger（纯展示）
└── checks/
    ├── yaml_schema.py      # yaml 字段完整性
    ├── pair_existence.py   # 图像-标签配对
    ├── label_format.py     # YOLO txt 行格式
    └── split_uniqueness.py # train/val/test 图像唯一性
```

四个 check 互相独立，任何一个抛异常不阻断其他 check。

---

## 五、D5 — 配置管理子系统

### 生成默认配置

```bash
# 生成训练配置（输出到 configs/train_config.yaml）
odp-config generate --task train

# 指定输出路径
odp-config generate --task train --output my_config.yaml
```

### 验证配置

```bash
# 验证正确配置
odp-config validate --config configs\train_config.yaml --task train

# 验证错误配置（nc=3 但 names 只有 2 个）
odp-config validate --config configs\bad_config.yaml --task train
# 期望：yaml_schema ERROR → 退出码 2
```

### 追踪配置来源

```bash
# 查看配置文件中每个字段的来源层级
odp-config trace --config configs\train_config.yaml
# 输出示例：classes → CLI > env var > default
```

### 配置快照（保存 & 恢复）

```bash
# 导出快照（保存当前配置的完整副本）
odp-config snapshot export --config configs\train_config.yaml

# 恢复快照（从历史快照恢复配置）
odp-config snapshot restore --snapshot runs\config_snapshots\<snapshot_name>.yaml

# 快照目录：runs/config_snapshots/
```

### 配置管理架构说明

```
config_manager/
├── registry.py       # @config_generator 注册装饰器
├── service.py        # 核心调度（generate/validate/trace/snapshot）
├── snapshot.py       # snapshot 导出与恢复
├── generator.py      # 配置生成器基类
├── validator.py      # 配置校验器
├── tracer.py         # 配置溯源
└── generators/
    └── train.py      # 训练配置生成器
```

---

## 六、WebUI — Gradio 前端

### CLI 命令

```bash
# 启动 WebUI（自动启动本地后端 + 数据库）
odp-webui

# 指定端口
odp-webui --port 7861

# 仅启动 FastAPI 后端（单独进程）
odp-backend

# 或通过 Python 直接启动
python apps\platform\src\odp_platform\webui\app.py
```

> **注意**：
> - `odp-webui` 在启动 Gradio 的同时，会**自动启动** FastAPI + SQLite 后端（subprocess）。
> - 后端可独立运行：`odp-backend`。
> - 访问地址：`http://localhost:7860`

### 双模式架构

项目提供**用户模式**和**管理员模式**两套界面，通过密码切换。

#### 用户模式 Tab

| Tab | 功能 | 源文件 |
|-----|------|--------|
| 单图检测 | 上传单张图片 + 选择模型进行推理 | `webui/user_tabs.py` |
| 文件夹检测 | 批量检测文件夹内图片 | `webui/user_tabs.py` |
| 视频检测 | 上传视频逐帧检测，显示预览图 + 明细 | `webui/user_tabs.py` |
| 实时摄像头 | OpenCV 摄像头实时检测，支持分辨率切换 | `webui/user_tabs.py` |
| 模型选择 | 刷新/上传 .pt / 手动路径 + 实验训练结果展示 | `webui/user_tabs.py` |
| LLM对话 | DeepSeek API（默认 deepseek-v4-flash）对话 | `webui/user_tabs.py` |

#### 管理员模式 Tab（额外）

| Tab | 功能 | 源文件 |
|-----|------|--------|
| Dashboard | 项目概览、后端状态 | `webui/dashboard.py` |
| 模型演示 | 加载模型 + 推理可视化 | `webui/model_demo.py` |
| 数据集浏览 | 查看图片 + 标注 | `webui/dataset_browser.py` |
| 训练 | 配置参数 + 启动训练 | `webui/training_tab.py` |
| 数据校验 | 运行质检 + 查看报告 | `webui/validation_tab.py` |
| 配置管理 | 生成/验证/追踪配置 | `webui/config_tab.py` |

### 模型选择详解

- 自动扫描 `data/models/checkpoints/` 目录下所有 `.pt` 文件
- **上传模型**：点击上传按钮选择 `.pt` 文件，自动复制到 checkpoints 目录
- **手动路径**：在文本框输入任意路径的 `.pt` 文件
- **实验训练结果**（折叠面板）：展开后展示所选实验的训练曲线、混淆矩阵、PR/F1 曲线、类别分布

### 实验可视化

实验文件夹位于 `data/runs/experiments/<实验名>/`，包含：

| 文件 | 说明 |
|------|------|
| `results.png` | 训练 Loss + 验证指标曲线 |
| `confusion_matrix.png` | 混淆矩阵热力图 |
| `confusion_matrix_normalized.png` | 归一化混淆矩阵 |
| `BoxPR_curve.png` | PR 曲线 |
| `BoxF1_curve.png` | F1 曲线 |
| `labels.jpg` | 类别分布柱状图 |
| `results.csv` | 逐轮训练指标（动态绘图用） |

### 架构说明

```
webui/
├── app.py               # create_app() + main() 入口
├── user_tabs.py         # 用户 Tab：检测/摄像头/模型/LLM + 实验可视化
├── dashboard.py         # Dashboard Tab
├── dataset_browser.py   # 数据集浏览 Tab
├── dataset_analysis.py  # 数据集分析（类分布/热力图）
├── training_tab.py      # 训练 Tab
├── model_demo.py        # 模型演示 Tab
├── validation_tab.py    # 数据校验 Tab
├── config_tab.py        # 配置管理 Tab
└── utils.py             # 通用 UI 工具函数（模型扫描/图片列表）
```

---

## 七、Backend API — FastAPI 后端

### 启动

```bash
# 方式一：WebUI 自动启动（推荐）
odp-webui

# 方式二：独立启动
odp-backend
```

后端默认监听 `http://localhost:8888`

### API 端点

| 方法 | 路径 | 功能 |
|------|------|------|
| GET | `/api/v1/experiments` | 获取实验列表 |
| POST | `/api/v1/experiments` | 创建实验 |
| GET | `/api/v1/experiments/{id}/epochs` | 获取实验的 epoch 列表 |
| GET | `/api/v1/experiments/{id}/models` | 获取实验的模型列表 |
| GET | `/api/v1/experiments/{id}/quality` | 获取实验的质检报告列表 |
| GET | `/api/v1/experiments/{id}/quality/{type}` | 获取特定类型质检报告 |
| GET | `/api/v1/health` | 健康检查（Dashboard 用） |

### 数据库（SQLite）

数据库文件位于：`data/backend/odp.db`

包含 4 张表：

| 表 | 说明 |
|----|------|
| `experiments` | 实验记录 |
| `epochs` | 训练轮次记录 |
| `models` | 模型记录 |
| `quality_reports` | 质检报告 |

---

## 八、全链路验证指南

启动后按以下流程验证各功能链路是否正常：

```bash
# 1. 启动
odp-webui

# 2. 打开 http://localhost:7860
```

### 验证清单

| 步骤 | Tab | 操作 | 预期结果 |
|------|-----|------|---------|
| 1 | Dashboard | 等待加载 | 显示"后端在线"、实验统计（0 条） |
| 2 | 数据集浏览 | 选择数据集 → 切换图片 | 下拉框可选、图片 + 标注正常渲染 |
| 3 | 模型演示 | 选择模型 → 加载 → 上传图片推理 | 推理结果可视化、框 + 标签正确 |
| 4 | 数据校验 | 选择数据集 → 运行质检 | 4 个 check 全部通过 |
| 5 | 配置管理 | 生成训练配置 → 验证 | 配置生成成功、验证通过 |
| 6 | 训练 | 配置参数（dry-run） | 准备流程正常执行 |

### 常见问题

| 现象 | 原因 | 解决 |
|------|------|------|
| Dashboard 显示"后端离线" | 后端未启动 | 确认 `odp-backend` 在运行 |
| 下拉框无法滚动/遮挡 | CSS 层级问题 | 刷新页面（Ctrl+F5） |
| 模型加载失败 | 模型文件不存在 | 先确认 `data/models/checkpoints/` 下有 .pt 文件 |
| 数据集浏览无数据 | 数据集不存在或路径不对 | 先运行 `odp-transform` 转换数据 |

---

## 九、集成命令（D3 + D4 + D5 串联）

### 端到端训练准备

```bash
# 一键运行：数据验证 → 配置生成 → 快照保存（不实际训练）
odp-train --dry-run

# 输出流程：
#   1. odp-transform D3 数据校验（内部调用）
#   2. odp-validate D4 数据质检
#   3. odp-config generate 自动生成配置
#   4. odp-config snapshot export 保存快照
#   5. 退出（--dry-run 跳过实际训练）
```

---

## 十、回归测试

```bash
# 从 apps/platform/src 目录运行
cd apps\platform\src
python -m pytest tests -v
```

---

## 十一、关键目录结构

```
ODPlatform/
├── apps/platform/
│   ├── src/odp_platform/
│   │   ├── cli/
│   │   │   ├── init_project.py     # odp-init
│   │   │   ├── reset_project.py    # odp-reset
│   │   │   ├── transform_data.py   # odp-transform （D3）
│   │   │   ├── validate_data.py    # odp-validate （D4）
│   │   │   ├── config_cli.py       # odp-config （D5）
│   │   │   ├── train.py            # odp-train （集成）
│   │   │   ├── backend.py          # odp-backend （FastAPI）
│   │   │   └── webui.py            # odp-webui （Gradio）
│   │   ├── common/
│   │   │   ├── paths.py
│   │   │   ├── constants.py
│   │   │   └── ...
│   │   ├── data_pipeline/          # D3 子系统
│   │   │   ├── registry.py
│   │   │   ├── service.py
│   │   │   ├── orchestrator.py
│   │   │   ├── core/
│   │   │   └── split/
│   │   ├── data_validation/        # D4 子系统
│   │   │   ├── registry.py
│   │   │   ├── snapshot.py
│   │   │   ├── service.py
│   │   │   ├── report.py
│   │   │   ├── render.py
│   │   │   └── checks/
│   │   ├── config_manager/         # D5 子系统
│   │   │   ├── registry.py
│   │   │   ├── service.py
│   │   │   ├── snapshot.py
│   │   │   ├── generator.py
│   │   │   ├── validator.py
│   │   │   ├── tracer.py
│   │   │   └── generators/
│   │   ├── webui/                  # Gradio 前端（PR #1）
│   │   │   ├── app.py              # create_app() 入口
│   │   │   ├── dashboard.py
│   │   │   ├── dataset_browser.py
│   │   │   ├── training_tab.py
│   │   │   ├── model_demo.py
│   │   │   ├── validation_tab.py
│   │   │   ├── config_tab.py
│   │   │   └── utils.py
│   │   ├── training/       模型训练
│   │   ├── evaluation/     模型评估
│   │   ├── inference/      模型推理
│   │   │   ├── engine.py      推理引擎
│   │   │   └── visualizer.py  可视化工具
│   │   └── __init__.py
│   ├── configs/
│   │   ├── datasets/               # 数据集 yaml
│   │   ├── train_config.yaml       # 训练配置
│   │   └── bad_config.yaml         # 坏配置（测试用）
│   └── pyproject.toml
├── data/
│   ├── raw/<dataset>/{images,annotations}/
│   ├── {train,val,test}/{images,labels}/
│   └── runs/
│       ├── data_validation/        # D4 质检报告
│       └── config_snapshots/       # D5 配置快照
├── scripts/
├── docs/
│   ├── architecture/                # ADR 决策记录
│   ├── srs/                         # 需求规格
│   ├── teaching/                    # 教学讲义
│   ├── ODPlatform_团队协作指南.md
│   ├── ODPlatform_答辩演练问题集.md
│   ├── ODPlatform_AI接手指南.md
│   └── ODPlatform_命令速查.md       # ← 就是这个文件
```