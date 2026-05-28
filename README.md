# ODPlatform - Object Detection Platform

基于 YOLO 的多格式目标检测开发平台，覆盖数据转换、模型训练、评估、推理和 WebUI 可视化全流程。

## 快速开始

```bash
# 1. 创建环境
conda create -n odp-gpu python=3.12 -y
conda activate odp-gpu
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 2. 安装项目
cd apps/platform
pip install -e .
cd ../..

# 3. 启动 WebUI（自动启动后端）
odp-webui
# 浏览器打开 http://localhost:7860
```

## CLI 命令速览

| 命令 | 功能 | 详细文档 |
|------|------|---------|
| `odp-init` | 初始化项目目录 | [命令速查](ODPlatform_命令速查.md#二项目初始化-odp-init) |
| `odp-reset` | 安全重置运行时产物 | [命令速查](ODPlatform_命令速查.md#三项目重置-odp-reset) |
| `odp-transform` | 端到端数据转换+划分+yaml | [命令速查](ODPlatform_命令速查.md#四数据格式转换与划分-odp-transform) |
| `odp-validate` | 数据集校验与分析 | [命令速查](ODPlatform_命令速查.md#六数据校验-odp-validate) |
| `odp-gen-config` | 生成配置模板 | [命令速查](#) |
| `odp-train` | 模型训练 | [命令速查](ODPlatform_命令速查.md#十训练-odp-train) |
| `odp-val` | 模型评估 | [命令速查](ODPlatform_命令速查.md#十一评估-odp-val) |
| `odp-infer detect` | 模型推理（图片/视频） | [命令速查](ODPlatform_命令速查.md#十二推理-odp-infer) |
| `odp-infer live` | **实时推理（摄像头/RTSP）** | 画面显示FPS/帧数/检测数信息面板 |
| `odp-infer benchmark` | **模型基准测试** | 测试模型推理速度 |
| **`odp-webui`** | **启动 Gradio Web 前端** | [命令速查](ODPlatform_命令速查.md#六webui--gradio-前端) |
| **`odp-backend`** | **启动 FastAPI 后端（单独）** | [命令速查](ODPlatform_命令速查.md#六webui--gradio-前端) |

### 实时推理

摄像头或 RTSP 流实时检测，带信息面板（FPS/推理耗时/帧数/检测数）：

```bash
# 默认摄像头
odp-infer live --model runs/train/exp/weights/best.pt

# 指定摄像头编号（0,1,2...）并配置参数
odp-infer live --model best.pt --source 0 --display-width 1280 --display-height 720 --camera-fps 60

# RTSP 网络流
odp-infer live --model best.pt --source rtsp://192.168.1.100:554/stream1

# 图片/视频检测
odp-infer detect --model best.pt --input test.jpg --output results/
odp-infer detect --model best.pt --input video.mp4 --output results/
```

## WebUI — 可视化操作界面

`odp-webui` 提供用户和管理员双模式 Web 界面：

### 用户模式 Tab

| Tab | 功能 |
|-----|------|
| 单图检测 | 上传单张图片 + 选择模型进行推理 |
| 文件夹检测 | 批量检测文件夹内图片，自动切换当前模型运行 |
| 视频检测 | 上传视频逐帧检测，显示预览图 + 明细表 |
| 实时摄像头 | 通过 OpenCV 读取摄像头实时流，支持分辨率切换 |
| 模型选择 | 选择/刷新可用模型，上传 .pt 文件，或手动输入路径；底部折叠展示实验训练结果 |
| LLM对话 | 集成 DeepSeek API（默认 deepseek-v4-flash），对话式协助 |

### 管理员模式 Tab（用户模式 + 额外）

| Tab | 功能 | 源文件 |
|-----|------|--------|
| Dashboard | 项目概览、实验状态 | `webui/dashboard.py` |
| 模型演示 | 加载模型 + 推理可视化 | `webui/model_demo.py` |
| 数据集浏览 | 查看图片 + 标注 | `webui/dataset_browser.py` |
| 训练 | 配置参数 + 启动训练 | `webui/training_tab.py` |
| 数据校验 | 运行质检 + 查看报告 | `webui/validation_tab.py` |
| 配置管理 | 生成/验证/追踪配置 | `webui/config_tab.py` |

**注意**：
- `odp-webui` 会自动启动本地后端（FastAPI + SQLite），Dashboard 需要后端连接
- 后端也可单独启动：`odp-backend`
- Gradio 默认端口 7860，浏览器访问 [http://localhost:7860](http://localhost:7860)

## 快速验证链路（UI → 模型 → 数据库）

```bash
# 1. 启动 WebUI
odp-webui

# 2. 浏览器打开 http://localhost:7860

# 3. 验证链路：
#    a) Dashboard Tab → 确认显示"后端在线"（验证后端+数据库链路）
#    b) 数据集浏览 Tab → 下拉框可选 → 切换图片（验证数据集链路）
#    c) 模型演示 Tab → 选择模型 → 加载 → 上传图片推理（验证模型链路）
#    d) 数据校验 Tab → 选择数据集 → 运行质检（验证校验链路）
#    e) 配置管理 Tab → 生成/验证配置（验证配置链路）
```

## 项目结构

```
ODPlatform/
├── apps/platform/              ← 核心引擎
│   └── src/odp_platform/
│       ├── common/             基础工具（路径/日志/性能）
│       ├── config/             配置管理
│       ├── data_pipeline/      数据管道（格式转换+划分）
│       ├── data_validation/    数据校验
│       ├── training/           模型训练
│       ├── evaluation/         模型评估
│       ├── inference/              模型推理（D8 推理子系统）
│       │   ├── __init__.py         公共 API（Detector、InferService、InferStats…）
│       │   ├── engine.py           推理引擎（Detector）
│       │   ├── service.py          推理服务编排（InferService，对标 odp-train）
│       │   ├── pipeline_config.py  帧源+美化 YAML 配置加载
│       │   ├── sources.py          输入源（图片/视频/摄像头/RTSP）
│       │   ├── frame_source/       统一帧输入源（摄像头配置/多线程/覆盖层）
│       │   │   ├── overlay.py      画面信息叠加（FPS、帧数、推理耗时）
│       │   │   └── sources/        各输入源实现（Camera/Video/Image）
│       │   ├── visualizer.py       检测结果绘制 + 信息面板
│       │   ├── benchmark.py        性能基准测试
│       │   └── utils.py            辅助工具
│       ├── webui/              Gradio 前端（用户+管理员双模式）
│       │   ├── app.py          入口，自动启动后端
│       │   ├── user_tabs.py    用户 Tab（检测/摄像头/模型/LLM）+ 实验可视化
│       │   ├── dashboard.py / model_demo.py / dataset_browser.py
│       │   ├── training_tab.py / validation_tab.py / config_tab.py
│       │   ├── dataset_analysis.py / experiment_viz.py（已并入 user_tabs.py）
│       │   └── utils.py        通用工具函数
│       └── cli/                命令行入口（含 odp-webui / odp-backend）
├── docs/                       文档 + ADR + SRS
├── data/                       数据集（.gitignore，大文件用 LFS）
│   └── models/checkpoints/     基线模型（Git LFS）
└── pyproject.toml              项目配置 + CLI 入口定义
```

## 支持的数据格式

| 输入 → 输出 | 命令 |
|-----------|------|
| Pascal VOC → YOLO | `odp-trans voc` / `odp-transform` |
| COCO → YOLO | `odp-trans coco` / `odp-transform` |
| LabelMe → YOLO | `odp-trans labelme` |
| YOLO → COCO | `odp-trans yolo2coco` |
| YOLO 重排+划分 | `odp-transform --format yolo` |

## 文档导航

| 文档 | 内容 |
|------|------|
| [命令速查](ODPlatform_命令速查.md) | 所有 CLI 命令及参数（日常使用） |
| [AI 接手指南](ODPlatform_AI接手指南.md) | 全链路验证 + AI 协作开发流程 |
| [学习指南](ODPlatform_学习指南.md) | 架构详解 + 学习路线（从零入门） |
| [企业软件开发指南](ODPlatform_企业软件开发指南.md) | 企业级工程规范与最佳实践 |
| [项目结构设计指南](ODPlatform_项目结构设计指南_V1_0.pdf) | 架构决策与演进路线 |

## 端到端流程示例

```bash
# 1. 初始化运行时目录
odp-init

# 2. 数据转换 + 划分
odp-transform --dataset RSOD --format pascal_voc \
    --source-dir data\raw\RSOD --output-dir data\yolo\RSOD \
    --classes "aircraft ship oiltank playground overpass"

# 3. 数据校验
odp-validate --dataset RSOD

# 4. 训练
odp-train --model yolo11n.pt --data configs\datasets\RSOD.yaml --epochs 100

# 5. 评估
odp-val --model data\runs\train\exp\weights\best.pt --data configs\datasets\RSOD.yaml

# 6. 推理（图片/视频/摄像头）
odp-infer --model best.pt --source data/test/images       # 图片文件夹
odp-infer --model best.pt --source demo.mp4 --conf 0.5    # 视频（覆盖置信度）
odp-infer --model best.pt --source 0 --show               # 摄像头实时检测
odp-infer --model best.pt --source 0 --threaded --show    # 摄像头 + 多线程流水线

# 7. WebUI 可视化
odp-webui
```

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🧪 给测试人员的操作指南

### 环境准备（新机器）

```bash
# 1. 克隆仓库（仅 main 分支）
git clone https://github.com/wuwo1979/ODPlatform.git
cd ODPlatform

# 2. 创建环境
conda create -n odp-gpu python=3.12 -y
conda activate odp-gpu
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. 安装项目
pip install -e ./apps/platform

# 4. 初始化目录
odp-init
```

### 启动测试

```bash
# 一键启动 WebUI（推荐）
odp-webui
# 浏览器打开 http://localhost:7860
# 首次启动会自动安装 Gradio 依赖，稍等 10-20 秒
```

### 功能测试点（按顺序）

| 序号 | 测试项 | 操作步骤 | 预期结果 |
|------|--------|---------|---------|
| 1 | **首页/导航** | 打开 http://localhost:7860 | 看到左侧导航栏（仪表盘/图像检测/模型选择等），点击 "低空智瞰" logo 回到首页 |
| 2 | **单图检测** | 图像检测 Tab → 选模型 → 上传图片 → 点"开始检测" | 右侧显示带检测框的图片 + 下方 JSON 列表 |
| 3 | **Web 摄像头** | 图像检测 Tab → 点击摄像头画面 → 允许权限 | 画面出现后自动逐帧推理，叠加检测框 + FPS 信息 |
| 4 | **服务器摄像头（OpenCV）** | 展开"服务器摄像头" → 点"启动" | 画面显示实时检测结果，左上角有 FPS/检测数/推理耗时面板 |
| 5 | **管理员模式** | 点击右下 ⚙️ → 输入密码 `0000` | 进入管理员面板，显示 8 个 Tab（Dashboard/训练/配置管理等） |
| 6 | **训练** | 管理员 → 训练 Tab → 选数据集 → 调参 → 开始训练 | 实时输出训练日志，训练结束后下方"训练结果可视化"显示 Loss/mAP 曲线 |
| 7 | **LLM 对话** | 用户模式 → LLM对话 → 填 API Key + 模型名 → 发送消息 | 返回 AI 回复 |
| 8 | **数据集浏览** | 管理员 → 数据集浏览 → 选数据集 | 显示图片缩略图 + 标注框 |

### 常见问题

```bash
# 端口被占用
odp-webui --port 7861

# 单独启动后端（如果 WebUI 的后端没启动成功）
odp-backend

# 清除所有运行时数据
odp-reset --yes --force
```