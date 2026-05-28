# ODPlatform - Object Detection Platform

> 一个让用户从原始标注数据到目标检测结果，全流程可视化的工程化平台。

基于 YOLO(Ultralytics) + Gradio + FastAPI + OpenCV 的多格式目标检测开发平台，覆盖数据转换、模型训练、评估、推理和 WebUI 可视化全流程。

## 快速开始

```bash
# 1. 克隆
git clone https://github.com/wuwo1979/ODPlatform.git
cd ODPlatform

# 2. 安装
pip install -e ./apps/platform

# 3. 初始化
odp-init

# 4. 启动 WebUI
odp-webui
# → http://localhost:7860
```

> 如需 GPU 环境，先安装 PyTorch CUDA 版再执行上述步骤。

## 核心架构：五层

```
CLI 层      (10 命令, 只解析参数)
Service 层  (编排: run_experiment / InferService)
Core 层     (单一逻辑: Detector / FrameSource)
Config 层   (三源合并 + 溯源 + 快照)
Common 层   (paths / logging / constants)
```

- **Monorepo**：一次 PR 完成跨模块变更
- **Marker 文件**：`.odp-workspace` 自动定位根目录
- **注册表模式**：新增格式/检项只加一行装饰器，不碰旧代码
- **三源合并**：CLI > YAML > 默认值，每字段可溯源

## CLI 命令

| 命令 | 功能 |
|------|------|
| `odp-init` | 初始化项目目录 |
| `odp-reset` | 清理运行时数据 |
| `odp-transform` | 数据集格式转换（VOC/COCO → YOLO） |
| `odp-validate` | 数据集质量检查 |
| `odp-config` | 配置生成/验证/溯源 |
| `odp-train` | 启动训练 |
| `odp-val` | 模型评估 |
| `odp-infer` | 推理（图片/视频/摄像头/RTSP） |
| `odp-webui` | 启动 Gradio WebUI |
| `odp-backend` | 启动 FastAPI 后端 |

## WebUI 功能

### 用户模式（6 Tab）

| Tab | 功能 |
|-----|------|
| 单图检测 | 上传图片 → 选模型 → 检测 |
| 文件夹检测 | 上传多图或输入路径 → 批量检测 + Gallery 展示 |
| 视频检测 | 上传视频 → 跳帧检测 → 下载结果 |
| 实时摄像头 | 启动摄像头实时流检测（支持分辨率切换） |
| 模型选择 | 选/传/输路径 .pt 文件 + **展开实验训练结果** |
| LLM 对话 | DeepSeek API + Agent 工具对话 |

### 管理员模式（齿轮按钮 → 密码 `0000`）

额外 5 个 Tab：Dashboard / 模型演示 / 训练 / 数据校验 / 配置管理

## 学习资源

| 文档 | 说明 |
|------|------|
| [学习指南](ODPlatform_学习指南.md) | 五层架构详解 + 每层源码对照 + 答辩 FAQ |
| [架构师讲解稿](ODPlatform_架构师讲解文稿.md) | 22 分钟逐字演讲稿，含演示节奏和应急话术 |
| [命令速查](ODPlatform_命令速查.md) | 所有 CLI 命令参数说明 |
| [学习指南——命令速查](ODPlatform_命令速查.md) | 命令快速参考 |

## 项目结构

```
ODPlatform/
├── apps/platform/src/odp_platform/
│   ├── cli/              CLI 命令入口
│   ├── common/           基础设施（路径/日志/常量）
│   ├── data_pipeline/    数据格式转换
│   ├── data_validation/  数据质量检查
│   ├── training/         训练引擎 + callbacks
│   ├── inference/        推理引擎 + 帧源
│   ├── webui/            Gradio Web 界面
│   └── run_config/       配置管理（三源合并）
├── apps/web-backend/     FastAPI 后端（可选）
├── configs/datasets/     数据集 YAML
├── data/runs/experiments/ 训练结果
└── docs/                 架构文档 / ADR
```
