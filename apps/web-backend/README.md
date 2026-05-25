# Web Backend — ODPlatform 实验数据存储与查询服务

FastAPI + SQLite 后端，负责实验元数据、训练指标、模型注册和质检报告的持久化与查询。

## 启动

```bash
cd apps/web-backend
pip install -r requirements.txt
python main.py
# → http://127.0.0.1:8000
# Swagger 文档: http://127.0.0.1:8000/docs
```

## API 一览

| 方法   | 路径                                  | 说明                         |
|--------|---------------------------------------|------------------------------|
| POST   | `/api/experiments`                    | 创建实验 → `{"id": 1}`       |
| GET    | `/api/experiments?dataset=&status=&limit=20` | 查询实验列表           |
| PATCH  | `/api/experiments/{id}`                | 更新状态 + 最终指标          |
| POST   | `/api/experiments/{id}/epochs`        | 写入 epoch 训练指标          |
| GET    | `/api/experiments/{id}/epochs`        | 查询训练曲线数据             |
| POST   | `/api/models`                         | 注册模型文件                 |
| GET    | `/api/models?experiment_id=`          | 查询模型列表                 |
| POST   | `/api/validation/reports`             | 写入质检报告                 |
| GET    | `/api/validation/reports?dataset=&limit=20` | 查询质检历史           |

## 快速验证

```bash
# 创建实验
curl -X POST http://127.0.0.1:8000/api/experiments \
  -H "Content-Type: application/json" \
  -d '{"name":"test","dataset":"rsod","model":"yolo11n.pt","config_json":"{}"}'
# → {"id":1,"name":"test"}

# 查询列表
curl http://127.0.0.1:8000/api/experiments
```

## 数据库

4 张表（SQLite，自动创建于 `odplatform.db`）：

| 表名                | 说明                               |
|---------------------|------------------------------------|
| `experiments`       | 实验主表（name UNIQUE）            |
| `training_epochs`   | 逐 epoch 训练指标 (experiment_id, epoch) UNIQUE |
| `models`            | 模型注册表（外键 → experiments）   |
| `validation_reports`| 数据质检报告历史                   |

## 跨模块集成

训练模块通过 `hooks.py`（本目录下）调用后端 API，无需关心 HTTP 细节：

```python
from hooks import on_training_start, on_epoch_end, on_training_end

exp_id = on_training_start("rsod_exp_001", config_json, "rsod", "yolo11n.pt")
on_epoch_end(exp_id, epoch=1, metrics={"train_loss": 2.3, ...})
on_training_end(exp_id, map50=0.872, model_path="checkpoints/best_rsod_exp_001.pt")
```
