# Web Backend — ODPlatform 实验数据存储与查询服务

FastAPI + SQLite 后端，负责实验元数据、训练指标、用户认证、检测任务、模型管理和 LLM 透传。

## 启动

```bash
cd apps/web-backend
pip install -r requirements.txt
python main.py
# → http://127.0.0.1:8000
# Swagger 文档: http://127.0.0.1:8000/docs
```

## API 一览

### 实验相关（`/api`）

| 方法   | 路径                                  | 说明                         | 认证 |
|--------|---------------------------------------|------------------------------|------|
| POST   | `/api/experiments`                    | 创建实验 → `{"id": 1}`       | 无   |
| GET    | `/api/experiments?dataset=&status=&limit=20` | 查询实验列表           | 无   |
| PATCH  | `/api/experiments/{id}`                | 更新状态 + 最终指标          | 无   |
| POST   | `/api/experiments/{id}/epochs`        | 写入 epoch 训练指标          | 无   |
| GET    | `/api/experiments/{id}/epochs`        | 查询训练曲线数据             | 无   |
| POST   | `/api/models`                         | 注册模型文件                 | 无   |
| GET    | `/api/models?experiment_id=`          | 查询模型列表                 | 无   |
| POST   | `/api/validation/reports`             | 写入质检报告                 | 无   |
| GET    | `/api/validation/reports?dataset=&limit=20` | 查询质检历史           | 无   |

### 认证（`/api/v1/auth`）

| 方法   | 路径                       | 说明                     |
|--------|----------------------------|--------------------------|
| POST   | `/api/v1/auth/register`    | 注册 → `{"id","username","token"}` |
| POST   | `/api/v1/auth/login`       | 登录 → `{"id","username","token"}` |

密码使用 PBKDF2-SHA256 哈希存储，登录后返回 64 位 hex token。后续请求通过 `Authorization: Bearer <token>` 头传递。

### 用户（`/api/v1/users`）🔒

| 方法 | 路径                        | 说明                     |
|------|-----------------------------|--------------------------|
| GET  | `/api/v1/users/me`          | 当前用户信息             |
| GET  | `/api/v1/users/me/history`  | 检测历史（含结果数量）   |

### 检测（`/api/v1/detection`）🔒

| 方法 | 路径                        | 说明                     |
|------|-----------------------------|--------------------------|
| POST | `/api/v1/detection`         | 提交检测（multipart: image + model_name + conf + iou） |
| GET  | `/api/v1/detection/{id}`    | 查询检测结果             |

### 模型管理（`/api/v1/models`）🔒

| 方法   | 路径                        | 说明                     |
|--------|-----------------------------|--------------------------|
| GET    | `/api/v1/models`            | 扫描 checkpoints 目录    |
| POST   | `/api/v1/models/upload`     | 上传 .pt 文件             |
| DELETE | `/api/v1/models/{filename}` | 删除模型文件             |

### LLM（`/api/v1/llm`）🔒

| 方法 | 路径                     | 说明                      |
|------|--------------------------|---------------------------|
| POST | `/api/v1/llm/chat        | 对话透传（OpenAI 兼容）    |
| GET  | `/api/v1/llm/models      | 后端可用模型列表           |

需设置环境变量 `LLM_API_KEY` 和 `LLM_BASE_URL`（默认 `https://api.openai.com/v1`）。

## 认证机制

```
注册/登录 → 返回 token → 后续请求带 Authorization: Bearer <token>
```

- `get_current_user` 依赖注入，在 API 函数签名中加 `user: dict = Depends(get_current_user)` 即可
- 无 token → 401，无效 token → 401，跨用户数据 → 403

## 数据库

7 张表（SQLite，自动创建于 `odplatform.db`）：

| 表名                | 说明                               |
|---------------------|------------------------------------|
| `experiments`       | 实验主表（name UNIQUE）            |
| `training_epochs`   | 逐 epoch 训练指标                  |
| `models`            | 模型注册表                         |
| `validation_reports`| 数据质检报告                       |
| `users`             | 用户表（token 列存会话令牌）       |
| `detection_tasks`   | 检测任务（user_id FK）             |
| `detection_results` | 检测结果（task_id FK）             |

## 项目结构

```
apps/web-backend/
├── main.py              # FastAPI 入口，注册所有路由
├── schemas.py           # Pydantic 请求/响应模型
├── requirements.txt     # 依赖
├── hooks.py             # 训练侧客户端 SDK
├── db/
│   ├── database.py      # 连接管理（线程本地 + WAL）
│   └── init_db.py       # 建表脚本（幂等 CREATE IF NOT EXISTS）
├── api/
│   ├── experiments.py   # 实验 CRUD + Epoch
│   ├── models.py        # 模型注册（实验相关）
│   ├── validation.py    # 质检报告
│   ├── auth.py          # 注册 / 登录
│   ├── users_api.py     # 用户信息 / 历史
│   ├── detection.py     # 检测任务
│   ├── models_api.py    # 模型文件管理
│   └── llm.py           # LLM 透传代理
└── uploads/             # 检测上传图片存储
```

## 快速验证

```bash
# 1. 注册
curl -X POST http://127.0.0.1:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username":"dev","password":"123456"}'

# 2. 登录
curl -X POST http://127.0.0.1:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"dev","password":"123456"}'
# → {"id":1,"username":"dev","token":"xxxx..."}

# 3. 用户信息（替换 TOKEN）
curl http://127.0.0.1:8000/api/v1/users/me \
  -H "Authorization: Bearer <TOKEN>"

# 4. 提交检测（上传图片）
curl -X POST http://127.0.0.1:8000/api/v1/detection \
  -H "Authorization: Bearer <TOKEN>" \
  -F "model_name=yolo11n" \
  -F "conf=0.3" \
  -F "iou=0.5" \
  -F "image=@test.jpg"

# 5. 上传模型
curl -X POST http://127.0.0.1:8000/api/v1/models/upload \
  -H "Authorization: Bearer <TOKEN>" \
  -F "file=@best_rsod.pt"
```

## 跨模块集成

训练模块通过 `hooks.py`（本目录下）调用后端 API，无需关心 HTTP 细节：

```python
from hooks import on_training_start, on_epoch_end, on_training_end

exp_id = on_training_start("rsod_exp_001", config_json, "rsod", "yolo11n.pt")
on_epoch_end(exp_id, epoch=1, metrics={"train_loss": 2.3, ...})
on_training_end(exp_id, map50=0.872, model_path="checkpoints/best_rsod_exp_001.pt")
```

## 开发约定

- **新增 API 文件** → `api/xxx.py`，创建 `APIRouter()`，在 `main.py` 中 `include_router`
- **新增 Schema** → 追加到 `schemas.py` 末尾，不动已有
- **新增表** → 追加到 `db/init_db.py` 的 `executescript` 中，`CREATE TABLE IF NOT EXISTS`
- **认证** → 需要登录的接口加 `user: dict = Depends(get_current_user)`
- **错误处理** → 用 `HTTPException(status_code=...)`，不裸抛异常
- **日志** → `logging.getLogger("odp-backend.xxx")`
