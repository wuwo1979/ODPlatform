#!/usr/bin/env python
# @FileName  : init_db.py
# @Time      : 2026/5/25
# @Project   : ODPlatform
# @Function  : 数据库建表 —— 4 张核心表（幂等执行）

from __future__ import annotations

import sqlite3


def create_tables(db: sqlite3.Connection) -> None:
    """在给定连接上执行 CREATE TABLE IF NOT EXISTS（幂等）。

    四张表：
      1. experiments      — 实验主表
      2. training_epochs  — 每个 epoch 的训练指标
      3. models           — 注册的训练产出模型
      4. validation_reports — 数据质检报告历史
    """
    db.executescript("""
        -- ============================================================
        -- 1. 实验主表
        -- ============================================================
        CREATE TABLE IF NOT EXISTS experiments (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            name            TEXT NOT NULL UNIQUE,
            dataset         TEXT NOT NULL,
            model           TEXT NOT NULL,
            task            TEXT DEFAULT 'detect',
            config_json     TEXT NOT NULL,
            status          TEXT DEFAULT 'running',
            best_map50      REAL,
            best_map50_95   REAL,
            best_epoch      INTEGER,
            start_time      TEXT,
            end_time        TEXT,
            model_path      TEXT,
            created_at      TEXT DEFAULT (datetime('now'))
        );

        -- ============================================================
        -- 2. 训练 epoch 指标
        -- ============================================================
        CREATE TABLE IF NOT EXISTS training_epochs (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id   INTEGER REFERENCES experiments(id) ON DELETE CASCADE,
            epoch           INTEGER NOT NULL,
            train_loss      REAL,
            val_loss        REAL,
            map50           REAL,
            map50_95        REAL,
            precision       REAL,
            recall          REAL,
            lr              REAL,
            UNIQUE(experiment_id, epoch)
        );

        -- ============================================================
        -- 3. 模型注册表
        -- ============================================================
        CREATE TABLE IF NOT EXISTS models (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id   INTEGER REFERENCES experiments(id) ON DELETE CASCADE,
            filename        TEXT NOT NULL,
            format          TEXT DEFAULT 'pt',
            map50           REAL,
            map50_95        REAL,
            file_size_mb    REAL,
            created_at      TEXT DEFAULT (datetime('now'))
        );

        -- ============================================================
        -- 4. 数据质检报告
        -- ============================================================
        CREATE TABLE IF NOT EXISTS validation_reports (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset         TEXT NOT NULL,
            run_id          TEXT NOT NULL,
            passed          INTEGER DEFAULT 0,
            warnings        INTEGER DEFAULT 0,
            errors          INTEGER DEFAULT 0,
            report_json     TEXT,
            created_at      TEXT DEFAULT (datetime('now'))
        );

        -- ============================================================
        -- 索引（加速常见查询）
        -- ============================================================
        CREATE INDEX IF NOT EXISTS idx_experiments_dataset
            ON experiments(dataset);
        CREATE INDEX IF NOT EXISTS idx_experiments_status
            ON experiments(status);
        CREATE INDEX IF NOT EXISTS idx_epochs_experiment
            ON training_epochs(experiment_id, epoch);
        CREATE INDEX IF NOT EXISTS idx_models_experiment
            ON models(experiment_id);
        CREATE INDEX IF NOT EXISTS idx_validation_dataset
            ON validation_reports(dataset);
    """)
