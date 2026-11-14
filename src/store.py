"""SQLite 数据层: 把流水线产物写入数据库, 供 FastAPI / Streamlit 查询。

选型说明: 实验文档提到 MySQL, 本项目默认使用 SQLite(单文件、零配置、可直接进仓库外),
schema 与 SQL 写法与 MySQL 兼容, 若需切换只需改 `SQLALCHEMY_URL` / 连接串。
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from typing import Any, Dict, Iterable, List

import pandas as pd

from src.config import DB_PATH, P
from src.logging_utils import get_logger

log = get_logger("store")

SCHEMA = """
DROP TABLE IF EXISTS news;
CREATE TABLE news (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pub_time TEXT NOT NULL,
    trade_date TEXT NOT NULL,
    title TEXT NOT NULL,
    content TEXT,
    source TEXT,
    symbol TEXT,
    sentiment_score_rule REAL,
    trend_signal INTEGER
);
CREATE INDEX idx_news_date ON news(trade_date);

DROP TABLE IF EXISTS price;
CREATE TABLE price (
    trade_date TEXT PRIMARY KEY,
    open REAL, high REAL, low REAL, close REAL,
    volume REAL, amount REAL, ret REAL
);

DROP TABLE IF EXISTS daily_signal;
CREATE TABLE daily_signal (
    trade_date TEXT PRIMARY KEY,
    n_news INTEGER,
    s_score_mean REAL,
    S REAL,
    S_lag1 REAL,
    position REAL,
    turnover REAL,
    strategy_ret REAL,
    bench_ret REAL,
    equity_strategy REAL,
    equity_bench REAL
);

DROP TABLE IF EXISTS model_metrics;
CREATE TABLE model_metrics (
    model TEXT PRIMARY KEY,
    macro_f1 REAL, weighted_f1 REAL, accuracy REAL,
    f1_neg REAL, f1_neu REAL, f1_pos REAL,
    train_seconds REAL, latency_ms_per_news REAL
);

DROP TABLE IF EXISTS backtest_metrics;
CREATE TABLE backtest_metrics (
    strategy TEXT PRIMARY KEY,
    total_return REAL, annual_return REAL, annual_vol REAL,
    sharpe REAL, max_drawdown REAL, calmar REAL,
    n_days INTEGER, win_rate_daily REAL, trades INTEGER, exposure REAL,
    excess_annual_return REAL
);

DROP TABLE IF EXISTS llm_stability;
CREATE TABLE llm_stability (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    agreement_rate REAL,
    mean_abs_deviation REAL,
    n_samples INTEGER,
    detail_json TEXT
);

DROP TABLE IF EXISTS pipeline_log;
CREATE TABLE pipeline_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    step TEXT, status TEXT, rows INTEGER, message TEXT,
    created_at TEXT DEFAULT (datetime('now', 'localtime'))
);
"""


@contextmanager
def connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with connect() as conn:
        conn.executescript(SCHEMA)
    log.info("数据库已初始化: %s", DB_PATH)


def _insert_df(conn, table: str, df: pd.DataFrame, cols: List[str]) -> int:
    d = df.copy()
    for c in cols:
        if c not in d.columns:
            d[c] = None
    rows = d[cols].astype(object).where(pd.notnull(d[cols]), None).values.tolist()
    ph = ",".join(["?"] * len(cols))
    conn.executemany(f"INSERT OR REPLACE INTO {table} ({','.join(cols)}) VALUES ({ph})", rows)
    return len(rows)


def load_all(news: pd.DataFrame, price: pd.DataFrame, sig: pd.DataFrame,
             ml_metrics: pd.DataFrame, perf: pd.DataFrame,
             stability: Dict[str, Any], steps: Iterable[Dict[str, Any]] = ()) -> Dict[str, int]:
    init_db()
    counts = {}
    with connect() as conn:
        counts["news"] = _insert_df(conn, "news", news.assign(
            pub_time=news["pub_time"].astype(str),
            trade_date=news["trade_date"].astype(str),
        ), ["pub_time", "trade_date", "title", "content", "source", "symbol",
            "sentiment_score_rule", "trend_signal"])
        counts["price"] = _insert_df(conn, "price", price.assign(
            trade_date=price["trade_date"].astype(str)), list(price.columns))
        counts["daily_signal"] = _insert_df(conn, "daily_signal", sig.assign(
            trade_date=sig["trade_date"].astype(str)), [
            "trade_date", "n_news", "s_score_mean", "S", "S_lag1", "position",
            "turnover", "strategy_ret", "bench_ret", "equity_strategy", "equity_bench"])
        counts["model_metrics"] = _insert_df(conn, "model_metrics", ml_metrics, [
            "model", "macro_f1", "weighted_f1", "accuracy", "f1_neg", "f1_neu", "f1_pos",
            "train_seconds", "latency_ms_per_news"])
        counts["backtest_metrics"] = _insert_df(conn, "backtest_metrics", perf, [
            "strategy", "total_return", "annual_return", "annual_vol", "sharpe",
            "max_drawdown", "calmar", "n_days", "win_rate_daily", "trades", "exposure",
            "excess_annual_return"])
        s = stability["summary"]
        conn.execute(
            "INSERT OR REPLACE INTO llm_stability (id, agreement_rate, mean_abs_deviation,"
            " n_samples, detail_json) VALUES (1,?,?,?,?)",
            (s["agreement_rate"], s["mean_abs_deviation"], s["n_samples"],
             json.dumps(stability["detail"], ensure_ascii=False)))
        counts["llm_stability"] = 1
        for st in steps:
            conn.execute("INSERT INTO pipeline_log (step, status, rows, message) VALUES (?,?,?,?)",
                         (st.get("step"), st.get("status"), st.get("rows"), st.get("message")))
    log.info("数据库写入完成: %s", counts)
    return counts


def query_df(sql: str, params: tuple = ()) -> pd.DataFrame:
    with connect() as conn:
        return pd.read_sql_query(sql, conn, params=params)


if __name__ == "__main__":
    init_db()
    print(query_df("SELECT name FROM sqlite_master WHERE type='table'"))
