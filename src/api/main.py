"""FastAPI 后端: 数据查询 + 回测结果 + 模型指标 接口。

启动: uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
文档: http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import json
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from src.config import CONFIG, DB_PATH, LOG_DIR, P

ROOT_LOGS = LOG_DIR
from src.store import query_df

app = FastAPI(
    title="新闻情感交易信号 API",
    description="实验后端: 新闻数据 / 情感信号 / 回测绩效 查询接口",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


def _records(df: pd.DataFrame) -> list[dict]:
    return json.loads(df.to_json(orient="records", force_ascii=False, date_format="iso"))


@app.get("/health")
def health():
    return {"status": "ok", "db": str(DB_PATH), "db_exists": DB_PATH.exists(),
            "index": CONFIG["data"]["index_name"]}


@app.get("/meta")
def meta():
    news = query_df("SELECT COUNT(*) n, MIN(trade_date) s, MAX(trade_date) e FROM news")
    price = query_df("SELECT COUNT(*) n, MIN(trade_date) s, MAX(trade_date) e FROM price")
    return {
        "news": news.to_dict("records")[0],
        "price": price.to_dict("records")[0],
        "clean_report": json.loads(P.clean_log.read_text(encoding="utf-8"))
        if P.clean_log.exists() else {},
        "config": {"theta": CONFIG["signal"]["theta"], "cost_bps": CONFIG["signal"]["cost_bps"],
                   "position_mode": CONFIG["signal"]["position_mode"]},
    }


@app.get("/news")
def news(
    limit: int = Query(50, le=500),
    offset: int = 0,
    q: Optional[str] = None,
    signal: Optional[int] = Query(None, ge=-1, le=1),
    start: Optional[str] = None,
    end: Optional[str] = None,
):
    sql = "SELECT * FROM news WHERE 1=1"
    params: list = []
    if q:
        sql += " AND (title LIKE ? OR content LIKE ?)"
        params += [f"%{q}%", f"%{q}%"]
    if signal is not None:
        sql += " AND trend_signal = ?"
        params.append(signal)
    if start:
        sql += " AND trade_date >= ?"
        params.append(start)
    if end:
        sql += " AND trade_date <= ?"
        params.append(end)
    total = query_df(sql.replace("SELECT *", "SELECT COUNT(*) AS n"), tuple(params)).iloc[0]["n"]
    sql += " ORDER BY pub_time DESC LIMIT ? OFFSET ?"
    params += [limit, offset]
    df = query_df(sql, tuple(params))
    return {"total": int(total), "limit": limit, "offset": offset, "items": _records(df)}


@app.get("/news/{news_id}")
def news_detail(news_id: int):
    df = query_df("SELECT * FROM news WHERE id = ?", (news_id,))
    if df.empty:
        raise HTTPException(404, "not found")
    return _records(df)[0]


@app.get("/price")
def price(start: Optional[str] = None, end: Optional[str] = None):
    sql, params = "SELECT * FROM price WHERE 1=1", []
    if start:
        sql += " AND trade_date >= ?"
        params.append(start)
    if end:
        sql += " AND trade_date <= ?"
        params.append(end)
    return _records(query_df(sql + " ORDER BY trade_date", tuple(params)))


@app.get("/signal")
def signal(start: Optional[str] = None, end: Optional[str] = None):
    sql, params = "SELECT * FROM daily_signal WHERE 1=1", []
    if start:
        sql += " AND trade_date >= ?"
        params.append(start)
    if end:
        sql += " AND trade_date <= ?"
        params.append(end)
    return _records(query_df(sql + " ORDER BY trade_date", tuple(params)))


@app.get("/backtest/metrics")
def backtest_metrics():
    df = query_df("SELECT * FROM backtest_metrics")
    if df.empty:
        raise HTTPException(404, "尚未运行回测")
    return _records(df)


@app.get("/backtest/equity")
def backtest_equity():
    df = query_df("""SELECT trade_date, equity_strategy, equity_bench, position,
                            strategy_ret, bench_ret FROM daily_signal ORDER BY trade_date""")
    return _records(df)


@app.get("/models/metrics")
def model_metrics():
    return {"ml": _records(query_df("SELECT * FROM model_metrics ORDER BY macro_f1 DESC")),
            "compare": _records(pd.read_csv(P.model_compare, encoding="utf-8-sig"))
            if P.model_compare.exists() else []}


@app.get("/llm/stability")
def llm_stability():
    df = query_df("SELECT * FROM llm_stability")
    if df.empty:
        raise HTTPException(404, "尚未运行 LLM 稳定性测试")
    row = df.iloc[0].to_dict()
    row["detail"] = json.loads(row.pop("detail_json"))
    return row


@app.get("/logs")
def logs(limit: int = 100):
    return _records(query_df(
        "SELECT * FROM pipeline_log ORDER BY id DESC LIMIT ?", (limit,)))


@app.get("/stats/daily")
def stats_daily():
    df = query_df("""SELECT trade_date, COUNT(*) n_news,
                            AVG(sentiment_score_rule) s_mean,
                            SUM(CASE WHEN trend_signal=1 THEN 1 ELSE 0 END) n_pos,
                            SUM(CASE WHEN trend_signal=-1 THEN 1 ELSE 0 END) n_neg
                     FROM news GROUP BY trade_date ORDER BY trade_date""")
    return _records(df)


# --------------------------------------------------------------------------- #
# 运维监控页(HTML): 数据库记录 + 流水线日志 + 后端日志尾部
# 用途: 答辩演示"数据层/后端日志", 也可作为部署自检页
# --------------------------------------------------------------------------- #
from fastapi.responses import HTMLResponse  # noqa: E402

TABLE_NAMES = ["news", "price", "daily_signal", "model_metrics",
               "backtest_metrics", "llm_stability", "pipeline_log"]


def _tail(path, n: int = 25) -> str:
    from pathlib import Path
    p = Path(path)
    if not p.exists():
        return "(暂无日志文件)"
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()[-n:]
    return "\n".join(lines)


@app.get("/monitor", response_class=HTMLResponse)
def monitor():
    counts = []
    for t in TABLE_NAMES:
        try:
            n = query_df(f"SELECT COUNT(*) AS n FROM {t}").iloc[0]["n"]
        except Exception:  # noqa: BLE001
            n = -1
        counts.append((t, int(n)))

    logs = query_df("""SELECT id, step, status, rows, message, created_at
                       FROM pipeline_log ORDER BY id DESC LIMIT 15""")
    rows = "".join(
        f"<tr><td>{r.id}</td><td>{r.step}</td><td class='{r.status}'>{r.status}</td>"
        f"<td>{'' if r.rows is None else r.rows}</td><td>{r.message}</td>"
        f"<td>{r.created_at}</td></tr>"
        for r in logs.itertuples())

    cards = "".join(f"<div class='card'><div class='k'>{t}</div>"
                    f"<div class='v'>{n:,}</div></div>" for t, n in counts)

    html = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>数据层监控</title><style>
 body{{font-family:-apple-system,'Microsoft YaHei',sans-serif;background:#f6f8fb;margin:0;padding:32px 40px;color:#1f2933}}
 h1{{font-size:22px;margin:0 0 4px}} .sub{{color:#6b7280;font-size:13px;margin-bottom:22px}}
 .grid{{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:26px}}
 .card{{background:#fff;border:1px solid #e5e8ee;border-radius:10px;padding:12px 18px;min-width:130px}}
 .card .k{{font-size:12px;color:#6b7280}} .card .v{{font-size:20px;font-weight:600;color:#d64545}}
 h2{{font-size:15px;margin:26px 0 10px}} table{{border-collapse:collapse;width:100%;background:#fff;font-size:12.5px;border:1px solid #e5e8ee}}
 th{{background:#f0f3f8;text-align:left;padding:7px 10px;font-weight:600;color:#374151}}
 td{{padding:6px 10px;border-top:1px solid #eef1f6}} .ok{{color:#15803d}} .error{{color:#b91c1c}}
 pre{{background:#0f172a;color:#cbd5e1;padding:14px 16px;border-radius:10px;font-size:11.5px;line-height:1.55;overflow:auto;max-height:330px}}
 .badge{{background:#d64545;color:#fff;border-radius:6px;padding:2px 8px;font-size:12px;margin-left:8px}}
</style></head><body>
<h1>数据层监控 <span class="badge">FastAPI :8000</span></h1>
<div class="sub">SQLite · {DB_PATH.name} · {CONFIG['data']['index_name']} · 接口 /health /news /signal /backtest/metrics /logs</div>
<div class="grid">{cards}</div>
<h2>流水线执行记录(pipeline_log)</h2>
<table><tr><th>id</th><th>step</th><th>status</th><th>rows</th><th>耗时</th><th>created_at</th></tr>{rows}</table>
<h2>后端日志 logs/pipeline.log(尾部 25 行)</h2>
<pre>{_tail(ROOT_LOGS / 'pipeline.log')}</pre>
<h2>API 访问日志 logs/api.log(尾部 25 行)</h2>
<pre>{_tail(ROOT_LOGS / 'api.log')}</pre>
</body></html>"""
    return HTMLResponse(html)
