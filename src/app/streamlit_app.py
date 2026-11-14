"""Streamlit 前端: 3 个页面(数据概览 / 情感分析 / 回测结果)。

启动: streamlit run src/app/streamlit_app.py
数据源: 默认直连 SQLite(零依赖); 侧栏可切换为调用 FastAPI。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.config import CONFIG, DB_PATH, P  # noqa: E402
from src.store import query_df  # noqa: E402

st.set_page_config(page_title="新闻情感交易信号实验台", page_icon="📈", layout="wide")

# ----------------------------------------------------------------- 数据访问
@st.cache_data(ttl=30)
def q(sql: str, params: tuple = ()) -> pd.DataFrame:
    return query_df(sql, params)


def api_get(path: str):
    import requests
    try:
        r = requests.get(f"{st.session_state.api_base}{path}", timeout=4)
        r.raise_for_status()
        return r.json()
    except Exception:  # noqa: BLE001
        return None


PAGES = ["1. 数据概览", "2. 情感分析结果", "3. 回测结果", "4. 系统与日志"]
_default_idx = 0
try:  # 支持 ?page=2 直接进入指定页面(便于截图与分享)
    _p = st.query_params.get("page")
    if _p and str(_p).isdigit():
        _default_idx = max(0, min(len(PAGES) - 1, int(_p) - 1))
except Exception:  # noqa: BLE001
    pass

# ----------------------------------------------------------------- 侧栏
with st.sidebar:
    st.title("📈 情感信号实验台")
    page = st.radio("页面", PAGES, index=_default_idx)
    st.divider()
    source = st.selectbox("数据源", ["SQLite (直连)", "FastAPI"])
    st.session_state.api_base = st.text_input(
        "API 地址", value=st.session_state.get("api_base", "http://127.0.0.1:8000"))
    st.caption(f"DB: {DB_PATH.name} · 状态: {'已就绪' if DB_PATH.exists() else '未初始化'}")
    st.caption(f"标的: {CONFIG['data']['index_name']} · θ={CONFIG['signal']['theta']}")


def use_api() -> bool:
    return source == "FastAPI"


# ================================================================= 页面 1
if page.startswith("1"):
    st.header("数据概览")
    news = q("SELECT * FROM news")
    price = q("SELECT * FROM price")
    funnel = json.loads(P.clean_log.read_text(encoding="utf-8")) if P.clean_log.exists() else {}

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("新闻条数", f"{len(news):,}")
    c2.metric("交易日数", f"{news['trade_date'].nunique() if len(news) else 0}")
    c3.metric("日均新闻", f"{len(news) / max(news['trade_date'].nunique(), 1):.1f}" if len(news) else "0")
    c4.metric("清洗保留率", f"{funnel.get('keep_rate', 0) * 100:.1f}%")
    c5.metric("行情行数", f"{len(price):,}")

    st.subheader("清洗漏斗(清洗前后对比)")
    if funnel:
        f = pd.DataFrame([
            {"阶段": "原始抓取", "条数": funnel["raw_rows"]},
            {"阶段": "去空/去重/异常文本后", "条数": funnel["clean_rows"]},
        ])
        col1, col2 = st.columns([1, 1])
        col1.dataframe(f, hide_index=True, use_container_width=True)
        label_map = {
            "bad_time": "时间解析失败", "empty": "标题/正文为空",
            "short_or_garbled": "过短或乱码", "dup_title_time": "标题+时间重复",
            "dup_title": "标题重复(转载)", "future_time": "未来时间",
            "non_trading_day": "非交易日(剔除)", "no_next_trading_day": "无对应交易日",
            "total": "合计",
        }
        drops = {label_map.get(k.replace("drop_", ""), k): v
                 for k, v in funnel.items() if k.startswith("drop_")}
        col2.dataframe(pd.DataFrame(list(drops.items()), columns=["剔除原因", "条数"]),
                       hide_index=True, use_container_width=True)
        st.caption(f"数据区间 {funnel.get('time_start', '')[:19]} ~ {funnel.get('time_end', '')[:19]}"
                   f" · 来源分布 {funnel.get('by_source', {})}")

    st.subheader("每日新闻量")
    daily = q("""SELECT trade_date, COUNT(*) n_news,
                        SUM(CASE WHEN trend_signal=1 THEN 1 ELSE 0 END) n_pos,
                        SUM(CASE WHEN trend_signal=-1 THEN 1 ELSE 0 END) n_neg
                 FROM news GROUP BY trade_date ORDER BY trade_date""")
    if len(daily):
        daily["trade_date"] = pd.to_datetime(daily["trade_date"])
        st.bar_chart(daily.set_index("trade_date")[["n_pos", "n_neg"]], height=260)

    st.subheader("行情与新闻覆盖")
    if len(price):
        price["trade_date"] = pd.to_datetime(price["trade_date"])
        st.line_chart(price.set_index("trade_date")["close"], height=240)

    st.subheader("新闻明细")
    kw = st.text_input("关键词搜索(标题/正文)", "")
    sig = st.selectbox("情感标签", ["全部", "利好(1)", "中性(0)", "利空(-1)"])
    sql = "SELECT pub_time, trade_date, title, source, symbol, sentiment_score_rule, trend_signal FROM news WHERE 1=1"
    params: list = []
    if kw:
        sql += " AND (title LIKE ? OR content LIKE ?)"
        params += [f"%{kw}%", f"%{kw}%"]
    if sig != "全部":
        sql += " AND trend_signal = ?"
        params.append(int(sig[-2]))
    df = q(sql + " ORDER BY pub_time DESC LIMIT 300", tuple(params))
    st.dataframe(df, hide_index=True, use_container_width=True, height=380)
    st.caption(f"当前显示 {len(df)} 条(上限 300)")

# ================================================================= 页面 2
elif page.startswith("2"):
    st.header("情感分析结果")
    news = q("SELECT trend_signal, sentiment_score_rule, trade_date FROM news")
    c1, c2, c3, c4 = st.columns(4)
    if len(news):
        c1.metric("利好占比", f"{(news['trend_signal'] == 1).mean() * 100:.1f}%")
        c2.metric("中性占比", f"{(news['trend_signal'] == 0).mean() * 100:.1f}%")
        c3.metric("利空占比", f"{(news['trend_signal'] == -1).mean() * 100:.1f}%")
        c4.metric("平均情感分", f"{news['sentiment_score_rule'].mean():.3f}")

    st.subheader("标签分布")
    dist = news["trend_signal"].value_counts().rename(
        index={-1: "利空(-1)", 0: "中性(0)", 1: "利好(1)"})
    st.bar_chart(dist, height=220)

    st.subheader("模型对比表(ML / LLM)")
    if P.model_compare.exists():
        cmp_df = pd.read_csv(P.model_compare, encoding="utf-8-sig")
        st.dataframe(cmp_df, hide_index=True, use_container_width=True)
        best = cmp_df[cmp_df["类别"] == "ML"].sort_values("Macro-F1", ascending=False).iloc[0]
        st.info(f"ML 最优: {best['模型']} · Macro-F1={best['Macro-F1']:.4f} · "
                f"单条延迟 {best['单条延迟(ms)']:.3f} ms")

    st.subheader("LLM 稳定性测试")
    try:
        st_json = q("SELECT * FROM llm_stability").iloc[0]
        s = json.loads(st_json["detail_json"])
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("样本数", int(st_json["n_samples"]))
        k2.metric("signal 一致率", f"{st_json['agreement_rate'] * 100:.1f}%")
        k3.metric("情感分平均绝对偏差", f"{st_json['mean_abs_deviation']:.4f}")
        k4.metric("重复运行次数", "2 次(T=0 / T=0.7)")
        st.dataframe(pd.DataFrame(s)[["title", "label_rule", "signal_t0", "signal_t07",
                                      "score_t0", "score_t07", "agree", "abs_diff"]]
                     .head(50), hide_index=True, use_container_width=True, height=320)
    except Exception:  # noqa: BLE001
        st.warning("尚无 LLM 稳定性结果, 请先运行流水线")

    st.subheader("情感信号时序")
    sig_df = q("SELECT trade_date, S, n_news FROM daily_signal ORDER BY trade_date")
    if len(sig_df):
        sig_df["trade_date"] = pd.to_datetime(sig_df["trade_date"])
        st.line_chart(sig_df.set_index("trade_date")[["S"]], height=260)
    st.caption("S_t = 0.6 × 日均情感分 + 0.4 × (利好占比 − 利空占比), 为次日交易的唯一输入。")

# ================================================================= 页面 3
elif page.startswith("3"):
    st.header("回测结果(策略 vs 基准)")
    perf = q("SELECT * FROM backtest_metrics")
    sig = q("SELECT * FROM daily_signal ORDER BY trade_date")
    if len(sig):
        sig["trade_date"] = pd.to_datetime(sig["trade_date"])

    if len(perf):
        st.subheader("绩效对比表")
        show = perf[["strategy", "total_return", "annual_return", "annual_vol", "sharpe",
                     "max_drawdown", "trades", "win_rate_daily", "exposure"]].copy()
        st.dataframe(show, hide_index=True, use_container_width=True)
        st_ = perf.iloc[0]
        b_ = perf.iloc[1]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("策略总收益", f"{st_['total_return'] * 100:.2f}%",
                  f"{(st_['total_return'] - b_['total_return']) * 100:+.2f}% vs 基准")
        c2.metric("策略夏普", f"{st_['sharpe']:.2f}", f"{st_['sharpe'] - b_['sharpe']:+.2f}")
        c3.metric("最大回撤", f"{st_['max_drawdown'] * 100:.2f}%",
                  f"{(st_['max_drawdown'] - b_['max_drawdown']) * 100:+.2f}%")
        c4.metric("交易次数", f"{int(st_['trades'])}")

    st.subheader("净值曲线")
    if len(sig):
        eq = sig.set_index("trade_date")[["equity_strategy", "equity_bench"]]
        eq.columns = ["情感策略", "Buy & Hold"]
        st.line_chart(eq, height=340)

        st.subheader("仓位与回撤")
        st.area_chart(sig.set_index("trade_date")[["position"]], height=140)
        dd = pd.DataFrame({
            "情感策略": sig.set_index("trade_date")["equity_strategy"].pipe(
                lambda s: s / s.cummax() - 1) * 100,
            "Buy & Hold": sig.set_index("trade_date")["equity_bench"].pipe(
                lambda s: s / s.cummax() - 1) * 100,
        })
        st.line_chart(dd, height=200)

    st.subheader("逐日信号与收益")
    if len(sig):
        cols = ["trade_date", "n_news", "S", "S_lag1", "position",
                "strategy_ret", "bench_ret", "equity_strategy", "equity_bench"]
        st.dataframe(sig[cols].sort_values("trade_date", ascending=False).head(200),
                     hide_index=True, use_container_width=True, height=340)
    st.caption("仓位由 S_{t-1} 决定(shift(1)), 当日收益为 t 日指数收益 → 严格无未来函数。")

# ================================================================= 页面 4
else:
    st.header("系统与日志")
    st.subheader("数据库表记录数")
    tables = ["news", "price", "daily_signal", "model_metrics",
              "backtest_metrics", "llm_stability", "pipeline_log"]
    rows = []
    for t in tables:
        try:
            n = q(f"SELECT COUNT(*) AS n FROM {t}").iloc[0]["n"]
        except Exception:  # noqa: BLE001
            n = -1
        rows.append({"表名": t, "记录数": int(n)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("DB 文件", DB_PATH.name)
    c2.metric("DB 大小(KB)", f"{DB_PATH.stat().st_size / 1024:.0f}" if DB_PATH.exists() else "0")
    health = api_get("/health") if use_api() else None
    c3.metric("API 状态", "在线" if health else ("未连接" if use_api() else "未启用"))

    st.subheader("流水线日志(pipeline_log)")
    logs = q("SELECT id, step, status, rows, message, created_at FROM pipeline_log ORDER BY id")
    st.dataframe(logs, hide_index=True, use_container_width=True)

    st.subheader("运行日志(文件尾)")
    if (ROOT / "logs" / "pipeline.log").exists():
        lines = (ROOT / "logs" / "pipeline.log").read_text(encoding="utf-8").splitlines()[-40:]
        st.code("\n".join(lines), language="log")
    else:
        st.info("尚无日志文件, 请先运行 python -m scripts.run_pipeline")

    st.subheader("当前配置")
    st.json(CONFIG)
