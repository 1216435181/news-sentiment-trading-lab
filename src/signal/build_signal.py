"""信号构建: 新闻情感 -> 日频信号 S_t (严格无未来函数)。

时间对齐约定(关键)
------------------
1. 一天内发布的新闻聚合为当日情感均值 S_t(按 `trade_date` 归集, 已剔除/前移非交易日)。
2. 交易发生在**次日**: 第 t 日的持仓由 S_{t-1} 决定, 收益取 t 日收益 r_t。
   代码上等价于 `signal.shift(1)`。
3. 收盘后(15:00 之后)发布的新闻, 在 T+1 开盘前已知, 因此可用于 T+1 交易;
   为保守起见本实验统一按"T 日新闻 -> T+1 日交易"处理, 不做日内抢跑。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import CONFIG, P
from src.logging_utils import get_logger

log = get_logger("signal")


def aggregate_daily(news: pd.DataFrame) -> pd.DataFrame:
    g = news.groupby("trade_date")
    daily = pd.DataFrame({
        "n_news": g.size(),
        "s_score_mean": g["sentiment_score_rule"].mean(),
        "s_score_std": g["sentiment_score_rule"].std(),
        "s_pos_ratio": g["trend_signal"].apply(lambda s: float((s == 1).mean())),
        "s_neg_ratio": g["trend_signal"].apply(lambda s: float((s == -1).mean())),
        "s_signal_sum": g["trend_signal"].sum(),
    }).reset_index()
    # 复合信号: 均值情感分 与 (利好-利空) 占比差 的等权组合
    daily["S_raw"] = 0.6 * daily["s_score_mean"] + 0.4 * (daily["s_pos_ratio"] - daily["s_neg_ratio"])
    return daily


def build_signal(news: pd.DataFrame, price: pd.DataFrame, tag: str = "main") -> pd.DataFrame:
    """tag=main 写主回测产物; tag=robust 写稳健性检验(仅财经早餐语料)产物。"""
    cfg = CONFIG["signal"]
    theta = cfg["theta"]

    daily = aggregate_daily(news)

    # 以交易日历为骨架, 缺新闻的交易日情感分记 0(中性), 并标记 n_news=0
    cal = price[["trade_date", "close", "ret"]].copy()
    df = cal.merge(daily, on="trade_date", how="left")
    for c in ["n_news", "s_signal_sum"]:
        df[c] = df[c].fillna(0).astype(int)
    for c in ["s_score_mean", "s_pos_ratio", "s_neg_ratio", "S_raw"]:
        df[c] = df[c].fillna(0.0)
    df["s_score_std"] = df["s_score_std"].fillna(0.0)

    # 平滑(可选): span=1 表示不平滑
    span = int(cfg.get("smoothing", 1) or 1)
    df["S"] = df["S_raw"].ewm(span=span, adjust=False).mean() if span > 1 else df["S_raw"]

    # ===== 无未来函数: 用 t-1 日信号决定 t 日持仓 =====
    df["S_lag1"] = df["S"].shift(1)
    df["n_news_lag1"] = df["n_news"].shift(1)

    pos = np.zeros(len(df))
    prev = 0.0
    for i, s in enumerate(df["S_lag1"].values):
        if np.isnan(s):
            pos[i] = 0.0
            continue
        if s > theta:
            prev = 1.0
        elif s < -theta:
            prev = -1.0 if cfg["position_mode"] == "long_short" else 0.0
        pos[i] = prev
    df["position"] = pos

    df["turnover"] = df["position"].diff().abs().fillna(df["position"].abs())
    cost = cfg["cost_bps"] / 10000.0
    df["strategy_ret"] = df["position"] * df["ret"] - df["turnover"] * cost
    df["bench_ret"] = df["ret"]

    # 信息系数 IC: S_t 与 r_{t+1} 的秩相关(诊断信号有效性)
    valid = df.dropna(subset=["S"]).copy()
    ic = valid["S"].corr(valid["ret"].shift(-1), method="spearman") if len(valid) > 5 else np.nan
    log.info("信号统计: theta=%.2f, 持仓天数=%d/%d, IC(S_t, r_t+1)=%.4f",
             theta, int((df["position"] != 0).sum()), len(df), ic)

    df["ic"] = ic
    out = P.daily_signal if tag == "main" else P.daily_signal_robust
    df.to_csv(out, index=False, encoding="utf-8-sig")
    return df


if __name__ == "__main__":
    news = pd.read_csv(P.labeled, encoding="utf-8-sig", parse_dates=["pub_time", "trade_date"])
    price = pd.read_csv(P.clean_price, encoding="utf-8-sig", parse_dates=["trade_date"])
    build_signal(news, price)
