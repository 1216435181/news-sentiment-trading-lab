"""回测引擎: 策略 vs Buy & Hold 基准。

指标
----
总收益率 / 年化收益 / 年化波动 / 夏普 / 最大回撤 / Calmar / 交易次数 / 胜率 / 持仓占比
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from src.config import CONFIG, P
from src.logging_utils import get_logger

log = get_logger("backtest")
TRADING_DAYS = CONFIG["backtest"]["trading_days"]
RF = CONFIG["backtest"]["rf_annual"]


def _max_drawdown(equity: pd.Series) -> tuple[float, str]:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    if len(dd) == 0:
        return 0.0, ""
    idx = dd.idxmin()
    return float(dd.min()), str(dd.index[idx].date()) if hasattr(dd.index[idx], "date") \
        else str(dd.index[idx])


def performance(ret: pd.Series, position: pd.Series | None = None,
                trades: int | None = None) -> Dict[str, float]:
    ret = pd.Series(ret).fillna(0.0)
    equity = (1.0 + ret).cumprod()
    n = len(ret)
    total = float(equity.iloc[-1] - 1.0) if n else 0.0
    years = n / TRADING_DAYS if n else 1.0
    ann = float((1.0 + total) ** (1.0 / years) - 1.0) if years > 0 and total > -1 else 0.0
    vol = float(ret.std(ddof=1) * np.sqrt(TRADING_DAYS)) if n > 1 else 0.0
    excess = ret - RF / TRADING_DAYS
    sharpe = float(excess.mean() / ret.std(ddof=1) * np.sqrt(TRADING_DAYS)) \
        if n > 1 and ret.std(ddof=1) > 0 else 0.0
    mdd, mdd_date = _max_drawdown(equity)
    calmar = float(ann / abs(mdd)) if mdd < 0 else 0.0
    wins = (ret > 0).sum()
    return {
        "total_return": round(total, 4),
        "annual_return": round(ann, 4),
        "annual_vol": round(vol, 4),
        "sharpe": round(sharpe, 3),
        "max_drawdown": round(mdd, 4),
        "max_dd_date": mdd_date,
        "calmar": round(calmar, 3),
        "n_days": n,
        "win_rate_daily": round(float(wins / max(n, 1)), 4),
        "best_day": round(float(ret.max()), 4) if n else 0.0,
        "worst_day": round(float(ret.min()), 4) if n else 0.0,
        "final_equity": round(float(equity.iloc[-1]), 4) if n else 1.0,
        "trades": int(trades) if trades is not None else int((ret != 0).sum()),
        "exposure": round(float((position.abs() > 0).mean()), 4) if position is not None else 1.0,
    }


def run(sig: pd.DataFrame, tag: str = "main") -> tuple[pd.DataFrame, pd.DataFrame]:
    df = sig.copy()
    df["equity_strategy"] = (1.0 + df["strategy_ret"]).cumprod()
    df["equity_bench"] = (1.0 + df["bench_ret"]).cumprod()
    df["dd_strategy"] = df["equity_strategy"] / df["equity_strategy"].cummax() - 1.0
    df["dd_bench"] = df["equity_bench"] / df["equity_bench"].cummax() - 1.0

    # 交易次数 = 仓位变动次数(单边计)
    n_round_trips = int((df["position"].diff().fillna(df["position"].abs()) != 0).sum())

    strat = performance(df["strategy_ret"], df["position"], n_round_trips)
    bench = performance(df["bench_ret"], pd.Series(1.0, index=df.index), 1)

    rows = []
    for name, m in [("情感策略(Sentiment L/S)", strat), ("基准(Buy & Hold)", bench)]:
        rows.append({"strategy": name, **m})
    perf = pd.DataFrame(rows)

    # 超额
    excess_ann = strat["annual_return"] - bench["annual_return"]
    perf.loc[perf.index[0], "excess_annual_return"] = round(excess_ann, 4)
    perf.loc[perf.index[1], "excess_annual_return"] = 0.0

    perf.to_csv(P.perf if tag == "main" else P.perf_robust, index=False, encoding="utf-8-sig")
    df.to_csv(P.equity if tag == "main" else P.equity_robust, index=False, encoding="utf-8-sig")

    log.info("回测结果(%d 个交易日):", len(df))
    for _, r in perf.iterrows():
        log.info("  %-26s 总收益=%7.2f%% 年化=%7.2f%% 夏普=%5.2f 最大回撤=%6.2f%% 交易=%d",
                 r["strategy"], r["total_return"] * 100, r["annual_return"] * 100,
                 r["sharpe"], r["max_drawdown"] * 100, r["trades"])
    return perf, df


if __name__ == "__main__":
    d = pd.read_csv(P.daily_signal, encoding="utf-8-sig", parse_dates=["trade_date"])
    perf, eq = run(d)
    print(perf.T)
