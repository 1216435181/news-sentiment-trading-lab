"""敏感性分析与信号诊断。

1. 二维参数敏感性: 平滑跨度 span ∈ {1,3,5} × 阈值 theta ∈ {0.02,0.05,0.10,0.15,0.20,0.30}
   × 仓位模式 {long_flat, long_short}
   —— 检验"跑输基准"的结论是否依赖某一组参数(避免调参至好看的单一设定)。
2. 多阶 IC: corr(S_t, r_{t+k}), k = 0..5
   —— 判断情感信号是否存在滞后/提前反应, 同时辅助排查未来函数。
"""
from __future__ import annotations

from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.backtest.engine import performance
from src.config import CONFIG, FIG_DIR, TABLE_DIR
from src.logging_utils import get_logger

log = get_logger("sensitivity")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

SPANS = [1, 3, 5]
THETAS = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
MODES = ["long_flat", "long_short"]


def _positions(s: pd.Series, theta: float, mode: str) -> pd.Series:
    pos, prev = [], 0.0
    for v in s.values:
        if np.isnan(v):
            pos.append(0.0)
            continue
        if v > theta:
            prev = 1.0
        elif v < -theta:
            prev = -1.0 if mode == "long_short" else 0.0
        pos.append(prev)
    return pd.Series(pos, index=s.index)


def theta_sensitivity(sig: pd.DataFrame) -> pd.DataFrame:
    cost = CONFIG["signal"]["cost_bps"] / 10000.0
    rows = []
    for span in SPANS:
        s = sig["S_raw"].ewm(span=span, adjust=False).mean() if span > 1 else sig["S_raw"]
        lag = s.shift(1)
        for mode in MODES:
            for theta in THETAS:
                pos = _positions(lag, theta, mode)
                turnover = pos.diff().abs().fillna(pos.abs())
                ret = pos * sig["ret"] - turnover * cost
                trades = int((pos.diff().fillna(pos.abs()) != 0).sum())
                m = performance(ret, pos, trades)
                rows.append({"position_mode": mode, "span": span, "theta": theta,
                             "total_return": m["total_return"],
                             "annual_return": m["annual_return"],
                             "sharpe": m["sharpe"],
                             "max_drawdown": m["max_drawdown"],
                             "trades": m["trades"], "exposure": m["exposure"]})
    df = pd.DataFrame(rows)
    df.to_csv(TABLE_DIR / "sensitivity_theta.csv", index=False, encoding="utf-8-sig")
    log.info("敏感性网格 %d 组完成", len(df))
    return df


def ic_decay(sig: pd.DataFrame, max_lag: int = 5) -> pd.DataFrame:
    rows: List[Dict] = []
    for k in range(max_lag + 1):
        fwd = sig["ret"].shift(-k) if k else sig["ret"]
        ic = sig["S"].corr(fwd, method="spearman")
        rows.append({"lag": k, "ic_spearman": round(float(ic), 4)})
    df = pd.DataFrame(rows)
    df.to_csv(TABLE_DIR / "ic_decay.csv", index=False, encoding="utf-8-sig")
    log.info("IC 衰减: %s", df.to_dict("records"))
    return df


def plot_all(sens: pd.DataFrame, ic: pd.DataFrame, bench: float) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.9))
    ax = axes[0]
    for (mode, span), g in sens.groupby(["position_mode", "span"]):
        ls = "-" if mode == "long_flat" else "--"
        ax.plot(g["theta"], g["annual_return"] * 100, marker="o", ms=3.5, ls=ls,
                label=f"{mode} span={span}")
    ax.axhline(bench * 100, color="#3b6fb6", lw=1.2, ls=":", label="基准年化")
    ax.axhline(0, color="#bbb", lw=0.8)
    ax.set_title("二维敏感性: 年化收益 vs 阈值 θ", fontsize=11, fontweight="bold")
    ax.set_xlabel("θ")
    ax.set_ylabel("年化收益 (%)")
    ax.legend(fontsize=6, frameon=False, ncol=2)
    ax.grid(True, color="#eee")

    ax2 = axes[1]
    ax2.bar(ic["lag"], ic["ic_spearman"], color="#3b6fb6", width=0.55)
    ax2.axhline(0, color="#999", lw=0.8)
    ax2.set_title("多阶 IC: corr(S_t, r_{t+k})", fontsize=11, fontweight="bold")
    ax2.set_xlabel("滞后阶数 k(交易日)")
    ax2.grid(True, color="#eee", axis="y")
    for i, v in enumerate(ic["ic_spearman"]):
        ax2.text(ic["lag"][i], v + (0.002 if v >= 0 else -0.005), f"{v:.3f}",
                 ha="center", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "sensitivity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("图已保存: %s", FIG_DIR / "sensitivity.png")


def run(sig: pd.DataFrame, bench_annual: float = 0.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    sens = theta_sensitivity(sig)
    ic = ic_decay(sig)
    plot_all(sens, ic, bench_annual)
    return sens, ic
