"""图表生成: 净值曲线 / 回撤 / 情感时序 / 混淆矩阵。"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import P
from src.logging_utils import get_logger

log = get_logger("plots")

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

C_STRAT = "#d64545"
C_BENCH = "#3b6fb6"
C_GRID = "#e6e6e6"


def _style(ax):
    ax.grid(True, color=C_GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def plot_equity(eq: pd.DataFrame, path=P.fig_equity) -> str:
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1]})
    x = pd.to_datetime(eq["trade_date"])
    ax.plot(x, eq["equity_strategy"], color=C_STRAT, lw=1.8, label="情感策略")
    ax.plot(x, eq["equity_bench"], color=C_BENCH, lw=1.6, ls="--", label="Buy & Hold")
    ax.axhline(1.0, color="#999", lw=0.8)
    ax.set_ylabel("净值")
    ax.set_title("净值曲线: 新闻情感策略 vs 买入持有基准", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", frameon=False)
    _style(ax)

    ax2.fill_between(x, 0, eq["position"], step="mid", color="#8a8f98", alpha=0.55, lw=0)
    ax2.set_ylabel("仓位")
    ax2.set_yticks([-1, 0, 1])
    ax2.set_ylim(-1.2, 1.2)
    _style(ax2)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("图已保存: %s", path)
    return str(path)


def plot_drawdown(eq: pd.DataFrame, path=P.fig_drawdown) -> str:
    fig, ax = plt.subplots(figsize=(11, 3.6))
    x = pd.to_datetime(eq["trade_date"])
    ax.fill_between(x, eq["dd_strategy"] * 100, 0, color=C_STRAT, alpha=0.35, label="情感策略")
    ax.plot(x, eq["dd_bench"] * 100, color=C_BENCH, lw=1.3, ls="--", label="Buy & Hold")
    ax.set_ylabel("回撤 (%)")
    ax.set_title("回撤对比", fontsize=12, fontweight="bold")
    ax.legend(loc="lower left", frameon=False)
    _style(ax)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_sentiment(sig: pd.DataFrame, path=P.fig_sentiment) -> str:
    fig, ax = plt.subplots(figsize=(11, 4.2))
    x = pd.to_datetime(sig["trade_date"])
    ax.bar(x, sig["n_news"], color="#cfd8e3", width=0.8, label="当日新闻数")
    ax.set_ylabel("新闻条数", color="#4a5568")
    ax2 = ax.twinx()
    ax2.plot(x, sig["S"], color=C_STRAT, lw=1.6, label="情感信号 S_t")
    ax2.axhline(0, color="#999", lw=0.7)
    theta = float(sig["S"].abs().quantile(0.7)) if len(sig) else 0.1
    ax2.axhline(theta, color=C_STRAT, lw=0.7, ls=":")
    ax2.axhline(-theta, color=C_STRAT, lw=0.7, ls=":")
    ax2.set_ylabel("情感分 S_t")
    ax.set_title("日频情感信号与新闻热度", fontsize=13, fontweight="bold")
    _style(ax)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_confusion(detail: dict, path=P.fig_confusion) -> str:
    cm = np.array(detail["confusion_matrix"])
    labels = ["利空(-1)", "中性(0)", "利好(1)"]
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(cm, cmap="Reds")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() * 0.6 else "#333", fontsize=11)
    ax.set_xticks(range(3), labels, fontsize=9)
    ax.set_yticks(range(3), labels, fontsize=9)
    ax.set_xlabel("预测")
    ax.set_ylabel("真实(弱标注)")
    ax.set_title(f"混淆矩阵: {detail['best_model']}", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def run_all(sig: pd.DataFrame) -> list[str]:
    eq = pd.read_csv(P.equity, encoding="utf-8-sig", parse_dates=["trade_date"])
    detail = json.loads((P.ml_metrics.parent / "ml_detail.json").read_text(encoding="utf-8"))
    paths = [plot_equity(eq), plot_drawdown(eq), plot_sentiment(sig), plot_confusion(detail)]
    return paths
