"""全局配置加载与路径管理。"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
DB_DIR = DATA_DIR / "db"
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"
TABLE_DIR = REPORT_DIR / "tables"
LOG_DIR = ROOT / "logs"

for _d in (RAW_DIR, PROC_DIR, DB_DIR, FIG_DIR, TABLE_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

DB_PATH = DB_DIR / "market.db"


def load_config(path: Path | None = None) -> Dict[str, Any]:
    cfg_path = path or (ROOT / "config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


CONFIG = load_config()


@dataclass
class Paths:
    raw_news: Path = RAW_DIR / "news_raw.csv"
    raw_price: Path = RAW_DIR / "price_raw.csv"
    clean_news: Path = PROC_DIR / "news_clean.csv"
    clean_price: Path = PROC_DIR / "price_clean.csv"
    clean_log: Path = PROC_DIR / "clean_report.json"
    labeled: Path = PROC_DIR / "news_labeled.csv"
    ml_metrics: Path = TABLE_DIR / "ml_metrics.csv"
    llm_stability: Path = TABLE_DIR / "llm_stability.csv"
    model_compare: Path = TABLE_DIR / "model_compare.csv"
    daily_signal: Path = PROC_DIR / "daily_signal.csv"
    equity: Path = PROC_DIR / "equity_curve.csv"
    perf: Path = TABLE_DIR / "performance.csv"
    daily_signal_robust: Path = PROC_DIR / "daily_signal_robust.csv"
    equity_robust: Path = PROC_DIR / "equity_curve_robust.csv"
    perf_robust: Path = TABLE_DIR / "performance_robust.csv"
    fig_equity: Path = FIG_DIR / "equity_curve.png"
    fig_sentiment: Path = FIG_DIR / "sentiment_timeseries.png"
    fig_confusion: Path = FIG_DIR / "confusion_matrix.png"
    fig_drawdown: Path = FIG_DIR / "drawdown.png"
    report_md: Path = REPORT_DIR / "REPORT.md"


P = Paths()


def get_env(key: str, default: str | None = None) -> str | None:
    """读取 .env 中的变量(不依赖 python-dotenv)。"""
    if key in os.environ:
        return os.environ[key]
    for name in (ROOT / ".env", ROOT.parent / ".env"):
        if name.exists():
            for line in name.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k.strip() == key:
                    return v.strip().strip('"').strip("'")
    return default
