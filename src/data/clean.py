"""数据清洗: 去重 / 去空 / 时间统一 / 异常文本 / 交易日对齐。

产出:
  data/processed/news_clean.csv
  data/processed/price_clean.csv
  data/processed/clean_report.json   (清洗漏斗, 供报告与前端"数据概览"页使用)
"""
from __future__ import annotations

import json
import re
from typing import Dict, Tuple

import pandas as pd

from src.config import CONFIG, P
from src.logging_utils import get_logger

log = get_logger("clean")

URL_RE = re.compile(r"https?://\S+|www\.\S+")
HTML_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")
# 乱码/无意义字符(非中文、非英文数字、非常用标点)
BAD_CHAR_RE = re.compile(r"[^\u4e00-\u9fa5A-Za-z0-9，。、；：？！“”‘’（）《》%\-—…·\s]")


def _normalize_title(s: str) -> str:
    s = HTML_RE.sub("", str(s))
    s = URL_RE.sub("", s)
    s = SPACE_RE.sub(" ", s).strip()
    return s.strip("【】[] ").strip()


def _normalize_content(s: str) -> str:
    s = HTML_RE.sub(" ", str(s))
    s = URL_RE.sub(" ", s)
    return SPACE_RE.sub(" ", s).strip()


def _garbled_ratio(s: str) -> float:
    if not s:
        return 1.0
    bad = len(BAD_CHAR_RE.findall(s))
    return bad / max(len(s), 1)


def clean_news(raw: pd.DataFrame, trading_days: pd.Series | None = None) -> Tuple[pd.DataFrame, Dict]:
    cfg = CONFIG["clean"]
    funnel = {"raw_rows": int(len(raw))}

    df = raw.copy()

    # 1) 时间格式统一 (强制 datetime, 无法解析的丢弃)
    df["pub_time"] = pd.to_datetime(df["pub_time"], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["pub_time"])
    funnel["drop_bad_time"] = int(before - len(df))
    df["pub_time"] = df["pub_time"].dt.floor("min")

    # 2) 文本规范化
    df["title"] = df["title"].map(_normalize_title)
    df["content"] = df["content"].map(_normalize_content)

    # 3) 去空(标题或正文为空)
    before = len(df)
    df = df[(df["title"].str.len() > 0) & (df["content"].str.len() > 0)]
    funnel["drop_empty"] = int(before - len(df))

    # 4) 异常文本: 过短 / 乱码 / 纯符号
    before = len(df)
    mask_len = (df["title"].str.len() >= cfg["min_title_len"]) & (
        df["content"].str.len() >= cfg["min_content_len"]
    )
    mask_garbled = df.apply(
        lambda r: _garbled_ratio(r["title"]) < 0.3 and _garbled_ratio(r["content"]) < 0.3, axis=1
    )
    df = df[mask_len & mask_garbled]
    funnel["drop_short_or_garbled"] = int(before - len(df))

    # 5) 超长正文截断(不改行数, 只截断)
    df["content"] = df["content"].str.slice(0, cfg["max_content_len"])

    # 6) 去重: 标题+时间(精确), 再按标题去重(同一稿多次转发, 保留最早发布时间)
    before = len(df)
    df["title_key"] = df["title"].str.replace(r"\s+", "", regex=True)
    df = df.drop_duplicates(subset=["title_key", "pub_time"], keep="first")
    funnel["drop_dup_title_time"] = int(before - len(df))
    before = len(df)
    df = df.sort_values("pub_time").drop_duplicates(subset=["title_key"], keep="first")
    funnel["drop_dup_title"] = int(before - len(df))

    # 7) 时间合理性: 不能晚于当前时间
    now = pd.Timestamp.now()
    before = len(df)
    df = df[df["pub_time"] <= now]
    funnel["drop_future_time"] = int(before - len(df))

    # 8) 交易日对齐
    df["trade_date"] = df["pub_time"].dt.normalize()
    if trading_days is not None and len(trading_days):
        tdays = pd.DatetimeIndex(sorted(pd.to_datetime(trading_days).unique()))
        if cfg["align_rule"] == "ffill":
            # 非交易日新闻归到下一个交易日
            idx = tdays.searchsorted(df["trade_date"], side="left")
            mapped = pd.Series(
                [tdays[i] if i < len(tdays) else pd.NaT for i in idx], index=df.index
            )
            before = len(df)
            df["trade_date"] = mapped
            df = df.dropna(subset=["trade_date"])
            funnel["drop_no_next_trading_day"] = int(before - len(df))
        else:
            before = len(df)
            df = df[df["trade_date"].isin(tdays)]
            funnel["drop_non_trading_day"] = int(before - len(df))

    df = df.sort_values("pub_time").reset_index(drop=True)
    funnel["clean_rows"] = int(len(df))
    funnel["drop_total"] = funnel["raw_rows"] - funnel["clean_rows"]
    funnel["keep_rate"] = round(funnel["clean_rows"] / max(funnel["raw_rows"], 1), 4)

    cols = ["pub_time", "trade_date", "title", "content", "source", "symbol"]
    df = df[[c for c in cols if c in df.columns]]

    # 文本长度统计
    funnel["title_len_mean"] = round(float(df["title"].str.len().mean()), 2) if len(df) else 0
    funnel["content_len_mean"] = round(float(df["content"].str.len().mean()), 2) if len(df) else 0
    funnel["time_start"] = str(df["pub_time"].min()) if len(df) else None
    funnel["time_end"] = str(df["pub_time"].max()) if len(df) else None
    funnel["n_days"] = int(df["trade_date"].nunique()) if len(df) else 0
    funnel["by_source"] = df["source"].value_counts().to_dict() if len(df) else {}
    log.info("清洗漏斗: %s", json.dumps(funnel, ensure_ascii=False))
    return df, funnel


def clean_price(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.normalize()
    before = len(df)
    df = df.drop_duplicates(subset=["trade_date"], keep="last").sort_values("trade_date")
    df = df[df["close"] > 0]
    df["ret"] = df["close"].pct_change().fillna(0.0)
    log.info("行情清洗: %d -> %d 行", before, len(df))
    return df.reset_index(drop=True)


def run() -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    raw_news = pd.read_csv(P.raw_news, encoding="utf-8-sig")
    raw_price = pd.read_csv(P.raw_price, encoding="utf-8-sig")
    price = clean_price(raw_price)
    news, funnel = clean_news(raw_news, price["trade_date"])

    news.to_csv(P.clean_news, index=False, encoding="utf-8-sig")
    price.to_csv(P.clean_price, index=False, encoding="utf-8-sig")
    P.clean_log.write_text(json.dumps(funnel, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("已写出: %s (%d 条) / %s (%d 行)", P.clean_news, len(news), P.clean_price, len(price))
    return news, price, funnel


if __name__ == "__main__":
    run()
