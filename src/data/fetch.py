"""数据采集: AkShare 新闻 + 指数日线。

新闻源(AkShare 免费接口, 无需 token)
------------------------------------
1. `stock_news_cjzc_em`  东方财富"财经早餐", **每日 06:00 发布, 覆盖 2025-03 至今**,
   正文含多条编号要闻, 按 "1、" "2、" 切分为单条新闻 —— 长历史、密度稳定的市场级语料,
   是回测主区间的数据基础。
2. `stock_news_em`       个股新闻(按股票池批量抓取), 近期高密度、个股级细节。
3. `stock_news_global_em` 东财全球财经快讯(近 12 小时滚动)。
4. `stock_news_global_sina` 新浪全球快讯(仅"时间/内容"两列, 用正文前 40 字作标题)。
5. `stock_news_global_cls` 财联社电报(当日)。

行情
----
`stock_zh_index_daily(symbol="sh000300")` 新浪沪深300 指数全历史日线(本地按日期截取);
失败时依次降级到 `index_zh_a_hist` / `stock_zh_a_hist("510300")`。

所有原始数据落盘 data/raw/, 后续步骤只读本地文件, 保证可复现。
"""
from __future__ import annotations

import re
import time
from typing import Iterable, List

import pandas as pd

from src.config import CONFIG, RAW_DIR
from src.logging_utils import get_logger

log = get_logger("fetch")

TIME_COLS = ["发布时间", "发布日期", "时间", "日期", "pub_time", "datetime"]
TITLE_COLS = ["新闻标题", "标题", "title"]
CONTENT_COLS = ["新闻内容", "内容", "摘要", "content"]

CJZC_ITEM_RE = re.compile(r"\s*\d{1,2}、")
CJZC_HEADER_RE = re.compile(r"^【[^】]*】")


def _pick_col(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _normalize(df: pd.DataFrame, source: str, symbol: str = "") -> pd.DataFrame:
    """把各来源异构字段统一成 pub_time/title/content/source/symbol。"""
    if df is None or len(df) == 0:
        return pd.DataFrame()

    tcol = _pick_col(df, TIME_COLS)
    title_col = _pick_col(df, TITLE_COLS)
    content_col = _pick_col(df, CONTENT_COLS)

    out = pd.DataFrame()
    if title_col is None and content_col is not None:
        # 新浪快讯只有正文, 用前 40 字作为标题
        out["title"] = df[content_col].astype(str).str.slice(0, 40)
    elif title_col is not None:
        out["title"] = df[title_col].astype(str)
    else:
        return pd.DataFrame()

    out["content"] = df[content_col].astype(str) if content_col else out["title"]
    if tcol is None:
        out["pub_time"] = pd.NaT
    else:
        raw_time = df[tcol].astype(str)
        if tcol == "发布时间" and "发布日期" in df.columns:      # 财联社: 日期 + 时间两列
            raw_time = df["发布日期"].astype(str) + " " + raw_time
        elif tcol == "时间" and "日期" in df.columns:
            raw_time = df["日期"].astype(str) + " " + raw_time
        out["pub_time"] = pd.to_datetime(raw_time, errors="coerce")
    out["source"] = source
    out["symbol"] = symbol
    return out


# ---------------------------------------------------------------- 新闻源
def fetch_cjzc() -> pd.DataFrame:
    """财经早餐: 一条记录切成多条要闻, 形成长历史语料。"""
    import akshare as ak

    try:
        df = ak.stock_news_cjzc_em()
    except Exception as exc:  # noqa: BLE001
        log.warning("  stock_news_cjzc_em 抓取失败: %s", exc)
        return pd.DataFrame()
    if df is None or not len(df):
        return pd.DataFrame()

    tcol = _pick_col(df, TIME_COLS)
    ccol = _pick_col(df, CONTENT_COLS)
    rows = []
    for _, r in df.iterrows():
        text = CJZC_HEADER_RE.sub("", str(r.get(ccol, "")))
        raw_time = pd.to_datetime(r.get(tcol), errors="coerce")
        if pd.isna(raw_time):
            continue
        for part in CJZC_ITEM_RE.split(text):
            part = part.strip("。 ；;")
            if len(part) < 12:      # 过滤"国内新闻"这类小标题
                continue
            rows.append({
                "pub_time": raw_time,
                "title": part[:60],
                "content": part,
                "source": "eastmoney_cjzc",
                "symbol": "",
            })
    out = pd.DataFrame(rows)
    log.info("  stock_news_cjzc_em: %d 篇早餐 -> 切分 %d 条要闻 (%s ~ %s)",
             len(df), len(out), out["pub_time"].min(), out["pub_time"].max())
    return out


def fetch_stock_news(symbols: List[str], sleep: float = 0.35) -> pd.DataFrame:
    import akshare as ak

    frames = []
    for i, code in enumerate(symbols, 1):
        try:
            df = ak.stock_news_em(symbol=code)
            norm = _normalize(df, "eastmoney_stock_news", code)
            if len(norm):
                frames.append(norm)
            if i % 20 == 0 or i == len(symbols):
                log.info("  个股新闻进度 [%d/%d], 累计 %d 条", i, len(symbols),
                         sum(len(f) for f in frames))
        except Exception as exc:  # noqa: BLE001
            log.warning("  [%02d/%02d] %s 抓取失败: %s", i, len(symbols), code, exc)
        time.sleep(sleep)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fetch_global_news() -> pd.DataFrame:
    import akshare as ak

    specs = [
        ("stock_news_global_em", "eastmoney_global", {}),
        ("stock_news_global_sina", "sina_global", {}),
        ("stock_news_global_cls", "cls_telegraph", {"symbol": "全部"}),
    ]
    frames = []
    for fn_name, src, kwargs in specs:
        fn = getattr(ak, fn_name, None)
        if fn is None:
            log.warning("AkShare 无接口 %s, 跳过", fn_name)
            continue
        try:
            norm = _normalize(fn(**kwargs), src)
            if len(norm):
                frames.append(norm)
            log.info("  %s 抓到 %d 条", fn_name, len(norm))
        except Exception as exc:  # noqa: BLE001
            log.warning("  %s 抓取失败: %s", fn_name, exc)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fetch_news() -> pd.DataFrame:
    cfg = CONFIG["data"]
    log.info("=== 开始抓取新闻 ===")
    parts = [fetch_cjzc(), fetch_global_news()]
    try:
        parts.append(fetch_stock_news(cfg["news_symbols"]))
    except Exception as exc:  # noqa: BLE001
        log.warning("个股新闻整体失败: %s", exc)
    parts = [p for p in parts if p is not None and len(p)]
    if not parts:
        raise RuntimeError("所有新闻接口均抓取失败, 请检查网络")

    df = pd.concat(parts, ignore_index=True)
    df = df.dropna(subset=["pub_time"]).sort_values("pub_time").reset_index(drop=True)
    out = RAW_DIR / "news_raw.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    log.info("新闻原始数据: %d 条 (%s ~ %s) -> %s", len(df),
             df["pub_time"].min(), df["pub_time"].max(), out)
    return df


# ---------------------------------------------------------------- 行情
def fetch_price() -> pd.DataFrame:
    import akshare as ak

    cfg = CONFIG["data"]
    symbol, start, end = cfg["index_symbol"], cfg["start_date"], cfg["end_date"]
    log.info("=== 开始抓取行情: %s ===", symbol)

    df = None
    # 1) 新浪指数全历史(最稳定, 不会被 push2 限流)
    try:
        df = ak.stock_zh_index_daily(symbol=f"sh{symbol}")
        log.info("  stock_zh_index_daily 成功, %d 行", len(df))
    except Exception as exc:  # noqa: BLE001
        log.warning("  stock_zh_index_daily 失败: %s", exc)
        df = None

    if df is not None and len(df):
        df = df.rename(columns={"date": "trade_date"})
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df = df[(df["trade_date"] >= pd.to_datetime(start)) &
                (df["trade_date"] <= pd.to_datetime(end))]
    else:
        for fn_name, kw in [
            ("index_zh_a_hist", {"symbol": symbol, "period": "daily",
                                 "start_date": start, "end_date": end}),
            ("stock_zh_a_hist", {"symbol": "510300", "period": "daily",
                                 "start_date": start, "end_date": end, "adjust": "qfq"}),
        ]:
            try:
                df = getattr(ak, fn_name)(**kw)
                if df is not None and len(df):
                    log.info("  使用接口 %s 成功, %d 行", fn_name, len(df))
                    df = df.rename(columns={
                        "日期": "trade_date", "开盘": "open", "收盘": "close",
                        "最高": "high", "最低": "low", "成交量": "volume", "成交额": "amount"})
                    break
            except Exception as exc:  # noqa: BLE001
                log.warning("  %s 失败: %s", fn_name, exc)
                df = None

    if df is None or not len(df):
        raise RuntimeError("行情抓取失败")

    keep = [c for c in ["trade_date", "open", "high", "low", "close", "volume", "amount"]
            if c in df.columns]
    df = df[keep].copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.normalize()
    for c in keep[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["close"]).drop_duplicates("trade_date")
    df = df.sort_values("trade_date").reset_index(drop=True)

    out = RAW_DIR / "price_raw.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    log.info("行情数据: %d 行 (%s ~ %s) -> %s", len(df),
             df["trade_date"].min().date(), df["trade_date"].max().date(), out)
    return df


if __name__ == "__main__":
    fetch_news()
    fetch_price()
