"""端到端流水线: 采集 -> 清洗 -> 弱标注 -> ML -> LLM 稳定性 -> 信号 -> 回测 -> 入库 -> 图表 -> 报告。

用法:
    python -m scripts.run_pipeline              # 全流程
    python -m scripts.run_pipeline --skip-fetch # 复用 data/raw 下的原始数据
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime

import pandas as pd

from src.backtest import engine as bt
from src.backtest import sensitivity
from src.config import CONFIG, P
from src.data import clean as cleaner
from src.data import fetch as fetcher
from src.logging_utils import get_logger
from src.nlp import ml_model
from src.nlp.llm_client import stability_test
from src.nlp.weak_label import label_news
from src.report import plots, report
from src.signal.build_signal import build_signal
from src.store import load_all

log = get_logger("pipeline")
STEPS = []


def step(name: str, fn, *args, **kwargs):
    t0 = time.perf_counter()
    log.info("========== [STEP] %s ==========", name)
    out = fn(*args, **kwargs)
    dur = time.perf_counter() - t0
    rows = len(out[0]) if isinstance(out, tuple) and hasattr(out[0], "__len__") else None
    STEPS.append({"step": name, "status": "ok", "rows": rows,
                  "message": f"{dur:.1f}s"})
    log.info("========== [DONE] %s (%.1fs) ==========", name, dur)
    return out


def main(skip_fetch: bool = False) -> None:
    t_start = datetime.now()
    log.info("流水线启动 %s", t_start)

    # 1) 采集
    if skip_fetch:
        raw_news = pd.read_csv(P.raw_news, encoding="utf-8-sig")
        raw_price = pd.read_csv(P.raw_price, encoding="utf-8-sig")
        log.info("跳过采集, 复用原始数据: %d 条新闻 / %d 行行情", len(raw_news), len(raw_price))
    else:
        raw_news = step("data_fetch_news", fetcher.fetch_news)
        raw_price = step("data_fetch_price", fetcher.fetch_price)

    # 2) 清洗
    news, price, funnel = step("data_clean", cleaner.run)

    # 3) 弱标注
    labeled = step("weak_label", label_news, news)
    labeled.to_csv(P.labeled, index=False, encoding="utf-8-sig")

    # 4) ML 基线(时间切分)
    ml_metrics, ml_detail, test_pred = step("ml_train_eval", ml_model.run, labeled)

    # 5) LLM 稳定性测试(测试集随机抽 50 条)
    cfg_llm = CONFIG["llm"]
    n = min(cfg_llm["stability_sample_size"], len(test_pred))
    sample = test_pred.sample(n=n, random_state=CONFIG["ml"]["random_state"])
    sample_full = labeled.merge(sample[["pub_time", "title"]], on=["pub_time", "title"], how="inner")
    sample_full = sample_full.drop_duplicates(subset=["title"]).head(n)
    st = step("llm_stability", stability_test, sample_full.to_dict("records"))
    pd.DataFrame(st["detail"]).to_csv(P.llm_stability, index=False, encoding="utf-8-sig")

    # 6) 模型对比表(ML + LLM)
    compare = report.build_model_compare(ml_metrics, st["summary"], st["detail"])
    compare.to_csv(P.model_compare, index=False, encoding="utf-8-sig")

    # 7) 信号 + 回测(主: 全语料)
    sig = step("build_signal", build_signal, labeled, price, "main")
    perf, eq = step("backtest", bt.run, sig, "main")

    # 7b) 稳健性检验: 仅用财经早餐语料(新闻密度稳定, 长历史), 排除"后期新闻变密"的干扰
    cjzc = labeled[labeled["source"] == "eastmoney_cjzc"]
    perf_r, eq_r = None, None
    if len(cjzc) > 200:
        sig_r = step("build_signal_robust", build_signal, cjzc, price, "robust")
        perf_r, eq_r = step("backtest_robust", bt.run, sig_r, "robust")
    else:
        log.warning("财经早餐语料不足(%d 条), 跳过稳健性检验", len(cjzc))

    # 8) 图表
    step("plots", plots.run_all, eq)

    # 8b) 敏感性分析 + 多阶 IC 诊断
    sens, ic_decay_df = step("sensitivity", sensitivity.run, sig, float(perf.iloc[1]["annual_return"]))

    # 9) 入库
    counts = step("db_load", load_all, labeled, price, eq, ml_metrics, perf, st, STEPS)

    # 10) 报告
    step("report", report.write_report, funnel, ml_metrics, ml_detail, st, perf, eq,
         compare, perf_r, sens, ic_decay_df)

    elapsed = (datetime.now() - t_start).total_seconds()
    log.info("流水线完成, 耗时 %.1fs, 数据库写入 %s", elapsed, counts)
    print("\n===== 关键结果 =====")
    print(compare.to_string(index=False))
    print()
    print(perf[["strategy", "total_return", "annual_return", "sharpe",
                "max_drawdown", "trades"]].to_string(index=False))
    if perf_r is not None:
        print("\n--- 稳健性检验(仅财经早餐语料) ---")
        print(perf_r[["strategy", "total_return", "annual_return", "sharpe",
                      "max_drawdown", "trades"]].to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-fetch", action="store_true")
    args = ap.parse_args()
    main(skip_fetch=args.skip_fetch)
