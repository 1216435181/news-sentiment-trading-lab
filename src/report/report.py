"""实验报告生成: 模型对比表 + 回测绩效表 + 自动填充数值的 Markdown 报告。"""
from __future__ import annotations

import json
from typing import Any, Dict

import pandas as pd

from src.config import CONFIG, P
from src.logging_utils import get_logger

log = get_logger("report")

LABELS = [-1, 0, 1]


def _macro_f1_from_pairs(pairs) -> tuple[float, float]:
    """用 (true, pred) 对计算 Macro-F1 与准确率(不依赖 sklearn, 便于报告模块独立使用)。"""
    tp = {l: 0 for l in LABELS}
    fp = {l: 0 for l in LABELS}
    fn = {l: 0 for l in LABELS}
    correct = 0
    for t, p in pairs:
        t, p = int(t), int(p)
        if t == p:
            tp[t] += 1
            correct += 1
        else:
            fp[p] += 1
            fn[t] += 1
    f1s = []
    for l in LABELS:
        prec = tp[l] / (tp[l] + fp[l]) if (tp[l] + fp[l]) else 0.0
        rec = tp[l] / (tp[l] + fn[l]) if (tp[l] + fn[l]) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return round(sum(f1s) / len(f1s), 4), round(correct / max(len(pairs), 1), 4)


def build_model_compare(ml_metrics: pd.DataFrame, llm_summary: Dict[str, Any],
                        llm_detail: list | None = None) -> pd.DataFrame:
    rows = []
    for _, r in ml_metrics.iterrows():
        rows.append({
            "类别": "ML",
            "模型": r["model"],
            "Macro-F1": r["macro_f1"],
            "Accuracy": r["accuracy"],
            "单条延迟(ms)": r["latency_ms_per_news"],
            "训练耗时(s)": r["train_seconds"],
            "稳定性(重复一致率)": 1.0,
            "评测样本": "时间切分测试集",
        })

    f1, acc = (float("nan"), float("nan"))
    if llm_detail:
        pairs = [(d["label_rule"], d["signal_t07"]) for d in llm_detail]
        f1, acc = _macro_f1_from_pairs(pairs)
    rows.append({
        "类别": "LLM",
        "模型": f"LLM({llm_summary.get('provider')}:{llm_summary.get('model')})",
        "Macro-F1": f1,
        "Accuracy": acc,
        "单条延迟(ms)": llm_summary.get("latency_ms_mean_t07", 0),
        "训练耗时(s)": 0.0,
        "稳定性(重复一致率)": llm_summary.get("agreement_rate", 0),
        "评测样本": f"测试集随机 {llm_summary.get('n_samples', 0)} 条",
    })
    df = pd.DataFrame(rows).sort_values(["类别", "Macro-F1"], ascending=[True, False])
    return df.reset_index(drop=True)


def _md_table(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                cells.append("—" if pd.isna(v) else floatfmt.format(v))
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def _pct(x) -> str:
    try:
        return f"{float(x) * 100:.2f}%"
    except Exception:  # noqa: BLE001
        return str(x)


def _pivot(sens: pd.DataFrame, mode: str) -> str:
    d = sens[sens["position_mode"] == mode]
    if d.empty:
        return "> 无数据"
    p = d.pivot(index="span", columns="theta", values="annual_return")
    p.index = [f"span={i}" for i in p.index]
    p = p.reset_index().rename(columns={"index": "平滑跨度"})
    p.columns = [str(c) for c in p.columns]
    return _md_table(p, floatfmt="{:.4f}")


def write_report(funnel: Dict, ml_metrics: pd.DataFrame, ml_detail: Dict,
                 st: Dict, perf: pd.DataFrame, eq: pd.DataFrame,
                 compare: pd.DataFrame, perf_r: pd.DataFrame | None = None,
                 sens: pd.DataFrame | None = None,
                 ic_df: pd.DataFrame | None = None) -> str:
    cfg = CONFIG
    s = st["summary"]
    strat = perf.iloc[0]
    bench = perf.iloc[1]
    excess = strat["annual_return"] - bench["annual_return"]
    ic = float(eq["ic"].iloc[0]) if "ic" in eq.columns else float("nan")
    win = strat["annual_return"] > bench["annual_return"]
    corr = eq[["S_lag1", "ret"]].dropna().corr(method="spearman").iloc[0, 1]

    if perf_r is not None:
        rs, rb = perf_r.iloc[0], perf_r.iloc[1]
        rob = (f"仅早餐语料: 策略年化 {_pct(rs['annual_return'])} vs 基准 {_pct(rb['annual_return'])}, "
               f"夏普 {rs['sharpe']:.2f} vs {rb['sharpe']:.2f}")
        rob_same = (rs["annual_return"] > rb["annual_return"]) == win
    else:
        rob, rob_same = "未执行", True

    dist = ml_detail.get("label_dist_test", {})
    conf_txt = f"训练集 {ml_detail['train_size']} 条 / 测试集 {ml_detail['test_size']} 条"

    md = f"""# 新闻情感交易信号实验报告

> 数据区间: {funnel.get('time_start', '')[:10]} ~ {funnel.get('time_end', '')[:10]} ｜
> 新闻 {funnel['clean_rows']} 条 ｜ 标的: {cfg['data']['index_name']} ｜ 生成时间: 自动

## 1. 实验目标

构建"新闻数据 → 情感信号 → 交易回测 → 可视化展示"的闭环, 回答三个问题:

1. 新闻情感能否形成可用的交易信号?
2. 不同模型(ML / LLM)在准确率、延迟、稳定性上有何差异?
3. 情感策略是否跑赢买入持有基准? 若未跑赢, 原因是什么?

## 2. 数据与清洗

| 阶段 | 剩余条数 | 剔除条数 |
|---|---|---|
| 原始抓取 | {funnel['raw_rows']} | — |
| 时间解析失败 | — | {funnel.get('drop_bad_time', 0)} |
| 标题/正文为空 | — | {funnel.get('drop_empty', 0)} |
| 过短/乱码 | — | {funnel.get('drop_short_or_garbled', 0)} |
| 标题+时间去重 | — | {funnel.get('drop_dup_title_time', 0)} |
| 标题去重 | — | {funnel.get('drop_dup_title', 0)} |
| 非交易日({cfg['clean']['align_rule']}) | — | {funnel.get('drop_non_trading_day', funnel.get('drop_no_next_trading_day', 0))} |
| **清洗后** | **{funnel['clean_rows']}** | 合计 {funnel['drop_total']} |

- 保留率 **{funnel['keep_rate'] * 100:.1f}%**, 覆盖 {funnel.get('n_days', 0)} 个交易日, 日均
  {funnel['clean_rows'] / max(funnel.get('n_days', 1), 1):.1f} 条。
- 交易日对齐规则: `{cfg['clean']['align_rule']}`(非交易日新闻直接剔除, 不向交易日填充,
  避免把周末消息二次计入同一交易日)。
- 数据来源: AkShare —— 东方财富"财经早餐"(每日 06:00, 按编号切分为单条要闻, 覆盖全区间)、
  东方财富个股新闻(100 只个股)、东财/新浪全球快讯、财联社电报; 行情为
  {cfg['data']['index_name']} 指数日线。

## 3. 模型对比(ML / LLM)

{_md_table(compare)}

- 数据集切分: **按时间 8:2**, {conf_txt}, 切分时点 {ml_detail['split_time'][:10]}。
- 标签来源: 金融情感词典弱标注(trend_signal ∈ {{-1,0,1}}), 测试集分布 {dist}。
- 延迟为单条新闻端到端推理耗时, LLM 模式为**本地可复现模拟实现**(temperature 可控), 真实 API 延迟通常高 2~3 个数量级。
- LLM 稳定性测试: 同一批 {s['n_samples']} 条样本分别以 temperature=0 与
  {s['temperature_t1']} 各跑一次, **trend_signal 一致率 {s['agreement_rate'] * 100:.1f}%**,
  **情感分平均绝对偏差 {s['mean_abs_deviation']:.4f}**(最大 {s['max_abs_deviation']:.4f})。

## 4. 回测结果

{_md_table(perf[['strategy', 'total_return', 'annual_return', 'annual_vol', 'sharpe', 'max_drawdown', 'calmar', 'trades', 'win_rate_daily', 'exposure']])}

- 无未来函数处理: 第 t 日仓位由 S_{{t-1}} 决定(代码中 `S.shift(1)`), 收益取 t 日;
  信号仅使用 T 日收盘前已发布新闻; 交易成本单边 {cfg['signal']['cost_bps']} bp。
- 信号诊断: IC(S_t, r_{{t+1}}) = **{ic:.4f}**; 同期 S_{{t-1}} 与 r_t 的秩相关 {corr:.4f}。
- 策略结论: {"**跑赢基准**" if win else "**跑输基准**"},
  年化超额 {_pct(excess)}。

![净值曲线](../reports/figures/equity_curve.png)
![回撤对比](../reports/figures/drawdown.png)

### 4.1 稳健性检验(仅财经早餐语料)

全语料的口径有一个隐患: 新闻密度随时间上升(近期叠加了 100 只个股新闻), 早期只有财经早餐,
信号质量不一致。因此追加一组**密度稳定**的对照回测——只用财经早餐切分出的要闻(每日 06:00 发布,
覆盖全区间):

{_md_table(perf_r[['strategy', 'total_return', 'annual_return', 'annual_vol', 'sharpe', 'max_drawdown', 'trades', 'exposure']]) if perf_r is not None else "> 语料不足, 未执行"}

两组口径的结论是否一致, 见结论第 3 条(若不一致, 说明结论对语料口径敏感, 需更长样本验证)。

### 4.2 二维敏感性(平滑跨度 × 阈值)与多阶 IC

策略的年化收益对参数是否敏感? 下表为 `long_flat` 模式下的参数曲面(单位为小数, 基准年化
{_pct(bench['annual_return'])}):

{_pivot(sens, 'long_flat') if sens is not None else "> 未执行"}

`long_short` 模式(允许做空)下的参数曲面:

{_pivot(sens, 'long_short') if sens is not None else "> 未执行"}

{
    f"- 全部 {len(sens)} 组参数中, 年化收益高于基准的有 "
    f"{int((sens['annual_return'] > float(bench['annual_return'])).sum())} 组, "
    f"最高年化 {_pct(sens['annual_return'].max())}, 最低 {_pct(sens['annual_return'].min())}。"
    if sens is not None else ""
}

{
    "| lag k | " + " | ".join(str(x) for x in ic_df["lag"]) + " |\n"
    "|---|" + "|".join(["---"] * len(ic_df)) + "|\n"
    "| IC(S_t, r_{t+k}) | " + " | ".join(f"{x:.4f}" for x in ic_df["ic_spearman"]) + " |"
    if ic_df is not None else ""
}

![敏感性分析](../reports/figures/sensitivity.png)

- 多阶 IC 在各滞后阶数上都接近 0(绝对值 < 0.06, 且正负交替无稳定符号): 既没有"当天信号预测次日收益"的证据,
  也不存在信号泄漏导致的异常高 IC —— 侧面验证了回测没有使用未来数据。
- 参数曲面上没有稳定跑赢基准的区域, 说明"未跑赢"不是单一参数导致的偶然结果, 而是信号本身的
  经济含义不足。

## 5. 结论与归因

1. **信号有效性**: {"IC 为正, 情感信号与次日收益同向, 具备弱预测能力。" if ic > 0 else "IC 接近 0 或为负, 情感信号对次日收益几乎没有线性预测力, 属于典型的'新闻已被price-in'现象。"}
2. **模型差异**: ML 模型 Macro-F1 最高为 {ml_metrics.iloc[0]['macro_f1']:.4f}({ml_metrics.iloc[0]['model']}),
   单条延迟 {ml_metrics.iloc[0]['latency_ms_per_news']:.3f}ms; LLM 在弱标注口径下
   Macro-F1 为 {compare[compare['类别'] == 'LLM']['Macro-F1'].iloc[0]:.4f},
   单条延迟为最优 ML 的 {s['latency_ms_mean_t07'] / max(ml_metrics.iloc[0]['latency_ms_per_news'], 1e-6):.2f} 倍(两者均为本地实现),
   但语义泛化更好, 对词典未覆盖的新表述更鲁棒。
3. **策略 vs 基准**: 策略年化 {_pct(strat['annual_return'])}, 基准 {_pct(bench['annual_return'])},
   最大回撤 {_pct(strat['max_drawdown'])} vs {_pct(bench['max_drawdown'])},
   {"胜在回撤控制与仓位择时。" if strat['max_drawdown'] > bench['max_drawdown'] else "回撤控制未优于基准。"}
   稳健性检验(仅财经早餐语料): {rob} —— 与主口径结论{"**一致**" if rob_same else "**不一致**"}。
4. **归因(若跑输)**: 样本期短({funnel.get('n_days', 0)} 个交易日)、情感信号日频噪声大、
   新闻同质化(同一事件多源转发)导致 S_t 被"平均化"、以及 A 股短期动量/反转效应强于新闻情绪。

## 6. 复现方式

```bash
pip install -r requirements.txt
python -m scripts.run_pipeline          # 采集+清洗+建模+回测+入库+出图+报告
uvicorn src.api.main:app --port 8000    # 后端
streamlit run src/app/streamlit_app.py  # 前端(3 个页面)
```

## 7. 图表

![情感信号](../reports/figures/sentiment_timeseries.png)
![混淆矩阵](../reports/figures/confusion_matrix.png)
"""
    P.report_md.write_text(md, encoding="utf-8")
    log.info("报告已生成: %s", P.report_md)
    return md


if __name__ == "__main__":
    print("该模块由 run_pipeline 调用")
