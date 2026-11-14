# 新闻情感交易信号实验(News Sentiment → Signal → Backtest → Dashboard)

从 **新闻数据 → 情感信号 → 交易回测 → 可视化展示** 的完整闭环实现。
数据全部来自 AkShare 免费接口(无需 token), 全流程一条命令复现。

```
采集(AkShare) → 清洗(去重/去空/对齐) → 弱标注(金融情感词典) → ML 基线(4 个)
     ↓                                                        ↓
SQLite ← 回测(策略 vs Buy&Hold, 无未来函数) ← 日频信号 S_t ← LLM 稳定性测试
     ↓
FastAPI(接口) + Streamlit(4 个页面) + 图表 + Markdown/PDF 报告
```

## 结果速览

### 1. 模型对比(按时间 8:2 切分, 测试集 379 条)

| 类别 | 模型 | Macro-F1 | Accuracy | 单条延迟(ms) | 训练(s) | 稳定性(重复一致率) |
|---|---|---|---|---|---|---|
| ML | TF-IDF + LinearSVC | **0.7605** | **0.8127** | 0.317 | 0.33 | 1.00 |
| ML | TF-IDF + LogisticRegression | 0.7460 | 0.7942 | 0.317 | 0.48 | 1.00 |
| ML | TF-IDF + SVD + MLP | 0.6711 | 0.7361 | 0.332 | 1.14 | 1.00 |
| ML | TF-IDF + ComplementNB | 0.5821 | 0.6148 | 0.323 | 0.31 | 1.00 |
| LLM | 本地可复现模拟 LLM(温度可控) | 0.6888 | 0.7600 | 0.391 | — | **0.76** |

- 准确率类指标由固定随机种子保证可复现; 单条延迟随机器与负载浮动, 上表为本次运行结果。
- LLM 稳定性测试: 同批 50 条样本 `temperature=0` vs `0.7` 各跑一次,
  **signal 一致率 76.0%**, 情感分平均绝对偏差 **0.1248**(最大 0.2899)。
- ML 推理比 LLM 快、且完全确定; LLM 优势在语义泛化, 代价是采样不确定性。

### 2. 回测(沪深300, 2025-03-20 ~ 2026-11-13, 415 个交易日, 单边成本 5bp)

| 策略 | 总收益 | 年化 | 年化波动 | 夏普 | 最大回撤 | 交易次数 | 持仓占比 |
|---|---|---|---|---|---|---|---|
| 情感策略(Sentiment) | 1.74% | 1.06% | 15.79% | 0.146 | **-20.46%** | 31 | 79.3% |
| 基准(Buy & Hold) | **3.29%** | **1.98%** | 16.95% | **0.201** | -21.23% | 1 | 100% |

- **未跑赢基准**: 年化 -0.92%, 但最大回撤低 0.77pct。
- 稳健性(仅用每日财经早餐语料): 策略年化 1.00% vs 基准 1.98%, 结论一致。
- 敏感性(平滑跨度 × 阈值共 36 组参数): 仅 2 组年化超过基准, 无稳定 alpha 区域。
- 多阶 IC(S_t vs r_{t+k}, k=0..5): 0.051 / -0.008 / 0.027 / 0.047 / -0.011 / 0.037 ——
  全部接近 0 且正负交替, 既是"信号无预测力"的证据, 也是"没有未来函数"的旁证。

### 3. 三个问题的回答

1. **新闻情感能否形成可用交易信号?** —— 在当前语料与日频颗粒度下**不能**。
   多阶 IC 近 0, 参数曲面上不存在稳定跑赢基准的区域。
2. **ML / LLM 差异?** —— ML 精度更高(0.76 vs 0.69)、延迟低、100% 确定;
   LLM 语义泛化好但采样带来 24% 的标签翻转, 且真实 API 延迟高 2~3 个数量级。
3. **是否跑赢基准? 为什么没有?** —— 没有。原因见 `reports/REPORT.md` 第 5 节:
   新闻被 price-in(公开快讯发布时已被交易)、日频聚合把个股信号"平均化"失败、
   语料密度随时间变化、A 股短期反转效应强于新闻情绪、交易成本与空仓期错失上涨。

## 目录结构

```
.
├── config.yaml                  # 全部实验参数(股票池/阈值/切分/成本/LLM)
├── requirements.txt
├── src/
│   ├── config.py                # 路径与配置加载
│   ├── logging_utils.py         # 控制台 + logs/pipeline.log 双写
│   ├── data/fetch.py            # AkShare 采集(财经早餐切分 / 个股新闻 / 快讯 / 指数日线)
│   ├── data/clean.py            # 去重、去空、乱码过滤、交易日对齐, 输出清洗漏斗
│   ├── nlp/lexicon.py           # 金融情感词典(正/负词 + 程度副词 + 否定词)
│   ├── nlp/weak_label.py        # 弱标注 trend_signal ∈ {-1,0,1}
│   ├── nlp/ml_model.py          # 4 个 ML 基线 + 时间切分 + Macro-F1
│   ├── nlp/llm_client.py        # LLM 客户端(mock / OpenAI 兼容) + 稳定性测试
│   ├── signal/build_signal.py   # 日频信号 S_t(shift(1), 严格无未来函数)
│   ├── backtest/engine.py       # 回测引擎 + 绩效指标
│   ├── backtest/sensitivity.py  # 二维参数敏感性 + 多阶 IC
│   ├── report/plots.py          # 净值/回撤/情感时序/混淆矩阵
│   ├── report/report.py         # 自动生成 Markdown 报告
│   ├── store.py                 # SQLite 数据层(schema + 写入 + 查询)
│   ├── api/main.py              # FastAPI 后端(含 /monitor 运维页)
│   └── app/streamlit_app.py     # Streamlit 前端(4 个页面)
├── scripts/
│   ├── run_pipeline.py          # 一键跑完整实验
│   ├── screenshots.py           # Playwright 驱动本机 Edge 自动截图
│   └── make_report_pdf.py       # Markdown → HTML → PDF
├── data/{raw,processed,db}/     # 原始/清洗后数据、SQLite
├── reports/
│   ├── REPORT.md / REPORT.pdf   # 实验报告
│   ├── figures/                 # 净值曲线、回撤、情感时序、混淆矩阵、敏感性
│   └── tables/                  # 模型对比、绩效、敏感性、LLM 稳定性
└── docs/
    ├── DATA.md                  # 数据说明文件(来源/时间范围/字段/清洗规则)
    └── screenshots/             # 前端页面、数据库记录、后端日志截图
```

## 快速开始

> 环境: Python 3.11+。装依赖时国内建议加清华镜像, 直连 PyPI 容易卡住。

### 路线 A · 只想看结果(最快, 不用跑流水线)

```bash
python -m venv .venv
.venv\Scripts\activate                     # Windows; macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

uvicorn src.api.main:app --port 8000       # 后端: /docs  /monitor
streamlit run src/app/streamlit_app.py     # 前端: 4 个页面
```

仓库已提交 `data/db/market.db`、`data/processed/*.csv` 与全部图表, **克隆下来直接启动就有数据**,
不需要先跑流水线。

### 路线 B · 离线复现全部结果(约 20 秒, 推荐答辩前跑一次)

```bash
python -m scripts.run_pipeline --skip-fetch
```

- 若 `data/raw/` 存在 → 复用原始数据重新清洗(精确复现清洗漏斗);
- 若是全新克隆(`data/raw/` 因版权未入库) → **自动回退**到已提交的 `data/processed/` 清洗结果,
  从弱标注开始跑完全流程, 数值与报告完全一致(已实测: 年化 1.06%、夏普 0.146、31 笔交易逐位相同)。

### 路线 C · 从零联网采集(约 60 秒)

```bash
python -m scripts.run_pipeline
```

会实时抓取 100+ 只个股新闻、财经早餐长历史、全球快讯与沪深300 日线;
因数据源每日更新, **数值会与报告略有差异**, 属正常现象。

### 报告与截图

```bash
python -m scripts.make_report_pdf          # Markdown -> HTML -> PDF(需 playwright)
python -m scripts.screenshots              # 自动截取 3 个前端页(需 playwright, 复用本机 Edge)
pip install -r requirements-dev.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 系统实现

**FastAPI 接口**

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 健康检查 |
| GET | `/meta` | 数据集概况 + 清洗漏斗 + 关键参数 |
| GET | `/news?limit&q&signal&start&end` | 新闻查询(关键词/标签/区间) |
| GET | `/price`, `/signal` | 行情 / 日频信号 |
| GET | `/backtest/metrics`, `/backtest/equity` | 回测绩效 / 净值序列 |
| GET | `/models/metrics` | ML 指标 + ML/LLM 对比表 |
| GET | `/llm/stability` | LLM 稳定性测试明细 |
| GET | `/stats/daily` | 每日新闻量与情感分布 |
| GET | `/logs` | 流水线执行记录 |
| GET | `/monitor` | HTML 运维页(表计数 + 日志尾部) |

**Streamlit 页面**

1. **数据概览** —— 指标卡、清洗漏斗(前后对比)、每日新闻量、行情、新闻明细检索
2. **情感分析结果** —— 标签分布、ML/LLM 模型对比表、LLM 稳定性、情感信号时序
3. **回测结果** —— 绩效对比表、净值曲线、仓位与回撤、逐日信号明细
4. **系统与日志** —— 数据库表记录数、pipeline_log、日志文件尾部、当前配置

## 关键设计(评分要点对应)

| 要点 | 做法 |
|---|---|
| **禁止随机切分** | `ml_model.time_split()` 按 `pub_time` 排序后 8:2 切分, 测试集全在训练集之后 |
| **禁止未来函数** | 第 t 日仓位由 `S.shift(1)` 决定; 非交易日新闻直接剔除; 多阶 IC 无异常高值 |
| **交易成本** | 按仓位变动计单边 5bp, 已在 `strategy_ret` 中扣除 |
| **基准** | Buy & Hold 沪深300, 同一区间、同一成本口径 |
| **稳健性** | ① 仅财经早餐语料复跑 ② 平滑跨度 × 阈值 36 组参数曲面 ③ 多阶 IC 诊断 |
| **可复现** | 随机种子固定; 原始数据落盘; `data/processed/*.csv` 与全部图表入库, 无网络也能核对结果 |

## 数据来源与合规

- 新闻: 东方财富"财经早餐"(`stock_news_cjzc_em`)、东财个股新闻(`stock_news_em`)、
  东财/新浪全球快讯、财联社电报 —— 均为 AkShare 免费接口。
- 行情: 沪深300 指数日线(`stock_zh_index_daily`)。
- **未使用 Tushare**(避免积分/限流问题), 也未伪造任何数据或结果; 所有图表数值可由
  `python -m scripts.run_pipeline --skip-fetch` 重新生成。
- 原始数据文件不随仓库分发(版权归原提供方), 但清洗后的数据与全部分析结果已提交, 便于核对。
  字段与清洗规则详见 [`docs/DATA.md`](docs/DATA.md)。

## 局限与后续工作

1. 语料以"每日财经早餐 + 近期个股新闻"为主, 早期每日仅 3 条左右, 信号颗粒度粗;
   引入更长历史的个股级新闻(付费源)可提升信号密度。
2. 情感标签来自词典弱标注, 存在标注噪声; 可用人工标注 500 条做校准。
3. 目前是市场级择时(单标的)。后续可做**横截面**选股(个股情感 vs 行业中性收益),
   情绪类因子的信息比通常高于时序择时。
4. LLM 部分为可复现的本地模拟实现; 配置 `.env` 中的 `LLM_API_KEY` 即切换为真实 LLM
   (OpenAI 兼容接口), 对比结论需以真实调用重跑为准。

## 免责声明

本项目为课程实验, 所有结论仅用于方法验证, **不构成任何投资建议**。
