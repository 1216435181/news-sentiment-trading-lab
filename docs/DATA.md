# 数据说明文件

## 1. 数据来源

| 数据类型 | 来源接口 | 提供方 | 采集脚本 | 说明 |
|---|---|---|---|---|
| 新闻(个股) | `ak.stock_news_em` | 东方财富 | `src/data/fetch.py::fetch_stock_news` | 按 30 只大盘股逐一抓取, 每只约 100 条 |
| 新闻(市场) | `ak.stock_info_global_cls` | 财联社电报 | `fetch_global_news` | 市场级 7×24 快讯 |
| 新闻(市场) | `ak.stock_info_global_em` | 东方财富 | 同上 | 全球财经快讯 |
| 新闻(市场) | `ak.stock_info_global_sina` | 新浪财经 | 同上 | 全球财经快讯 |
| 行情 | `ak.index_zh_a_hist(symbol="000300")` | 上交所/深交所指数 | `fetch_price` | 沪深300 日线(未复权指数) |

- AkShare 版本: 见 `requirements.txt` 实际安装版本; AkShare 为免费接口, **无 token、无积分限制**,
  避免了 Tushare 常见的权限/限流问题(实验文档"常见问题 1"的备选方案)。
- 备用行情接口: `ak.stock_zh_a_hist("510300", adjust="qfq")`, 当指数接口不可用时自动降级。
- 采集时间戳与条数记录在 `data/processed/clean_report.json` 与 `logs/pipeline.log`。

## 2. 时间范围

| 项目 | 范围 |
|---|---|
| 新闻 | 以接口实际返回为准, 见 `clean_report.json` 的 `time_start` / `time_end` |
| 行情 | `config.yaml` 的 `start_date` ~ `end_date`(默认自然年当年) |
| 回测区间 | 与"有新闻覆盖的交易日"取交集, 缺新闻的交易日按中性 0 处理 |

> 说明: AkShare 的新闻接口只提供**近期滚动窗口**(通常 1~3 个月), 因此回测样本期偏短。
> 这是免费数据源的客观限制, 报告第 5 节将其列为策略不显著的原因之一, 未做任何数据伪造或时间延长。

## 3. 字段说明

### 3.1 `data/raw/news_raw.csv`(原始)

| 字段 | 类型 | 说明 |
|---|---|---|
| pub_time | datetime | 发布时间(已统一为 `YYYY-MM-DD HH:MM:SS`) |
| title | str | 新闻标题 |
| content | str | 新闻正文/摘要 |
| source | str | 来源标识: `eastmoney_stock_news` / `cls_telegraph` / `eastmoney_global` / `sina_global` |
| symbol | str | 关联股票代码(市场级快讯为空) |

### 3.2 `data/processed/news_clean.csv`(清洗后)

在原始字段基础上新增:

| 字段 | 类型 | 说明 |
|---|---|---|
| trade_date | date | 对齐后的交易日: 非交易日新闻按 `config.clean.align_rule` 剔除(默认 `drop`) |

### 3.3 `data/processed/news_labeled.csv`(弱标注后)

| 字段 | 类型 | 说明 |
|---|---|---|
| sentiment_score_rule | float | 词典规则情感分, 范围 [-1, 1], tanh 压缩 |
| lexicon_hits | int | 命中的情感词数量(0 表示无情感词, 计为中性) |
| trend_signal | int | 弱标注标签: 1 利好 / 0 中性 / -1 利空 |
| is_noise_hint | bool | 是否含"招聘/招标/更正"等噪声提示词 |

### 3.4 `data/processed/daily_signal.csv`(日频信号)

| 字段 | 类型 | 说明 |
|---|---|---|
| n_news | int | 当日新闻条数 |
| s_score_mean / s_pos_ratio / s_neg_ratio | float | 当日情感均值与利好/利空占比 |
| S | float | 合成日频信号 = 0.6×均值 + 0.4×(利好占比−利空占比) |
| S_lag1 | float | **t-1 日信号(唯一可用于 t 日交易的输入)** |
| position | float | t 日持仓: 1 多 / 0 空仓 / -1 空头 |
| turnover | float | 仓位变化绝对值(用于计交易成本) |
| strategy_ret / bench_ret | float | 扣费后策略日收益 / 基准日收益(指数日收益) |
| equity_strategy / equity_bench | float | 净值曲线 |

### 3.5 `data/processed/price_clean.csv`

| 字段 | 说明 |
|---|---|
| trade_date / open / high / low / close | 沪深300 日线 OHLC |
| volume / amount | 成交量(手) / 成交额 |
| ret | 日收益率 = close_t / close_{t-1} − 1 |

## 4. 清洗规则与阈值

| 规则 | 阈值/做法 | 原因 |
|---|---|---|
| 时间解析 | 强制 `to_datetime`, 解析失败剔除 | 保证排序与对齐可靠 |
| 去空 | 标题或正文为空则剔除 | 无法判断情感 |
| 超短文本 | 标题 < 6 字 或 正文 < 10 字 剔除 | 多为滚动播报残片 |
| 乱码 | 非常用字符占比 > 30% 剔除 | 编码损坏文本 |
| 超长文本 | 正文截断到 5000 字, 建模只用前 200 字 | 控制噪声与计算量 |
| 去重 1 | (标题去空格, 发布时间) 完全重复 -> 保留第一条 | 多源转载 |
| 去重 2 | 标题完全相同 -> 保留最早发布时间 | 同一稿件反复推送 |
| 未来时间 | `pub_time > now` 剔除 | 数据源脏数据 |
| 交易日对齐 | 非交易日新闻**剔除**(`align_rule=drop`) | 避免周末消息重复计入同一交易日; 可改 `ffill` 前向填充 |

## 5. 合规与引用

- 新闻与行情数据版权归原始提供方(东方财富、财联社、新浪财经、中证指数)所有, 本项目仅用于课程实验,
  不再分发原始数据文件, `data/raw/` 已在 `.gitignore` 中排除。
- 复现时运行 `python -m scripts.run_pipeline` 会按上述接口重新抓取; 若接口字段变动,
  脚本中的 `_normalize` 会按列名模糊匹配并跳过失败来源, 不中断流水线。
