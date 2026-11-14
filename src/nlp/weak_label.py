"""弱标注(trend_signal: -1/0/1)生成。

规则情感打分 = 词典命中(带程度副词与否定翻转) -> tanh 压缩到 [-1,1]
再按 config 阈值离散化为 -1/0/1, 作为 ML 模型的训练标签与 LLM 的对照标准。
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import List

import jieba
import pandas as pd

from src.config import CONFIG
from src.logging_utils import get_logger
from src.nlp.lexicon import (DEGREE, NEG_WORDS, NEGATION, NOISE_HINTS, POS_WORDS)

log = get_logger("weak_label")

jieba.setLogLevel(60)


@lru_cache(maxsize=1)
def _dicts():
    return POS_WORDS, NEG_WORDS


def _raw_score(tokens: List[str]) -> float:
    pos, neg = _dicts()
    score = 0.0
    hits = 0
    for i, tok in enumerate(tokens):
        # 最长匹配优先: 尝试 2-3 字组合, 保证"净利润增长"这类词被识别
        w = 0.0
        for size in (4, 3, 2):
            if i + size <= len(tokens):
                cand = "".join(tokens[i:i + size])
                if cand in pos or cand in neg:
                    w = pos.get(cand, 0.0) - neg.get(cand, 0.0)
                    break
        if w == 0.0:
            if tok in pos:
                w = pos[tok]
            elif tok in neg:
                w = -neg[tok]
            else:
                continue

        # 程度副词(前 2 个 token 内)
        for j in range(max(0, i - 2), i):
            if tokens[j] in DEGREE:
                w *= DEGREE[tokens[j]]
                break
        # 否定词翻转(前 2 个 token 内)
        window = tokens[max(0, i - 2):i]
        if any(t in NEGATION for t in window):
            w = -w * 0.8
        score += w
        hits += 1
    return score, hits


def score_text(title: str, content: str = "") -> tuple[float, int]:
    """返回 (情感分 in [-1,1], 命中的情感词数量)。"""
    text = (str(title) + "。") * 2 + str(content)[:300]
    tokens = [t.strip() for t in jieba.lcut(text) if t.strip()]
    raw, hits = _raw_score(tokens)
    if hits == 0:
        return 0.0, 0
    return math.tanh(raw / 2.5), hits


def to_signal(score: float, pos_thr: float | None = None, neg_thr: float | None = None) -> int:
    pos_thr = CONFIG["label"]["pos_threshold"] if pos_thr is None else pos_thr
    neg_thr = CONFIG["label"]["neg_threshold"] if neg_thr is None else neg_thr
    if score > pos_thr:
        return 1
    if score < neg_thr:
        return -1
    return 0


def label_news(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    scores, hits = [], []
    for title, content in zip(out["title"].tolist(), out["content"].tolist()):
        s, h = score_text(title, content)
        scores.append(s)
        hits.append(h)
    out["sentiment_score_rule"] = scores
    out["lexicon_hits"] = hits
    out["trend_signal"] = [to_signal(s) for s in scores]
    # 宏观噪声样本降权标识
    out["is_noise_hint"] = out["title"].str.contains("|".join(NOISE_HINTS), regex=True)

    dist = out["trend_signal"].value_counts().sort_index().to_dict()
    log.info("弱标注分布: %s, 平均情感分 %.4f, 词典命中率 %.1f%%",
             dist, out["sentiment_score_rule"].mean(), (out["lexicon_hits"] > 0).mean() * 100)
    return out


if __name__ == "__main__":
    from src.config import P
    df = pd.read_csv(P.clean_news, encoding="utf-8-sig")
    labeled = label_news(df)
    labeled.to_csv(P.labeled, index=False, encoding="utf-8-sig")
