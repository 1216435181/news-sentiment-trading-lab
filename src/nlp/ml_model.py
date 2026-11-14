"""ML 基线模型: TF-IDF(词 + 字符) 三分类, 按时间 8:2 切分。

对比模型
--------
M1  TF-IDF + LogisticRegression   (线性, 强基线)
M2  TF-IDF + LinearSVC            (线性 SVM, 间隔最大化)
M3  TF-IDF + ComplementNB         (朴素贝叶斯, 极快, 低延迟代表)
M4  TF-IDF -> SVD(256) + MLP      (浅层神经网络, 深度学习对照)
"""
from __future__ import annotations

import json
import time
from typing import Dict, List, Tuple

import jieba
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.naive_bayes import ComplementNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from src.config import CONFIG, P
from src.logging_utils import get_logger

log = get_logger("ml")

jieba.setLogLevel(60)
LABELS = [-1, 0, 1]


def _tok(text: str) -> List[str]:
    return [w for w in jieba.lcut(str(text)) if len(w.strip()) > 0]


def build_text(df: pd.DataFrame) -> pd.Series:
    """标题权重更高(重复两次), 正文取前 200 字控制噪声与长度。"""
    return (df["title"].astype(str) + "。" ) * 2 + df["content"].astype(str).str.slice(0, 200)


def time_split(df: pd.DataFrame, test_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """严格按 pub_time 排序后切分, 禁止随机切分(避免未来信息泄漏)。"""
    d = df.sort_values("pub_time").reset_index(drop=True)
    cut = int(len(d) * (1 - test_ratio))
    return d.iloc[:cut].copy(), d.iloc[cut:].copy()


def _make_features(cfg: dict) -> TfidfVectorizer:
    return TfidfVectorizer(
        tokenizer=_tok,
        lowercase=False,
        ngram_range=(1, 2),
        min_df=cfg["min_df"],
        max_features=cfg["tfidf_word_max_features"],
        sublinear_tf=True,
    )


def get_models(cfg: dict) -> Dict[str, Pipeline]:
    return {
        "TF-IDF + LogisticRegression": Pipeline([
            ("tfidf", _make_features(cfg)),
            ("clf", LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced",
                                       random_state=cfg["random_state"], n_jobs=None)),
        ]),
        "TF-IDF + LinearSVC": Pipeline([
            ("tfidf", _make_features(cfg)),
            ("clf", LinearSVC(C=0.5, class_weight="balanced",
                              random_state=cfg["random_state"], max_iter=5000)),
        ]),
        "TF-IDF + ComplementNB": Pipeline([
            ("tfidf", _make_features(cfg)),
            ("clf", ComplementNB(alpha=0.3)),
        ]),
        "TF-IDF+SVD+MLP(浅层神经网络)": Pipeline([
            ("tfidf", _make_features(cfg)),
            ("svd", TruncatedSVD(n_components=200, random_state=cfg["random_state"])),
            ("scale", StandardScaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=(96,), max_iter=400,
                                  early_stopping=True, random_state=cfg["random_state"])),
        ]),
    }


def evaluate(y_true, y_pred) -> Dict[str, float]:
    return {
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro",
                                         labels=LABELS, zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(y_true, y_pred, average="weighted",
                                           labels=LABELS, zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "f1_neg": round(float(f1_score(y_true, y_pred, labels=[-1], average="macro",
                                       zero_division=0)), 4),
        "f1_neu": round(float(f1_score(y_true, y_pred, labels=[0], average="macro",
                                       zero_division=0)), 4),
        "f1_pos": round(float(f1_score(y_true, y_pred, labels=[1], average="macro",
                                       zero_division=0)), 4),
    }


def run(labeled: pd.DataFrame) -> Tuple[pd.DataFrame, Dict, pd.DataFrame]:
    cfg = CONFIG["ml"]
    df = labeled.dropna(subset=["trend_signal"]).copy()
    df["trend_signal"] = df["trend_signal"].astype(int)
    df["text"] = build_text(df)

    train, test = time_split(df, cfg["test_ratio"])
    log.info("时间切分: train=%d (%s ~ %s) | test=%d (%s ~ %s)",
             len(train), train["pub_time"].min(), train["pub_time"].max(),
             len(test), test["pub_time"].min(), test["pub_time"].max())

    y_tr, y_te = train["trend_signal"].values, test["trend_signal"].values
    rows, preds_store, trained = [], {}, {}

    for name, pipe in get_models(cfg).items():
        t0 = time.perf_counter()
        pipe.fit(train["text"].values, y_tr)
        train_sec = time.perf_counter() - t0

        t1 = time.perf_counter()
        pred = pipe.predict(test["text"].values)
        infer_sec = time.perf_counter() - t1
        latency_ms = infer_sec / max(len(test), 1) * 1000

        row = {"model": name, "train_seconds": round(train_sec, 3),
               "latency_ms_per_news": round(latency_ms, 4), **evaluate(y_te, pred)}
        rows.append(row)
        preds_store[name] = pred
        trained[name] = pipe
        log.info("%-32s Macro-F1=%.4f Acc=%.4f 训练=%.2fs 单条=%.3fms",
                 name, row["macro_f1"], row["accuracy"], train_sec, latency_ms)

    metrics = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    best_name = metrics.iloc[0]["model"]
    best_pred = preds_store[best_name]
    cm = confusion_matrix(y_te, best_pred, labels=LABELS)

    detail = {
        "best_model": best_name,
        "train_size": int(len(train)),
        "test_size": int(len(test)),
        "split_time": str(test["pub_time"].min()),
        "label_dist_train": {str(k): int(v) for k, v in train["trend_signal"].value_counts().items()},
        "label_dist_test": {str(k): int(v) for k, v in test["trend_signal"].value_counts().items()},
        "confusion_matrix": cm.tolist(),
        "labels": LABELS,
        "report": classification_report(y_te, best_pred, labels=LABELS,
                                        zero_division=0, output_dict=True),
    }
    metrics.to_csv(P.ml_metrics, index=False, encoding="utf-8-sig")
    (P.ml_metrics.parent / "ml_detail.json").write_text(
        json.dumps(detail, ensure_ascii=False, indent=2), encoding="utf-8")

    test_out = test[["pub_time", "trade_date", "title", "trend_signal"]].copy()
    test_out["pred_best"] = best_pred
    return metrics, detail, test_out


if __name__ == "__main__":
    labeled = pd.read_csv(P.labeled, encoding="utf-8-sig", parse_dates=["pub_time", "trade_date"])
    m, d, t = run(labeled)
    print(m.to_string(index=False))
