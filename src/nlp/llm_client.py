"""LLM 情感分析客户端(支持 mock 与真实 OpenAI 兼容接口)。

provider = "mock"
    本地可复现模拟 LLM:
      * temperature=0  -> 完全确定性(同输入同输出), 对应真实 LLM 的贪心解码
      * temperature>0  -> 以文本哈希为种子的高斯扰动, 模拟采样带来的不确定性
    这是实验要求中"可复现"与"不依赖外部 API"的折中方案, 报告中明确标注为**模拟 LLM**。
provider = "openai"
    调用 OpenAI 兼容 /chat/completions 接口(DeepSeek、Moonshot、DashScope 等均可),
    需在 .env 配置 LLM_API_KEY / LLM_BASE_URL / LLM_MODEL。
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

from src.config import CONFIG, get_env
from src.logging_utils import get_logger
from src.nlp.weak_label import score_text, to_signal

log = get_logger("llm")

SYSTEM_PROMPT = (
    "你是一名A股市场新闻情感分析师。请阅读新闻, 判断它对次日相关股票/大盘的影响, "
    "只输出 JSON: {\"trend_signal\": -1|0|1, \"sentiment_score\": -1~1, \"reason\": \"不超过20字\"}。"
    "-1 利空, 0 中性/无关, 1 利好。"
)


@dataclass
class LLMResult:
    trend_signal: int
    sentiment_score: float
    reason: str = ""
    latency_ms: float = 0.0
    raw: Dict[str, Any] = field(default_factory=dict)


class LLMClient:
    def __init__(self, provider: str | None = None, model: str | None = None,
                 temperature: float | None = None):
        cfg = CONFIG["llm"]
        self.provider = (provider or cfg["provider"]).lower()
        self.model = model or cfg["model"]
        self.temperature = cfg["temperature_default"] if temperature is None else temperature
        self.api_key = get_env("LLM_API_KEY")
        self.base_url = (get_env("LLM_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        if self.provider == "openai" and not self.api_key:
            log.warning("未检测到 LLM_API_KEY, 自动回退到 mock 模式")
            self.provider = "mock"

    # ------------------------------------------------------------------
    def analyze(self, title: str, content: str = "") -> LLMResult:
        t0 = time.perf_counter()
        if self.provider == "openai":
            result = self._call_openai(title, content)
        else:
            result = self._mock(title, content)
        result.latency_ms = (time.perf_counter() - t0) * 1000
        return result

    # ------------------------------------------------------------------
    def _mock(self, title: str, content: str) -> LLMResult:
        """模拟 LLM: 词典分作为"语义先验", 叠加可注入的采样噪声。

        与规则弱标注的差别(让对比有意义):
          1) 模拟 LLM 对标题给更高权重(标题重复 3 次), 更接近真实 LLM 抓重点的行为;
          2) temperature=0 时做轻微"校准"(向 0.7x 收缩), 体现 LLM 输出的保守性;
          3) temperature>0 时注入以 (文本, 温度) 为种子的高斯噪声。
        """
        s_title, _ = score_text(title, "")
        s_full, _ = score_text(title, content)
        base = 0.65 * s_title + 0.35 * s_full

        if self.temperature <= 0:
            score = base * 0.7           # 确定性 + 轻微收缩
        else:
            seed_src = f"{self.model}|{title}|{content[:80]}|{self.temperature:.3f}"
            seed = int(hashlib.md5(seed_src.encode("utf-8")).hexdigest()[:12], 16)
            rng = random.Random(seed)
            sigma = 0.22 * min(self.temperature, 1.0)
            score = base * 0.7 + rng.gauss(0.0, sigma)

        score = max(-1.0, min(1.0, score))
        signal = to_signal(score)
        reason = "词典语义匹配" if abs(score) > 1e-6 else "无明显情感倾向"
        return LLMResult(trend_signal=signal, sentiment_score=round(score, 4), reason=reason)

    # ------------------------------------------------------------------
    def _call_openai(self, title: str, content: str) -> LLMResult:
        import requests

        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"标题: {title}\n正文: {str(content)[:800]}"},
            ],
            "response_format": {"type": "json_object"},
        }
        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}",
                         "Content-Type": "application/json"},
                json=payload, timeout=30,
            )
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"]
            data = json.loads(text)
            return LLMResult(
                trend_signal=int(data.get("trend_signal", 0)),
                sentiment_score=float(data.get("sentiment_score", 0.0)),
                reason=str(data.get("reason", ""))[:40],
                raw=data,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("LLM 调用失败, 回退到中性: %s", exc)
            return LLMResult(trend_signal=0, sentiment_score=0.0, reason="api_error")


def stability_test(sample: List[Dict], client: LLMClient | None = None) -> Dict[str, Any]:
    """同一批样本用 temperature=0 与默认温度各跑一次, 统计一致率与情感分偏差。"""
    cfg = CONFIG["llm"]
    client = client or LLMClient()
    det = LLMClient(provider=client.provider, model=client.model, temperature=0.0)
    stoch = LLMClient(provider=client.provider, model=client.model,
                      temperature=cfg["temperature_default"])

    rows, lat_det, lat_sto = [], [], []
    for item in sample:
        r0 = det.analyze(item["title"], item["content"])
        r1 = stoch.analyze(item["title"], item["content"])
        lat_det.append(r0.latency_ms)
        lat_sto.append(r1.latency_ms)
        rows.append({
            "title": item["title"][:60],
            "label_rule": int(item.get("trend_signal", 0)),
            "signal_t0": r0.trend_signal,
            "signal_t07": r1.trend_signal,
            "score_t0": r0.sentiment_score,
            "score_t07": r1.sentiment_score,
            "agree": int(r0.trend_signal == r1.trend_signal),
            "abs_diff": abs(r0.sentiment_score - r1.sentiment_score),
            "hit_t0": int(r0.trend_signal == int(item.get("trend_signal", 0))),
            "hit_t07": int(r1.trend_signal == int(item.get("trend_signal", 0))),
        })

    n = max(len(rows), 1)
    summary = {
        "n_samples": len(rows),
        "agreement_rate": round(sum(r["agree"] for r in rows) / n, 4),
        "mean_abs_deviation": round(sum(r["abs_diff"] for r in rows) / n, 4),
        "max_abs_deviation": round(max([r["abs_diff"] for r in rows] or [0]), 4),
        "agreement_with_rule_t0": round(sum(r["hit_t0"] for r in rows) / n, 4),
        "agreement_with_rule_t07": round(sum(r["hit_t07"] for r in rows) / n, 4),
        "latency_ms_mean_t0": round(sum(lat_det) / n, 3),
        "latency_ms_mean_t07": round(sum(lat_sto) / n, 3),
        "temperature_t0": 0.0,
        "temperature_t1": cfg["temperature_default"],
        "provider": client.provider,
        "model": client.model,
    }
    log.info("LLM 稳定性: 一致率=%.2f%% 平均绝对偏差=%.4f 延迟=%.3fms",
             summary["agreement_rate"] * 100, summary["mean_abs_deviation"],
             summary["latency_ms_mean_t07"])
    return {"summary": summary, "detail": rows}


if __name__ == "__main__":
    import pandas as pd
    from src.config import P
    df = pd.read_csv(P.labeled, encoding="utf-8-sig").tail(50)
    res = stability_test(df.to_dict("records"))
    print(json.dumps(res["summary"], ensure_ascii=False, indent=2))
