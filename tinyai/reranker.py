"""学習型リランカー: 👍/👎 と「教えた答え」から、検索候補の良さをロジスティック回帰で学ぶ。

特徴量は検索スコア・カバー率・質問タイプ適合・意味類似・文の品質・出典の種類など 10 個程度。
オンライン SGD (1 サンプル数マイクロ秒) なので会話中に即時更新できる。
重みは保存され、進化のパラメータ `learned_weight` が最終的な確信度への混ぜ方を決める。
"""
from __future__ import annotations

import math

FEATURES = ("bias", "score", "cover", "qtype", "semantic", "quality", "trust", "src_user", "src_web", "length", "expanded", "fact")


class Reranker:
    def __init__(self, lr: float = 0.05, l2: float = 1e-3):
        self.w = {f: 0.0 for f in FEATURES}
        self.w["score"] = 1.0
        self.w["cover"] = 1.0
        self.lr = lr
        self.l2 = l2
        self.samples = 0

    @staticmethod
    def features(score: float, cover: float, qtype_bonus: float, semantic: float, quality: float, trust: float, source: str, length: int, expanded: bool, fact: bool) -> dict:
        return {
            "bias": 1.0,
            "score": min(score / 10.0, 3.0),
            "cover": cover,
            "qtype": qtype_bonus,
            "semantic": max(semantic, 0.0),
            "quality": quality,
            "trust": max(-1.0, min(1.0, trust / 5.0)),
            "src_user": 1.0 if source in ("user", "chat") else 0.0,
            "src_web": 1.0 if source.startswith("http") else 0.0,
            "length": min(length, 300) / 300.0,
            "expanded": 1.0 if expanded else 0.0,
            "fact": 1.0 if fact else 0.0,
        }

    def predict(self, x: dict) -> float:
        z = sum(self.w[k] * v for k, v in x.items())
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))

    def update(self, x: dict, y: float) -> float:
        """y = 1 (良い) / 0 (悪い)。損失を返す。"""
        p = self.predict(x)
        g = p - y
        lr = self.lr
        for k, v in x.items():
            self.w[k] -= lr * (g * v + self.l2 * self.w[k])
        self.samples += 1
        return -(y * math.log(max(p, 1e-9)) + (1 - y) * math.log(max(1 - p, 1e-9)))

    def state(self) -> dict:
        return {"w": dict(self.w), "samples": self.samples}

    @classmethod
    def from_state(cls, st: dict) -> "Reranker":
        r = cls()
        r.w.update({k: v for k, v in st.get("w", {}).items() if k in r.w})
        r.samples = st.get("samples", 0)
        return r
