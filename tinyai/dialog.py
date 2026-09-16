"""会話データ: (発話, 応答) のペアを集めて、ニューラル LM の学習と応答例の検索に使う。

出どころ:
  * 自分の会話 (ユーザー発話 → 応答。👍 が付いたものは重み高、👎 は除外)
  * 公開データセット (Hugging Face datasets-server の対話/指示データ)
  * Stack Exchange の質問 → 採用回答
  * 青空文庫などの文学作品の「」の応酬 (会話らしい言い回しの学習)

ペアは上限付きで保持し、保存される。個人情報らしいもの (メールアドレス・電話番号) は落とす。
"""
from __future__ import annotations

import random
import re
from collections import deque

_PII_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+|\+?\d[\d\-() ]{8,}\d")
_QUOTE_RE = re.compile(r"「([^「」]{2,80})」")


class DialogStore:
    def __init__(self, capacity: int = 20000):
        self.pairs: deque = deque(maxlen=capacity)   # (user, bot, source, weight)
        self.seen: set[int] = set()

    def add(self, user: str, bot: str, source: str = "chat", weight: float = 1.0) -> bool:
        user, bot = user.strip(), bot.strip()
        if len(user) < 2 or len(bot) < 2 or len(user) > 300 or len(bot) > 600:
            return False
        if _PII_RE.search(user) or _PII_RE.search(bot):
            return False
        h = hash((user, bot))
        if h in self.seen:
            return False
        if len(self.seen) > 60000:
            self.seen.clear()
        self.seen.add(h)
        self.pairs.append((user, bot, source[:60], weight))
        return True

    def add_many(self, pairs, source: str, weight: float = 1.0) -> int:
        return sum(1 for u, b in pairs if self.add(u, b, source, weight))

    def sample(self, n: int, rng: random.Random | None = None) -> list[tuple[str, str, str, float]]:
        rng = rng or random
        if not self.pairs:
            return []
        return rng.sample(list(self.pairs), min(n, len(self.pairs)))

    def __len__(self) -> int:
        return len(self.pairs)

    def by_source(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for _, _, s, _ in self.pairs:
            out[s] = out.get(s, 0) + 1
        return out

    def state(self) -> list:
        return list(self.pairs)

    @classmethod
    def from_state(cls, items, capacity: int = 20000) -> "DialogStore":
        ds = cls(capacity)
        for u, b, s, w in items:
            ds.add(u, b, s, w)
        return ds


def extract_quote_pairs(text: str, max_pairs: int = 200) -> list[tuple[str, str]]:
    """文学作品などから「」で囲まれた発話の連続を (発話, 応答) のペアにする。
    地の文が長く挟まる場合はペアにしない。"""
    out = []
    last_end = None
    last_q = None
    for m in _QUOTE_RE.finditer(text):
        q = m.group(1).strip()
        if last_q is not None and last_end is not None and m.start() - last_end <= 40:
            if q != last_q:
                out.append((last_q, q))
                if len(out) >= max_pairs:
                    break
        last_q, last_end = q, m.end()
    return out
