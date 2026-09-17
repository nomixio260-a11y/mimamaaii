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
        # (user, bot, source, weight, history)。history は直前までのやり取り ((発話, 応答), ...)。
        # 多ターンの会話データは「その場で 1 回学ぶ」だけでは身に付かないので、履歴ごと保存して
        # 再生バッファに何度も流す (会話のキャッチボールを学ぶのはこの履歴つきの系列だけ)。
        self.pairs: deque = deque(maxlen=capacity)
        self.seen: set[int] = set()

    MAX_BOT = 220      # 小さなモデルは長い応答を覚えきれない: 先頭の数文に切り詰めて学ぶ
    MAX_HISTORY = 2    # 保存する過去のやり取りの数
    MAX_HISTORY_CHARS = 120   # 過去のやり取りは短く持つ (記憶量を抑える)

    @staticmethod
    def _shorten(text: str, limit: int) -> str:
        """句点・改行の区切りで limit 文字以内に切り詰める (途中で切れた文は捨てる)。"""
        if len(text) <= limit:
            return text
        cut = text[: limit + 40]
        best = -1
        for mark in ("。", "！", "？", ".", "!", "?", "\n"):
            i = cut.rfind(mark)
            if i > best:
                best = i
        return cut[: best + 1].strip() if best >= 20 else text[:limit].rstrip()

    def _trim_history(self, history) -> tuple:
        """過去のやり取りを (発話, 応答) の組に整えて短く切り詰める。"""
        out = []
        for item in list(history or [])[-self.MAX_HISTORY:]:
            if not item or len(item) < 2:
                continue
            u, b = (str(item[0] or "").strip(), str(item[1] or "").strip())
            if len(u) < 1 and len(b) < 1:
                continue
            out.append((self._shorten(u, self.MAX_HISTORY_CHARS), self._shorten(b, self.MAX_HISTORY_CHARS)))
        return tuple(out)

    @staticmethod
    def clean_reply(text: str) -> str:
        """応答から書式の記号を落とし、箇条書きが始まる手前までにする。

        指示データの応答は「説明文 + 番号つきの箇条書き」が多い。応答は 220 字で切り詰めるので、
        そのままだと「1. **バッテリーの温度管理**」のような途中で切れた断片を学ぶことになり、
        生成にも「1.**…**」という壊れた書式が現れる (実測: 会話の 8% に書式混入、6% が途中で終了)。
        小さなモデルには書式より地の文を学ばせる。"""
        t = re.sub(r"\*\*|__|`+", "", text)
        t = re.sub(r"^\s*#{1,6}\s*", "", t, flags=re.M)
        m = re.search(r"\n\s*(?:\d+[.)]|[-*・])\s*", t)
        if m and m.start() >= 30:            # 箇条書きの手前に十分な説明文があれば、そこまでを学ぶ
            t = t[: m.start()]
        t = re.sub(r"[ \t]+", " ", t)
        t = re.sub(r"\n{2,}", "\n", t).strip()
        return t or text.strip()

    def add(self, user: str, bot: str, source: str = "chat", weight: float = 1.0, history=None) -> bool:
        user, bot = user.strip(), self.clean_reply(bot)
        if len(bot) > self.MAX_BOT:
            bot = self._shorten(bot, self.MAX_BOT)
        if len(user) > 300:
            user = self._shorten(user, 300)
        if len(user) < 2 or len(bot) < 2:
            return False
        if _PII_RE.search(user) or _PII_RE.search(bot):
            return False
        h = hash((user, bot))
        if h in self.seen:
            return False
        if len(self.seen) > 60000:
            self.seen.clear()
        self.seen.add(h)
        self.pairs.append((user, bot, source[:60], weight, self._trim_history(history)))
        return True

    def add_many(self, pairs, source: str, weight: float = 1.0) -> int:
        return sum(1 for u, b in pairs if self.add(u, b, source, weight))

    def sample(self, n: int, rng: random.Random | None = None) -> list[tuple]:
        rng = rng or random
        if not self.pairs:
            return []
        return rng.sample(list(self.pairs), min(n, len(self.pairs)))

    def __len__(self) -> int:
        return len(self.pairs)

    def with_history(self) -> int:
        """履歴つきで保存されている会話の数 (多ターン学習がどれだけできるかの指標)。"""
        return sum(1 for item in self.pairs if len(item) > 4 and item[4])

    def by_source(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for _, _, s, _, *_rest in self.pairs:
            out[s] = out.get(s, 0) + 1
        return out

    def state(self) -> list:
        return list(self.pairs)

    @classmethod
    def from_state(cls, items, capacity: int = 20000) -> "DialogStore":
        ds = cls(capacity)
        for item in items:
            u, b, s, w = item[:4]
            ds.add(u, b, s, w, history=item[4] if len(item) > 4 else None)
        return ds


def extract_quote_pairs(text: str, max_pairs: int = 200, with_history: bool = False) -> list[tuple]:
    """文学作品などから「」で囲まれた発話の連続を (発話, 応答) のペアにする。
    地の文が長く挟まる場合はペアにしない。

    with_history=True なら、同じ応酬の中で手前に続いていたやり取りを履歴として付け、
    (発話, 応答, None, 1.0, 履歴) の形で返す。台詞の応酬は人間どうしの会話そのものなので、
    多ターンの練習に使える数少ない自然なデータ源になる。"""
    out: list[tuple] = []
    chain: list[tuple[str, str]] = []      # 続いている応酬 (地の文で切れたら空にする)
    last_end = None
    last_q = None
    for m in _QUOTE_RE.finditer(text):
        q = m.group(1).strip()
        linked = last_q is not None and last_end is not None and m.start() - last_end <= 40
        if linked:
            if q != last_q:
                if with_history and chain:
                    out.append((last_q, q, None, 1.0, list(chain[-2:])))
                else:
                    out.append((last_q, q))
                chain.append((last_q, q))
                if len(out) >= max_pairs:
                    break
        else:
            chain = []                      # 応酬が途切れた: 履歴を引き継がない
        last_q, last_end = q, m.end()
    return out
