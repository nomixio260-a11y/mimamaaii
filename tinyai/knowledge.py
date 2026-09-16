"""文単位の知識ベース + BM25 転置インデックス。

各文書は (id, text, source, added, score) を持つ。`score` は会話中の
フィードバック (👍/👎) や検索ヒットで増減し、メモリ逼迫時の追い出し順に
使う (古くて役に立たない文から消える)。
"""
from __future__ import annotations

import hashlib
import math
import random
import time
from collections import Counter
from typing import Iterable

from .tokenizer import term_weight, terms

DOC_BASE_COST = 220


class Doc:
    __slots__ = ("id", "text", "source", "added", "score", "length")

    def __init__(self, id: int, text: str, source: str, added: float, length: int):
        self.id = id
        self.text = text
        self.source = source
        self.added = added
        self.score = 0.0
        self.length = length


class KnowledgeBase:
    def __init__(self, k1: float = 1.4, b: float = 0.6, phrase_bonus: float = 1.5):
        self.k1 = k1
        self.b = b
        self.phrase_bonus = phrase_bonus  # 漢字/カタカナ連続語 (>=2 文字) が一致した時の重み
        self.docs: dict[int, Doc] = {}
        self.index: dict[str, dict[int, int]] = {}
        self.hashes: set[str] = set()
        self.next_id = 1
        self.total_len = 0
        self._est_bytes = 0

    # ------------------------------------------------------------ 追加/削除
    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.blake2b(text.strip().lower().encode("utf-8"), digest_size=8).hexdigest()

    def add(self, text: str, source: str = "") -> Doc | None:
        text = text.strip()
        if len(text) < 4 or len(text) > 600:
            return None
        h = self._hash(text)
        if h in self.hashes:
            return None
        tf = Counter(terms(text))
        if not tf:
            return None
        doc = Doc(self.next_id, text, source[:120], time.time(), sum(tf.values()))
        self.next_id += 1
        self.docs[doc.id] = doc
        self.hashes.add(h)
        for t, c in tf.items():
            post = self.index.get(t)
            if post is None:
                post = self.index[t] = {}
            post[doc.id] = c
        self.total_len += doc.length
        self._est_bytes += DOC_BASE_COST + len(text) * 2 + len(tf) * 90
        return doc

    def remove(self, doc_id: int) -> None:
        doc = self.docs.pop(doc_id, None)
        if doc is None:
            return
        self.hashes.discard(self._hash(doc.text))
        self.total_len -= doc.length
        tf = Counter(terms(doc.text))
        for t in tf:
            post = self.index.get(t)
            if post:
                post.pop(doc_id, None)
                if not post:
                    del self.index[t]
        self._est_bytes -= DOC_BASE_COST + len(doc.text) * 2 + len(tf) * 90

    def __len__(self) -> int:
        return len(self.docs)

    def estimated_bytes(self) -> int:
        return max(0, self._est_bytes)

    # ------------------------------------------------------------ 検索
    def search(self, query: str, k: int = 5, exclude_sources: Iterable[str] = ()) -> list[tuple[float, Doc]]:
        q_terms = terms(query)
        if not q_terms or not self.docs:
            return []
        n = len(self.docs)
        avgdl = self.total_len / n if n else 1.0
        scores: dict[int, float] = {}
        excl = set(exclude_sources)
        qtf = Counter(q_terms)
        for t, qc in qtf.items():
            post = self.index.get(t)
            if not post:
                continue
            df = len(post)
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            w = idf * term_weight(t) * (1.0 + (self.phrase_bonus - 1.0) * (len(t) >= 3 or (len(t) == 2 and not _is_kana_bigram(t))))
            for doc_id, tf in post.items():
                doc = self.docs[doc_id]
                denom = tf + self.k1 * (1 - self.b + self.b * doc.length / avgdl)
                s = w * tf * (self.k1 + 1) / denom
                scores[doc_id] = scores.get(doc_id, 0.0) + s * min(qc, 2)
        if not scores:
            return []
        # クエリ語のカバー率で正規化 (どれだけの語が一致したか)
        ranked = []
        total_w = sum(term_weight(t) for t in qtf) or 1.0
        for doc_id, s in scores.items():
            doc = self.docs[doc_id]
            if doc.source in excl:
                continue
            matched = sum(term_weight(t) for t in qtf if doc_id in self.index.get(t, ()))
            cover = matched / total_w
            s = s * (0.5 + cover) + 0.15 * doc.score
            ranked.append((s, doc))
        ranked.sort(key=lambda x: (-x[0], -x[1].added))
        return ranked[:k]

    def feedback(self, doc_id: int, delta: float) -> None:
        d = self.docs.get(doc_id)
        if d:
            d.score = max(-5.0, min(20.0, d.score + delta))

    # ------------------------------------------------------------ 圧縮
    def evict(self, n: int) -> int:
        """役に立っていない古い文から n 件削除。"""
        if n <= 0 or not self.docs:
            return 0
        now = time.time()
        # 低スコア・古い順。ユーザーが教えた文は残しやすくする。
        def key(d: Doc):
            age = (now - d.added) / 86400.0
            protect = 3.0 if d.source in ("user", "chat", "seed") else 0.0
            return d.score + protect - age * 0.2
        victims = sorted(self.docs.values(), key=key)[:n]
        for d in victims:
            self.remove(d.id)
        return len(victims)

    def shrink_to(self, budget_bytes: int, max_docs: int | None = None) -> int:
        removed = 0
        if max_docs is not None and len(self.docs) > max_docs:
            removed += self.evict(len(self.docs) - max_docs)
        while self.estimated_bytes() > budget_bytes and self.docs:
            removed += self.evict(max(1, len(self.docs) // 20))
        return removed

    # ------------------------------------------------------------ 探索支援
    def random_docs(self, n: int, rng: random.Random) -> list[Doc]:
        if not self.docs:
            return []
        ids = list(self.docs.keys())
        return [self.docs[i] for i in rng.sample(ids, min(n, len(ids)))]

    def sparse_terms(self, n: int, rng: random.Random, min_len: int = 2) -> list[str]:
        """出現文書数が少ない (知識が薄い) 語をサンプリングして返す。"""
        cands = [t for t, post in self.index.items() if len(post) <= 2 and len(t) >= min_len and not _is_kana_bigram(t)]
        if not cands:
            return []
        return rng.sample(cands, min(n, len(cands)))

    def sources(self) -> Counter:
        return Counter(d.source.split("/")[2] if d.source.startswith("http") and d.source.count("/") >= 2 else d.source for d in self.docs.values())

    def stats(self) -> dict:
        return {
            "docs": len(self.docs),
            "terms": len(self.index),
            "k1": round(self.k1, 3),
            "b": round(self.b, 3),
            "phrase_bonus": round(self.phrase_bonus, 3),
            "est_mb": round(self.estimated_bytes() / 1048576, 1),
        }


def _is_kana_bigram(t: str) -> bool:
    return len(t) == 2 and all("぀" <= ch <= "ヿ" for ch in t)
