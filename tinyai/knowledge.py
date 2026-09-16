"""文単位の知識ベース + BM25 転置インデックス。

高速化・省メモリの工夫:
  * 転置索引の投稿リストは、文書が 1 件だけの間は 1 個の整数 (doc_id << 8 | tf) で持ち、
    2 件以上になった時だけ {doc_id: tf} の辞書に昇格する (語の大半は 1 件だけ)
  * 出現文書率が高すぎる語 (かな bigram など) は他に語があれば無視する (動的ストップ語)
  * 検索結果は小さな LRU キャッシュに入れ、追加/削除で無効化する
  * 追い出し時の転置索引の更新は文書のテキストから語を再計算する (文書ごとに語列を持たない)
  * ジャンク文 (記号だらけ、数字だらけ、表の断片) は最初から取り込まない

知能面の工夫:
  * associate(): 質問文の語を答えの文に結び付ける (訂正・👍 から学ぶ)
  * related_terms(): 共起から関連語を求める (クエリ拡張に使う。学習時のコストはゼロ)
"""
from __future__ import annotations

import math
import random
import re
import time
from collections import Counter, OrderedDict
from typing import Iterable

from .tokenizer import is_phrase, term_weight, terms

DOC_BASE_COST = 200
TF_BITS = 8
TF_MAX = (1 << TF_BITS) - 1
_JUNK_RE = re.compile(r"[\[\]{}|<>=_*#@^~\\]")
_NON_ALNUM_RE = re.compile(r"[\W_]+")
_DIGIT_RE = re.compile(r"\d")


def is_junk(text: str) -> bool:
    """学習する価値の低い文かどうか (正規表現だけで判定、1 文あたり数 µs)。"""
    n = len(text)
    if n < 4:
        return True
    alnum = n - sum(len(m) for m in _NON_ALNUM_RE.findall(text))
    if alnum / n < 0.55:
        return True
    if len(_DIGIT_RE.findall(text)) / n > 0.4:
        return True
    if len(_JUNK_RE.findall(text)) > 2:
        return True
    # 同じ文字の異常な繰り返し
    if n >= 12 and len(set(text)) < n / 4:
        return True
    return False


def _post_items(post):
    if type(post) is int:
        return ((post >> TF_BITS, post & TF_MAX),)
    return post.items()


def _post_len(post) -> int:
    return 1 if type(post) is int else len(post)


def _post_has(post, doc_id: int) -> bool:
    if type(post) is int:
        return (post >> TF_BITS) == doc_id
    return doc_id in post


class Doc:
    __slots__ = ("id", "text", "source", "added", "score", "length", "quality", "hits")

    def __init__(self, id: int, text: str, source: str, added: float, length: int):
        self.id = id
        self.text = text
        self.source = source
        self.added = added
        self.score = 0.0      # フィードバックによる信頼度
        self.length = length
        self.quality = 0.5    # 文の品質 (0..1): 定義文・数値・固有名詞を含むほど高い
        self.hits = 0         # 検索で上位に出た回数 (使われる知識は残る)


class KnowledgeBase:
    def __init__(self, k1: float = 1.4, b: float = 0.6, phrase_bonus: float = 1.5, stop_ratio: float = 0.2):
        self.k1 = k1
        self.b = b
        self.phrase_bonus = phrase_bonus
        self.stop_ratio = stop_ratio      # これ以上の文書に出る語は (他に語があれば) 無視
        self.docs: dict[int, Doc] = {}
        self.index: dict[str, dict[int, int]] = {}
        self.hashes: set[int] = set()
        self.content_keys: set[int] = set()  # 近似重複判定用 (情報量の高い語の集合のハッシュ)
        self.assoc: dict[int, list[str]] = {}  # doc_id -> 結び付けた追加の語
        self.next_id = 1
        self.total_len = 0
        self._est_bytes = 0
        self._cache: OrderedDict = OrderedDict()
        self._version = 0
        self.on_remove = None  # 文書削除時のフック (事実ストアの同期用)

    # ------------------------------------------------------------ 追加/削除
    @staticmethod
    def _hash(text: str) -> int:
        # プロセス内で一貫していればよい (保存時は文書を再追加して作り直す)
        return hash(text.strip().lower())

    @staticmethod
    def _content_key(tf: Counter) -> int | None:
        """情報量の高い語 (句) の集合から作る鍵。語順や助詞が違うだけの文は同じ鍵になる。"""
        # 句 (漢字/カタカナ語・英単語) と数字・英数字トークンが文の「中身」。かな bigram は無視
        items = sorted(t for t in tf if is_phrase(t) or (t.isascii() and len(t) >= 2))
        if len(items) < 3:
            return None
        return hash(tuple(items))

    def add(self, text: str, source: str = "", tf: Counter | None = None, quality: float = 0.5) -> Doc | None:
        text = text.strip()
        if len(text) < 4 or len(text) > 600:
            return None
        h = self._hash(text)
        if h in self.hashes or is_junk(text):
            return None
        if tf is None:
            tf = Counter(terms(text))
        if not tf:
            return None
        ck = self._content_key(tf)
        if ck is not None:
            if ck in self.content_keys:
                return None  # 近似重複 (同じ句の集合を持つ文が既にある)
            self.content_keys.add(ck)
        doc = Doc(self.next_id, text, source[:120], time.time(), sum(tf.values()))
        doc.quality = quality
        self.next_id += 1
        self.docs[doc.id] = doc
        self.hashes.add(h)
        index = self.index
        for t, c in tf.items():
            self._post_add(index, t, doc.id, c)
        self.total_len += doc.length
        self._est_bytes += DOC_BASE_COST + len(text) * 2 + len(tf) * 60
        self._version += 1
        return doc

    @staticmethod
    def _post_add(index: dict, t: str, doc_id: int, c: int) -> None:
        c = min(c, TF_MAX)
        post = index.get(t)
        if post is None:
            index[t] = (doc_id << TF_BITS) | c
        elif type(post) is int:
            index[t] = {post >> TF_BITS: post & TF_MAX, doc_id: c}
        else:
            post[doc_id] = c

    def _post_remove(self, t: str, doc_id: int) -> None:
        post = self.index.get(t)
        if post is None:
            return
        if type(post) is int:
            if (post >> TF_BITS) == doc_id:
                del self.index[t]
            return
        post.pop(doc_id, None)
        if not post:
            del self.index[t]
        elif len(post) == 1:
            (d, c), = post.items()
            self.index[t] = (d << TF_BITS) | c

    def posting_ids(self, t: str) -> list[int]:
        post = self.index.get(t)
        return [d for d, _ in _post_items(post)] if post is not None else []

    def remove(self, doc_id: int) -> None:
        doc = self.docs.pop(doc_id, None)
        if doc is None:
            return
        if self.on_remove is not None:
            self.on_remove(doc_id)
        self.hashes.discard(self._hash(doc.text))
        self.total_len -= doc.length
        tf = Counter(terms(doc.text))
        ck = self._content_key(tf)
        if ck is not None:
            self.content_keys.discard(ck)
        extra = self.assoc.pop(doc_id, ())
        for t in list(tf) + list(extra):
            self._post_remove(t, doc_id)
        self._est_bytes -= DOC_BASE_COST + len(doc.text) * 2 + len(tf) * 60
        self._version += 1

    def associate(self, doc_id: int, text: str, weight: int = 2) -> int:
        """質問文の語をこの文書に結び付ける。次から同じ聞き方で直接ヒットする。"""
        if doc_id not in self.docs:
            return 0
        added = 0
        lst = self.assoc.setdefault(doc_id, [])
        for t in set(terms(text)):
            if term_weight(t) < 1.0:
                continue
            post = self.index.get(t)
            if post is not None and _post_has(post, doc_id):
                continue
            self._post_add(self.index, t, doc_id, weight)
            lst.append(t)
            added += 1
            self._est_bytes += 80
        if added:
            self._version += 1
        return added

    def __len__(self) -> int:
        return len(self.docs)

    def estimated_bytes(self) -> int:
        return max(0, self._est_bytes)

    # ------------------------------------------------------------ 検索
    def search(self, query: str, k: int = 5, extra: dict[str, float] | None = None) -> list[tuple[float, Doc]]:
        """BM25 検索。extra は拡張語 -> 重み (0..1)。戻り値は (スコア, 文書)。"""
        q_terms = terms(query)
        if not q_terms or not self.docs:
            return []
        key = (query, k, tuple(sorted(extra.items())) if extra else None, self._version)
        hit = self._cache.get(key)
        if hit is not None:
            self._cache.move_to_end(key)
            return list(hit)
        res = self._search_terms(Counter(q_terms), k, extra)
        self._cache[key] = res
        if len(self._cache) > 64:
            self._cache.popitem(last=False)
        return list(res)

    def _search_terms(self, qtf: Counter, k: int, extra: dict[str, float] | None) -> list[tuple[float, Doc]]:
        n = len(self.docs)
        avgdl = self.total_len / n if n else 1.0
        index = self.index
        docs = self.docs
        k1, b = self.k1, self.b
        # 動的ストップ語: 文書の stop_ratio 以上に出る語は、他に語があれば飛ばす
        limit = max(50, int(n * self.stop_ratio))
        weighted: list[tuple[str, float, int]] = []
        for t, qc in qtf.items():
            post = index.get(t)
            if post is None:
                continue
            weighted.append((t, min(qc, 2), _post_len(post)))
        if extra:
            for t, w in extra.items():
                post = index.get(t)
                if post is not None and t not in qtf:
                    weighted.append((t, w, _post_len(post)))
        if not weighted:
            return []
        informative = [x for x in weighted if x[2] <= limit]
        if informative:
            weighted = informative
        scores: dict[int, float] = {}
        cover: dict[int, float] = {}
        total_w = sum(term_weight(t) for t in qtf) or 1.0
        for t, qw, df in weighted:
            post = index[t]
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            tw = term_weight(t)
            w = idf * tw * (1.0 + (self.phrase_bonus - 1.0) * (len(t) >= 3 or (len(t) == 2 and not _is_kana_bigram(t))))
            in_query = t in qtf
            for doc_id, tf in _post_items(post):
                doc = docs.get(doc_id)
                if doc is None:
                    continue
                denom = tf + k1 * (1 - b + b * doc.length / avgdl)
                scores[doc_id] = scores.get(doc_id, 0.0) + w * qw * tf * (k1 + 1) / denom
                if in_query:
                    cover[doc_id] = cover.get(doc_id, 0.0) + tw
        if not scores:
            return []
        ranked = []
        for doc_id, s in scores.items():
            doc = docs[doc_id]
            c = cover.get(doc_id, 0.0) / total_w
            ranked.append((s * (0.5 + c) * (0.85 + 0.3 * doc.quality) + 0.15 * doc.score, doc))
        ranked.sort(key=lambda x: (-x[0], -x[1].added))
        top = ranked[:k]
        for _, d in top[:2]:
            d.hits += 1
        return top

    def coverage(self, doc_id: int, query: str) -> float:
        """クエリ語の情報量重み付きカバー率 (0..1)。"""
        q = set(terms(query))
        if not q:
            return 0.0
        total = sum(term_weight(t) for t in q) or 1.0
        index = self.index
        got = 0.0
        for t in q:
            post = index.get(t)
            if post is not None and _post_has(post, doc_id):
                got += term_weight(t)
        return got / total

    def related_terms(self, term: str, k: int = 3, max_docs: int = 40) -> list[tuple[str, float]]:
        """term と共起しやすい語 (分布的な関連語)。学習時には何も計算しない。"""
        post = self.index.get(term)
        if post is None or len(self.docs) < 20:
            return []
        n = len(self.docs)
        df_t = _post_len(post)
        ids = [d for d, _ in _post_items(post)][-max_docs:]
        co: Counter = Counter()
        for doc_id in ids:
            doc = self.docs.get(doc_id)
            if doc is None:
                continue
            for u in set(terms(doc.text)):
                if u != term and is_phrase(u) and u not in term and term not in u:
                    co[u] += 1
        # 部分文字列 (カタカナ bigram など) は、それを含むより長い候補があれば捨てる
        top = [u for u, _ in co.most_common(60)]
        keep = [u for u in top if not any(len(v) > len(u) and u in v for v in top)]
        out = []
        for u in keep:
            c = co[u]
            pu = self.index.get(u)
            df_u = _post_len(pu) if pu is not None else 0
            if df_u < 2 or c < 2:
                continue
            # PMI 風: 共起 / 期待共起
            pmi = math.log((c * n) / (df_t * df_u) + 1e-9)
            if pmi > 0.5:
                out.append((u, pmi * min(1.0, c / 5.0)))
        out.sort(key=lambda x: -x[1])
        return out[:k]

    def feedback(self, doc_id: int, delta: float) -> None:
        d = self.docs.get(doc_id)
        if d:
            d.score = max(-5.0, min(20.0, d.score + delta))
            self._version += 1

    # ------------------------------------------------------------ 圧縮
    def evict(self, n: int) -> int:
        """役に立っていない古い文から n 件削除。"""
        if n <= 0 or not self.docs:
            return 0
        now = time.time()

        def key(d: Doc):
            age = (now - d.added) / 86400.0
            protect = 3.0 if d.source in ("user", "chat", "seed") else 0.0
            return d.score + protect + d.quality + 0.3 * min(d.hits, 10) - age * 0.2

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
        cands = [t for t, post in self.index.items() if _post_len(post) <= 2 and len(t) >= min_len and is_phrase(t)]
        if not cands:
            return []
        return rng.sample(cands, min(n, len(cands)))

    def sources(self) -> Counter:
        return Counter(d.source.split("/")[2] if d.source.startswith("http") and d.source.count("/") >= 2 else d.source for d in self.docs.values())

    def stats(self) -> dict:
        return {
            "docs": len(self.docs),
            "terms": len(self.index),
            "assoc": len(self.assoc),
            "avg_quality": round(sum(d.quality for d in self.docs.values()) / len(self.docs), 3) if self.docs else None,
            "k1": round(self.k1, 3),
            "b": round(self.b, 3),
            "phrase_bonus": round(self.phrase_bonus, 3),
            "est_mb": round(self.estimated_bytes() / 1048576, 1),
        }


def _is_kana_bigram(t: str) -> bool:
    return len(t) == 2 and all("぀" <= ch <= "ヿ" for ch in t)
