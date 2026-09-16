"""意味ベクトル: Random Indexing (Kanerva 1988, Sahlgren 2005) による語の分散表現。

* 各語は決定的なハッシュから作る疎な ±1 のインデックス（乱択）ベクトルを持つ (保存不要)
* 文中で近くに現れた語のインデックスベクトルを足し込んだものが、その語の文脈ベクトル (int16 × DIM)
* 似た文脈に現れる語は似たベクトルになる (word2vec の超軽量・オンライン版、学習は加算だけ)
* 近傍探索は 64 bit の符号スケッチ (SimHash) でハミング距離の候補を絞ってから正確なコサインを計算

学習は「後回し」でよい設計: Brain は学習した文書の ID をキューに積み、
自律ループの空き時間 (または応答の合間) に少しずつ取り込む。会話の即時応答経路は遅くならない。
"""
from __future__ import annotations

import array
import hashlib
import math
import random
from typing import Iterable

DIM = 128
NONZERO = 8          # インデックスベクトルの非ゼロ数
WINDOW = 3
MAX_ABS = 30000      # int16 の飽和を避けるための縮小しきい値


def _index_vector(term: str) -> list[tuple[int, int]]:
    """語ごとに決定的な疎ベクトル [(位置, ±1), ...]。"""
    h = hashlib.blake2b(term.encode("utf-8"), digest_size=NONZERO * 2).digest()
    out = []
    for i in range(NONZERO):
        pos = ((h[2 * i] << 8) | h[2 * i + 1]) % DIM
        sign = 1 if (h[2 * i] & 1) else -1
        out.append((pos, sign))
    return out


class SemanticSpace:
    def __init__(self, dim: int = DIM):
        self.dim = dim
        self.vec: dict[str, array.array] = {}
        self.sketch: dict[str, int] = {}
        self._dirty: set[str] = set()
        self._planes: list[list[int]] | None = None
        self.updates = 0
        self._ivcache: dict[str, list[tuple[int, int]]] = {}
        self._unit: dict[str, tuple[int, list[float]]] = {}  # 語 -> (世代, 単位ベクトル) キャッシュ

    # ------------------------------------------------------------ 学習
    def _iv(self, term: str) -> list[tuple[int, int]]:
        v = self._ivcache.get(term)
        if v is None:
            if len(self._ivcache) > 20000:
                self._ivcache.clear()
            v = self._ivcache[term] = _index_vector(term)
        return v

    def learn(self, phrases: list[str], window: int = WINDOW) -> None:
        """1 文の句列 (漢字/カタカナ語・英単語) から文脈ベクトルを更新する。"""
        n = len(phrases)
        if n < 2:
            return
        vec = self.vec
        for i, t in enumerate(phrases):
            v = vec.get(t)
            if v is None:
                v = vec[t] = array.array("h", bytes(2 * self.dim))
            lo, hi = max(0, i - window), min(n, i + window + 1)
            for j in range(lo, hi):
                if j == i:
                    continue
                w = 2 if abs(j - i) == 1 else 1
                for pos, sign in self._iv(phrases[j]):
                    v[pos] += sign * w
            self._dirty.add(t)
        self.updates += 1
        if self.updates % 5000 == 0:
            self._rescale()

    def _rescale(self) -> None:
        for v in self.vec.values():
            m = max(abs(x) for x in v)
            if m > MAX_ABS:
                for k in range(len(v)):
                    v[k] //= 2

    # ------------------------------------------------------------ 類似
    def vector(self, term: str) -> array.array | None:
        return self.vec.get(term)

    def _unit_vector(self, t: str) -> list[float] | None:
        v = self.vec.get(t)
        if v is None:
            return None
        gen = self.updates // 2000
        hit = self._unit.get(t)
        if hit is not None and hit[0] == gen:
            return hit[1]
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        u = [x / norm for x in v]
        if len(self._unit) > 20000:
            self._unit.clear()
        self._unit[t] = (gen, u)
        return u

    def text_vector(self, phrases: Iterable[str]) -> list[float] | None:
        acc = None
        for t in phrases:
            u = self._unit_vector(t)
            if u is None:
                continue
            if acc is None:
                acc = list(u)
            else:
                for k, x in enumerate(u):
                    acc[k] += x
        return acc

    @staticmethod
    def cosine(a, b) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a)) or 1.0
        nb = math.sqrt(sum(y * y for y in b)) or 1.0
        return dot / (na * nb)

    def _ensure_planes(self) -> list[tuple[list[int], list[int]]]:
        """疎な超平面 64 枚: それぞれ +1 の位置 16 個と -1 の位置 16 個 (射影が 32 回の加減算で済む)。"""
        if self._planes is None:
            rng = random.Random(12345)
            planes = []
            for _ in range(64):
                idx = rng.sample(range(self.dim), 32)
                planes.append((idx[:16], idx[16:]))
            self._planes = planes
        return self._planes

    def refresh_sketches(self, limit: int = 2000) -> int:
        """更新のあった語の符号スケッチを計算し直す (整理のタイミングで呼ぶ)。"""
        planes = self._ensure_planes()
        done = 0
        for t in list(self._dirty)[:limit]:
            v = self.vec.get(t)
            self._dirty.discard(t)
            if v is None:
                continue
            bits = 0
            for pos_idx, neg_idx in planes:
                proj = 0
                for k in pos_idx:
                    proj += v[k]
                for k in neg_idx:
                    proj -= v[k]
                bits = (bits << 1) | (1 if proj > 0 else 0)
            self.sketch[t] = bits
            done += 1
        return done

    def similar(self, term: str, k: int = 5, candidates: int = 40) -> list[tuple[str, float]]:
        """term に意味的に近い語。スケッチのハミング距離で候補を絞り、コサインで並べ替える。"""
        v = self.vec.get(term)
        sk = self.sketch.get(term)
        if v is None or sk is None or len(self.sketch) < 10:
            return []
        near = []
        for u, s in self.sketch.items():
            if u == term:
                continue
            d = (s ^ sk).bit_count()
            if d <= 20:
                near.append((d, u))
        if not near:
            return []
        near.sort()
        out = []
        for _, u in near[:candidates]:
            c = self.cosine(v, self.vec[u])
            if c > 0.3:
                out.append((u, c))
        out.sort(key=lambda x: -x[1])
        return out[:k]

    # ------------------------------------------------------------ 容量
    def estimated_bytes(self) -> int:
        return len(self.vec) * (2 * self.dim + 160)

    def shrink_to(self, budget_bytes: int, keep_terms: Iterable[str] = ()) -> int:
        """予算に収まるまで、ノルムの小さい (情報の少ない) 語から消す。"""
        if self.estimated_bytes() <= budget_bytes:
            return 0
        keep = set(keep_terms)
        target = max(0, int(budget_bytes / (2 * self.dim + 160)))
        scored = sorted(((sum(abs(x) for x in v), t) for t, v in self.vec.items() if t not in keep))
        removed = 0
        for _, t in scored:
            if len(self.vec) <= target:
                break
            del self.vec[t]
            self.sketch.pop(t, None)
            self._dirty.discard(t)
            removed += 1
        return removed

    def state(self) -> dict:
        return {"dim": self.dim, "vec": {t: v.tobytes() for t, v in self.vec.items()}, "sketch": self.sketch}

    @classmethod
    def from_state(cls, st: dict) -> "SemanticSpace":
        sp = cls(st.get("dim", DIM))
        for t, b in st.get("vec", {}).items():
            a = array.array("h")
            a.frombytes(b)
            sp.vec[t] = a
        sp.sketch = dict(st.get("sketch", {}))
        sp._dirty = set(t for t in sp.vec if t not in sp.sketch)
        return sp

    def stats(self) -> dict:
        return {"terms": len(self.vec), "sketched": len(self.sketch), "pending": len(self._dirty), "est_mb": round(self.estimated_bytes() / 1048576, 1)}
