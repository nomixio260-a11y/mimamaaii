"""補間付き絶対ディスカウント n-gram 言語モデル。

`ctx[context_tuple] = [total, {token: count}]` の入れ子辞書で保持する。
メモリ制限に合わせて `prune()` で低頻度エントリを削り、`use_order` で
実際に使う次数を進化 (evolve) で変えられる。
"""
from __future__ import annotations

import math
import random
from typing import Iterable, Sequence

from .tokenizer import SENT_END, SENT_START

# 1 エントリ (context->token->count) あたりのおおよそのメモリ (bytes)
ENTRY_COST = 110
CONTEXT_COST = 240


class NGramLM:
    def __init__(self, max_order: int = 4, discount: float = 0.75, use_order: int | None = None):
        self.max_order = max(1, int(max_order))
        self.use_order = min(self.max_order, use_order or self.max_order)
        self.discount = discount
        self.ctx: dict[tuple, list] = {(): [0, {}]}
        self.entries = 0
        self.sentences = 0
        self.prune_threshold = 1  # プルーニングで削る最大カウント (段階的に上がる)

    # ------------------------------------------------------------ 学習
    def learn(self, tokens: Sequence[str]) -> int:
        if not tokens:
            return 0
        toks = [SENT_START] + list(tokens) + [SENT_END]
        ctxs = self.ctx
        new = 0
        for i in range(1, len(toks)):
            tok = toks[i]
            for n in range(self.max_order):
                if i - n < 0:
                    break
                key = tuple(toks[i - n : i]) if n else ()
                e = ctxs.get(key)
                if e is None:
                    e = ctxs[key] = [0, {}]
                d = e[1]
                if tok in d:
                    d[tok] += 1
                else:
                    d[tok] = 1
                    new += 1
                e[0] += 1
        self.entries += new
        self.sentences += 1
        return new

    # ------------------------------------------------------------ 確率
    @property
    def vocab_size(self) -> int:
        return len(self.ctx[()][1]) + 1

    def prob(self, context: Sequence[str], token: str) -> float:
        """P(token | context) 補間付き絶対ディスカウント。"""
        order = self.use_order
        ctx = tuple(context[-(order - 1):]) if order > 1 else ()
        return self._prob(ctx, token)

    def _prob(self, ctx: tuple, token: str) -> float:
        if not ctx:
            root = self.ctx[()]
            total = root[0]
            v = self.vocab_size
            return (root[1].get(token, 0) + 1.0) / (total + v)
        e = self.ctx.get(ctx)
        lower = self._prob(ctx[1:], token)
        if e is None or e[0] == 0:
            return lower
        total, d = e
        c = d.get(token, 0)
        disc = self.discount
        p = max(c - disc, 0.0) / total
        backoff = disc * len(d) / total
        return p + backoff * lower

    def perplexity(self, sentences: Iterable[Sequence[str]]) -> float:
        logp = 0.0
        n = 0
        for tokens in sentences:
            toks = [SENT_START] + list(tokens) + [SENT_END]
            for i in range(1, len(toks)):
                p = self.prob(toks[:i], toks[i])
                logp += math.log(max(p, 1e-12))
                n += 1
        if n == 0:
            return float("inf")
        return math.exp(-logp / n)

    # ------------------------------------------------------------ 生成
    def generate(
        self,
        seed: Sequence[str] = (),
        max_len: int = 40,
        temperature: float = 0.8,
        rng: random.Random | None = None,
        min_len: int = 4,
    ) -> list[str]:
        rng = rng or random
        out = [SENT_START] + list(seed)
        order = self.use_order
        for step in range(max_len):
            cands = None
            for n in range(order - 1, -1, -1):
                key = tuple(out[-n:]) if n else ()
                e = self.ctx.get(key)
                if e and e[0] >= (2 if n else 1):
                    cands = list(e[1].keys())
                    break
            if not cands:
                break
            if len(cands) > 64:
                # 高頻度候補から絞る
                e = self.ctx[key]
                cands = sorted(cands, key=lambda t: e[1][t], reverse=True)[:64]
            weights = []
            for t in cands:
                p = self.prob(out, t)
                if t == SENT_END and len(out) - 1 < min_len:
                    p *= 0.05
                # 同じ語の繰り返しを抑制
                if t in out[-6:]:
                    p *= 0.3
                weights.append(p ** (1.0 / max(temperature, 0.05)))
            tot = sum(weights)
            if tot <= 0:
                break
            r = rng.random() * tot
            acc = 0.0
            pick = cands[-1]
            for t, w in zip(cands, weights):
                acc += w
                if acc >= r:
                    pick = t
                    break
            if pick == SENT_END:
                break
            out.append(pick)
        return out[1:]

    def knows(self, token: str) -> bool:
        return token in self.ctx[()][1]

    # ------------------------------------------------------------ 圧縮
    def estimated_bytes(self) -> int:
        return self.entries * ENTRY_COST + len(self.ctx) * CONTEXT_COST

    def prune(self, min_count: int | None = None, keep_orders: int | None = None) -> int:
        """カウントが min_count 未満のエントリ (unigram 以外) を削除。削除数を返す。"""
        if min_count is None:
            min_count = self.prune_threshold + 1
        removed = 0
        dead = []
        for key, (total, d) in self.ctx.items():
            if not key:
                continue
            if keep_orders is not None and len(key) >= keep_orders:
                dead.append(key)
                removed += len(d)
                continue
            drop = [t for t, c in d.items() if c < min_count]
            if drop:
                lost = 0
                for t in drop:
                    lost += d.pop(t)
                removed += len(drop)
                total -= lost
                self.ctx[key][0] = total
            if not d:
                dead.append(key)
        for key in dead:
            self.ctx.pop(key, None)
        self.entries -= removed
        self.prune_threshold = max(self.prune_threshold, min_count - 1)
        return removed

    def shrink_to(self, budget_bytes: int) -> int:
        """推定サイズが予算内に収まるまで段階的にプルーニング。"""
        removed = 0
        rounds = 0
        while self.estimated_bytes() > budget_bytes and rounds < 8:
            rounds += 1
            before = self.entries
            removed += self.prune(min_count=self.prune_threshold + 1)
            if self.entries == before:
                self.prune_threshold += 1
        if self.estimated_bytes() > budget_bytes and self.max_order > 2:
            removed += self.prune(min_count=1, keep_orders=self.max_order)
            self.max_order -= 1
            self.use_order = min(self.use_order, self.max_order)
        return removed

    def stats(self) -> dict:
        return {
            "entries": self.entries,
            "contexts": len(self.ctx),
            "vocab": self.vocab_size,
            "sentences": self.sentences,
            "max_order": self.max_order,
            "use_order": self.use_order,
            "discount": round(self.discount, 3),
            "prune_threshold": self.prune_threshold,
            "est_mb": round(self.estimated_bytes() / 1048576, 1),
        }
