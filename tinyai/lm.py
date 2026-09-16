"""補間付き絶対ディスカウント + Kneser-Ney 継続カウントの n-gram 言語モデル。

メモリ効率のため、トークンは整数 ID に変換し、文脈 (直前 n トークン) は
1 個の整数にパックして辞書のキーにする:

    key = n | (t[i-1] << 4) | (t[i-2] << 24) | (t[i-3] << 44) ...   (20 bit / トークン)

これにより文字列タプルをキーにするより 1 エントリあたりのメモリが半分以下になり、
文脈キーの計算も加算とシフトだけになる。

各文脈の分布は、後続トークンが 1 種類だけの間は 1 個の整数 (tok | count << 20) で持ち
(高次の文脈はほとんどこれ)、2 種類以上になった時だけ {token_id: count, -1: 合計} の
辞書に昇格する。辞書 1 個は約 200 bytes なので、これで実メモリがさらに半減する。
"""
from __future__ import annotations

import math
import random
from typing import Iterable, Sequence

from .tokenizer import SENT_END, SENT_START

UNK, BOS, EOS = 0, 1, 2
TOKEN_BITS = 20
TOKEN_MASK = (1 << TOKEN_BITS) - 1
MAX_VOCAB = TOKEN_MASK  # 約 100 万語

# 1 エントリ (context->token->count) / 1 文脈あたりのおおよそのメモリ (bytes)。tools/bench.py で較正。
ENTRY_COST = 60
CONTEXT_COST = 50


class NGramLM:
    def __init__(self, max_order: int = 4, discount: float = 0.75, use_order: int | None = None):
        self.max_order = max(1, int(max_order))
        self.use_order = min(self.max_order, use_order or self.max_order)
        self.discount = discount
        self.vocab: dict[str, int] = {SENT_START: BOS, SENT_END: EOS}
        self.words: list[str] = ["<unk>", SENT_START, SENT_END]
        self.ctx: dict[int, dict[int, int]] = {0: {-1: 0}}
        self.cont: dict[int, int] = {}   # token -> 直前トークンの種類数 (KN 継続カウント)
        self.cont_total = 0
        self.entries = 0
        self.sentences = 0
        self.prune_threshold = 1

    # ------------------------------------------------------------ 語彙
    def _id(self, tok: str) -> int:
        i = self.vocab.get(tok)
        if i is None:
            if len(self.words) >= MAX_VOCAB:
                return UNK
            i = len(self.words)
            self.vocab[tok] = i
            self.words.append(tok)
        return i

    def ids(self, tokens: Iterable[str]) -> list[int]:
        v = self.vocab
        return [v.get(t, UNK) for t in tokens]

    @property
    def vocab_size(self) -> int:
        return len(self.words)

    def knows(self, token: str) -> bool:
        i = self.vocab.get(token)
        return i is not None and i in self.ctx[0]

    # ------------------------------------------------------------ 学習
    def learn(self, tokens: Sequence[str]) -> int:
        """次数ごとにストリーミングでカウントする。文脈キーは前の位置のキーから
        シフトと加算で作るので、位置ごとにタプルを組み立てるより速い。"""
        if not tokens:
            return 0
        vocab = self.vocab
        words = self.words
        ids = [BOS]
        for t in tokens:
            i = vocab.get(t)
            if i is None:
                if len(words) >= MAX_VOCAB:
                    i = UNK
                else:
                    i = len(words)
                    vocab[t] = i
                    words.append(t)
            ids.append(i)
        ids.append(EOS)
        ctx = self.ctx
        cont = self.cont
        one = 1 << TOKEN_BITS
        mask = TOKEN_MASK
        L = len(ids)
        new = 0
        # 次数 0 (unigram)
        uni = ctx[0]
        for i in range(1, L):
            tok = ids[i]
            c = uni.get(tok)
            if c is None:
                uni[tok] = 1
                new += 1
            else:
                uni[tok] = c + 1
        uni[-1] += L - 1
        # 次数 n ≥ 1: key = n | ids[i-1]<<4 | ids[i-2]<<24 | ...
        for n in range(1, self.max_order):
            if L - 1 < n + 1:
                break
            shift_new = 4 + TOKEN_BITS * (n - 1)  # 最も古いトークンのスロット
            # 位置 i = n のキーを直接作る
            key = n
            for j in range(1, n + 1):
                key |= ids[n - j] << (4 + TOKEN_BITS * (j - 1))
            for i in range(n, L):
                if i > n:
                    # 1 つ進める: 最も新しいトークン ids[i-1] を最下位スロットへ、最古を捨てる
                    key = n | ((key >> 4) << (4 + TOKEN_BITS) & ((1 << (4 + TOKEN_BITS * n)) - 1)) | (ids[i - 1] << 4)
                tok = ids[i]
                d = ctx.get(key)
                if d is None:
                    ctx[key] = tok | one
                    new += 1
                    if n == 1:
                        cont[tok] = cont.get(tok, 0) + 1
                        self.cont_total += 1
                elif type(d) is int:
                    if (d & mask) == tok:
                        ctx[key] = d + one
                    else:
                        cnt = d >> TOKEN_BITS
                        ctx[key] = {-1: cnt + 1, d & mask: cnt, tok: 1}
                        new += 1
                        if n == 1:
                            cont[tok] = cont.get(tok, 0) + 1
                            self.cont_total += 1
                else:
                    c = d.get(tok)
                    if c is None:
                        d[tok] = 1
                        new += 1
                        if n == 1:
                            cont[tok] = cont.get(tok, 0) + 1
                            self.cont_total += 1
                    else:
                        d[tok] = c + 1
                    d[-1] += 1
        self.entries += new
        self.sentences += 1
        return new

    # ------------------------------------------------------------ 確率
    def _keys(self, hist: Sequence[int]) -> list[int]:
        """履歴 (ID 列) から各次数の文脈キー [key0, key1, ...] を作る。"""
        keys = [0]
        key = 0
        for n in range(1, self.use_order):
            if n > len(hist):
                break
            key += 1 + (hist[-n] << (4 + TOKEN_BITS * (n - 1)))
            keys.append(key)
        return keys

    def _prob_keys(self, keys: list[int], n: int, tok: int) -> float:
        """低次から高次へ反復で補間する (再帰より速い)。"""
        p = (self.cont.get(tok, 0) + 1.0) / (self.cont_total + self.vocab_size)
        ctx = self.ctx
        disc = self.discount
        for i in range(1, n + 1):
            d = ctx.get(keys[i])
            if d is None:
                continue
            if type(d) is int:
                total = d >> TOKEN_BITS
                c = total if (d & TOKEN_MASK) == tok else 0
                p = max(c - disc, 0.0) / total + disc / total * p
            else:
                total = d[-1]
                c = d.get(tok, 0)
                p = max(c - disc, 0.0) / total + disc * (len(d) - 1) / total * p
        return p

    def prob(self, context: Sequence[str], token: str) -> float:
        hist = self.ids(context)
        keys = self._keys(hist)
        return self._prob_keys(keys, len(keys) - 1, self.vocab.get(token, UNK))

    def perplexity(self, sentences: Iterable[Sequence[str]]) -> float:
        logp = 0.0
        n = 0
        for tokens in sentences:
            ids = [BOS] + self.ids(tokens) + [EOS]
            for i in range(1, len(ids)):
                keys = self._keys(ids[:i])
                p = self._prob_keys(keys, len(keys) - 1, ids[i])
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
        focus: set[str] | None = None,
        focus_bonus: float = 2.0,
    ) -> list[str]:
        """seed の続きを生成。focus に含まれる語 (検索で見つかった知識の語) を優先する。"""
        rng = rng or random
        out = [BOS] + self.ids(seed)
        focus_ids = {self.vocab[t] for t in focus if t in self.vocab} if focus else set()
        for _ in range(max_len):
            keys = self._keys(out)
            cands = None
            n_used = 0
            for n in range(len(keys) - 1, -1, -1):
                d = self.ctx.get(keys[n])
                if d is None:
                    continue
                if type(d) is int:
                    if (d >> TOKEN_BITS) >= 2:
                        cands = [d & TOKEN_MASK]
                        n_used = n
                        break
                elif d[-1] >= (2 if n else 1):
                    cands = [t for t in d if t != -1]
                    n_used = n
                    break
            if not cands:
                break
            if len(cands) > 64:
                d = self.ctx[keys[n_used]]
                cands = sorted(cands, key=lambda t: d[t], reverse=True)[:64]
            weights = []
            recent = out[-6:]
            inv_t = 1.0 / max(temperature, 0.05)
            for t in cands:
                p = self._prob_keys(keys, len(keys) - 1, t)
                if t == EOS and len(out) - 1 < min_len:
                    p *= 0.05
                if t in recent:
                    p *= 0.3
                if t in focus_ids:
                    p *= focus_bonus
                weights.append(p ** inv_t)
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
            if pick == EOS:
                break
            out.append(pick)
        words = self.words
        return [words[i] for i in out[1:]]

    # ------------------------------------------------------------ 圧縮
    def estimated_bytes(self) -> int:
        return self.entries * ENTRY_COST + len(self.ctx) * CONTEXT_COST + len(self.words) * 70

    def prune(self, min_count: int | None = None, keep_orders: int | None = None) -> int:
        """カウントが min_count 未満のエントリ (unigram 以外) を削除。削除数を返す。"""
        if min_count is None:
            min_count = self.prune_threshold + 1
        removed = 0
        dead = []
        cont = self.cont
        for key, d in self.ctx.items():
            if key == 0:
                continue
            n = key & 15
            if type(d) is int:
                if (keep_orders is not None and n >= keep_orders) or (d >> TOKEN_BITS) < min_count:
                    dead.append(key)
                    removed += 1
                    if n == 1:
                        cont[d & TOKEN_MASK] -= 1
                        self.cont_total -= 1
                continue
            if keep_orders is not None and n >= keep_orders:
                dead.append(key)
                removed += len(d) - 1
                if n == 1:
                    for t in d:
                        if t != -1:
                            cont[t] -= 1
                            self.cont_total -= 1
                continue
            drop = [t for t, c in d.items() if t != -1 and c < min_count]
            if drop:
                lost = 0
                for t in drop:
                    lost += d.pop(t)
                    if n == 1:
                        cont[t] -= 1
                        self.cont_total -= 1
                removed += len(drop)
                d[-1] -= lost
            if len(d) == 1:
                dead.append(key)
        for key in dead:
            self.ctx.pop(key, None)
        self.entries -= removed
        self.prune_threshold = max(self.prune_threshold, min_count - 1)
        return removed

    def shrink_to(self, budget_bytes: int) -> int:
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

    # ------------------------------------------------------------ 保存
    def state(self) -> dict:
        return {
            "words": self.words, "ctx": self.ctx, "cont": self.cont, "cont_total": self.cont_total,
            "entries": self.entries, "sentences": self.sentences, "max_order": self.max_order,
            "prune_threshold": self.prune_threshold, "format": 2,
        }

    @classmethod
    def from_state(cls, st: dict) -> "NGramLM":
        lm = cls(max_order=st["max_order"])
        lm.words = st["words"]
        lm.vocab = {w: i for i, w in enumerate(lm.words)}
        lm.ctx = st["ctx"]
        lm.cont = st["cont"]
        lm.cont_total = st["cont_total"]
        lm.entries = st["entries"]
        lm.sentences = st["sentences"]
        lm.prune_threshold = st.get("prune_threshold", 1)
        return lm

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


class CacheLM:
    """会話キャッシュ LM (Kuhn & De Mori 1990 の系譜)。直近のトークン列から減衰付きの
    unigram / bigram 分布を持ち、静的な n-gram と補間して「今の会話の流れ」に沿った
    確率を与える。数百トークンしか持たないので実質メモリゼロ。"""

    def __init__(self, capacity: int = 300, decay: float = 0.995):
        self.capacity = capacity
        self.decay = decay
        self.uni: dict[int, float] = {}
        self.bi: dict[int, dict[int, float]] = {}
        self.total = 0.0
        self.last: int | None = None
        self.n = 0

    def push(self, ids) -> None:
        for tok in ids:
            self.uni[tok] = self.uni.get(tok, 0.0) + 1.0
            self.total += 1.0
            if self.last is not None:
                d = self.bi.setdefault(self.last, {})
                d[tok] = d.get(tok, 0.0) + 1.0
            self.last = tok
            self.n += 1
            if self.n % 200 == 0:
                self._decay()
        if len(self.uni) > self.capacity * 3:
            self._decay(hard=True)

    def _decay(self, hard: bool = False) -> None:
        f = self.decay ** 200 if not hard else 0.5
        self.total *= f
        for k in list(self.uni):
            self.uni[k] *= f
            if self.uni[k] < 0.05:
                del self.uni[k]
        for a in list(self.bi):
            d = self.bi[a]
            for b in list(d):
                d[b] *= f
                if d[b] < 0.05:
                    del d[b]
            if not d:
                del self.bi[a]

    def prob(self, prev: int | None, tok: int) -> float:
        """キャッシュ内での P(tok | prev)。無ければ 0。"""
        if self.total <= 0:
            return 0.0
        p_uni = self.uni.get(tok, 0.0) / self.total
        if prev is not None:
            d = self.bi.get(prev)
            if d:
                tot = sum(d.values())
                return 0.6 * d.get(tok, 0.0) / tot + 0.4 * p_uni
        return p_uni

    def clear(self) -> None:
        self.uni.clear()
        self.bi.clear()
        self.total = 0.0
        self.last = None
