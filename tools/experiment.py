"""研究・実験ハーネス。コーパスを与えると、以下を比較して表にする。

  1. n-gram 次数 (2/3/4) とディスカウントのパープレキシティ・メモリ・学習時間
  2. Count-Min Sketch で高次 n-gram を近似する「乱択言語モデル」(Talbot & Osborne 2007 の系譜)
     - 正確な辞書との比較: メモリを固定して、パープレキシティがどれだけ悪化するか
  3. 接尾辞配列 (Infini-gram, Liu et al. 2024 の考え方) による任意長 n-gram カウント
     - 構築時間・メモリ・カウント問い合わせの速度

    python tools/experiment.py corpus.txt [--limit 3000]
"""
from __future__ import annotations

import argparse
import array
import bisect
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tinyai.lm import NGramLM, BOS, EOS  # noqa: E402
from tinyai.memory import rss_bytes  # noqa: E402
from tinyai.tokenizer import split_sentences, tokenize  # noqa: E402

MB = 1024 * 1024


# ---------------------------------------------------------------- 1. 次数・ディスカウント
def exp_orders(train, hold):
    rows = []
    for order in (2, 3, 4, 5):
        for disc in (0.6, 0.75, 0.9):
            lm = NGramLM(max_order=order, discount=disc)
            r0 = rss_bytes()
            t = time.perf_counter()
            for s in train:
                lm.learn(s)
            dt = time.perf_counter() - t
            rows.append((order, disc, round(lm.perplexity(hold), 2), lm.entries, round((rss_bytes() - r0) / MB, 1), round(dt, 3)))
            del lm
    print("\n[1] n-gram 次数 / ディスカウント")
    print(f"{'order':>5} {'disc':>5} {'ppl':>8} {'entries':>8} {'rssMB':>6} {'sec':>6}")
    for r in rows:
        print(f"{r[0]:>5} {r[1]:>5} {r[2]:>8} {r[3]:>8} {r[4]:>6} {r[5]:>6}")


# ---------------------------------------------------------------- 2. Count-Min Sketch LM
class CountMinLM:
    """次数 1 は正確に、次数 2 以上は Count-Min Sketch (幅 W, 深さ D の配列) で近似する。
    メモリは W*D*4 bytes に固定される。過大評価しか起きないので確率は少し楽観的になる。"""

    def __init__(self, base: NGramLM, width: int, depth: int = 3, max_order: int = 4, discount: float = 0.75):
        self.base = base  # 語彙と unigram/継続カウントを借りる
        self.W, self.D = width, depth
        self.tables = [array.array("I", [0]) * width for _ in range(depth)]
        self.totals = [array.array("I", [0]) * width for _ in range(depth)]   # 文脈ごとの合計
        self.types = [array.array("I", [0]) * width for _ in range(depth)]    # 文脈ごとの種類数 (近似)
        self.max_order = max_order
        self.discount = discount
        self.seeds = [0x9E3779B1 * (i + 1) for i in range(depth)]

    def _idx(self, key: int, d: int) -> int:
        return ((key * 0x9E3779B97F4A7C15) ^ (self.seeds[d] * 0xC2B2AE3D)) % self.W

    def learn(self, tokens):
        ids = [BOS] + self.base.ids(tokens) + [EOS]
        L = len(ids)
        for i in range(1, L):
            tok = ids[i]
            for n in range(1, self.max_order):
                if i - n < 0:
                    break
                ctx_key = n
                for j in range(1, n + 1):
                    ctx_key |= ids[i - j] << (4 + 20 * (j - 1))
                pair = ctx_key * 1_000_003 + tok
                for d in range(self.D):
                    ip = self._idx(pair, d)
                    ic = self._idx(ctx_key, d)
                    if self.tables[d][ip] == 0:
                        self.types[d][ic] += 1
                    self.tables[d][ip] += 1
                    self.totals[d][ic] += 1

    def _count(self, ctx_key: int, tok: int):
        pair = ctx_key * 1_000_003 + tok
        c = min(self.tables[d][self._idx(pair, d)] for d in range(self.D))
        tot = min(self.totals[d][self._idx(ctx_key, d)] for d in range(self.D))
        ty = min(self.types[d][self._idx(ctx_key, d)] for d in range(self.D))
        return c, tot, max(ty, 1)

    def prob(self, hist, tok: int) -> float:
        p = (self.base.cont.get(tok, 0) + 1.0) / (self.base.cont_total + self.base.vocab_size)
        for n in range(1, min(self.max_order, len(hist) + 1)):
            ctx_key = n
            for j in range(1, n + 1):
                ctx_key |= hist[-j] << (4 + 20 * (j - 1))
            c, tot, ty = self._count(ctx_key, tok)
            if tot == 0:
                break
            disc = self.discount
            p = max(c - disc, 0.0) / tot + disc * ty / tot * p
        return p

    def perplexity(self, sents) -> float:
        logp = 0.0
        n = 0
        for toks in sents:
            ids = [BOS] + self.base.ids(toks) + [EOS]
            for i in range(1, len(ids)):
                logp += math.log(max(self.prob(ids[:i], ids[i]), 1e-12))
                n += 1
        return math.exp(-logp / max(n, 1))


def exp_countmin(train, hold):
    exact = NGramLM(max_order=4)
    r0 = rss_bytes()
    for s in train:
        exact.learn(s)
    exact_mb = (rss_bytes() - r0) / MB
    ppl_exact = exact.perplexity(hold)
    print("\n[2] Count-Min Sketch LM (次数 2〜4 を固定メモリで近似)  正確な辞書: ppl=%.2f rss=%.1fMB entries=%d" % (ppl_exact, exact_mb, exact.entries))
    print(f"{'width':>8} {'depth':>5} {'MB':>6} {'ppl':>8} {'vs exact':>9} {'sec':>6}")
    for width in (1 << 14, 1 << 16, 1 << 18):
        for depth in (2, 3):
            cm = CountMinLM(exact, width, depth)
            t = time.perf_counter()
            for s in train:
                cm.learn(s)
            dt = time.perf_counter() - t
            mb = width * depth * 4 * 3 / MB
            ppl = cm.perplexity(hold)
            print(f"{width:>8} {depth:>5} {mb:>6.1f} {ppl:>8.2f} {ppl / ppl_exact:>9.2f} {dt:>6.2f}")
    print("注: 幅が小さいと衝突でカウントが過大になり確率が 1 を超えて「見かけの ppl」が下がる (正しい分布ではない)。"
          "\n    正規化なしの Count-Min LM は不正な分布になるため本体には採用しない (負の結果)。")


# ---------------------------------------------------------------- 3. 接尾辞配列
class SuffixArrayLM:
    """トークン ID 列全体の接尾辞配列。任意長 n-gram の出現回数を二分探索 2 回で数える
    (Infini-gram の考え方の最小実装)。メモリ = 4 bytes × トークン数 (ID 列) + 4 bytes × トークン数 (配列)。"""

    def __init__(self, seqs: list[list[int]]):
        data = array.array("I")
        for s in seqs:
            data.extend(s)
            data.append(EOS)
        self.data = data
        n = len(data)
        # 接尾辞のソート: 純 Python では O(n log n) 比較 × 比較コスト。学習コーパス数万トークンなら現実的
        key = data
        self.sa = array.array("I", sorted(range(n), key=lambda i: key[i : i + 64]))  # 近似: 64 トークンまでで比較

    def count(self, pattern: list[int]) -> int:
        data, sa = self.data, self.sa
        m = len(pattern)
        pat = array.array("I", pattern)

        def at(i):
            return data[i : i + m]

        lo = bisect.bisect_left(sa, 0, key=lambda i: (at(i) >= pat) and 1 or 0) if False else None
        # 手書き二分探索 (bisect の key は 3.10+ だが、比較の向きを自前で制御)
        lo, hi = 0, len(sa)
        while lo < hi:
            mid = (lo + hi) // 2
            if at(sa[mid]) < pat:
                lo = mid + 1
            else:
                hi = mid
        left = lo
        lo, hi = left, len(sa)
        while lo < hi:
            mid = (lo + hi) // 2
            if at(sa[mid]) <= pat and at(sa[mid])[:m] == pat:
                lo = mid + 1
            else:
                hi = mid
        return lo - left

    def prob(self, hist: list[int], tok: int, max_ctx: int = 8) -> float:
        # 最長一致文脈から後退 (バックオフ)。単純な相対頻度 + 1 平滑化
        for n in range(min(max_ctx, len(hist)), -1, -1):
            ctx = hist[len(hist) - n :]
            denom = self.count(ctx) if n else len(self.data)
            if denom >= 2 or n == 0:
                num = self.count(ctx + [tok])
                return (num + 0.1) / (denom + 0.1 * 1000)
        return 1e-6


def exp_suffix(train, hold, base: NGramLM):
    seqs = [base.ids(s) for s in train]
    r0 = rss_bytes()
    t = time.perf_counter()
    sa = SuffixArrayLM(seqs)
    build = time.perf_counter() - t
    mb = (rss_bytes() - r0) / MB
    ntok = len(sa.data)
    t = time.perf_counter()
    q = 0
    for toks in hold[:20]:
        ids = base.ids(toks)
        for i in range(1, len(ids)):
            sa.count(ids[max(0, i - 4) : i])
            q += 1
    qps = q / (time.perf_counter() - t)
    logp = 0.0
    n = 0
    for toks in hold[:40]:
        ids = [BOS] + base.ids(toks) + [EOS]
        for i in range(1, len(ids)):
            logp += math.log(max(sa.prob(ids[:i], ids[i]), 1e-12))
            n += 1
    print("\n[3] 接尾辞配列 (任意長 n-gram カウント)")
    print(f"tokens={ntok} build={build:.2f}s rss+={mb:.1f}MB ({mb * MB / max(ntok, 1):.0f} bytes/token) count-queries={qps:.0f}/s ppl(40文, 素朴な平滑化)={math.exp(-logp / max(n, 1)):.2f}")
    print("→ 構築は純 Python では遅いが、メモリ/トークンは n-gram 辞書より小さく、次数に上限が無い。整理 (consolidate) の裏で再構築する用途に向く。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--limit", type=int, default=4000)
    ap.add_argument("--only", choices=["orders", "countmin", "suffix"], default=None)
    args = ap.parse_args()
    text = Path(args.corpus).read_text(encoding="utf-8", errors="replace")
    sents = [tokenize(s) for s in split_sentences(text)][: args.limit]
    rng = random.Random(0)
    rng.shuffle(sents)
    hold = sents[: max(20, len(sents) // 20)]
    train = sents[len(hold):]
    print(f"train={len(train)} 文 hold={len(hold)} 文")
    if args.only in (None, "orders"):
        exp_orders(train, hold)
    if args.only in (None, "countmin"):
        exp_countmin(train, hold)
    if args.only in (None, "suffix"):
        base = NGramLM(max_order=2)
        for s in train:
            base.learn(s)
        exp_suffix(train, hold, base)


if __name__ == "__main__":
    main()
