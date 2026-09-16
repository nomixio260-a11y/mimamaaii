"""接尾辞配列による「最長一致の続き」(Infini-gram, Liu et al. 2024 の考え方)。

知識ベースの全文をトークン ID 列にして接尾辞配列を作り、生成時には
「今の出力の末尾が、コーパスのどこにどれだけ長く一致するか」を二分探索で求め、
その続きに現れたトークンの分布を返す。次数に上限が無いので、長い一致があれば
コーパスの文をそのまま滑らかに続けられる (n-gram の 2〜3 語より一貫した文になる)。

構築は O(n log n) 比較で純 Python だと 100 万トークンで数秒。整理 (consolidate) の裏で作り直す。
メモリ = 4 bytes × トークン数 × 2 (列 + 配列)。
"""
from __future__ import annotations

import array
from collections import Counter
from typing import Sequence

SEP = 2  # EOS を区切りに使う


class SuffixIndex:
    def __init__(self, data: array.array | None = None):
        self.data = data if data is not None else array.array("I")
        self.sa = array.array("I")
        self.built_docs = 0

    @classmethod
    def build(cls, seqs: Sequence[Sequence[int]], max_tokens: int = 1_500_000) -> "SuffixIndex":
        data = array.array("I")
        for s in seqs:
            if len(data) + len(s) + 1 > max_tokens:
                break
            data.extend(s)
            data.append(SEP)
        idx = cls(data)
        n = len(data)
        # 比較は最大 48 トークンまでの接尾辞スライスで (それ以上は一致長として扱わない)
        idx.sa = array.array("I", sorted(range(n), key=lambda i: data[i : i + 48]))
        idx.built_docs = len(seqs)
        return idx

    def __len__(self) -> int:
        return len(self.data)

    def _range(self, pat: array.array) -> tuple[int, int]:
        """pat で始まる接尾辞の [lo, hi) 範囲。"""
        data, sa = self.data, self.sa
        m = len(pat)
        lo, hi = 0, len(sa)
        while lo < hi:
            mid = (lo + hi) // 2
            if data[sa[mid] : sa[mid] + m] < pat:
                lo = mid + 1
            else:
                hi = mid
        left = lo
        hi = len(sa)
        while lo < hi:
            mid = (lo + hi) // 2
            s = data[sa[mid] : sa[mid] + m]
            if s == pat:
                lo = mid + 1
            elif s < pat:
                lo = mid + 1
            else:
                hi = mid
        return left, lo

    def continuations(self, hist: Sequence[int], min_ctx: int = 2, max_ctx: int = 12, max_occ: int = 200) -> tuple[int, Counter]:
        """hist の末尾に最も長く一致する文脈 (長さ ≥ min_ctx) の続きのトークン分布。
        戻り値 (一致長, Counter)。一致が無ければ (0, 空)。"""
        if not len(self.sa) or not hist:
            return 0, Counter()
        data = self.data
        n = len(data)
        best = (0, 0, 0)
        for L in range(min(max_ctx, len(hist)), min_ctx - 1, -1):
            pat = array.array("I", hist[-L:])
            lo, hi = self._range(pat)
            if hi > lo:
                best = (L, lo, hi)
                break
        L, lo, hi = best
        if L == 0:
            return 0, Counter()
        cnt: Counter = Counter()
        sa = self.sa
        step = max(1, (hi - lo) // max_occ)
        for k in range(lo, hi, step):
            p = sa[k] + L
            if p < n:
                cnt[data[p]] += 1
        return L, cnt

    def stats(self) -> dict:
        return {"tokens": len(self.data), "docs": self.built_docs, "est_mb": round(8 * len(self.data) / 1048576, 1)}
