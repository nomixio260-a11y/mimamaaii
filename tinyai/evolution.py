"""進化: パラメータの変異と自己評価 ((1+1)-ES + 1/5 成功則の超軽量版)。

適応度 = 検索の自己テスト + 0.5 × ユーザーフィードバックの再現率 − 0.1 × log(パープレキシティ)
変異は 1 パラメータずつ。採用されたら変異幅を 1.5 倍、却下されたら 0.9 倍。
Brain から独立させてあるので、別の適応度や別の最適化法に差し替えやすい。
"""
from __future__ import annotations

import logging
import math
import random
import time
from collections import deque
from dataclasses import dataclass, asdict, replace

from .brain_types import question_type
from .tokenizer import keywords, terms

log = logging.getLogger("tinyai.evolution")


# ---------------------------------------------------------------- 進化するパラメータ
@dataclass
class Params:
    use_order: int = 3            # LM で使う n-gram 次数
    discount: float = 0.75        # 絶対ディスカウント
    k1: float = 1.4               # BM25
    b: float = 0.6
    phrase_bonus: float = 1.5
    expand_weight: float = 0.4    # 関連語によるクエリ拡張の重み
    rerank_weight: float = 0.1    # 質問タイプ別リランクの重み (👍/👎 を通じて進化で調整)
    answer_threshold: float = 0.45
    temperature: float = 0.8
    semantic_weight: float = 0.2  # 意味ベクトルのコサインを確信度に混ぜる重み
    suffix_weight: float = 0.7    # 生成で接尾辞配列 (最長一致) の分布を混ぜる重み
    cache_weight: float = 0.15    # 生成で会話キャッシュ LM を混ぜる重み
    learned_weight: float = 0.3   # 学習型リランカーの確率を確信度に混ぜる重み

    def mutate(self, rng: random.Random, max_order: int, sigma: float = 1.0) -> "Params":
        p = replace(self)
        which = rng.choice(["use_order", "discount", "k1", "b", "phrase_bonus", "expand_weight", "rerank_weight", "discount", "k1",
                            "semantic_weight", "suffix_weight", "cache_weight", "learned_weight"])
        g = lambda sd: rng.gauss(0, sd * sigma)  # noqa: E731
        if which == "use_order":
            p.use_order = max(2, min(max_order, p.use_order + rng.choice([-1, 1])))
        elif which == "discount":
            p.discount = min(0.98, max(0.3, p.discount + g(0.08)))
        elif which == "k1":
            p.k1 = min(3.0, max(0.5, p.k1 + g(0.2)))
        elif which == "b":
            p.b = min(1.0, max(0.0, p.b + g(0.1)))
        elif which == "phrase_bonus":
            p.phrase_bonus = min(4.0, max(1.0, p.phrase_bonus + g(0.3)))
        elif which == "expand_weight":
            p.expand_weight = min(1.0, max(0.0, p.expand_weight + g(0.1)))
        elif which == "rerank_weight":
            p.rerank_weight = min(0.6, max(0.0, p.rerank_weight + g(0.05)))
        elif which == "semantic_weight":
            p.semantic_weight = min(0.6, max(0.0, p.semantic_weight + g(0.08)))
        elif which == "suffix_weight":
            p.suffix_weight = min(0.95, max(0.0, p.suffix_weight + g(0.1)))
        elif which == "cache_weight":
            p.cache_weight = min(0.5, max(0.0, p.cache_weight + g(0.05)))
        elif which == "learned_weight":
            p.learned_weight = min(0.8, max(0.0, p.learned_weight + g(0.1)))
        return p




class Evolution:
    def __init__(self, brain):
        self.brain = brain
        self.generation = 0
        self.sigma = 1.0          # 変異幅 (適応)
        self.fitness = None
        self.log: deque = deque(maxlen=200)

    def retrieval_selftest(self, rng: random.Random, n: int = 40) -> float:
        b = self.brain
        docs = b.kb.random_docs(n, rng)
        if not docs:
            return 0.0
        hit = 0
        for d in docs:
            ts = terms(d.text)
            if len(ts) < 2:
                continue
            q = [t for i, t in enumerate(ts) if i % 2 == 0]
            res = b.kb.search(" ".join(q), k=3)
            if any(x.id == d.id for _, x in res):
                hit += 1
        return hit / len(docs)

    def qa_score(self) -> float:
        b = self.brain
        """ユーザーの 👍/👎 と教えた答えが、今のパラメータで再現できる割合 (-1..1)。"""
        if not b.qa_log:
            return 0.0
        s = 0.0
        n = 0
        for q, doc_id, sign in list(b.qa_log)[-60:]:
            if doc_id not in b.kb.docs:
                continue
            hits = b._search(q, qtype=question_type(q), subject=(keywords(q, limit=1) or [None])[0], k=3)
            top = hits[0][1].id == doc_id if hits else False
            s += sign if top else -sign * 0.5
            n += 1
        return s / n if n else 0.0

    def evaluate(self, seed: int | None = None) -> dict:
        b = self.brain
        rng = random.Random(seed if seed is not None else b.rng.random())
        ppl = b.lm.perplexity(b.holdout) if b.holdout else float("nan")
        hit = self.retrieval_selftest(rng)
        qa = self.qa_score()
        f = hit + 0.5 * qa - (0.1 * math.log(ppl) if ppl == ppl else 0.0)
        return {"perplexity": round(ppl, 2) if ppl == ppl else None, "retrieval_hit": round(hit, 3), "qa": round(qa, 3), "fitness": round(f, 4)}

    def step(self) -> dict:
        b = self.brain
        """パラメータを 1 つ変異させ、自己評価が上がれば採用する。変異幅は適応する。"""
        with b.lock:
            seed = b.rng.randrange(1 << 30)
            base = self.evaluate(seed)
            old = b.params
            cand = old.mutate(b.rng, b.lm.max_order, self.sigma)
            b.params = cand
            b._apply_params()
            new = self.evaluate(seed)
            accepted = new["fitness"] > base["fitness"] + 1e-6
            if accepted:
                self.generation += 1
                self.fitness = new
                self.sigma = min(3.0, self.sigma * 1.5)
                b.stats["evolutions_accepted"] += 1
            else:
                b.params = old
                b._apply_params()
                self.fitness = base
                self.sigma = max(0.2, self.sigma * 0.9)
            b.stats["evolutions_tried"] += 1
            rec = {"time": time.time(), "generation": self.generation, "accepted": accepted, "before": base, "after": new, "params": asdict(b.params), "sigma": round(self.sigma, 3)}
            self.log.append(rec)
            log.info("進化 gen=%d accepted=%s %s -> %s", self.generation, accepted, base, new)
            return rec

