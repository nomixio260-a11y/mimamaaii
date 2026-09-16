"""Brain と Transformer をつなぐラッパー: 語彙の構築、学習データの供給、少しずつの学習 (継続学習)、
チェックポイント、生成・採点。numpy が無ければ `available` が False になり、何もしない。

学習の方針 (人が寝ている間に復習するように):
  * 応答の合間や自律ループの空き時間に数ステップずつ学習する (1 ステップ ≈ 0.1〜0.2 秒)
  * 知識文 (最近学んだもの優先) と会話ペア (<usr> 発話 <bot> 応答 <eos>) を混ぜる。👍 のペアは 3 倍
  * 取り置き文で ppl を測り、n-gram より良くなったら生成に使う (それまでは採点だけ)
"""
from __future__ import annotations

import logging
import math
import random
import time
from pathlib import Path

from . import neural
from .tokenizer import tokenize

log = logging.getLogger("tinyai.neural")


class NeuralLM:
    def __init__(self, data_dir: Path, vocab_size: int = 4096, d: int = 128, heads: int = 4, layers: int = 2, ctx: int = 64, seed: int = 0):
        self.available = neural.available()
        self.data_dir = Path(data_dir)
        self.path = self.data_dir / "neural.npz"
        self.cfg = dict(vocab_size=vocab_size, d=d, heads=heads, layers=layers, ctx=ctx)
        self.min_vocab = 1500      # 語彙を固定してよい最小の異なり語数
        self.min_tokens = 20000    # 同、延べトークン数
        self.model = None
        self.vocab = None
        self.pool = neural.SequencePool(seed=seed) if self.available else None
        self.rng = random.Random(seed)
        self.nprng = neural.np.random.default_rng(seed) if self.available else None
        self.trained_tokens = 0
        self.last_loss = None
        self.holdout_ppl = None
        self.ngram_ppl = None
        self.ready = False        # 生成に使ってよいか (ppl が n-gram に近づいたら)
        self.pending: list[list[int]] = []
        self._last_save = 0.0
        self._holdout: list[list[int]] = []

    # ------------------------------------------------------------ 構築
    def ensure_model(self, unigram_counts: dict[str, int]) -> bool:
        """語彙とモデルを用意する。既存のチェックポイントがあれば読む。"""
        if not self.available:
            return False
        if self.model is not None:
            return True
        if self.path.exists():
            try:
                self.model, self.vocab, meta = neural.TinyTransformer.load(self.path)
                self.trained_tokens = int(meta.get("trained_tokens", 0))
                self.holdout_ppl = meta.get("holdout_ppl")
                self.ready = bool(meta.get("ready", False))
                log.info("ニューラル LM を読込: %d params, step=%d", self.model.n_params(), self.model.step)
                return True
            except Exception as e:  # 壊れたチェックポイントは作り直す
                log.warning("ニューラル LM の読込失敗 (作り直します): %s", e)
        # 語彙は初期化時に固定されるので、十分な量の文を読んでから作る (小さすぎる語彙は後で困る)
        if len(unigram_counts) < self.min_vocab or sum(unigram_counts.values()) < self.min_tokens:
            return False
        self.vocab = neural.NeuralVocab.from_counts(unigram_counts, size=self.cfg["vocab_size"])
        self.model = neural.TinyTransformer(len(self.vocab), d=self.cfg["d"], heads=self.cfg["heads"], layers=self.cfg["layers"], ctx=self.cfg["ctx"])
        log.info("ニューラル LM を初期化: vocab=%d params=%d", len(self.vocab), self.model.n_params())
        return True

    # ------------------------------------------------------------ データ
    def encode_text(self, text: str) -> list[int]:
        return [neural.BOS] + self.vocab.encode(tokenize(text)) + [neural.EOS]

    def encode_dialog(self, user: str, bot: str) -> list[int]:
        return [neural.BOS, neural.USR] + self.vocab.encode(tokenize(user))[:40] + [neural.BOT] + self.vocab.encode(tokenize(bot))[:80] + [neural.EOS]

    def add_text(self, text: str) -> None:
        if self.vocab is None:
            return
        ids = self.encode_text(text)
        if len(ids) >= 4:
            if self.rng.random() < 0.02 and len(self._holdout) < 300:
                self._holdout.append(ids)
            else:
                self.pool.add(ids)

    def add_dialog(self, user: str, bot: str, weight: float = 1.0) -> None:
        if self.vocab is None:
            return
        ids = self.encode_dialog(user, bot)
        for _ in range(max(1, int(round(weight)))):
            self.pool.add(ids)

    # ------------------------------------------------------------ 学習
    def train_some(self, steps: int = 4, batch: int = 32) -> dict | None:
        if self.model is None or len(self.pool) < 32:
            return None
        r = neural.train_steps(self.model, self.pool, steps=steps, batch=batch)
        self.trained_tokens += steps * batch * self.model.T
        self.last_loss = r.get("loss")
        return r

    def evaluate(self, ngram_ppl: float | None = None) -> dict:
        if self.model is None or not self._holdout:
            return {}
        self.holdout_ppl = round(neural.perplexity(self.model, self._holdout), 2)
        self.ngram_ppl = ngram_ppl
        # 生成に使う基準: n-gram の 1.5 倍以内に入ったら (それ以前は採点のみ)
        if ngram_ppl is not None and self.holdout_ppl is not None:
            self.ready = self.holdout_ppl <= ngram_ppl * 1.5
        return {"neural_ppl": self.holdout_ppl, "ngram_ppl": ngram_ppl, "ready": self.ready}

    def save(self) -> None:
        if self.model is None or self.vocab is None:
            return
        self.model.save(self.path, self.vocab, meta={"trained_tokens": self.trained_tokens, "holdout_ppl": self.holdout_ppl, "ready": self.ready})
        self._last_save = time.time()

    # ------------------------------------------------------------ 利用
    def score(self, text: str) -> float | None:
        if self.model is None:
            return None
        return self.model.logprob(self.encode_text(text))

    def generate_reply(self, user: str, max_new: int = 40, temperature: float = 0.8, n: int = 3) -> list[str]:
        """会話形式で応答候補を n 本生成 (語列を文字列に戻す)。"""
        if self.model is None:
            return []
        prompt = [neural.BOS, neural.USR] + self.vocab.encode(tokenize(user))[:40] + [neural.BOT]
        out = []
        for _ in range(n):
            ids = self.model.generate(prompt, max_new=max_new, temperature=temperature, rng=self.nprng)
            words = self.vocab.decode(ids)
            if words:
                out.append(words)
        return out

    def continue_text(self, seed_tokens: list[str], max_new: int = 30, temperature: float = 0.8) -> list[str]:
        if self.model is None:
            return []
        prompt = [neural.BOS] + self.vocab.encode(seed_tokens)
        ids = self.model.generate(prompt, max_new=max_new, temperature=temperature, rng=self.nprng)
        return self.vocab.decode(ids)

    def stats(self) -> dict:
        if not self.available:
            return {"available": False}
        return {
            "available": True,
            "params": self.model.n_params() if self.model else 0,
            "vocab": len(self.vocab) if self.vocab else 0,
            "steps": self.model.step if self.model else 0,
            "trained_tokens": self.trained_tokens,
            "pool": len(self.pool),
            "holdout": len(self._holdout),
            "last_loss": round(self.last_loss, 3) if self.last_loss is not None else None,
            "holdout_ppl": self.holdout_ppl,
            "ngram_ppl": self.ngram_ppl,
            "ready": self.ready,
        }
