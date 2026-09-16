"""Brain と Transformer をつなぐ層 (numpy が無ければ available=False で何もしない)。

* トークナイザ: 知識文からサブワード語彙を学習 (十分な量が溜まってから固定)
* 学習データ (再生バッファ):
    平文      <bos> 文 <eos>
    会話      <bos><usr> 発話 <bot> 応答 <eos>
    RAG 会話  <bos><ctx> 検索で得た文 <usr> 発話 <bot> 応答 <eos>   ← 応答時と同じ形式
    合成 QA   事実ストア (主語, 関係, 目的語) から質問と答えを作る = 記号処理系からの蒸留
* 学習は空き時間に少しずつ (継続学習)。ウォームアップ + コサイン減衰
* 応答: 検索した文を <ctx> に入れて生成 (RAG)。取り置き文の ppl が n-gram の閾値以内で「使用可」
"""
from __future__ import annotations

import logging
import random
import time
from pathlib import Path

from . import neural
from .bpe import BOS, BOT, CTX, EOS, USR, SubwordTokenizer
from .neural_parallel import ParallelTrainer

log = logging.getLogger("tinyai.neural")

_QA_TEMPLATES = {
    "definition": ["{s}とは？", "{s}って何？", "{s}について教えて"],
    "is": ["{s}とは？", "{s}は何？"],
    "location": ["{s}はどこ？", "{s}はどこにある？", "{s}の場所は？"],
    "event": ["{s}はいつ？", "{s}はいつ頃？"],
}


class NeuralLM:
    def __init__(self, data_dir: Path, size: str = "base", vocab_size: int = 6000, seed: int = 0):
        self.available = neural.available()
        self.data_dir = Path(data_dir)
        self.path = self.data_dir / "neural.npz"
        self.size = size if size in neural.PRESETS else "base"
        self.vocab_size = vocab_size
        self.model = None
        self.tok: SubwordTokenizer | None = None
        self.pool = neural.SequencePool(seed=seed) if self.available else None
        self.rng = random.Random(seed)
        self.nprng = neural.np.random.default_rng(seed) if self.available else None
        self.trained_tokens = 0
        self.last_loss = None
        self.holdout_ppl = None
        self.ngram_ppl = None
        self.ready = False
        self.min_sentences = 2000     # 語彙を固定してよい最小の文数
        self.min_chars = 150_000      # 同、文字数
        self.ready_ratio = 2.0        # 取り置き ppl が n-gram の何倍以内なら生成に使うか (サブワードと文字で単位が違うため緩め)
        self.ready_abs = 40.0         # n-gram の取り置きが無い時: サブワード ppl がこれ以下なら生成に使う
        self.total_steps = 30000      # コサイン減衰の想定総ステップ
        self._holdout: list[list[int]] = []
        self._last_save = 0.0
        self.batch = neural.PRESETS[self.size]["batch"] if self.available else 8
        self.lr = neural.PRESETS[self.size]["lr"] if self.available else 5e-4
        self.workers = 1                       # >1 ならデータ並列 (train コマンドで使う)
        self._parallel: ParallelTrainer | None = None

    # ------------------------------------------------------------ 構築
    def ensure_model(self, texts=None) -> bool:
        """トークナイザとモデルを用意する。チェックポイントがあれば読む。texts は語彙学習用の文の列。"""
        if not self.available:
            return False
        if self.model is not None:
            return True
        if self.path.exists():
            try:
                self.model, self.tok, meta = neural.TinyTransformer.load(self.path)
                self.trained_tokens = int(meta.get("trained_tokens", 0))
                self.holdout_ppl = meta.get("holdout_ppl")
                self.ready = bool(meta.get("ready", False))
                self.size = meta.get("size", self.size)
                self._holdout = [list(x) for x in meta.get("holdout", [])][:300]
                self.batch = neural.PRESETS.get(self.size, {}).get("batch", self.batch)
                self.lr = neural.PRESETS.get(self.size, {}).get("lr", self.lr)
                log.info("ニューラル LM を読込: %s %d params, step=%d", self.size, self.model.n_params(), self.model.step)
                return True
            except Exception as e:
                log.warning("ニューラル LM の読込失敗 (作り直します): %s", e)
        texts = list(texts or [])
        if len(texts) < self.min_sentences or sum(len(t) for t in texts) < self.min_chars:
            return False
        self.tok = SubwordTokenizer.train(texts, size=self.vocab_size)
        self.model = neural.TinyTransformer.from_preset(len(self.tok), self.size)
        log.info("ニューラル LM を初期化: %s vocab=%d params=%d", self.size, len(self.tok), self.model.n_params())
        return True

    # ------------------------------------------------------------ 系列の作り方
    def seq_text(self, text: str) -> list[int]:
        return [BOS] + self.tok.encode(text, max_tokens=self.model.T - 2) + [EOS]

    def seq_dialog(self, user: str, bot: str, context: str | None = None) -> list[int]:
        T = self.model.T
        ctx_ids = self.tok.encode(context, max_tokens=T // 2) if context else []
        u = self.tok.encode(user, max_tokens=T // 4)
        room = T - len(ctx_ids) - len(u) - 5
        b = self.tok.encode(bot, max_tokens=max(8, room))
        seq = [BOS]
        if ctx_ids:
            seq += [CTX] + ctx_ids
        return seq + [USR] + u + [BOT] + b + [EOS]

    def prompt_dialog(self, user: str, context: str | None = None) -> list[int]:
        T = self.model.T
        ctx_ids = self.tok.encode(context, max_tokens=T // 2) if context else []
        u = self.tok.encode(user, max_tokens=T // 4)
        seq = [BOS]
        if ctx_ids:
            seq += [CTX] + ctx_ids
        return seq + [USR] + u + [BOT]

    # ------------------------------------------------------------ データ供給
    def add_text(self, text: str) -> None:
        if self.model is None:
            return
        ids = self.seq_text(text)
        if len(ids) < 4:
            return
        if self.rng.random() < 0.02 and len(self._holdout) < 300:
            self._holdout.append(ids)
        else:
            self.pool.add(ids)

    def add_dialog(self, user: str, bot: str, context: str | None = None, weight: float = 1.0) -> None:
        if self.model is None:
            return
        ids = self.seq_dialog(user, bot, context)
        for _ in range(max(1, int(round(weight)))):
            self.pool.add(ids)

    def add_copy_example(self, keyword: str, sentence: str, neighbors: str | None = None) -> None:
        """RAG の「文脈から抜き出す」練習: 文脈 (その文 + 周辺) を与え、キーワードについて聞かれたらその文を答える。"""
        if self.model is None:
            return
        context = f"{sentence} {neighbors}" if neighbors else sentence
        q = self.rng.choice([f"{keyword}について教えて", f"{keyword}とは？", f"{keyword}は？", f"{keyword}について"])
        self.pool.add(self.seq_dialog(q, sentence, context))

    def add_synthetic_qa(self, subject: str, relation: str, obj: str, answer: str, context: str) -> int:
        """事実から質問文を作り (テンプレート)、文脈付き/無しの両方で会話例にする。"""
        if self.model is None:
            return 0
        if relation in _QA_TEMPLATES:
            qs = [t.format(s=subject) for t in _QA_TEMPLATES[relation]]
        else:
            qs = [f"{subject}の{relation}は？", f"{subject}の{relation}を教えて"]
        n = 0
        for q in qs[:2]:
            self.pool.add(self.seq_dialog(q, answer, context))
            self.pool.add(self.seq_dialog(q, answer, None))
            n += 2
        return n

    # ------------------------------------------------------------ 学習
    def set_workers(self, n: int) -> int:
        """データ並列のワーカー数を設定 (1 で単一プロセス)。モデルが無い間は予約だけ。"""
        self.workers = max(1, int(n))
        if self._parallel is not None and self._parallel.workers != self.workers:
            self._parallel.stop()
            self._parallel = None
        return self.workers

    def stop_parallel(self) -> None:
        if self._parallel is not None:
            self._parallel.stop()
            self._parallel = None

    def train_some(self, steps: int = 4, batch: int | None = None) -> dict | None:
        if self.model is None or len(self.pool) < 32:
            return None
        batch = batch or self.batch
        if self.workers > 1:
            if self._parallel is None:
                self._parallel = ParallelTrainer(self.model, self.pool, workers=self.workers)
                if not self._parallel.start():
                    self._parallel = None
                    self.workers = 1
            if self._parallel is not None:
                try:
                    r = self._parallel.train(steps=steps, batch=batch, lr=self.lr, total=self.total_steps)
                except MemoryError:
                    self.batch = max(2, batch // 2)
                    return None
                self.trained_tokens += steps * batch * self._parallel.workers * self.model.T
                self.last_loss = r.get("loss")
                return r
        try:
            r = neural.train_steps(self.model, self.pool, steps=steps, batch=batch, lr=self.lr, total=self.total_steps)
        except MemoryError:
            # メモリ上限に当たったらバッチを半分にして続ける
            self.batch = max(2, batch // 2)
            log.warning("ニューラル LM: メモリ不足のためバッチを %d に縮小", self.batch)
            return None
        self.trained_tokens += steps * batch * self.model.T
        self.last_loss = r.get("loss")
        return r

    def evaluate(self, ngram_ppl: float | None = None) -> dict:
        if self.model is None or not self._holdout:
            return {}
        self.holdout_ppl = round(neural.perplexity(self.model, self._holdout), 2)
        self.ngram_ppl = ngram_ppl
        if self.holdout_ppl is not None:
            if ngram_ppl is not None:
                self.ready = self.holdout_ppl <= ngram_ppl * self.ready_ratio
            else:
                self.ready = self.holdout_ppl <= self.ready_abs  # n-gram の取り置きが無い時の絶対基準
        return {"neural_ppl": self.holdout_ppl, "ngram_ppl": ngram_ppl, "ready": self.ready}

    def save(self) -> None:
        if self.model is None or self.tok is None:
            return
        self.model.save(self.path, self.tok, meta={"trained_tokens": self.trained_tokens, "holdout_ppl": self.holdout_ppl, "ready": self.ready, "size": self.size, "holdout": self._holdout[:300]})
        self._last_save = time.time()

    # ------------------------------------------------------------ 利用
    def score(self, text: str) -> float | None:
        if self.model is None:
            return None
        return self.model.logprob(self.seq_text(text))

    def chat(self, user: str, context: str | None = None, n: int = 2, max_new: int = 48, temperature: float = 0.7) -> list[str]:
        """RAG 形式で応答候補を n 本生成。"""
        if self.model is None:
            return []
        prompt = self.prompt_dialog(user, context)
        out = []
        for ids in self.model.generate_batch(prompt, n=n, max_new=max_new, temperature=temperature, rng=self.nprng):
            text = self.tok.decode(ids).strip()
            if len(text) >= 2:
                out.append(text)
        return out

    def continue_text(self, seed_text: str, max_new: int = 40, temperature: float = 0.8) -> str:
        if self.model is None:
            return ""
        prompt = [BOS] + self.tok.encode(seed_text, max_tokens=self.model.T // 2)
        ids = self.model.generate(prompt, max_new=max_new, temperature=temperature, rng=self.nprng)
        return self.tok.decode(ids)

    def stats(self) -> dict:
        if not self.available:
            return {"available": False}
        return {
            "available": True,
            "size": self.size,
            "params": self.model.n_params() if self.model else 0,
            "vocab": len(self.tok) if self.tok else 0,
            "steps": self.model.step if self.model else 0,
            "trained_tokens": self.trained_tokens,
            "pool": len(self.pool),
            "pool_tokens": self.pool.total_tokens,
            "holdout": len(self._holdout),
            "last_loss": round(self.last_loss, 3) if self.last_loss is not None else None,
            "holdout_ppl": self.holdout_ppl,
            "ngram_ppl": self.ngram_ppl,
            "ready": self.ready,
        }
