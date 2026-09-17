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
import threading
import time
from collections import deque
from pathlib import Path

from . import neural
from .bpe import BOS, BOT, CTX, EOS, USR, SubwordTokenizer
from .neural_parallel import ParallelTrainer

log = logging.getLogger("tinyai.neural")

# 復号パラメータの既定値の世代。上げると、古いチェックポイントが持っている値のうち
# 研究で見直した項目 (現在は copy_bonus) を捨てて新しい既定値から再開する。
DECODE_VERSION = 1

_QA_TEMPLATES = {
    "definition": ["{s}とは？", "{s}って何？", "{s}について教えて"],
    "is": ["{s}とは？", "{s}は何？"],
    "location": ["{s}はどこ？", "{s}はどこにある？", "{s}の場所は？"],
    "event": ["{s}はいつ？", "{s}はいつ頃？"],
}


class NeuralLM:
    @staticmethod
    def size_for_memory(memory_mb: int) -> str:
        """メモリ上限から最大のプリセットを選ぶ (学習バッファと Adam 状態を含めた概算)。"""
        if memory_mb >= 1024:
            return "xl"
        if memory_mb >= 450:
            return "large"
        if memory_mb >= 200:
            return "base"
        return "small"

    def __init__(self, data_dir: Path, size: str = "base", vocab_size: int = 6000, seed: int = 0, dropout: float = 0.1, pool_capacity: int = 0):
        self.available = neural.available()
        self.dropout = float(dropout)
        # 学習スレッドと会話スレッドが同じモデルを触るためのロック (EMA への切替中に更新が走ると壊れる)
        self.lock = threading.RLock()
        self.data_dir = Path(data_dir)
        self.path = self.data_dir / "neural.npz"
        self.pool_path = self.data_dir / "neural.pool.npz"   # 再生バッファ (再起動しても作り直さない)
        self.size = size if size in neural.PRESETS else "base"
        # 進化する復号パラメータ (👍/👎 の割合で山登り)
        self.decode = {"temperature": 0.7, "top_p": 0.9, "repetition_penalty": 1.3, "copy_bonus": 1.0,
                       "no_repeat_ngram": 3, "min_new": 6}
        self._decode_trial: dict | None = None
        self._fb = [0, 0]           # 現在の設定での (👍, 👎)
        self._fb_best = 0.5         # 採用済み設定の 👍 率
        self.loss_hist: list[float] = []
        self.ppl_hist: list[float] = []      # 取り置き ppl の推移 (成長の判断に使う)
        self.use_ema = True                  # EMA (平均重み) を推論に使うか。評価で悪ければ自動で切る
        self.ema_ppl = None
        self.grown = 0
        self.widened = 0
        self.vocab_added = 0
        self.online_steps = 0
        self.vocab_size = vocab_size
        self.model = None
        self.tok: SubwordTokenizer | None = None
        # 再生バッファの容量は使えるメモリに比例させる。容量が小さいと同じ系列を何十周も学ぶことになり
        # (実測: 3 万系列 = 276 万トークンに対し学習済み 1.13 億トークン = 約 41 周)、学習損失は下がるのに
        # 取り置き ppl が上がる = 過学習になる。系列 1 本あたり約 370 バイト (平均 92 トークン × int32)。
        self.pool = neural.SequencePool(capacity=pool_capacity or 30000, seed=seed) if self.available else None
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
        self._holdout: list[list[int]] = []          # 初期に固定した取り置き (忘却の検出用)
        self._holdout_recent: deque = deque(maxlen=150)  # 最近の文から入れ替わる取り置き (今の分布での汎化)
        self.recent_ppl: float | None = None
        self.recent_hist: list[float] = []           # 入れ替わる取り置き ppl の推移 (成長の判断に使う)
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
                self.model.dropout = self.dropout
                want_ctx = neural.PRESETS.get(meta.get("size", self.size), {}).get("ctx", self.model.T)
                if want_ctx > self.model.T:   # プリセットが伸びていれば、学習済みの重みのまま文脈を伸ばす
                    log.info("文脈長を %d -> %d に拡張", self.model.T, want_ctx)
                    self.model.extend_context(want_ctx)
                self.trained_tokens = int(meta.get("trained_tokens", 0))
                self.holdout_ppl = meta.get("holdout_ppl")
                self.ready = bool(meta.get("ready", False))
                self.size = meta.get("size", self.size)
                self._holdout = [list(x) for x in meta.get("holdout", [])][:300]
                self._holdout_recent = deque([list(x) for x in meta.get("holdout_recent", [])], maxlen=150)
                saved = dict(meta.get("decode", {}))
                # 既定値を変えた項目は、山登りで動かした形跡がない限り新しい既定値を使う
                # (古いチェックポイントの復号設定が新しい研究結果を上書きしてしまうのを防ぐ)
                if int(meta.get("decode_version", 0)) < DECODE_VERSION:
                    for k in ("copy_bonus",):
                        saved.pop(k, None)
                self.decode.update({k: v for k, v in saved.items() if k in self.decode})
                self.grown = int(meta.get("grown", 0))
                self.vocab_added = int(meta.get("vocab_added", 0))
                self.online_steps = int(meta.get("online_steps", 0))
                self.batch = neural.PRESETS.get(self.size, {}).get("batch", self.batch)
                self.lr = neural.PRESETS.get(self.size, {}).get("lr", self.lr)
                try:
                    n_pool = self.pool.load(self.pool_path)
                except Exception as e:
                    n_pool, _ = 0, log.warning("再生バッファの読込失敗 (作り直します): %s", e)
                log.info("ニューラル LM を読込: %s %d params, step=%d, 再生バッファ %d 系列", self.size, self.model.n_params(), self.model.step, n_pool)
                return True
            except Exception as e:
                log.warning("ニューラル LM の読込失敗 (作り直します): %s", e)
        texts = list(texts or [])
        if len(texts) < self.min_sentences or sum(len(t) for t in texts) < self.min_chars:
            return False
        self.tok = SubwordTokenizer.train(texts, size=self.vocab_size)
        self.model = neural.TinyTransformer.from_preset(len(self.tok), self.size)
        self.model.dropout = self.dropout
        log.info("ニューラル LM を初期化: %s vocab=%d params=%d", self.size, len(self.tok), self.model.n_params())
        return True

    # ------------------------------------------------------------ 系列の作り方
    def seq_text(self, text: str) -> list[int]:
        return [BOS] + self.tok.encode(text, max_tokens=self.model.T - 2) + [EOS]

    def _history_ids(self, history, budget: int) -> list[int]:
        """直前の会話 (新しいものを優先) を <usr> 発話 <bot> 応答 … の形で budget トークン以内に詰める。"""
        out: list[int] = []
        for u, b in reversed(list(history or [])):
            if not u or not b:
                continue
            turn = [USR] + self.tok.encode(u, max_tokens=budget // 3) + [BOT] + self.tok.encode(b, max_tokens=budget // 2)
            if len(out) + len(turn) > budget:
                break
            out = turn + out          # 古い順に並べ直す
        return out

    def seq_dialog(self, user: str, bot: str, context: str | None = None, history=None) -> list[int]:
        """会話系列。history があれば直前のやり取りを含める (人間のようにキャッチボールを学ぶ)。
        <bos> [<ctx> 検索文] [<usr> 過去発話 <bot> 過去応答]… <usr> 今の発話 <bot> 応答 <eos>"""
        T = self.model.T
        ctx_ids = self.tok.encode(context, max_tokens=T // 3) if context else []
        u = self.tok.encode(user, max_tokens=T // 4)
        hist = self._history_ids(history, max(0, T // 3)) if history else []
        room = T - len(ctx_ids) - len(hist) - len(u) - 5
        if room < 8 and hist:                       # 入りきらなければ過去を削る
            hist = hist[-max(0, T // 6):]
            room = T - len(ctx_ids) - len(hist) - len(u) - 5
        b = self.tok.encode(bot, max_tokens=max(8, room))
        seq = [BOS]
        if ctx_ids:
            seq += [CTX] + ctx_ids
        return seq + hist + [USR] + u + [BOT] + b + [EOS]

    @staticmethod
    def loss_from(seq: list[int]) -> int:
        """会話系列で本来の重みで学習し始める位置 (最後の <bot> の次 = 今回の応答の先頭)。
        過去のやり取りは文脈なのでプロンプト側の弱い重みで学ぶ。"""
        for i in range(len(seq) - 1, -1, -1):
            if seq[i] == BOT:
                return i + 1
        return 0

    def prompt_dialog(self, user: str, context: str | None = None, history=None) -> list[int]:
        T = self.model.T
        ctx_ids = self.tok.encode(context, max_tokens=T // 3) if context else []
        u = self.tok.encode(user, max_tokens=T // 4)
        hist = self._history_ids(history, max(0, T // 3)) if history else []
        seq = [BOS]
        if ctx_ids:
            seq += [CTX] + ctx_ids
        return seq + hist + [USR] + u + [BOT]

    # ------------------------------------------------------------ データ供給
    def add_text(self, text: str) -> None:
        if self.model is None:
            return
        ids = self.seq_text(text)
        if len(ids) < 4:
            return
        if self.rng.random() < 0.02:
            # 取り置きは学習に使わない。固定の 300 本は「昔の分布を忘れていないか」、
            # 入れ替わる 150 本は「今の分布にどれだけ汎化しているか」を測るためのもの
            if len(self._holdout) < 300:
                self._holdout.append(ids)
            else:
                self._holdout_recent.append(ids)
            return
        self.pool.add(ids, kind="text")

    def add_dialog(self, user: str, bot: str, context: str | None = None, weight: float = 1.0, history=None) -> None:
        if self.model is None:
            return
        ids = self.seq_dialog(user, bot, context, history=history)
        if weight < 0:      # 選好データの「選ばれなかった応答」: unlikelihood で出しにくくする
            self.pool.add(ids, weight, loss_from=self.loss_from(ids), kind="dialog")
            return
        for _ in range(max(1, int(round(weight)))):
            self.pool.add(ids, loss_from=self.loss_from(ids), kind="dialog")

    def add_copy_example(self, keyword: str, sentence: str, neighbors: str | None = None) -> None:
        """RAG の「文脈から抜き出す」練習: 文脈 (その文 + 周辺) を与え、キーワードについて聞かれたらその文を答える。"""
        if self.model is None:
            return
        # 文脈の中での位置も散らす (先頭固定だと「最初の文を写す」だけ覚える)
        context = (f"{sentence} {neighbors}" if self.rng.random() < 0.5 else f"{neighbors} {sentence}") if neighbors else sentence
        q = self.rng.choice([f"{keyword}について教えて", f"{keyword}とは？", f"{keyword}は？", f"{keyword}について", f"{keyword}を説明して", f"{keyword}って何？"])
        ids = self.seq_dialog(q, sentence, context)
        self.pool.add(ids, loss_from=self.loss_from(ids), kind="copy")

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
            for ctx in (context, None):
                ids = self.seq_dialog(q, answer, ctx)
                self.pool.add(ids, loss_from=self.loss_from(ids), kind="qa")
                n += 1
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

    # ------------------------------------------------------------ リアルタイム学習 (1 ターンごと)
    def learn_turn(self, user: str, bot: str, context: str | None = None, weight: float = 1.0, steps: int = 2, history=None) -> dict | None:
        """今の対話を即座に学習する。weight > 0 は正例、< 0 は unlikelihood (その答えを出しにくくする)。
        その系列 + 再生バッファからの少量を混ぜて数ステップ更新 (忘却を防ぐ)。"""
        with self.lock:
            if self.model is None:
                return None
            ids = self.seq_dialog(user, bot, context, history=history)
            lf = self.loss_from(ids)
            self.pool.add(ids, weight, loss_from=lf)
            T = self.model.T
            B = 4 if len(self.pool) >= 8 else 1   # 再生バッファがまだ無ければその系列だけで学ぶ
            x = neural.np.full((B, T), neural.PAD, dtype=neural.np.int64)
            y = neural.np.full((B, T), neural.PAD, dtype=neural.np.int64)
            w = neural.np.ones((B, T), dtype=neural.np.float32)
            seq = ids[: T + 1]
            L = len(seq) - 1
            x[0, :L] = seq[:L]
            y[0, :L] = seq[1 : L + 1]
            tw = neural.SequencePool.token_weights(L, lf, weight)
            w[0, :L] = tw if weight > 0 else -tw
            if B > 1:
                rx, ry, rw = self.pool.batch(B - 1, T)
                x[1:], y[1:], w[1:] = rx, ry, rw
            last = None
            for _ in range(steps):
                loss, g = self.model.loss_and_grads(x, y, w)
                self.model.adamw(g, lr=self.lr * 0.5)
                last = loss
            self.online_steps += steps
            self.trained_tokens += steps * B * T
            return {"loss": last, "weight": weight}

        # ------------------------------------------------------------ 進化 (成長・語彙・復号)
    def maybe_grow(self, memory_ok: bool = True) -> bool:
        """損失が停滞していて容量に余裕があれば、関数を保ったまま層を 1 つ追加する。"""
        with self.lock:
            if self.model is None or not memory_ok:
                return False
            max_layers = neural.MAX_LAYERS.get(self.size, 6)
            max_ff = neural.PRESETS.get(self.size, {}).get("ff", self.model.ff) * 3
            if self.model.L >= max_layers and self.model.ff >= max_ff:
                return False
            h = self.loss_hist
            if len(h) < 20:
                return False
            recent, before = sum(h[-10:]) / 10, sum(h[-20:-10]) / 10
            # 取り置き ppl が悪化し続けている = 過学習なので、容量を増やしても意味がない。
            # 判断には「入れ替わる取り置き」を使う: 固定の取り置きは学習データから外れた古い文を含むので、
            # 忘却による悪化と過学習による悪化を区別できない (忘却なら容量を増やす方が効く)。
            ph = self.recent_hist if len(self.recent_hist) >= 4 else self.ppl_hist
            if len(ph) >= 4 and sum(ph[-2:]) / 2 > sum(ph[-4:-2]) / 2 * 1.15:
                return False
            if before - recent < 0.02 and recent > 1.5:  # 改善が止まり、まだ十分に低くない
                self.stop_parallel()  # パラメータの形が変わるので並列ワーカーは作り直す (次の train_some で再開)
                if self.model.L < max_layers:
                    self.model.grow_layer()
                    log.info("ニューラル LM: 層を追加 -> %d 層 (%d params)", self.model.L, self.model.n_params())
                else:
                    self.model.grow_width()   # 層が上限なら幅を広げる (どちらも関数を保つ)
                    self.widened += 1
                    log.info("ニューラル LM: 中間次元を拡張 -> ff=%d (%d params)", self.model.ff, self.model.n_params())
                self.grown += 1
                self.loss_hist = []
                return True
            return False

    def evolve_vocab(self, texts, top: int = 100) -> int:
        """新しいテキストに頻出する未知の単位を語彙に足す (モデルの埋め込みも拡張)。"""
        with self.lock:
            if self.model is None or self.tok is None:
                return 0
            units = self.tok.frequent_new_units(texts, top=top)
            n = self.tok.add_tokens(units)
            if n:
                self.stop_parallel()  # 埋め込みの形が変わるので並列ワーカーは作り直す
                self.model.add_tokens(n)
                self.vocab_added += n
            return n

    def feedback(self, positive: bool) -> None:
        """👍/👎 で復号パラメータを山登り: 試行中の設定が採用済みより良ければ採用、悪ければ戻す。"""
        self._fb[0 if positive else 1] += 1
        n = sum(self._fb)
        if n < 6:
            return
        rate = self._fb[0] / n
        if self._decode_trial is not None:
            if rate >= self._fb_best:
                self._fb_best = rate
            else:
                self.decode = self._decode_trial  # 戻す
            self._decode_trial = None
        else:
            self._fb_best = rate
        # 次の試行: 1 つのパラメータを少し動かす
        cand = dict(self.decode)
        key = self.rng.choice([k for k in ("temperature", "top_p", "repetition_penalty", "copy_bonus") if k in cand])
        step = {"temperature": 0.1, "top_p": 0.05, "repetition_penalty": 0.1, "copy_bonus": 0.5}[key]
        hi = {"temperature": 1.2, "top_p": 0.99, "repetition_penalty": 2.0, "copy_bonus": 5.0}[key]
        lo = {"temperature": 0.3, "top_p": 0.5, "repetition_penalty": 1.0, "copy_bonus": 0.0}[key]
        cand[key] = round(min(hi, max(lo, cand[key] + self.rng.choice([-step, step]))), 3)
        self._decode_trial = dict(self.decode)
        self.decode = cand
        self._fb = [0, 0]

    def train_some(self, steps: int = 4, batch: int | None = None) -> dict | None:
        with self.lock:
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
                    self.loss_hist.append(r["loss"])
                    if len(self.loss_hist) > 200:
                        del self.loss_hist[:100]
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
            self.loss_hist.append(r["loss"])
            if len(self.loss_hist) > 200:
                del self.loss_hist[:100]
            return r

    def evaluate(self, ngram_ppl: float | None = None) -> dict:
        with self.lock:
            if self.model is None or not self._holdout:
                return {}
            # EMA と生の重みを比べ、良い方を推論に使う (学習が速いと EMA が遅れて悪くなることがある)
            raw_ppl = round(neural.perplexity(self.model, self._holdout), 2)
            ema_ppl = None
            if self.model.ema is not None:
                with self.model.use_ema():
                    ema_ppl = round(neural.perplexity(self.model, self._holdout), 2)
            if ema_ppl is not None and ema_ppl > raw_ppl * 1.05:
                if self.use_ema:
                    log.info("EMA 重みが劣るため生の重みに切替 (EMA %.1f > 生 %.1f)。EMA を作り直します", ema_ppl, raw_ppl)
                self.use_ema = False
                self.model.ema = {k: v.copy() for k, v in self.model.p.items()}   # 作り直す
            elif ema_ppl is not None:
                self.use_ema = True
            self.ema_ppl = ema_ppl
            self.holdout_ppl = min(raw_ppl, ema_ppl) if ema_ppl is not None else raw_ppl
            self.ppl_hist.append(self.holdout_ppl)
            if len(self.ppl_hist) > 100:
                del self.ppl_hist[:50]
            if len(self._holdout_recent) >= 30:
                with self._infer():
                    self.recent_ppl = round(neural.perplexity(self.model, list(self._holdout_recent)), 2)
                self.recent_hist.append(self.recent_ppl)
                if len(self.recent_hist) > 100:
                    del self.recent_hist[:50]
            self.ngram_ppl = ngram_ppl
            if self.holdout_ppl is not None:
                if ngram_ppl is not None:
                    self.ready = self.holdout_ppl <= ngram_ppl * self.ready_ratio
                else:
                    self.ready = self.holdout_ppl <= self.ready_abs  # n-gram の取り置きが無い時の絶対基準
            return {"neural_ppl": self.holdout_ppl, "recent_ppl": self.recent_ppl, "ngram_ppl": ngram_ppl, "ready": self.ready}

    def save(self) -> None:
        with self.lock:
            if self.model is None or self.tok is None:
                return
            try:
                self.pool.save(self.pool_path)      # 再生バッファごと持ち越す (再開しても混ざり具合を失わない)
            except Exception as e:
                log.warning("再生バッファの保存に失敗: %s", e)
            self.model.save(self.path, self.tok, meta={"trained_tokens": self.trained_tokens, "holdout_ppl": self.holdout_ppl, "ready": self.ready, "size": self.size, "holdout": self._holdout[:300], "holdout_recent": [list(x) for x in self._holdout_recent],
                                                       "decode": self.decode, "decode_version": DECODE_VERSION, "grown": self.grown, "vocab_added": self.vocab_added, "online_steps": self.online_steps})
            self._last_save = time.time()

    def _infer(self):
        """推論に使う重みの文脈 (EMA が良ければ EMA、悪ければ生の重み)。"""
        import contextlib

        if self.use_ema and self.model is not None and self.model.ema is not None:
            return self.model.use_ema()
        return contextlib.nullcontext()

    # ------------------------------------------------------------ データの価値 (驚き)
    def surprise(self, texts, max_texts: int = 6) -> float | None:
        """文の集合の平均トークン損失 (nat)。モデルにとって新しい情報ほど大きい。
        収集ソースの評価に使う: 低すぎる = 既知 (学ぶ価値が低い)、極端に高い = ジャンクや別言語。"""
        with self.lock:
            if self.model is None or self.tok is None:
                return None
            texts = [t for t in texts if len(t) >= 8][:max_texts]
            if not texts:
                return None
            drop, self.model.dropout = self.model.dropout, 0.0
            try:
                lp = [self.model.logprob(self.seq_text(t)) for t in texts]
            finally:
                self.model.dropout = drop
            return round(-sum(lp) / len(lp), 3)

        # ------------------------------------------------------------ 利用
    def score(self, text: str) -> float | None:
        if self.model is None:
            return None
        with self._infer():
            return self.model.logprob(self.seq_text(text))

    def chat(self, user: str, context: str | None = None, n: int = 2, max_new: int = 48, temperature: float = 0.7, history=None, copy_bonus: float | None = None) -> list[str]:
        """RAG 形式で応答候補を n 本生成。history があれば直前のやり取りを踏まえて答える。"""
        with self.lock:
            if self.model is None:
                return []
            prompt = self.prompt_dialog(user, context, history=history)
            out = []
            dec = self.decode
            # 文脈に出てくるトークンを少し出やすくする (検索した文を実際に使わせる)
            copy_ids = self.tok.encode(context, max_tokens=self.model.T) if context else None
            bonus = dec.get("copy_bonus", 0.0) if copy_bonus is None else copy_bonus
            with self._infer():
                gens = self.model.generate_batch(prompt, n=n, max_new=max_new, temperature=dec["temperature"], top_p=dec["top_p"], repetition_penalty=dec["repetition_penalty"], rng=self.nprng,
                                                 copy_ids=copy_ids, copy_bonus=bonus,
                                                 no_repeat_ngram=int(dec.get("no_repeat_ngram", 0)),
                                                 min_new=min(int(dec.get("min_new", 0)), max(max_new // 4, 1)))
            for ids in gens:
                text = self.tok.decode(ids).strip()
                if len(text) >= 2:
                    out.append(text)
            return out

    def continue_text(self, seed_text: str, max_new: int = 40, temperature: float = 0.8) -> str:
        with self.lock:
            if self.model is None:
                return ""
            prompt = [BOS] + self.tok.encode(seed_text, max_tokens=self.model.T // 2)
            with self._infer():
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
            "holdout": len(self._holdout), "holdout_recent": len(self._holdout_recent), "recent_ppl": self.recent_ppl,
            "last_loss": round(self.last_loss, 3) if self.last_loss is not None else None,
            "holdout_ppl": self.holdout_ppl,
            "ngram_ppl": self.ngram_ppl,
            "ready": self.ready,
            "layers": self.model.L if self.model else 0,
            "grown_layers": self.grown,
            "widened": self.widened,
            "ff": self.model.ff if self.model else 0,
            "vocab_added": self.vocab_added,
            "online_steps": self.online_steps,
            "decode": dict(self.decode), "decode_version": DECODE_VERSION,
            "dropout": self.model.dropout if self.model else self.dropout,
            "ema": bool(self.use_ema and self.model is not None and self.model.ema is not None),
            "ema_ppl": self.ema_ppl,
            "priority_mean": round(float(self.pool.priority[: len(self.pool)].mean()), 3) if len(self.pool) else None,
            "pool_kinds": dict(self.pool.kind_counts),
        }
