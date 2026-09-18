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
import math
import random
import re
import threading
import time
from collections import deque
from pathlib import Path

from . import neural
from .bpe import BOS, BOT, CTX, EOS, USR, SubwordTokenizer
from .neural_parallel import ParallelTrainer
from .textquality import good_prose

log = logging.getLogger("tinyai.neural")

# 復号パラメータの既定値の世代。上げると、古いチェックポイントが持っている値のうち
# 研究で見直した項目 (現在は copy_bonus) を捨てて新しい既定値から再開する。
DECODE_VERSION = 3
# 会話系列の作り方の版。上げると、次回の起動時に手持ちの会話を作り直して再生バッファに入れ直す
# (古い系列は切り詰めた応答に <eos> が付いており、「短く終わる」癖を教え続けてしまうため)
SEQ_VERSION = 3

# 本文を指す言い回し (読解データ由来。文脈なしで学ぶと雑談にも出てくる)
_PASSAGE_RE = re.compile(r"文章(に|では|から|によ)|文中|この記事(に|では)|上記の|与えられた文|本文(に|では)|記載されてい")

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

    def __init__(self, data_dir: Path, size: str = "base", vocab_size: int = 6000, seed: int = 0, dropout: float = 0.1, pool_capacity: int = 0, corpus_tokens: int = 0):
        self.available = neural.available()
        self.dropout = float(dropout)
        # 学習スレッドと会話スレッドが同じモデルを触るためのロック (EMA への切替中に更新が走ると壊れる)
        self.lock = threading.RLock()
        self.data_dir = Path(data_dir)
        self.path = self.data_dir / "neural.npz"
        self.pool_path = self.data_dir / "neural.pool.npz"   # 再生バッファ (再起動しても作り直さない)
        self.size = size if size in neural.PRESETS else "base"
        # 進化する復号パラメータ (👍/👎 の割合で山登り)
        # 自由な文生成のための復号既定値。温度と top_p を上げて言い回しの幅を広げる。
        # 繰り返しペナルティは一度 1.15 まで緩めたが、新しい温度で測り直すと 1.3 の方が
        # 長さ・多様性・繰り返しのすべてで優れていた (3 つの乱数種で確認):
        #   罰 1.15 → 平均長 85.9 / distinct-2 0.542 / 繰り返し 0.037
        #   罰 1.30 → 平均長 75.8 / distinct-2 0.577 / 繰り返し 0.020
        # 温度を変えたら、他の復号パラメータも測り直す必要がある。
        self.decode = {"temperature": 0.85, "top_p": 0.95, "repetition_penalty": 1.3, "copy_bonus": 1.0,
                       "no_repeat_ngram": 3, "min_new": 6, "temp_spread": 0.25}
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
        # 読んだ文はディスクのコーパスにも貯める (RAM の再生バッファに入りきらない分の置き場)。
        # 1 トークン 4 バイトなので、数千万トークンでも RAM を使わずに回せる。
        self.corpus = neural.TokenCorpus(self.data_dir / "corpus.bin", max_tokens=corpus_tokens or 16_000_000) if self.available else None
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
        self.recent_hist: list[float] = []           # 入れ替わる取り置き ppl の推移 (表示用)
        self.recent_bpc: float | None = None         # 同じ取り置きの 1 文字あたりビット数
        self.recent_bpc_hist: list[float] = []       # 成長の判断はこちらを使う (語彙を増やしても比べられる)
        self._last_grow_step = 0                     # 直近で成長したステップ (連続した成長を避ける)
        self._pregrow_ppl: float | None = None       # 成長直前の ppl (成長が裏目に出ていないかの判定用)
        self._pregrow_dialog: float | None = None    # 成長直前の対話 ppl
        self.seq_version = 0                         # 読み込んだチェックポイントの会話系列の版
        self.dialog_hist: list[float] = []           # 対話 ppl の推移 (成長の判断に使う)
        self._grow_block: str | None = None          # 直近に成長を見送った理由 (ログに 1 度だけ出す)
        self.growth_records: list[dict] = []         # 成長の前後で品質がどう動いたか (成長が効いたかの記録)
        self.lr_scale = 1.0                          # 学習率の調整倍率 (不安定な時に下げる)
        self._last_damp_step = 0
        self._pregrow_path = self.data_dir / "neural.pregrow.npz"
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
                    for k in ("copy_bonus", "temperature", "top_p", "repetition_penalty"):
                        saved.pop(k, None)
                self.decode.update({k: v for k, v in saved.items() if k in self.decode})
                self.grown = int(meta.get("grown", 0))
                self._last_grow_step = int(meta.get("last_grow_step", 0))
                self.lr_scale = float(meta.get("lr_scale", 1.0))
                # 品質の履歴も引き継ぐ。10 分ごとに再開する運用では、履歴が消えると
                # 「続けて悪化したら学習率を下げる」ような規則が一度も発火しない
                self.dialog_hist = self._single_scale([float(x) for x in meta.get("dialog_hist", [])])
                self.recent_hist = [float(x) for x in meta.get("recent_hist", [])]
                self.recent_bpc_hist = [float(x) for x in meta.get("recent_bpc_hist", [])]
                self.growth_records = [dict(x) for x in meta.get("growth_records", [])]
                self.seq_version = int(meta.get("seq_version", 0))
                self._pregrow_ppl = meta.get("pregrow_ppl")
                self._pregrow_dialog = meta.get("pregrow_dialog")
                self._last_damp_step = int(meta.get("last_damp_step", 0))
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
        # 応答の場所を先に確保する。文脈と履歴で埋めてしまうと応答が数十トークンに切られ、
        # 「短く答えて止める」ことを学んでしまう (実測: 生成される応答が常に 44 文字前後だった)
        floor = T // 2
        room = T - len(ctx_ids) - len(hist) - len(u) - 5
        if room < floor:
            hist = hist[-max(0, T // 6):] if hist else hist
            room = T - len(ctx_ids) - len(hist) - len(u) - 5
        if room < floor and ctx_ids:
            ctx_ids = ctx_ids[: max(0, T // 5)]
            room = T - len(ctx_ids) - len(hist) - len(u) - 5
        full = self.tok.encode(bot, max_tokens=max(8, room) + 1)
        truncated = len(full) > max(8, room)
        b = full[: max(8, room)]
        seq = [BOS]
        if ctx_ids:
            seq += [CTX] + ctx_ids
        seq = seq + hist + [USR] + u + [BOT] + b
        # 途中で切った応答に <eos> を付けると「ここで終わってよい」と教えることになる。切れた時は付けない
        return seq + ([EOS] if not truncated else [])

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
        if self.corpus is not None:
            self.corpus.append(ids, kind="text")

    def add_dialog(self, user: str, bot: str, context: str | None = None, weight: float = 1.0, history=None) -> None:
        if self.model is None:
            return
        # 「文章には記載されていません」のように本文を指す応答は、文脈を一緒に学ばないと
        # 「文脈が無いのに本文の話をする」という対応づけを覚えてしまう (実測: 雑談でこの言い回しが出る)。
        # 文脈つきの読解データとして学ぶ分には問題ないので、文脈が無い時だけ落とす。
        if not context and _PASSAGE_RE.search(bot):
            return
        ids = self.seq_dialog(user, bot, context, history=history)
        if weight < 0:      # 選好データの「選ばれなかった応答」: unlikelihood で出しにくくする
            self.pool.add(ids, weight, loss_from=self.loss_from(ids), kind="dialog")
            return
        # 履歴つき (多ターン) の会話は、人が実際にやり取りしている数少ないデータなので厚めに学ぶ。
        # 指示データの 1 往復に埋もれると、前の発話を踏まえて答える練習がほとんどできない。
        reps = max(1, int(round(weight)))
        if history:
            reps = max(reps, 2)
        for _ in range(reps):
            self.pool.add(ids, loss_from=self.loss_from(ids), kind="dialog")
        if self.corpus is not None:
            self.corpus.append(ids, loss_from=self.loss_from(ids), kind="dialog")

    def add_corpus_text(self, text: str, max_chars: int = 400_000) -> int:
        """ページ本文をそのままディスクのコーパスへ (知識ベースには入れない)。

        知識ベースは重複判定・品質判定・事実抽出・意味ベクトルまで行うので重く、メモリ上限もある。
        一方、言語モデルの学習に必要なのはトークン列だけで、1 トークン 4 バイトで済む。
        読んだページの本文を丸ごとコーパスに流し込めば、知識ベースを太らせずに学習量だけを増やせる。
        戻り値は追加したトークン数。"""
        with self.lock:
            if self.model is None or self.corpus is None or not text:
                return 0
            budget = self.model.T - 2
            added = 0
            for para in re.split(r"\n{2,}", text[:max_chars]):
                para = para.strip()
                if len(para) < 40 or not good_prose(para):
                    continue
                ids = self.tok.encode(para)
                for i in range(0, len(ids), budget):    # 文脈長で切って詰める (段落の流れは保つ)
                    chunk = ids[i : i + budget]
                    if len(chunk) < 16:
                        break
                    seq = [BOS] + list(chunk) + [EOS]
                    if self.corpus.append(seq, kind="text"):
                        added += len(seq)
            return added

    def refresh_from_corpus(self, k: int) -> int:
        """ディスクのコーパスから k 本引いて再生バッファへ入れ直す。
        バッファに入りきらない分を少しずつ循環させることで、同じ系列を何十周もするのを防ぐ。"""
        with self.lock:
            if self.model is None or self.corpus is None or k <= 0 or not len(self.corpus):
                return 0
            n = 0
            for ids, lf, kind in self.corpus.sample(k, self.nprng, prefer_long=True):
                if len(ids) >= 4:
                    self.pool.add(ids, loss_from=int(lf), kind=str(kind))
                    n += 1
            return n

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
    GROW_COOLDOWN = 2000         # 成長してから次の成長までに最低限回すステップ数
                                 # (800 では 1 時間で層 3 段 + 中間次元 2 段まで進み、会話の質が崩れた)
    TOKENS_PER_PARAM = 20        # Chinchilla 則の目安 (一から学習する場合の計算最適)
    TOKENS_PER_PARAM_SOFT = 5    # 継続学習でデータが増え続ける場合の、容量不足を疑い始める線

    GROW_WARMUP = 400            # 成長直後に学習率を戻していくステップ数

    LR_FLOOR_RATIO = 0.5         # 学習が不安定な時に下げる倍率

    def _depth_lr(self) -> float:
        """大きさに応じた学習率。プリセットの学習率は「その層数・その幅」で調整した値なので、
        成長して大きくなったらそのままでは大きすぎる (実測: 4 層想定の 6e-4 のまま 9 層まで増やしたら、
        対話 ppl 75 → 118、接地率 0.95 → 0.83 と崩れた)。深さと幅それぞれの平方根に反比例させる
        (層が増えるほど残差の重なりが深くなり、幅が広いほど 1 つの出力に足し込む項が増えるため、
        同じ更新幅でも出力の変化が大きくなる)。"""
        preset = neural.PRESETS.get(self.size, {})
        lr = self.lr * self.lr_scale
        if self.model is None:
            return lr
        base_layers = preset.get("layers", self.model.L)
        base_ff = preset.get("ff", self.model.ff)
        if self.model.L > base_layers:
            lr *= (base_layers / self.model.L) ** 0.5
        if self.model.ff > base_ff:
            lr *= (base_ff / self.model.ff) ** 0.5
        return lr

    def maybe_damp_lr(self) -> bool:
        """会話の質が続けて落ちていたら学習率を下げる。

        成長や分布の変化で学習が不安定になった時、放っておくと質が落ち続ける。
        対話 ppl (固定の取り置きで測る) の直近 2 回が、その前 4 回の中央値より 25% 以上悪ければ、
        学習率を半分にして落ち着かせる。下げるのは 1,500 ステップに 1 回まで。"""
        with self.lock:
            if self.model is None or len(self.dialog_hist) < 6:
                return False
            if self.model.step - self._last_damp_step < 1500:
                return False
            recent = sum(self.dialog_hist[-2:]) / 2
            prev = sorted(self.dialog_hist[-6:-2])
            base = (prev[1] + prev[2]) / 2
            if recent <= base * 1.25:
                return False
            self._last_damp_step = self.model.step
            self.damp_lr("対話 ppl %.1f -> %.1f" % (base, recent))
            return True

    def damp_lr(self, reason: str = "") -> float:
        """学習が不安定な時に学習率をさらに半分にする (下限 1/8 まで)。
        自動で下げたことを記録し、書き出しにも残す (なぜ遅くなったかを後から追えるように)。"""
        with self.lock:
            if self.lr_scale <= 0.125:
                return self.lr_scale
            self.lr_scale *= self.LR_FLOOR_RATIO
            log.warning("学習率を下げました (x%.3f) %s", self.lr_scale, reason)
            return self.lr_scale

    def _effective_lr(self) -> float:
        """成長直後は学習率を下げてから戻す。

        層を足した直後のモデルは、追加した層が恒等写像の状態から学び始める。そこへ通常の学習率を
        かけると既に学んだ重みの方が崩れ、取り置き ppl と対話 ppl が悪化する
        (実測: 12 分で 3 層追加した後、対話 ppl 75 → 98、損失 1.93 → 2.04)。
        0.3 倍から始めて 400 ステップかけて戻す。"""
        lr = self._depth_lr()
        if not self.grown or self.model is None:
            return lr
        since = self.model.step - self._last_grow_step
        if since >= self.GROW_WARMUP or since < 0:
            return lr
        return lr * (0.3 + 0.7 * since / self.GROW_WARMUP)

    @staticmethod
    def _single_scale(hist: list[float]) -> list[float]:
        """尺度の違う値が混ざった履歴は捨てる。

        以前は 1 回の評価で 2 つの尺度 (bpc×100 ≒ 410 と削減率から作った値 ≒ 49) を入れていた。
        古いチェックポイントを読むと、直したあとも混ざった履歴が残り、成長の取り消しが誤爆する。"""
        if len(hist) >= 2 and max(hist) > min(hist) * 4:
            return []
        return hist

    def note_dialog_ppl(self, value: float | None) -> None:
        """自己評価で測った対話 ppl を記録する (成長の判断に使う)。"""
        if value is None:
            return
        with self.lock:
            self.dialog_hist.append(float(value))
            if len(self.dialog_hist) > 60:
                del self.dialog_hist[:30]

    def _no_grow(self, reason: str) -> bool:
        """成長しなかった理由を残す (同じ理由は 1 度だけ記録する)。

        「なぜ成長しないのか」を後から追えないと、圧力の判定ミスのような詰まりに何時間も気付けない
        (実測: メモリの圧力で 7 MB の拡張が拒否され続けていたのに、ログには何も出ていなかった)。"""
        if reason != self._grow_block:
            log.info("成長は見送り: %s", reason)
            self._grow_block = reason
        return False

    def growth_bytes(self) -> int:
        """次の成長で増えるメモリの見積り (重み + Adam の 1 次/2 次 + EMA)。"""
        if self.model is None:
            return 0
        d, ff = self.model.d, self.model.ff
        per_layer = d * 3 * d + d * d + d * ff * 2 + ff * d + 2 * d     # wqkv, wo, w1+wg, w2, rms×2
        return int(per_layer * 4 * 4)                                   # float32 × (重み, m, v, EMA)

    def maybe_grow(self, memory_ok: bool = True, data_tokens: int = 0) -> bool:
        """容量を増やす判断。次のどちらかで、関数を保ったまま層 (または中間次元) を増やす。
          1. 損失が停滞した = 今の容量で学べることは学び切った
          2. 手持ちのトークン数がモデルの容量に対して多すぎる (20 トークン/パラメータ超)
        2 は Chinchilla 則の考え方で、データが増えたなら先回りして大きくする、という判断。
        どちらの場合も「今の分布での取り置き ppl が悪化していない」ことを条件にする。"""
        with self.lock:
            if self.model is None:
                return self._no_grow("モデルがまだ無い")
            if not memory_ok:
                return self._no_grow("メモリに余裕がない")
            # 成長の直後は、増えた容量を使えるようになるまで時間がかかる。間を置かずに続けて増やすと
            # 「増やす → 一時的に悪化 → 悪化を見てまた増やす」の悪循環になる (実測: 12 分で 3 層増えた)。
            if self.grown and self.model.step - self._last_grow_step < self.GROW_COOLDOWN:
                return self._no_grow(f"成長の間隔 (あと {self.GROW_COOLDOWN - (self.model.step - self._last_grow_step)} step)")
            max_layers = neural.MAX_LAYERS.get(self.size, 6)
            max_ff = neural.PRESETS.get(self.size, {}).get("ff", self.model.ff) * 3
            if self.model.L >= max_layers and self.model.ff >= max_ff:
                return self._no_grow(f"上限に到達 (層 {self.model.L}, ff {self.model.ff})")
            h = self.loss_hist
            if len(h) < 20:
                return self._no_grow(f"損失の履歴が足りない ({len(h)}/20)")
            recent, before = sum(h[-10:]) / 10, sum(h[-20:-10]) / 10
            # 取り置き ppl が悪化し続けている = 過学習なので、容量を増やしても意味がない。
            # 判断には「入れ替わる取り置き」を使う: 固定の取り置きは学習データから外れた古い文を含むので、
            # 忘却による悪化と過学習による悪化を区別できない (忘却なら容量を増やす方が効く)。
            ph = self.recent_bpc_hist if len(self.recent_bpc_hist) >= 4 else (self.recent_hist if len(self.recent_hist) >= 4 else self.ppl_hist)
            worse = sum(ph[-2:]) / 2 / max(sum(ph[-4:-2]) / 2, 1e-9) if len(ph) >= 4 else 1.0
            if worse > 1.25:        # 急激に悪化している = 学習が不安定。容量の問題ではない
                return self._no_grow(f"取り置き ppl が急に悪化 (x{worse:.2f})")
            # 平文の指標だけで判断すると、会話の質が落ちているのに成長を続けてしまう
            # (実測: 層 6 → 9 + 中間次元の拡張で、平文の ppl は回復したのに対話 ppl は 75 → 141)。
            dh = self.dialog_hist
            if len(dh) >= 4 and sum(dh[-2:]) / 2 > sum(dh[-4:-2]) / 2 * 1.10:
                return self._no_grow("会話の質が落ちている")
            plateau = before - recent < 0.02 and recent > 1.5     # 改善が止まり、まだ十分に低くない
            n_params = self.model.n_params()
            # データ過多 (強): 計算最適の目安を超えた
            data_rich = bool(data_tokens) and data_tokens > self.TOKENS_PER_PARAM * n_params
            # データ過多 (弱): データは増え続けているのに、今の分布での ppl が良くなっていない。
            # 継続学習では「データが増えても良くならない」ことが容量不足のいちばん素直な証拠になる
            # (計算最適の目安 20 は、一から学習する場合の話であって、飽和の判定基準ではない)。
            if not data_rich and data_tokens > self.TOKENS_PER_PARAM_SOFT * n_params and len(ph) >= 4:
                data_rich = worse > 1.0
            if plateau or data_rich:
                self.stop_parallel()  # パラメータの形が変わるので並列ワーカーは作り直す (次の train_some で再開)
                try:
                    # 成長は元に戻せない操作なので、直前の重みを取っておく。
                    # 成長が裏目に出た時 (実測: 3 層追加で対話 ppl 75 → 118) に戻せるようにする。
                    self.model.save(self._pregrow_path, self.tok, meta={"step": self.model.step, "ppl": self.recent_ppl, "bpc": self.recent_bpc})
                except Exception as e:
                    log.warning("成長前の重みの保存に失敗: %s", e)
                if self.model.L < max_layers:
                    kind = "layer"
                    self.model.grow_layer()
                    log.info("ニューラル LM: 層を追加 -> %d 層 (%d params)", self.model.L, self.model.n_params())
                else:
                    kind = "width"
                    self.model.grow_width()   # 層が上限なら幅を広げる (どちらも関数を保つ)
                    self.widened += 1
                    log.info("ニューラル LM: 中間次元を拡張 -> ff=%d (%d params)", self.model.ff, self.model.n_params())
                self.grown += 1
                self._last_grow_step = self.model.step
                # 「増やしたら良くなったのか」を後から言えるように、前の値を残す
                self.growth_records.append({
                    "step": self.model.step, "kind": kind,
                    "layers": self.model.L, "ff": self.model.ff, "params": self.model.n_params(),
                    "bpc_before": self.recent_bpc, "dialog_before": self.dialog_hist[-1] if self.dialog_hist else None,
                })
                del self.growth_records[:-20]
                self._pregrow_ppl = self.recent_bpc or self.recent_ppl or self.holdout_ppl
                self._pregrow_dialog = self.dialog_hist[-1] if self.dialog_hist else None
                if data_rich and not plateau:
                    log.info("データ量が容量を超えたので成長 (%.1fM トークン / %.1fM パラメータ)",
                             data_tokens / 1e6, self.model.n_params() / 1e6)
                self.loss_hist = []
                self._grow_block = None
                return True
            return self._no_grow(f"条件を満たさない (改善 {before - recent:.3f}, {data_tokens / 1e6:.0f}M トークン)")

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

    @property
    def decode_stable(self) -> dict:
        """採用済みの復号設定。試行中は self.decode が「試している設定」なので、配布や表示には
        こちらを使う (実測: 6 票で動く山登りの途中の値がそのままブラウザ版に配られていた)。"""
        return dict(self._decode_trial or self.decode)

    def feedback(self, positive: bool) -> None:
        """👍/👎 で復号パラメータを山登り: 試行中の設定が採用済みより良ければ採用、悪ければ戻す。

        判定に使う票数が少ないと、山登りではなく酔歩になる。12 票 (だいたい 6 往復ぶん) 貯めてから判定する。"""
        self._fb[0 if positive else 1] += 1
        n = sum(self._fb)
        if n < 12:
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
                        r = self._parallel.train(steps=steps, batch=batch, lr=self._effective_lr(), total=self.total_steps)
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
                r = neural.train_steps(self.model, self.pool, steps=steps, batch=batch, lr=self._effective_lr(), total=self.total_steps)
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

    GROW_ROLLBACK_RATIO = 1.4    # 成長前より ppl がこの倍率を超えて悪ければ、成長を取り消す

    def _record_growth_outcome(self, now: float | None) -> None:
        """成長から GROW_COOLDOWN ステップ後の品質を記録して残す。

        取り消すほど悪くない成長でも「効いたのか」は別の話で、記録が無いと次の判断ができない。"""
        if not self.growth_records or self.growth_records[-1].get("bpc_after") is not None:
            return
        rec = self.growth_records[-1]
        rec["bpc_after"] = now
        rec["dialog_after"] = self.dialog_hist[-1] if self.dialog_hist else None
        b0, b1 = rec.get("bpc_before"), rec.get("bpc_after")
        d0, d1 = rec.get("dialog_before"), rec.get("dialog_after")
        log.info("成長の結果 (%d step 後): 平文 %s -> %s、会話 %s -> %s",
                 self.GROW_COOLDOWN,
                 f"{b0:.3f}" if b0 else "-", f"{b1:.3f}" if b1 else "-",
                 f"{d0:.1f}" if d0 else "-", f"{d1:.1f}" if d1 else "-")

    def check_growth(self) -> bool:
        """成長の結果を確かめ、明らかに裏目なら成長前の重みに戻す。

        関数を保つ成長でも、その後の学習で崩れることがある。「増やしたら必ず良くなる」とは限らないので、
        増やした後に確かめて、駄目なら戻せるようにしておく。戻す判断は成長前の ppl との比較で行う。"""
        with self.lock:
            if self.model is None or not self.grown or self._pregrow_ppl is None:
                return False
            since = self.model.step - self._last_grow_step
            if since < self.GROW_COOLDOWN:            # まだ馴染ませている途中
                return False
            now = self.recent_bpc or self.recent_ppl or self.holdout_ppl   # 成長前と同じ尺度で比べる
            bad_text = now is not None and now > self._pregrow_ppl * self.GROW_ROLLBACK_RATIO
            bad_dialog = (self._pregrow_dialog is not None and self.dialog_hist
                          and self.dialog_hist[-1] > self._pregrow_dialog * self.GROW_ROLLBACK_RATIO)
            self._record_growth_outcome(now)
            if not bad_text and not bad_dialog:
                self._pregrow_ppl = self._pregrow_dialog = None   # 問題なし。以後は判定しない
                return False
            if not self._pregrow_path.exists():
                self._pregrow_ppl = None
                return False
            try:
                self.stop_parallel()
                model, tok, meta = neural.TinyTransformer.load(self._pregrow_path)
                model.dropout = self.dropout
                self.model, self.tok = model, tok
                log.warning("成長が裏目に出たため取り消し (ppl %.1f -> %.1f、%d 層へ戻す)",
                            self._pregrow_ppl, now, self.model.L)
                self.grown = max(0, self.grown - 1)
                self._last_grow_step = self.model.step
                self._pregrow_ppl = self._pregrow_dialog = None
                self.recent_hist, self.ppl_hist, self.loss_hist, self.recent_bpc_hist = [], [], [], []
                return True
            except Exception as e:
                log.warning("成長の取り消しに失敗: %s", e)
                self._pregrow_ppl = self._pregrow_dialog = None
                return False

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
                seqs = list(self._holdout_recent)
                with self._infer():
                    nats, ntok = neural.holdout_nats(self.model, seqs)
                self.recent_ppl = round(math.exp(nats / max(ntok, 1)), 2)
                self.recent_hist.append(self.recent_ppl)
                if len(self.recent_hist) > 100:
                    del self.recent_hist[:50]
                # 語彙を増やすと per-token の ppl は機械的に上がる。成長の判断は文字あたりで見る
                # (実測: 語彙 +300 で ppl が 1.52 倍になり、「悪化した」と誤判定して成長が止まっていた)。
                chars = sum(len(self.tok.decode([int(t) for t in sq[1:]])) for sq in seqs) if self.tok else 0
                if chars:
                    self.recent_bpc = round(nats / chars / math.log(2), 4)
                    self.recent_bpc_hist.append(self.recent_bpc)
                    if len(self.recent_bpc_hist) > 100:
                        del self.recent_bpc_hist[:50]
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
                if self.corpus is not None:
                    self.corpus.save()
                self.pool.save(self.pool_path)      # 再生バッファごと持ち越す (再開しても混ざり具合を失わない)
            except Exception as e:
                log.warning("再生バッファの保存に失敗: %s", e)
            self.model.save(self.path, self.tok, meta={"trained_tokens": self.trained_tokens, "holdout_ppl": self.holdout_ppl, "ready": self.ready, "size": self.size, "holdout": self._holdout[:300], "holdout_recent": [list(x) for x in self._holdout_recent],
                                                       "decode": self.decode, "decode_version": DECODE_VERSION, "grown": self.grown, "last_grow_step": self._last_grow_step, "lr_scale": self.lr_scale,
                                                       "dialog_hist": self.dialog_hist[-20:], "recent_hist": self.recent_hist[-20:], "recent_bpc_hist": self.recent_bpc_hist[-20:], "growth_records": self.growth_records[-20:], "seq_version": SEQ_VERSION,
                                                       # 成長直前の品質も保存する。これが消えると「成長が裏目なら戻す」判定が
                                                       # 再起動のたびに取り消され、安全網が一度も働かない
                                                       "pregrow_ppl": self._pregrow_ppl, "pregrow_dialog": self._pregrow_dialog,
                                                       "last_damp_step": self._last_damp_step, "vocab_added": self.vocab_added, "online_steps": self.online_steps})
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
                                                 temp_spread=float(dec.get("temp_spread", 0.25)),
                                                 no_repeat_ngram=int(dec.get("no_repeat_ngram", 0)),
                                                 min_new=min(int(dec.get("min_new", 0)), max(max_new // 4, 1)))
            for ids in gens:
                text = self.tok.decode(ids).strip()
                if len(text) >= 2 and text not in out:      # 同じ文が並んでも選ぶ意味が無い
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
            "corpus_seqs": len(self.corpus) if self.corpus else 0,
            "corpus_tokens": self.corpus.tokens if self.corpus else 0,
            "holdout": len(self._holdout), "holdout_recent": len(self._holdout_recent), "recent_ppl": self.recent_ppl,
            "lr_scale": self.lr_scale, "lr": round(self._effective_lr(), 7) if self.model else None,
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
