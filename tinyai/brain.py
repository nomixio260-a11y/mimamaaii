"""Brain: 言語モデル + 知識ベース + 進化パラメータ + メモリ制御をまとめた中核。

会話の流れ:
  発話 -> コマンド判定 -> 語に分解 -> (文脈が無ければ前の話題を補う)
       -> 知識検索 (+ 確信が低ければ関連語でクエリ拡張)
       -> 質問タイプ別リランク (定義/いつ/どこ/いくつ/誰)
       -> recall / guess / generate (知識に接地した生成)
       -> 分からなかった話題を調査キューへ

学習コストの削減:
  * 知識ベースに入らなかった文 (重複・ジャンク) は言語モデルにも入れない
  * 探索戦略 (調査キュー / 好奇心 / 薄い知識 / 種) の「収穫量」を記録し、
    収穫の多い戦略を優先する (バンディット)
  * 進化は変異幅を適応させる (成功したら広げ、失敗したら狭める)
"""
from __future__ import annotations

import gzip
import json
import logging
import math
import os
import pickle
import random
import re
import threading
import time
import re as _re
from collections import Counter, deque

_CONTENT_RE = _re.compile(r"[一-鿿㐀-䶿]+|[゠-ヿ]{2,}")  # 漢字の連続、カタカナ 2 文字以上
from dataclasses import dataclass, asdict, replace
from pathlib import Path
from typing import Iterable

from .config import Config
from .brain_types import question_type, _rerank_bonus  # noqa: F401
from .evolution import Evolution, Params
from .facts import FactStore, attr_synonyms, extract_facts, parse_question
from .knowledge import KnowledgeBase, Doc
from .agent import Agent, apply_format, strip_format
from .dialog import DialogStore
from .lm import NGramLM, CacheLM, EOS
from .memory import MemoryGuard, MB
from .neural_lm import SEQ_VERSION as neural_lm_SEQ_VERSION
from .neural_lm import NeuralLM
from .reranker import Reranker
from .semantic import SemanticSpace
from .suffix import SuffixIndex
from .tokenizer import (
    analyze,
    detokenize,
    is_phrase,
    is_question,
    keywords,
    normalize,
    phrases,
    split_sentences,
    term_weight,
    terms,
    tokenize,
)

log = logging.getLogger("tinyai.brain")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SAVE_NAME = "brain.pkl.gz"
MAX_TIGHTEN = 6  # 予算を締める回数の上限 (0.8^6 ≈ 26%)
STRATEGIES = ("gap", "curiosity", "sparse", "seed", "interest")
_NOT_TOPICS = {"tinyai", "http", "https", "www", "com", "html", "wiki", "wikipedia", "inbox", "sources", "feeds"}
_QUALITY_DEF_RE = re.compile(r"とは|である|です|のこと|を指す|is a|is an|is the|refers to|was a|は、")
_QUALITY_NUM_RE = re.compile(r"\d")
_QUALITY_PHRASE_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\u30a0-\u30ff]{2,}|[A-Za-z][A-Za-z0-9]{2,}")


@dataclass
class Reply:
    text: str
    confidence: float
    mode: str                 # recall / guess / generate / command
    sources: list
    doc_ids: list
    learned_topics: list


def sentence_quality(text: str, has_fact: bool = False) -> float:
    """文の学習価値 (0..1)。定義文・数値・話題語を含み、長さが適度なものを高く。"""
    n = len(text)
    q = 0.3
    if 20 <= n <= 160:
        q += 0.2
    elif n < 12 or n > 300:
        q -= 0.15
    if _QUALITY_DEF_RE.search(text):
        q += 0.15
    if _QUALITY_NUM_RE.search(text):
        q += 0.1
    q += 0.08 * min(len(_QUALITY_PHRASE_RE.findall(text)), 3)
    if has_fact:
        q += 0.15
    return max(0.0, min(1.0, q))


def _same_utterance(a: str, b: str) -> bool:
    strip = lambda x: "".join(ch for ch in x.lower() if ch.isalnum())  # noqa: E731
    return strip(a) == strip(b)


def _cjk_ratio(text: str) -> float:
    if not text:
        return 0.0
    cjk = sum(1 for ch in text if "぀" <= ch <= "ヿ" or "一" <= ch <= "鿿")
    return cjk / len(text)


_GARBLED_RE = re.compile(r"[?？!！][^。！？!?]{2,}|」[^「]*」|\)[^(]*\)|<unk>|[、,]{2}|[。.]{2}")
_MORE_RE = re.compile(r"^(もっと|詳しく|もっと詳しく|続けて|続き|他には|ほかには|それで|それから|more|tell me more|continue|go on|and\??)[。!！?？]*$", re.I)
_FOLLOWUP_RE = re.compile(r"^(それ|これ|あれ|そこ|そいつ|彼|彼女|it|that|this|they|he|she)")
# 雑談の合図: 挨拶・お礼・気持ち・体調。知識文をそのまま返すと会話にならない発話
_CHAT_RE = re.compile(
    r"こんにちは|こんばんは|おはよう|やあ|ありがとう|ありがと|よろしく|おやすみ|またね|さようなら"
    r"|元気|疲れ|つかれ|眠い|ねむい|しんどい|つらい|辛い|悲し|嬉し|うれし|楽し|たのし|寂し|さびし|不安|心配"
    r"|けんか|喧嘩|失恋|落ち込|むかつ|腹が立|hello|hi\b|thanks|thank you|good morning|good night")


def neural_perplexity(nl) -> float:
    """取り置き文のパープレキシティ (EMA 重みで)。"""
    from . import neural as _neural

    with nl.lock, nl._infer():
        return _neural.perplexity(nl.model, nl._holdout)


class Brain:
    def __init__(self, config: Config | None = None, rng: random.Random | None = None):
        self.cfg = config or Config()
        self.rng = rng or random.Random(self.cfg.seed)
        self.guard = MemoryGuard(self.cfg.memory_mb, hard=self.cfg.hard_limit)
        self.lock = threading.RLock()
        self.lm = NGramLM(max_order=self.cfg.max_order)
        self.kb = KnowledgeBase()
        self.facts = FactStore()
        self.kb.on_remove = self.facts.remove_doc
        self.facts.source_of = self._source_of
        self.semantic = SemanticSpace()           # 意味ベクトル (後回しで学習)
        self._semantic_queue: deque = deque(maxlen=50000)  # 意味ベクトル未学習の文書 ID
        self.suffix = SuffixIndex()               # 最長一致生成用の接尾辞配列 (整理時に再構築)
        self._suffix_dirty = 0                    # 前回構築以降に増えた文書数
        self.cache_lm = CacheLM()                 # 会話キャッシュ LM
        self.reranker = Reranker()                # 👍/👎 から学ぶリランカー
        self.dialogs = DialogStore(self.cfg.max_dialogs)   # 会話データ (自分の会話 + 公開データ)
        self.agent = Agent(self)                  # 道具 (計算・日付・換算・比較・列挙・要約・調査・プロファイル)
        size = self.cfg.neural_size if self.cfg.neural_size != "auto" else NeuralLM.size_for_memory(self.cfg.memory_mb)
        # 再生バッファはメモリ 1 MB あたり 100 系列 (1 GB で約 10 万系列 ≒ 37 MB)
        self.neural = NeuralLM(self.cfg.data_dir, size=size, seed=self.cfg.seed or 0,
                               dropout=self.cfg.neural_dropout,
                               pool_capacity=max(30000, int(self.cfg.memory_mb) * 100),
                               # コーパスはディスクなので、メモリではなく空き容量で決める。
                               # 1 トークン 4 バイト = メモリ 1 MB あたり 20 万トークンで 800 KB のディスク。
                               # 上限に達すると古い素材から上書きされるので、余裕を持たせる。
                               corpus_tokens=max(8_000_000, int(self.cfg.memory_mb) * 200_000))  # Transformer LM (numpy)
        self.last_self_eval: dict | None = None
        self.dialog_holdout: list = []             # 評価用に固定した会話 (比較できるように)
        self._last_growth_check = 0                # 最後に成長・語彙を点検したステップ
        self._rag_docs: list[int] = []             # RAG 忠実性を測る文 (一定期間は同じ文で測る)
        self._rag_docs_step = 0
        self._fresh_shift = 0.0                    # 取り置きを入れ替えた時の段差の累積 (値を連続させる)
        self._fresh_holdout: list = []             # 最近の会話から採った取り置き (一定期間は固定して比べる)
        self._fresh_holdout_step = 0
        self._followup = False                     # 直前の発話が指示語・情報量の乏しい問いか
        self.last_thought: dict | None = None      # 直近のニューラル応答の思考過程 (下書き → 再検索 → 検証)
        self._neural_pending_text: deque = deque(maxlen=5000)
        self._neural_pending_dialog: deque = deque(maxlen=2000)   # (発話, 応答, 文脈, 重み)
        self._qa_done: set[int] = set()                           # 合成 QA を作った文書 ID
        self._last_features: dict[int, dict] = {}  # 直前の候補 doc_id -> 特徴量 (学習用)
        self.timers: Counter = Counter()          # 段階ごとの累積秒 (コスト計測)
        self._docvec_cache: dict[int, tuple[int, list[float] | None]] = {}  # doc_id -> (世代, ベクトル)
        self._last_cover: dict[int, float] = {}
        self._guard_note: tuple[str, str, str] | None = None  # (種類, 主語, 属性) 直前の質問で「知らない」と判定した内容
        self._last_pair: tuple[str, str] | None = None
        self._last_pair_ctx: str | None = None
        self.params = Params(use_order=min(3, self.cfg.max_order))
        self._apply_params()
        self.evolution = Evolution(self)          # 世代・適応度・変異幅はここが持つ
        self.holdout: list[list[str]] = []
        self._holdout_counter = 0
        self.gaps: deque = deque(maxlen=200)      # 調べたい話題
        self.explored: dict[str, float] = {}      # 話題 -> 最終探索時刻
        self.strategy_stats: dict[str, list] = {s: [0, 0.0] for s in STRATEGIES}  # [試行, 収穫合計]
        self._topic_strategy: dict[str, str] = {}
        self.qa_log: deque = deque(maxlen=200)    # (質問, doc_id, +1/-1)
        self.interest: dict[str, float] = {}      # 関心プロファイル: 話題語 -> 重み (減衰)
        self.notices: deque = deque(maxlen=20)    # 会話に反映する「さっき学んだこと」
        self.on_gap = None                        # 話題が追加された時に呼ぶ (収集を即起動)
        self.admission = 0.0                      # メモリ逼迫時に上がる取り込み品質しきい値
        self.stats: Counter = Counter()
        self.history: deque = deque(maxlen=20)
        self.last_docs: list[int] = []
        self._said: deque = deque(maxlen=12)      # 最近答えに使った文書 (「もっと詳しく」で繰り返さない)
        self.last_mode = ""
        self.last_question = ""
        self.last_topics: list[str] = []
        self._tighten = 0
        self._enforce_counter = 0
        self.created = time.time()
        self.seed_topics: list[str] = self._load_lines(DATA_DIR / "topics.txt")

    # ------------------------------------------------------------ 補助
    @staticmethod
    def _load_lines(path: Path) -> list[str]:
        try:
            return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")]
        except OSError:
            return []

    def _apply_params(self) -> None:
        p = self.params
        self.lm.use_order = min(p.use_order, self.lm.max_order)
        self.lm.discount = p.discount
        self.kb.k1, self.kb.b, self.kb.phrase_bonus = p.k1, p.b, p.phrase_bonus

    def bootstrap(self) -> int:
        """知識が空なら同梱のシード文を学習する。"""
        if len(self.kb) > 0:
            return 0
        n = 0
        for name in ("seed_ja.txt", "seed_en.txt"):
            p = DATA_DIR / name
            if p.exists():
                n += self.learn_text(p.read_text(encoding="utf-8"), source="seed")
        return n

    # ------------------------------------------------------------ 学習
    def learn_text(self, text: str, source: str = "") -> int:
        """文に分けて知識ベースと LM に取り込む。取り込んだ文数を返す。
        大きなテキストは段落ごとに処理して一時メモリを抑える。"""
        if len(text) > 200_000:
            total = 0
            buf: list[str] = []
            size = 0
            for line in text.splitlines(keepends=True):
                buf.append(line)
                size += len(line)
                if size >= 100_000:
                    total += self.learn_text("".join(buf), source)
                    buf, size = [], 0
            if buf:
                total += self.learn_text("".join(buf), source)
            return total
        added = 0
        with self.lock:
            try:
                added = self._learn_sentences(split_sentences(text), source)
            except MemoryError:
                log.warning("MemoryError: 緊急プルーニング")
                self.lm.shrink_to(self.lm.estimated_bytes() // 2)
                self.kb.shrink_to(self.kb.estimated_bytes() // 2, self.cfg.max_docs)
                self._tighten = min(MAX_TIGHTEN, self._tighten + 1)
                self.guard.collect()
                self.stats["memory_tightened"] += 1
            self.stats["sentences_learned"] += added
            self._maybe_enforce()
        return added

    def _learn_sentences(self, sentences, source: str) -> int:
        """学習パイプライン (文単位): 解析 1 回 → 取り込み判定 → 知識ベース → 事実 → LM → 後回しキュー。
        各段階の所要時間を self.timers に積む。"""
        added = 0
        # 自己評価用の取り置きは、長い Web 文書からだけ (短い入力は 1 文が重要なので全部学ぶ)
        holdout_ok = source.startswith("http") and len(sentences) >= 40
        kb, lm, timers = self.kb, self.lm, self.timers
        protected = source in ("user", "chat", "seed")
        perf = time.perf_counter
        # ニューラル LM には「続きもの」で渡す。1 文ずつ (平均 92 トークン) だと段落の流れを学べず、
        # 生成が 1 文ごとに話題を変えてしまう。同じ出典の連続した文をまとめ、文脈長に近い長さで渡す。
        span: list[str] = []
        span_len = 0
        SPAN_CHARS = 350        # 文脈 256 トークン ≒ 日本語 300〜400 文字
        for s in sentences:
            self._holdout_counter += 1
            t0 = perf()
            info = analyze(s)
            timers["analyze"] += perf() - t0
            if holdout_ok and self._holdout_counter % 40 == 0 and len(info.tokens) >= 4:
                if len(self.holdout) < self.cfg.holdout_size:
                    self.holdout.append(info.tokens)
                else:
                    self.holdout[self.rng.randrange(len(self.holdout))] = info.tokens
                continue
            t0 = perf()
            facts = extract_facts(s) if len(s) <= 200 else []
            q = sentence_quality(s, bool(facts))
            timers["facts+quality"] += perf() - t0
            if q < self.admission and not facts and not protected:
                self.stats["skipped_low_quality"] += 1
                continue
            t0 = perf()
            doc = kb.add(s, source, tf=Counter(info.terms), quality=q, content_key=info.content_key)
            timers["kb"] += perf() - t0
            if doc is None:
                dup = kb.last_dup_id
                if dup is not None:
                    orig = kb.docs.get(dup)
                    if orig is not None and orig.source != source:
                        # 別の出典が同じことを言っている = 裏付け。文書の信頼度と事実の支持数に反映
                        orig.score = min(20.0, orig.score + 0.5)
                        self.facts.corroborate(dup, source.split("/")[2] if source.startswith("http") else source)
                        self.stats["corroborated"] += 1
                continue
            added += 1
            for subj, rel, obj in facts:
                if self.facts.add(subj, rel, obj, doc.id):
                    self.stats["facts_learned"] += 1
            if len(info.tokens) >= 2:
                t0 = perf()
                lm.learn(info.tokens)
                timers["lm"] += perf() - t0
            self._semantic_queue.append((doc.id, info.phrases))
            self._suffix_dirty += 1
            if self.neural.available:
                span.append(s)
                span_len += len(s)
                if span_len >= SPAN_CHARS:
                    self._neural_pending_text.append("".join(span))
                    span, span_len = [], 0
            if added % 500 == 0:
                self._maybe_enforce()
        if span_len >= 40:                  # 端数も渡す (短すぎるものは捨てる)
            self._neural_pending_text.append("".join(span))
        return added

    # ------------------------------------------------------------ 後回しの学習 (意味ベクトル・接尾辞配列)
    def background_step(self, budget_docs: int = 300, sketches: bool = True) -> dict:
        """空き時間に呼ぶ: 意味ベクトルの取り込み、スケッチ更新、必要なら接尾辞配列の再構築。
        応答経路からは sketches=False・小さな budget で呼び、数百 µs に抑える。"""
        with self.lock:
            n = 0
            t0 = time.perf_counter()
            while self._semantic_queue and n < budget_docs:
                item = self._semantic_queue.popleft()
                if isinstance(item, tuple):
                    doc_id, phr = item
                    if doc_id not in self.kb.docs:
                        continue
                else:  # 読込直後は ID だけ
                    doc = self.kb.docs.get(item)
                    if doc is None:
                        continue
                    phr = phrases(doc.text)
                self.semantic.learn(phr)
                n += 1
            self.timers["semantic"] += time.perf_counter() - t0
            sk = self.semantic.refresh_sketches(limit=400) if sketches and (n or self.semantic._dirty) else 0
            rebuilt = False
            if sketches and (self._suffix_dirty >= max(50, len(self.kb) // 10) or (self._suffix_dirty and not len(self.suffix))):
                self.rebuild_suffix()
                rebuilt = True
            self.stats["semantic_docs"] += n
            return {"semantic_docs": n, "sketches": sk, "suffix_rebuilt": rebuilt}

    def _cite(self, reply: Reply) -> str:
        """Web 由来の答えには出典 (ホスト名) を添える (cfg.cite)。"""
        if not self.cfg.cite or reply.mode not in ("recall", "summary", "neural", "fact"):
            return reply.text
        hosts = []
        for src in reply.sources:
            if isinstance(src, str) and src.startswith("http"):
                h = src.split("/")[2]
                if h not in hosts:
                    hosts.append(h)
        return f"{reply.text}（出典: {', '.join(hosts[:2])}）" if hosts else reply.text

    @staticmethod
    def _repeat_ratio(text: str) -> float:
        """繰り返しの割合 (0〜1)。小さなモデルは「〜は、〜は、〜は」と壊れやすいので、
        句 (名詞句) と文字 3-gram の両方で見て、繰り返しの多い方を返す。"""
        ph = phrases(text)
        by_phrase = 1.0 - len(set(ph)) / len(ph) if len(ph) >= 3 else 0.0
        t = "".join(text.split())
        grams = [t[i : i + 3] for i in range(len(t) - 2)]
        by_gram = 1.0 - len(set(grams)) / len(grams) if len(grams) >= 6 else 0.0
        return max(by_phrase, by_gram)

    @staticmethod
    def _term_overuse(text: str) -> float:
        """同じ語の使いすぎ (0〜1)。「犬と猫はどちらも犬よりも大きく、犬は猫と…」のように、
        小さなモデルは短い応答の中で同じ名詞を何度も持ち出し、結果として矛盾したことを言う。
        繰り返しの割合 (句・文字 3-gram) では捉えられないので、語の頻度で別に測る。"""
        words = _CONTENT_RE.findall(text)             # 漢字・カタカナの連続 = 内容語のだいたいの単位
        if len(words) < 5:
            return 0.0
        top = Counter(words).most_common(1)[0][1]
        share = top / len(words)
        # 同じ語が 4 回以上、かつ内容語の 1/4 以上を占めるとき「使いすぎ」とみなす
        if top < 4 or share < 0.25:
            return 0.0
        return min(1.0, (top - 3) / 3)

    @staticmethod
    def _longest_common_run(key: str, body: str) -> int:
        """key の部分文字列のうち body に現れる最長のものの長さ。"""
        best = 0
        for i in range(len(key)):
            for j in range(i + best + 1, len(key) + 1):
                if key[i:j] in body:
                    best = j - i
                else:
                    break
        return best

    @classmethod
    def _key_match(cls, key: str, body: str) -> bool:
        """肝になる語が文脈に「ほぼ」出てくるか。完全一致だと表記ゆれで落ちる
        (「ヴォルフガング・パウリ」が本文では「パウリ」と略される)。3 文字以上 (短い語はその全部が)
        連続して現れたら出てきたとみなす。割合で緩めると「カレー」が「レー」の 2 文字で通ってしまう。"""
        return cls._longest_common_run(key, body) >= min(len(key), 3)

    @staticmethod
    def _question_keys(text: str) -> tuple[list[str], bool]:
        """質問を決めている語と、それが主語かどうか。

        「X の Y は?」の形なら X (主語) だけを見る。属性側の語 (作り方・仕組み・意味) は
        どんな話題にも出てくるので、これを数に入れると「カレーの作り方」に
        「即席爆発装置の作り方」が一致してしまう。形が取れない時は珍しい句を 2 つ使う。"""
        parsed = parse_question(text)
        if parsed:
            subj_keys = [p for p in phrases(parsed[0]) if is_phrase(p)]
            if subj_keys:
                return sorted(subj_keys, key=lambda p: -term_weight(p))[:2], True
        return sorted({p for p in phrases(text) if is_phrase(p)}, key=lambda p: -term_weight(p))[:2], False

    @classmethod
    def _context_relevance(cls, text: str, docs) -> float:
        """質問の「肝になる語」が検索した文脈に現れるかどうか (0〜1)。

        文字 2-gram で測ると「の作」「を教」のような助詞混じりの断片まで一致してしまい、
        「カレーの作り方」に「即席爆発装置の作り方」が 0.86 で一致する。主語が分かる質問では
        主語が出てくることを必須にし、分からない質問では珍しい句がいくつ出てくるかで測る。"""
        keys, is_subject = cls._question_keys(text)
        if not keys or not docs:
            return 0.0
        body = " ".join(d.text for d in docs)
        hits = [cls._key_match(k, body) for k in keys]
        if is_subject:                                  # 主語が 1 つも出てこない文脈は、その質問の答えではない
            return 1.0 if any(hits) else 0.0
        return sum(hits) / len(hits)

    def _relevant_history(self, text: str, k: int = 2) -> list[tuple[str, str]]:
        """直近のやり取りのうち、今の発話と関係のあるものだけを渡す。

        話題が変わったのに履歴を渡すと、生成が前の話題に引きずられる
        (実測: 「カレーの作り方を教えて」に「ブロックチェーンとは…」と答えた)。
        今の発話に内容語があり、履歴とまったく重ならなければ「新しい話題」とみなして履歴を捨てる。
        指示語で始まる発話 (「それで?」など) は内容語が無いので、そのまま履歴を渡す。"""
        hist = self.recent_turns(k)
        if not hist:
            return hist
        low = text.lower().strip()
        if _MORE_RE.match(low) or _FOLLOWUP_RE.match(low):   # 「もっと詳しく」「それで?」は前の話の続き
            return hist
        now = {t for t in terms(text) if term_weight(t) >= 1.0}
        if not now:                                  # 内容語が無い相槌も履歴が頼り
            return hist
        kept = []
        for u, b in hist:
            past = {t for t in terms(u + " " + b) if term_weight(t) >= 1.0}
            if now & past:
                kept.append((u, b))
        return kept

    @staticmethod
    def _chatty(text: str) -> bool:
        """雑談か (= 知識文をそのまま返すべきでない発話か)。

        話題になる語がまったく無い発話、挨拶、気持ちを述べる発話を雑談とみなす。
        こういう発話に検索の 1 位を返すと、正しい文であっても会話としては噛み合わない。"""
        low = text.strip().lower()
        if _CHAT_RE.search(low):
            return True
        return not [k for k in keywords(text, limit=3) if is_phrase(k)]

    def _unsupported(self, ctx_docs, info) -> bool:
        """根拠の無い作文か (= 「知らない」と答えるべきか)。

        学習していない話題を訊かれた時に、それらしい文を作って出すのが一番たちが悪い
        (実測: 「ヴォルフガング・パウリの排他律とは」に森林破壊の話を返した)。
        次の 3 つが揃った時だけ黙る:
          (a) 使える文脈が無い (検索が空、または関連度の検査で捨てられた)
          (b) 知識に質問の主語がまったく無い (_guard_unknown の判定)
          (c) 出来た文が質問の語にほとんど触れていない、か日本語として苦しい
        (c) を見るのは、知識が無くても答えられる問い (挨拶・言い換え・常識) まで塞がないため。"""
        if ctx_docs or self._guard_note is None:
            return False
        fluency, q_overlap = info
        return q_overlap < 0.34 or fluency < -2.0

    def recent_turns(self, k: int = 2) -> list[tuple[str, str]]:
        """直近のやり取りを (発話, 応答) の組で返す (会話のキャッチボール用)。"""
        pairs: list[tuple[str, str]] = []
        pending = None
        for who, txt in self.history:
            if who == "user":
                pending = txt
            elif pending:
                pairs.append((pending, txt))
                pending = None
        return pairs[-k:]

    def _neural_reply(self, text: str, hits, fallback: Reply, strict: bool = True) -> Reply | None:
        """検索した文を文脈にして Transformer で応答を生成 (RAG)。「考える」= 2 段階:
          1. 下書き: 検索文脈で候補を生成
          2. 読み直し: 下書きの語で検索をやり直して文脈を広げ、もう一度候補を生成 (自分の答えを手掛かりにした再検索)
          3. 検証: 全候補を 接地率 (文脈・質問との句の重なり) + 自然さ (自己対数確率) + 文末の整い で採点し最良を返す
        strict=True: 接地率と自然さで厳しく検査 (検索応答と併用する時)。
        strict=False: ニューラル専用モード。明らかに壊れた文だけ捨てる。過程は self.last_thought に残す (UI / API 用)。"""
        ctx_docs = [d for c, d in hits[:3] if c >= self.params.answer_threshold * 0.5]
        # 検索は点数が付いていても見当違いのことがある (「カレーの作り方」に数列の説明が返る)。
        # 質問の内容語が文脈にどれだけ出てくるかで確かめ、関係が薄ければ文脈として使わない。
        # 無関係な文を写し取るくらいなら、何も見ずに書く方がまだ話が通じる。
        relevance = self._context_relevance(text, ctx_docs)
        if relevance < 0.5:                      # 肝になる語が 1 つも出てこない文脈は使わない
            ctx_docs = []
        chatty = self._chatty(text)
        if chatty:
            # 雑談に知識文を渡すと、その文を書き写してしまう (実測: 「こんにちは。今日は何を
            # していましたか」に青空文庫の一節をなぞった応答)。挨拶や気持ちの話は文脈なしで書く。
            ctx_docs = []
        context = " ".join(d.text for d in ctx_docs)[:240] or None
        history = self._relevant_history(text)
        # 検索が弱い時は文脈への寄せを緩める (雑談まで検索文を写すと会話にならない)
        best_score = hits[0][0] if hits else 0.0
        base_bonus = self.neural.decode.get("copy_bonus", 0.0)
        weak = best_score < self.params.answer_threshold * 1.5 or getattr(self, "_followup", False)
        bonus = base_bonus * 0.3 if weak else base_bonus
        n = 8 if not strict else 6                  # 候補を増やして選ぶ幅を広げる
        # 検索が弱い = 雑談なので長く自由に書かせる。検索が効いている時は短く的確に答える
        max_new = 110 if weak else 60
        cands = self.neural.chat(text, context, n=n, history=history, copy_bonus=bonus, max_new=max_new)
        thought = {"query": text, "context": [d.text[:80] for d in ctx_docs], "draft": list(cands), "rethink": [], "context2": [], "scores": [],
                   "history": [u for u, _ in history], "copy_bonus": round(bonus, 2), "retrieval_score": round(float(best_score), 3)}
        # 2. 読み直し: 下書きに出てきた句で再検索 (質問だけでは引けなかった文が見つかる)
        # 雑談では読み直しをしない。せっかく文脈を外しても、ここで知識文が戻ってきてしまう
        if self.cfg.neural_rethink and cands and not chatty:
            extra_terms = [p for c in cands[:2] for p in phrases(c) if is_phrase(p) and p not in text][:4]
            if extra_terms:
                hits2 = self._search(text + " " + " ".join(extra_terms))
                new_docs = [d for c, d in hits2[:3] if d not in ctx_docs and c >= self.params.answer_threshold * 0.5]
                if new_docs:
                    ctx_docs = (ctx_docs + new_docs)[:4]
                    context2 = " ".join(d.text for d in ctx_docs)[:240]
                    more = self.neural.chat(text, context2, n=max(2, n - 1), history=history, copy_bonus=bonus, max_new=max_new)
                    thought["rethink"], thought["context2"] = list(more), [d.text[:80] for d in new_docs]
                    cands = cands + more
                    context = context2
        if not cands:
            self.last_thought = thought
            return None
        # 3. 検証
        qphr = set(phrases(text))
        cphr = set(phrases(context)) if context else set()
        best, best_s, best_info = None, -1e9, (0.0, 0.0)
        seen = set()
        for c in cands:
            if c in seen:
                continue
            seen.add(c)
            ph = set(phrases(c))
            grounded = len(ph & (cphr | qphr)) / max(len(ph), 1) if ph else 0.0
            fluency = self.neural.score(c)
            if fluency is None or len(c) < 4 or _GARBLED_RE.search(c):
                thought["scores"].append((c[:60], None))
                continue
            if strict:
                if (context and grounded < 0.6) or fluency < -2.5:
                    continue
                if len(c) < 6 or c in text or not c.endswith(("。", "！", "？", ".", "!", "?", "です", "ます", "である")):
                    continue
            rep_ratio = self._repeat_ratio(c)
            if rep_ratio > 0.5:                    # 同じ句の繰り返しだらけの候補は捨てる
                thought["scores"].append((c[:60], None))
                continue
            overuse = self._term_overuse(c)        # 同じ語を何度も持ち出す候補も落とす
            if overuse >= 1.0:
                thought["scores"].append((c[:60], None))
                continue
            # 検索が弱い (雑談・指示語) 時は接地率より自然さを見る: 検索文の寄せ集めを選ばないため
            if weak:
                sc = fluency / 3.0 + min(len(c), 40) * 0.01 + grounded * 0.3
            else:
                sc = grounded + fluency / 5.0
            sc += 0.3 if c.endswith(("。", "！", "？", ".", "!", "?")) else 0.0
            sc -= rep_ratio * 0.5 + overuse * 0.4
            thought["scores"].append((c[:60], round(sc, 3)))
            if sc > best_s:
                best, best_s = c, sc
                best_info = (fluency, len(ph & qphr) / max(len(qphr), 1))
        thought["best"] = best
        # 根拠が無いのに作文していないか最後に確かめる (「知らない」と言う方が正しい場面)
        if best is not None and self._unsupported(ctx_docs, best_info):
            thought["best"], thought["unsupported"] = None, True
            self.last_thought = thought
            self.stats["neural_unknown"] += 1
            return None
        self.last_thought = thought
        if best is None:
            self.stats["neural_rejected"] += 1
            return None
        self.stats["neural_replies"] += 1
        conf = max(fallback.confidence, 0.6) if context else 0.5
        # 出典は「その文を使って書いた」時だけ付ける。雑談の返事に出典を付けると、
        # 引用していない文を引用したことにしてしまう
        return Reply(best, round(conf, 3), "neural", [d.source for d in ctx_docs], [d.id for d in ctx_docs], fallback.learned_topics)

    def _context_for(self, doc_ids, max_chars: int = 240) -> str | None:
        parts = []
        for i in doc_ids[:3]:
            d = self.kb.docs.get(i)
            if d:
                parts.append(d.text)
        ctx = " ".join(parts)
        return ctx[:max_chars] if ctx else None

    def feed_copy_examples(self, docs, retrieved_ratio: float = 0.5) -> int:
        """RAG の写し取り練習を再生バッファへ。
        文脈の作り方は 2 通り: (a) その文 (+ 隣の文、時々は無関係な文) を並べたもの、
        (b) **実際にその質問で検索した結果** — 本番と同じ条件で「検索結果の中から使う文を選ぶ」練習になる。
        (a) だけで学ぶと「文脈の先頭を写す」癖がつき、本番の検索結果 (順番も内容も違う) で効かない。"""
        nl = self.neural
        n = 0
        for d in docs:
            if d.source == "chat" or len(d.text) < 12:
                continue
            ks = [k for k in keywords(d.text, limit=2) if is_phrase(k)]
            if not ks:
                continue
            neighbor = None
            if self.rng.random() < retrieved_ratio:
                # 本番と同じ検索を通す: 取れた文を文脈にし、答えの文が無ければ混ぜ込む
                hits = self._search(f"{ks[0]}について教えて", k=3)
                texts = [h.text for _, h in hits if h.id != d.id][:2]
                if texts:
                    neighbor = " ".join(texts)
            if neighbor is None:
                nb = self.kb.docs.get(d.id + 1)
                neighbor = nb.text if nb and nb.source == d.source else None
                if self.rng.random() < 0.3:
                    # 無関係な文を混ぜる (前でも後ろでも): 文脈の中から正しい文を選ぶ練習
                    other = next(iter(self.kb.random_docs(1, self.rng)), None)
                    if other is not None and other.id != d.id:
                        neighbor = f"{other.text} {neighbor}" if neighbor and self.rng.random() < 0.5 else other.text
            nl.add_copy_example(ks[0], d.text, neighbor)
            n += 1
        return n

    def _rebuild_dialog_sequences(self, limit: int = 20000) -> int:
        """会話系列の作り方を直した時に、手持ちの会話を作り直して再生バッファへ入れ直す。

        再生バッファとコーパスの中身は「作った時の作り方」で固まっているので、作り方を直しても
        古い系列を学び続けてしまう (今回は、切り詰めた応答に <eos> が付いた系列が 40 万件あった)。"""
        nl = self.neural
        if nl.model is None or nl.seq_version >= neural_lm_SEQ_VERSION:
            return 0
        n = 0
        for item in list(self.dialogs.pairs)[-limit:]:
            u, b, _src, w = item[:4]
            if w <= 0:
                continue
            nl.add_dialog(u, b, weight=w, history=item[4] if len(item) > 4 else None)
            n += 1
        nl.seq_version = neural_lm_SEQ_VERSION
        if n:
            log.info("会話系列を作り直しました: %d 件 (作り方の版 %d)", n, neural_lm_SEQ_VERSION)
        return n

    def _feed_neural(self, max_qa_docs: int = 60) -> None:
        """再生バッファへ: 平文、会話 (文脈付き)、事実からの合成 QA。"""
        nl = self.neural
        self._rebuild_dialog_sequences()
        while self._neural_pending_text:
            nl.add_text(self._neural_pending_text.popleft())
        while self._neural_pending_dialog:
            item = self._neural_pending_dialog.popleft()
            u, b, ctx, w = item[:4]
            nl.add_dialog(u, b, ctx, weight=w, history=item[4] if len(item) > 4 else None)
        # ディスクのコーパスから少しずつ戻す: バッファに入りきらない過去の文も巡回して学ぶ
        nl.refresh_from_corpus(max(64, nl.pool.capacity // 100))
        # 文脈からの抽出練習: 最近の知識文をキーワード付きで (RAG で「検索文を使う」ことを学ぶ)
        self.feed_copy_examples(self.kb.random_docs(min(60, len(self.kb)), self.rng))
        n = 0
        for key, lst in self.facts.by_subject.items():
            for rel, obj, doc_id in lst:
                if doc_id in self._qa_done or doc_id not in self.kb.docs:
                    continue
                subj = key
                answer = self.facts._render(subj, rel, obj, True)
                nl.add_synthetic_qa(subj, rel, obj, answer, self.kb.docs[doc_id].text)
                self._qa_done.add(doc_id)
                n += 1
                if n >= max_qa_docs:
                    return
        if len(self._qa_done) > 200000:
            self._qa_done.clear()

    def neural_step(self, steps: int = 4, budget_seconds: float = 1.0) -> dict | None:
        """空き時間に呼ぶ: ニューラル LM の準備・データ供給・数ステップの学習。"""
        nl = self.neural
        if not nl.available:
            return None
        with self.lock:
            if nl.model is None:
                texts = [d.text for d in self.kb.docs.values()] if len(self.kb) >= nl.min_sentences else []
                if not nl.ensure_model(texts):
                    return None
                # バッファが容量の半分に満たなければ、持っている知識文と会話で埋める。
                # 同じ系列を何十周も学ぶより、手持ちの文をできるだけ一度ずつ通す方が過学習が少ない。
                room = nl.pool.capacity - len(nl.pool)
                if room > nl.pool.capacity * 0.5:
                    # まずディスクのコーパスから戻す (段落単位の長い系列が入っている)。
                    # 知識ベースの文書は 1 文ずつなので、そのまま入れると平文の平均長が
                    # 63 トークンまで落ちて段落の流れを学べない。文をつないでから渡す。
                    got = nl.refresh_from_corpus(int(room * 0.7))
                    rest = max(0, int(room * 0.7) - got)
                    if rest > 0:
                        span, span_len = [], 0
                        for d in self.kb.random_docs(min(rest * 4, len(self.kb)), self.rng):
                            span.append(d.text)
                            span_len += len(d.text)
                            if span_len >= 350:
                                nl.add_text("".join(span))
                                span, span_len = [], 0
                    for item in list(self.dialogs.pairs)[-int(room * 0.3):]:
                        u, b, _, w = item[:4]
                        nl.add_dialog(u, b, weight=w, history=item[4] if len(item) > 4 else None)
                    log.info("再生バッファを補充: %d 系列 (容量 %d, コーパス %d 系列)", len(nl.pool), nl.pool.capacity, len(nl.corpus or []))
            self._feed_neural()
        t0 = time.perf_counter()
        r = None
        while time.perf_counter() - t0 < budget_seconds:
            r = nl.train_some(steps=steps)
            if r is None:
                break
        if r:
            self.stats["neural_steps"] += r["steps"]
            self.timers["neural"] += time.perf_counter() - t0
            # 200 ステップごとに点検する。以前は `step % 200 < steps` で見ていたが、1 回の
            # neural_step は 60 秒ぶん (数百ステップ) 進めてから 1 度だけ判定するので、
            # 剰余が 10 未満に落ちる確率は 5% しかなく、成長の点検はほとんど回っていなかった
            # (実測: 640 ステップ進んでも一度も判定されなかった)。経過量で判定する。
            if nl.model.step - self._last_growth_check >= 200:
                self._last_growth_check = nl.model.step
                ngram = self.lm.perplexity(self.holdout) if self.holdout else None
                nl.evaluate(ngram)
                # 進化: 損失が停滞したら層を追加、新語が増えていれば語彙を拡張
                # 成長に必要なメモリはモデル自身の分だけ (1 層で約 44 万パラメータ =
                # 重み・Adam の 1 次/2 次・EMA で 7 MB 程度)。知識ベースや再生バッファが
                # 上限に近いことを理由に成長を止めると、いちばん容量が要る局面で成長できない
                # (実測: 圧力 0.77 で閾値 0.7 に阻まれ、層 6 のまま止まっていた)。
                # データ側は自前の刈り込みで縮むので、モデルの成長は別枠で判断する。
                grow_cost = self.neural.growth_bytes()
                mem_ok = self.guard.can_afford(grow_cost)
                nl.maybe_damp_lr()                         # 会話の質が落ち続けていれば学習率を下げる
                if nl.check_growth():                      # 前回の成長が裏目なら取り消す
                    self.stats["neural_rollback"] += 1
                elif nl.maybe_grow(mem_ok, data_tokens=nl.corpus.tokens if nl.corpus else 0):
                    self.stats["neural_grown"] += 1
                # 語彙の進化: 学習に使っている文 (コーパス) から候補を採る。知識ベースだけを見ると
                # 実際に学んでいる分布とずれる。1 語の追加コストは埋め込み 192 次元 × 4 系列 = 約 3 KB
                # なので、まとめて増やしても安い (1 トークンあたりの文字数が増えれば、同じ文脈長で
                # より多くの文が入る = 実質的に文脈が伸びる)。
                vocab_texts = [d.text for d in self.kb.random_docs(min(300, len(self.kb)), self.rng)]
                if nl.corpus is not None and len(nl.corpus):
                    vocab_texts += [nl.tok.decode([int(t) for t in ids]) for ids, _, _ in nl.corpus.sample(300, nl.nprng)]
                added = nl.evolve_vocab(vocab_texts, top=150)
                if added:
                    self.stats["neural_vocab_added"] += added
                log.info("ニューラル LM: step=%d loss=%.3f %s", nl.model.step, r["loss"], nl.stats())
            if time.time() - nl._last_save > 60:   # 強制終了されても失う学習は 1 分以内
                nl.save()
        return r

    def _dialog_gain(self, pairs, with_bpc: bool = False):
        """会話の取り置きに対する「文字あたりの予測の良さ」。

        返すのは削減率 (0 = 文字の出現頻度だけを知っている状態、1 = 完全予測)。
        with_bpc=True なら (削減率, 1 文字あたりビット数) を返す。"""
        nl = self.neural
        if nl.model is None or len(pairs) < 5:
            return None
        nats = chars = 0.0
        with nl.lock, nl._infer():
            for u, bb in pairs:
                ids = nl.seq_dialog(u, bb)
                st = nl.loss_from(ids)
                seq = ids[max(0, st - 1):]
                if len(seq) < 3:
                    continue
                nats += -nl.model.logprob(seq) * (len(seq) - 1)
                chars += len(bb)
        if not chars:
            return None
        from collections import Counter as _C
        cnt = _C("".join(bb for _, bb in pairs))
        tot = sum(cnt.values())
        uni = -sum(n / tot * math.log2(n / tot) for n in cnt.values()) if tot else 0.0
        if not uni:
            return None
        bpc = nats / chars / math.log(2)
        gain = 1 - bpc / uni
        return (gain, bpc) if with_bpc else gain

    def self_evaluate(self, n_docs: int = 24, n_dialogs: int = 40) -> dict:
        """学習中に自分で品質を測る (自動評価)。取り置き文の ppl だけでなく、
        会話としての ppl と RAG 忠実性 (文脈の句をどれだけ使うか) を測り、学習ログに残す。"""
        nl = self.neural
        if nl.model is None or nl.tok is None:
            return {}
        out: dict = {"step": nl.model.step}
        t0 = time.perf_counter()
        # 成長・学習率・取り消しの規則が見る「会話の質」の履歴は 1 回の評価で 1 つだけ入れる。
        # 以前は固定の取り置き (bpc×100 ≒ 410) と入れ替わる取り置き (削減率 ≒ 49) の**両方**を
        # 同じ履歴に入れていたので、隣り合う値が別の尺度になり、規則が比べていたのは雑音だった。
        # 入れ替わる取り置きの方が今の分布に近いので、両方測れた時はそちらを採る。
        dialog_metric: float | None = None
        # 1. 取り置き文の ppl (言語としての予測力)
        if nl._holdout:
            out["ppl"] = round(neural_perplexity(nl), 2)
        if nl.recent_ppl is not None:
            out["recent_ppl"] = nl.recent_ppl      # 入れ替わる取り置き: 今の分布での汎化 (ppl との差が忘却の量)
        # 2. 会話の ppl (応答部だけ)。比較できるよう取り置き会話は一度決めたら固定する
        if not self.dialog_holdout and len(self.dialogs) >= n_dialogs * 2:
            rng = random.Random(12345)
            # 文学作品の会話 (文脈が無いと予測しようがない台詞) は評価から外す
            cand = [(u, b) for u, b, src, w, *_ in list(self.dialogs.pairs) if w > 0 and 10 <= len(b) <= 200 and not src.startswith("aozora")]
            self.dialog_holdout = rng.sample(cand, min(n_dialogs, len(cand)))
        pairs = self.dialog_holdout or [(u, b) for u, b, _, w, *_ in list(self.dialogs.pairs)[-2000:] if w > 0][-n_dialogs:]
        if pairs:
            ppls = []
            nats = chars = 0.0
            with nl.lock, nl._infer():
                for u, b in pairs:
                    ids = nl.seq_dialog(u, b)
                    start = nl.loss_from(ids)
                    if len(ids) - start < 2:
                        continue
                    seq = ids[max(0, start - 1):]
                    mean_nat = -nl.model.logprob(seq)         # 1 トークンあたりの負の対数尤度
                    ppls.append(math.exp(mean_nat))
                    nats += mean_nat * (len(seq) - 1)
                    chars += max(len(b), 1)
            if ppls:
                # 中央値で報告する: 固有名詞を含む 1 件が平均を 2 倍に押し上げる (実測: 平均 106 / 中央値 52)
                ppls.sort()
                out["dialog_ppl"] = round(ppls[len(ppls) // 2], 2)
                out["dialog_ppl_mean"] = round(sum(ppls) / len(ppls), 2)
                if chars:
                    # 絶対的な物差し: 同じ応答の文字ユニグラム分布のエントロピー。
                    # 「文字の出現頻度だけ知っている」状態が何ビット必要かで、モデルの上限側の基準になる
                    # (ppl や bpc の数字だけでは、良いのか悪いのか判断できない)。
                    from collections import Counter as _C
                    cnt = _C("".join(b for _, b in pairs))
                    tot = sum(cnt.values())
                    uni = -sum(n / tot * math.log2(n / tot) for n in cnt.values()) if tot else None
                    # 1 文字あたりのビット数。語彙を増やすとトークンの区切りが変わり、
                    # 1 トークンあたりの ppl は機械的に上がる (1 トークンが多くの文字を担うため)。
                    # 文字あたりで測れば語彙の変更をまたいで比較できる。成長や学習率の判断にはこちらを使う。
                    out["dialog_bpc"] = round(nats / chars / math.log(2), 4)
                    if uni:
                        out["dialog_bpc_unigram"] = round(uni, 3)
                        out["dialog_bpc_gain"] = round(1 - out["dialog_bpc"] / uni, 3)   # 0 = 頻度だけ、1 = 完全予測
                    dialog_metric = out["dialog_bpc"] * 100       # 規則は同じ尺度で扱う
                else:
                    dialog_metric = out["dialog_ppl"]
        # 2b. 最近の会話でも同じ測り方をする。固定の取り置きは時間が経つほど今の分布から離れるので、
        # そこだけを見ると「分布が動いた」のを「質が落ちた」と取り違える
        # (実測: 固定 4.23 bpc / 削減 45.7% に対し、最近の会話では 3.35 bpc / 削減 49.0%)。
        # 抽出しなおすたびに中身が変わると、評価のたびに値が動いて比べられない
        # (実測: 同じモデルで 0.529 → 0.501 → 0.477)。一定期間は同じ集合を使い、古くなったら入れ替える。
        step_now = nl.model.step
        if not self._fresh_holdout or step_now - self._fresh_holdout_step > 1500:
            cand = [(u, bb) for u, bb, src, w, *_ in list(self.dialogs.pairs)[-4000:]
                    if w > 0 and 10 <= len(bb) <= 200 and not src.startswith("aozora")]
            if len(cand) >= 20:
                fresh = random.Random(step_now).sample(cand, min(n_dialogs * 2, len(cand)))
                # 取り置きを入れ替える瞬間に、**同じモデルで**新旧の両方を測り、その差を覚えておく。
                # これをしないと、入れ替えのたびに値が跳ねて前後が比べられない
                # (実測: 0.507 → 0.441。モデルは何も変わっていないのに 0.066 動いた)。
                if self._fresh_holdout:
                    old_g = self._dialog_gain(self._fresh_holdout)
                    new_g = self._dialog_gain(fresh)
                    if old_g is not None and new_g is not None:
                        self._fresh_shift += old_g - new_g
                self._fresh_holdout, self._fresh_holdout_step = fresh, step_now
        recent_pairs = list(self._fresh_holdout)
        if len(recent_pairs) >= 20:
            measured = self._dialog_gain(recent_pairs, with_bpc=True)
            if measured is not None:
                gain, bpc = measured
                out["dialog_bpc_fresh"] = round(bpc, 4)
                out["dialog_gain_fresh"] = round(gain, 3)
                # 入れ替えの差を足し戻した値。UI と規則はこちらを使う (取り置きが変わっても連続する)
                out["dialog_gain_fresh_adj"] = round(gain + self._fresh_shift, 3)
                # 規則には「基準からの削減率」を使う: 語彙が変わっても、取り置きの中身が変わっても比較できる
                dialog_metric = (1 - out["dialog_gain_fresh_adj"]) * 100

        if dialog_metric is not None:
            nl.note_dialog_ppl(dialog_metric)

        # 3. RAG 忠実性: 知識文を文脈に、その文のキーワードを質問にして、答えが文脈の句をどれだけ含むか。
        # 毎回別の文で測ると値が揺れて比べられない (実測: 同じ時期に 0.785〜0.873)。
        # 会話の取り置きと同じく、一定期間は同じ文で測る。
        if not self._rag_docs or step_now - self._rag_docs_step > 5000:
            picked = [d.id for d in self.kb.random_docs(min(n_docs * 4, len(self.kb)), self.rng)
                      if 20 <= len(d.text) <= 200]
            if len(picked) >= n_docs:
                self._rag_docs, self._rag_docs_step = picked, step_now
        rag_pool = [self.kb.docs[i] for i in self._rag_docs if i in self.kb.docs]
        if len(rag_pool) < n_docs:                 # 取り置きの文が刈り込まれていたら足す
            rag_pool += list(self.kb.random_docs(min(n_docs * 3, len(self.kb)), self.rng))
        grounded = kw = n = 0
        # 生成の乱数も固定する。同じ文・同じモデルでも引くたびに文が変われば、測るたびに値が動く
        saved_rng = getattr(nl, "nprng", None)
        if saved_rng is not None:
            from . import neural as _nn

            nl.nprng = _nn.np.random.default_rng(20260918)
        for d in rag_pool:
            if n >= n_docs:
                break
            if not (20 <= len(d.text) <= 200):
                continue
            ks = [k for k in keywords(d.text, limit=1) if is_phrase(k)]
            if not ks:
                continue
            cands = nl.chat(f"{ks[0]}について教えて", d.text, n=2, max_new=40)
            if not cands:
                continue
            cph = set(phrases(d.text))
            best = max(cands, key=lambda c: len(set(phrases(c)) & cph))
            ph = set(phrases(best))
            grounded += len(ph & cph) / max(len(ph), 1)
            kw += ks[0] in best
            n += 1
        if saved_rng is not None:
            nl.nprng = saved_rng
        if n:
            out["rag_grounded"] = round(grounded / n, 3)
            out["rag_keyword"] = round(kw / n, 3)
        out["seconds"] = round(time.perf_counter() - t0, 1)
        self.stats["self_evals"] += 1
        self.last_self_eval = out
        return out

    def rebuild_suffix(self) -> None:
        """知識文全体から接尾辞配列を作り直す (数万文で 0.1 秒程度)。"""
        seqs = []
        for d in self.kb.docs.values():
            ids = self.lm.ids(tokenize(d.text))
            if len(ids) >= 3:
                seqs.append(ids)
        self.suffix = SuffixIndex.build(seqs)
        self._suffix_dirty = 0
        self.stats["suffix_rebuilds"] += 1

    def learn_batch(self, batch, collector=None) -> int:
        """収集システムの 1 バッチ (複数ページ) を学習し、リンクをフロンティアへ、収穫を報告する。
        収穫は「文数」ではなく「新しく覚えた語の数」も含めて報告する (新規性駆動)。"""
        total = 0
        best: tuple[int, str, str] | None = None
        for src, text, anchors in batch.pages:
            before = len(self.kb.index)
            surprise = self.neural.surprise(split_sentences(text)[:40:7]) if self.neural.model is not None else None
            n = self.learn_text(text, source=src)
            # 本文は丸ごとディスクのコーパスへ: 知識ベースの上限とは別に、学習トークンだけを増やす
            self.stats["corpus_tokens"] += self.neural.add_corpus_text(text)
            novelty = len(self.kb.index) - before
            total += n
            self.stats["new_terms"] += novelty
            if collector is not None:
                collector.report(batch.source or src, n, novelty, surprise)
                if anchors:
                    collector.push_links(anchors, depth=1, base_url=src)
            if n and (best is None or n > best[0]):
                best = (n, src, text)
            log.info("学習 [%s] %s <- %s (%d 文)", batch.kind, batch.topic, src, n)
        if getattr(batch, "dialogs", None):
            with self.lock:
                n_d = 0
                for item in batch.dialogs:
                    q, a = item[0], item[1]
                    ctx = item[2] if len(item) > 2 else None     # 読解データは文脈付き (RAG の練習)
                    w = item[3] if len(item) > 3 else 1.0        # w < 0 = 選好データの不採用応答 (unlikelihood)
                    hist = item[4] if len(item) > 4 else None    # 多ターン会話: これまでのやり取り
                    if self.dialogs.add(q, a, source=batch.source or "web", weight=w, history=hist):
                        n_d += 1
                        self._neural_pending_dialog.append((q, a, ctx, w, hist))
                self.stats["dialogs_collected"] += n_d
        with self.lock:
            if batch.kind == "topic":
                self.explored[batch.topic] = time.time()
                self.stats["topics_explored"] += 1
                strategy = self._topic_strategy.pop(batch.topic, None)
                if strategy in self.strategy_stats:
                    st = self.strategy_stats[strategy]
                    st[0] += 1
                    st[1] += min(total, 400) / 100.0
                if strategy == "gap" and total:
                    self._add_notice(batch.topic)
            self.stats["pages_read"] += len(batch.pages)
        return total

    def _add_notice(self, topic: str) -> None:
        """調べた話題について、最も関連の高い 1 文を「さっき学んだこと」として貯める。"""
        hits = self.kb.search(topic, k=1)
        if hits:
            self.notices.append((topic, hits[0][1].text, hits[0][1].source))

    def take_notices(self) -> list[tuple[str, str, str]]:
        with self.lock:
            out = list(self.notices)
            self.notices.clear()
            return out

    # ------------------------------------------------------------ 関心
    def _bump_interest(self, topics: Iterable[str], weight: float = 1.0) -> None:
        for t in topics:
            self.interest[t] = min(5.0, self.interest.get(t, 0.0) * 0.9 + weight)
        if len(self.interest) > 300:
            for k in sorted(self.interest, key=self.interest.get)[:100]:
                del self.interest[k]

    def interest_score(self, text: str) -> float:
        """収集システム用: アンカー文字列などがどれだけ関心に近いか (-1..1)。"""
        ks = [k for k in keywords(text, limit=3) if is_phrase(k)] or ([text] if is_phrase(text) else [])
        if not ks:
            return -0.5
        score = 0.0
        for k in ks:
            if k in self.gaps:
                score = max(score, 1.0)
            if k in self.interest:
                score = max(score, min(1.0, self.interest[k] / 3.0))
            if k in self.explored:
                score -= 0.5
            elif not self.kb.posting_ids(k):
                score += 0.2  # 未知の語は少し好奇心
        return max(-1.0, min(1.0, score))

    def learn_file(self, path: str | Path, max_bytes: int = 64 * MB) -> int:
        """ファイルを少しずつ読みながら学習する (一時メモリを抑えるため)。"""
        p = Path(path)
        source = f"file:{p.name}"
        if p.suffix.lower() in (".html", ".htm"):
            from .web import html_to_text

            text = html_to_text(p.read_bytes()[: 4 * MB].decode("utf-8", "replace"))[0]
            return self.learn_text(text, source=source)
        total = 0
        read = 0
        with p.open("r", encoding="utf-8", errors="replace") as f:
            buf: list[str] = []
            size = 0
            for line in f:
                read += len(line)
                if read > max_bytes:
                    break
                buf.append(line)
                size += len(line)
                if size >= 100_000:
                    total += self.learn_text("".join(buf), source)
                    buf, size = [], 0
            if buf:
                total += self.learn_text("".join(buf), source)
        return total

    def learn_from_web(self, topic: str, fetcher) -> int:
        """話題を検索して読み、学習する。学習文数を返し、戦略の収穫として記録する。"""
        total = 0
        for src, txt in fetcher.search_and_read(topic, langs=self.cfg.languages):
            n = self.learn_text(txt, source=src)
            total += n
            log.info("学習 %s <- %s (%d 文)", topic, src, n)
        with self.lock:
            self.explored[topic] = time.time()
            self.stats["topics_explored"] += 1
            self.stats["pages_read"] += 1 if total else 0
            strategy = self._topic_strategy.pop(topic, None)
            if strategy in self.strategy_stats:
                st = self.strategy_stats[strategy]
                st[0] += 1
                st[1] += min(total, 400) / 100.0  # 収穫 (文数を 100 で割った値、上限 4)
            if len(self.explored) > 2000:
                for k in sorted(self.explored, key=self.explored.get)[:500]:
                    del self.explored[k]
        return total

    # ------------------------------------------------------------ 話題選択
    def gap_first(self, topic: str) -> bool:
        """「知らない」と答えた語を、調べる順番の先頭に置く。

        普通の add_gap は末尾に積むので、雑談で拾った話題に埋もれて何時間も後回しになる。
        知らないと**言ってしまった**話題は、次に同じことを訊かれた時に答えられるべき最優先の宿題。"""
        topic = topic.strip()
        if len(topic) < 2 or not self._good_topic(topic):
            return False
        if time.time() - self.explored.get(topic, 0) < 86400:
            return False
        if topic in self.gaps:
            self.gaps.remove(topic)
        self.gaps.appendleft(topic)
        self._topic_strategy[topic] = "gap"
        return True

    def add_gap(self, topic: str) -> None:
        topic = topic.strip()
        if len(topic) < 2 or topic in self.gaps:
            return
        if time.time() - self.explored.get(topic, 0) < 86400:
            return
        self.gaps.append(topic)
        if self.on_gap is not None:
            try:
                self.on_gap(topic)
            except Exception:  # 通知先の失敗で会話を止めない
                pass

    def _pick_strategy(self) -> str:
        """収穫量に基づく UCB1 で探索戦略を選ぶ。"""
        total = sum(st[0] for st in self.strategy_stats.values()) + 1
        best, best_v = "seed", -1.0
        for name in ("interest", "curiosity", "sparse", "seed"):
            tries, gain = self.strategy_stats[name]
            if tries == 0:
                return name
            v = gain / tries + math.sqrt(2 * math.log(total) / tries)
            if v > best_v:
                best, best_v = name, v
        return best

    def _good_topic(self, t: str) -> bool:
        """探索する価値のある話題語か。英語の一般語 (things, lovers) を避ける。"""
        if not is_phrase(t) or t.lower() in _NOT_TOPICS:
            return False
        if t.isascii():
            return len(t) >= 4 and not t.isdigit() and len(self.kb.posting_ids(t)) >= 2
        return len(t) >= 2

    def _candidates(self, strategy: str, now: float) -> list[str]:
        if strategy == "interest":
            # 会話の関心に近い語のうち、まだ調べていないもの
            cands = [t for t, w in sorted(self.interest.items(), key=lambda x: -x[1]) if now - self.explored.get(t, 0) > 86400 * 3]
            out = []
            for t in cands[:5]:
                for u, _ in self.kb.related_terms(t, k=3):
                    if now - self.explored.get(u, 0) > 86400 * 3:
                        out.append(u)
                if not self.kb.posting_ids(t):
                    out.append(t)
            return out[:10]
        if strategy == "curiosity":
            out = []
            for d in self.kb.random_docs(5, self.rng):
                for k in keywords(d.text, limit=3):
                    if self._good_topic(k) and now - self.explored.get(k, 0) > 86400 * 3:
                        out.append(k)
            return out
        if strategy == "sparse":
            return [t for t in self.kb.sparse_terms(10, self.rng, min_len=3) if self._good_topic(t) and now - self.explored.get(t, 0) > 86400 * 3]
        cands = [t for t in self.seed_topics if now - self.explored.get(t, 0) > 86400 * 7]
        return [self.rng.choice(cands)] if cands else (list(self.seed_topics) if self.seed_topics else [])

    def next_topic(self) -> str | None:
        with self.lock:
            now = time.time()
            while self.gaps:  # 会話で分からなかった話題が最優先
                t = self.gaps.popleft()
                if now - self.explored.get(t, 0) > 86400:
                    self._topic_strategy[t] = "gap"
                    return t
            tried = []
            strategy = self._pick_strategy()
            for _ in range(3):
                cands = self._candidates(strategy, now)
                if cands:
                    t = self.rng.choice(cands)
                    self._topic_strategy[t] = strategy
                    return t
                tried.append(strategy)
                self.strategy_stats[strategy][0] += 1  # 候補なし = 収穫ゼロの試行
                rest = [s for s in ("interest", "curiosity", "sparse", "seed") if s not in tried]
                if not rest:
                    break
                strategy = rest[0]
            return None

    # ------------------------------------------------------------ 会話
    def reply(self, user_text: str) -> Reply:
        with self.lock:
            text = normalize(user_text)
            self.stats["turns"] += 1
            ja = _cjk_ratio(text) > 0.2 or not text.isascii()
            cmd = self._command(text, ja)
            if cmd is not None:
                self.history.append(("user", text))
                self.history.append(("ai", cmd.text))
                return cmd
            self.history.append(("user", text))
            # 道具 (計算・日付・換算・比較・列挙・要約・調査・プロファイル) は cfg.tools の時だけ
            fmt_instruction = text
            text = strip_format(text)
            neural_mode = self.cfg.neural_only and self.neural.ready
            tool = self.agent.handle(text, ja) if self.cfg.tools else None
            if tool is not None:
                tool.text = apply_format(tool.text, fmt_instruction)
                self.stats["tool_" + tool.mode.split(":")[-1]] += 1
                self.last_docs = tool.doc_ids
                self.last_mode = tool.mode
                self.last_question = text
                self.history.append(("ai", tool.text))
                return tool

            toks = tokenize(text)
            if toks:
                self.lm.learn(toks)  # 会話の文体を学ぶ
                self.cache_lm.push(self.lm.ids(toks))
            if self._semantic_queue:
                self.background_step(budget_docs=8, sketches=False)
            question = is_question(text)
            new_doc = None
            if not question and len(text) >= 8:
                new_doc = self.kb.add(text, "chat")

            # 文脈: 話題語が無い発話は前の話題を引き継ぐ
            topics = [k for k in keywords(text, limit=3) if is_phrase(k)]
            self._bump_interest(topics)
            query = text
            # 引き継ぎは「指示語で始まる」か「情報量のある語がほとんど無い」時だけ (「光の速さは？」には不要)
            info = sum(term_weight(t) for t in set(terms(text)))
            # 指示語で始まる・情報量が乏しい発話は検索が当てにならない (生成を文脈に寄せすぎない)
            self._followup = bool(_FOLLOWUP_RE.match(text.lower()) or (question and not topics and info < 1.5))
            if self.last_topics and (_FOLLOWUP_RE.match(text.lower()) or (question and not topics and info < 1.5)):
                query = text + " " + " ".join(self.last_topics)
                topics = topics or list(self.last_topics)
            qtype = question_type(text)
            reply = None
            if topics and not neural_mode:
                m = re.match(r"^(.+?)(について|の話を|のこと|に関して)\s*(教えて|話して|説明して|聞かせて|知りたい|まとめて)", text)
                if m and is_phrase(m.group(1).lower()):
                    summ = self.summarize(m.group(1), ja=ja)
                    if summ is not None:
                        s_text, s_ids, s_srcs = summ
                        self.stats["summaries"] += 1
                        reply = Reply(s_text, 0.75, "summary", list(dict.fromkeys(s_srcs)), s_ids, [])
            if reply is None and not neural_mode:
                reply = self._answer_from_facts(text, ja)
            if reply is None:
                hits = self._search(query, exclude_id=new_doc.id if new_doc else None, qtype=qtype, subject=topics[0] if topics else None)
                hits = self._guard_unknown(text, hits, ja)
                reply = self._compose(text, hits, topics, ja, question, qtype)
                if neural_mode:
                    # 応答は常にニューラル生成。検索結果 (と事実) は文脈として渡すだけ
                    fact_ans = self.facts.answer(text)
                    if fact_ans is not None and fact_ans[1] in self.kb.docs:
                        hits = [(1.0, self.kb.docs[fact_ans[1]])] + [h for h in hits if h[1].id != fact_ans[1]]
                    neural_reply = self._neural_reply(text, hits, reply, strict=False)
                    if neural_reply is not None:
                        reply = neural_reply
                elif self.cfg.neural_first and self.neural.ready and self._guard_note is None:
                    # 雑談には検索文をそのまま返さない。挨拶や気持ちの話に知識文を当てると
                    # 会話にならない (実測: 「こんにちは。今日は何をしていましたか」に青空文庫の一節、
                    # 「疲れたときはどうすればいいですか」に論文の一文が返っていた)。
                    weak = (reply.mode in ("guess", "generate") or self._chatty(text)
                            or (reply.mode == "recall" and reply.confidence < self.cfg.neural_override_conf))
                    if weak:
                        # 雑談では厳しい検査 (文脈との重なり 0.6 以上など) を外す。あれは知識を答える時の規則で、
                        # 「こんにちは」に文脈との重なりを求めても意味が無く、生成が全部捨てられてしまう
                        neural_reply = self._neural_reply(text, hits, reply, strict=not self._chatty(text))
                        if neural_reply is not None:
                            reply = neural_reply
            reply = self._attach_notices(reply, topics, ja)
            reply.text = self._cite(reply)
            if reply.confidence >= 0.3:  # 「まだ知りません」のような短い定型には書式指示を適用しない
                reply.text = apply_format(reply.text, fmt_instruction)
            for t in topics:
                if reply.confidence < 0.7 and (not t.isascii() or len(t) >= 4):
                    self.add_gap(t)
            reply.learned_topics = [t for t in topics if t in self.gaps]
            self.last_docs = reply.doc_ids
            self._said.extend(reply.doc_ids)
            self.last_mode = reply.mode
            self.cache_lm.push(self.lm.ids(tokenize(reply.text))[:60])
            if reply.mode in ("fact", "recall", "summary", "neural") and reply.confidence >= 0.6:
                if self.dialogs.add(text, reply.text, source="chat", history=self.recent_turns(2)):
                    ctx_now = self._context_for(reply.doc_ids)
                    if self.cfg.online_learning and self.neural.model is not None:
                        # リアルタイム学習: このターンで即座に勾配更新 (数十〜数百 ms)
                        t0 = time.perf_counter()
                        self.neural.learn_turn(text, reply.text, ctx_now, weight=1.0, steps=1, history=self.recent_turns(2))
                        self.timers["online"] += time.perf_counter() - t0
                        self.stats["online_turns"] += 1
                    else:
                        self._neural_pending_dialog.append((text, reply.text, ctx_now, 1.0))
            self._last_pair = (text, reply.text)
            self._last_pair_ctx = self._context_for(reply.doc_ids)
            self.last_question = text
            if topics:
                self.last_topics = topics[:2]
            self.history.append(("ai", reply.text))
            self._maybe_enforce()
            return reply

    def _answer_from_facts(self, text: str, ja: bool) -> Reply | None:
        """抽出済みの事実で直接答えられる質問 (XのYは? / Xとは?) なら即答。"""
        try:
            ans = self.facts.answer(text)
        except Exception:
            return None
        if ans is None:
            return None
        sentence, doc_id = ans
        doc = self.kb.docs.get(doc_id)
        if doc is None:
            return None
        self.stats["fact_answers"] += 1
        return Reply(sentence, 0.9, "fact", [doc.source], [doc.id], [])

    def _attach_notices(self, reply: Reply, topics: list[str], ja: bool) -> Reply:
        """裏で調べ終えた話題があれば、次の返答に一言添える (人が「さっき調べたんだけど」と言うように)。"""
        if not self.notices:
            return reply
        related = [n for n in self.notices if any(t in n[0] or n[0] in t for t in topics)]
        picked = related[0] if related else (self.notices[0] if reply.mode in ("generate", "guess") else None)
        if picked is None:
            return reply
        self.notices.remove(picked)
        topic, sentence, _ = picked
        note = f"（さっき「{topic}」を調べました: {sentence}）" if ja else f"(I just looked into '{topic}': {sentence})"
        if reply.mode in ("generate", "guess") and related:
            return Reply(sentence, 0.6, "recall", [picked[2]], reply.doc_ids, reply.learned_topics)
        reply.text = f"{reply.text} {note}"
        self.stats["notices_delivered"] += 1
        return reply

    def _search(self, text: str, exclude_id: int | None = None, k: int = 5, qtype: str = "none", subject: str | None = None):
        p = self.params
        qw = self.kb.query_weights(text)
        hits = self.kb.search(text, k=k + 1)
        conf = self._confidences(hits, text, exclude_id, qw=qw)
        cover_map = dict(self._last_cover)
        # 確信が低ければ関連語でクエリを広げてもう一度
        if (not conf or conf[0][0] < p.answer_threshold) and p.expand_weight > 0:
            extra: dict[str, float] = {}
            for t in [t for t in terms(text) if is_phrase(t)][:3]:
                for u, w in self.kb.related_terms(t, k=3):
                    extra[u] = max(extra.get(u, 0.0), min(1.0, w) * p.expand_weight)
            if extra:
                hits2 = self.kb.search(text, k=k + 1, extra=extra)
                conf2 = self._confidences(hits2, text, exclude_id, penalty=0.9, qw=qw)
                cover_map.update(self._last_cover)
                seen = {d.id for _, d in conf}
                conf.extend(x for x in conf2 if x[1].id not in seen)
                self.stats["expanded"] += 1
        # 意味ベクトル: 質問と文の句ベクトルのコサイン
        qvec = self.semantic.text_vector(phrases(text)) if p.semantic_weight > 0 and self.semantic.vec else None
        expanded = self.stats.get("expanded", 0)
        rescored = []
        self._last_features = {}
        for c, d in conf:
            qb = _rerank_bonus(qtype, d.text, subject) if qtype != "none" else 0.0
            sem = 0.0
            if qvec is not None:
                dvec = self._doc_vector(d)
                if dvec is not None:
                    sem = self.semantic.cosine(qvec, dvec)
            c2 = c + p.rerank_weight * qb + p.semantic_weight * max(sem, 0.0) * 0.5
            x = Reranker.features(c * 10, cover_map.get(d.id, 0.0), qb, sem, d.quality, d.score, d.source, len(d.text), False, d.id in self.facts.by_doc)
            if self.reranker.samples >= 20 and p.learned_weight > 0:
                # 学習サンプルが少ないうちは弱く混ぜる (50 件で満額)
                lw = p.learned_weight * min(1.0, self.reranker.samples / 50.0)
                c2 = (1 - lw) * c2 + lw * self.reranker.predict(x)
            self._last_features[d.id] = x
            rescored.append((min(1.0, c2), d))
        rescored.sort(key=lambda x: -x[0])
        return rescored[:k]

    def _source_of(self, doc_id: int) -> str:
        d = self.kb.docs.get(doc_id)
        if d is None:
            return "?"
        src = d.source
        if src.startswith("http"):
            return src.split("/")[2] + "/" + src.rsplit("/", 1)[-1][:40]  # ホスト + ページ
        return src

    def summarize(self, topic: str, max_sentences: int = 3, ja: bool = True) -> tuple[str, list[int], list[str]] | None:
        """「X について教えて」向けの抽出的要約: 事実 (定義など) + 出典の異なる関連文を句の重なりで重複除去して並べる。"""
        parts: list[str] = []
        ids: list[int] = []
        srcs: list[str] = []
        used_phr: set[str] = set()
        for rel, obj, doc_id in self.facts.lookup(topic)[:2]:
            if doc_id in self.kb.docs:
                sent = self.facts._render(topic, rel, obj, ja)
                parts.append(sent)
                ids.append(doc_id)
                srcs.append(self.kb.docs[doc_id].source)
                used_phr |= set(phrases(sent))
        hits = self._search(topic, k=8)
        seen_src = set(srcs)
        for conf, d in hits:
            if len(parts) >= max_sentences:
                break
            if conf < self.params.answer_threshold * 0.8 or d.id in ids or d.source == "chat":
                continue  # 会話ログの文 (質問そのもの) は要約に入れない
            ph = set(phrases(d.text))
            if used_phr and len(ph & used_phr) / max(len(ph), 1) > 0.6:
                continue  # 既に言った内容とほぼ同じ
            if d.source in seen_src and len(parts) >= 2:
                continue  # 出典の多様性を優先
            parts.append(d.text)
            ids.append(d.id)
            srcs.append(d.source)
            seen_src.add(d.source)
            used_phr |= ph
        if not parts:
            return None
        return " ".join(parts), ids, srcs

    def _guard_unknown(self, text: str, hits, ja: bool):
        """「知らないことは知らないと言う」ための確信度の抑制。
        (a) 質問の主語 (句) を知識がまったく含まない → 候補の確信度を下げる
        (b) 「X の Y は?」で X は知っているが属性 Y を含む文が無い → 確信度を下げる (別の属性で答えない)"""
        parsed = parse_question(text)
        subj_ok = True
        attr = None
        self._guard_note = None
        subj = ""
        if parsed:
            subj, attr = parsed
            subj_terms = [t for t in terms(subj) if is_phrase(t)] or [t for t in terms(subj) if term_weight(t) >= 1.0]
            if subj_terms and not any(self.kb.posting_ids(t) for t in subj_terms):
                subj_ok = False
        else:
            ks = [k for k in keywords(text, limit=2) if is_phrase(k)]
            if ks and not any(self.kb.posting_ids(k) for k in ks):
                subj_ok = False
        # (c) 検索の点数は高いのに、出てきた文が質問の肝になる語に触れていない
        # 転置索引の点数は助詞混じりの断片でも上がるので、点数だけ見ると見当違いの文を読み上げてしまう
        # (実測: 「カレーの作り方を教えて」→ 爆発装置の作り方、「ゾンビ星ペンタクロンの公転周期は?」→ 月の公転周期)
        keys, _ = self._question_keys(text)
        if subj_ok and keys and is_question(text) and hits:
            on_topic = [(c, d) for c, d in hits if any(self._key_match(k, d.text) for k in keys)]
            if not on_topic:
                self.stats["off_topic_hits"] += 1
                self._guard_note = ("subject", subj or keys[0], "")
                return [(min(c, 0.2), d) for c, d in hits]
            if len(on_topic) < len(hits):
                # 触れている文を前に出す (点数 1 位が見当違いでも、下位に本物があればそちらを使う)
                off = [(min(c, self.params.answer_threshold * 0.5), d) for c, d in hits if (c, d) not in on_topic]
                self.stats["off_topic_hits"] += 1
                hits = on_topic + off
        if not subj_ok:
            self.stats["unknown_subject"] += 1
            self._guard_note = ("subject", subj or " ".join(keywords(text, limit=1)), "")
            return [(min(c, 0.2), d) for c, d in hits]
        if attr and len(attr) >= 2 and attr not in ("definition", "event", "location", "who"):
            keys = attr_synonyms(attr)
            kanji = [ch for ch in attr if "\u4e00" <= ch <= "\u9fff"]
            out = []
            kept = 0
            for c, d in hits:
                t = d.text
                # 属性語・同義語を含む、または属性の漢字を含む (高さ ↔ 高い) 文は残す
                if any(k in t for k in keys) or (kanji and any(ch in t for ch in kanji)):
                    out.append((c, d))
                    kept += 1
                else:
                    out.append((min(c, self.params.answer_threshold * 0.5), d))  # 属性が無い文は「たぶん」以下に
            self.stats["attr_guard"] += 1
            if not kept:
                self._guard_note = ("attr", subj, attr)
            return out
        return hits

    def _doc_vector(self, doc: Doc):
        """文書の意味ベクトル (意味空間の世代ごとにキャッシュ)。"""
        gen = self.semantic.updates // 2000
        hit = self._docvec_cache.get(doc.id)
        if hit is not None and hit[0] == gen:
            return hit[1]
        vec = self.semantic.text_vector(phrases(doc.text))
        if len(self._docvec_cache) > 5000:
            self._docvec_cache.clear()
        self._docvec_cache[doc.id] = (gen, vec)
        return vec

    def _confidences(self, hits, text: str, exclude_id: int | None, penalty: float = 1.0, qw=None):
        out = []
        qw = qw or self.kb.query_weights(text)
        self._last_cover = {}
        for score, doc in hits:
            if doc.id == exclude_id:
                continue
            cover = self.kb.coverage(doc.id, text, qw)
            self._last_cover[doc.id] = cover
            conf = (0.65 * cover + 0.35 * (1 - math.exp(-score / 6.0))) * penalty
            out.append((conf, doc))
        return out

    def _continuation(self, ja: bool) -> Reply | None:
        """「もっと詳しく」: 直前に答えた文の続きを同じ出典から返す。"""
        if not self.last_docs:
            return None
        last = self.kb.docs.get(self.last_docs[-1])
        if last is None:
            return None
        parts = []
        ids = []
        anchor = set(phrases(last.text)) | set(self.last_topics)
        for i in range(1, 4):
            nxt = self.kb.docs.get(last.id + i)
            if nxt is None or nxt.source != last.source:
                break
            # 隣の文でも話題が違えば続きではない (同じ出典に別の話題が混ざることがある)
            if anchor and not (anchor & set(phrases(nxt.text))) or nxt.id in self._said:
                break
            parts.append(nxt.text)
            ids.append(nxt.id)
            if sum(len(x) for x in parts) > 240:
                break
        if not parts and self.last_topics:
            # 続きの文が無ければ、同じ話題についてまだ言っていない文を探す
            query = " ".join(self.last_topics)
            for conf, d in self._confidences(self.kb.search(query, k=8), query, None):
                if conf >= self.params.answer_threshold * 0.8 and d.id not in self._said and d.id != last.id:
                    parts.append(d.text)
                    ids.append(d.id)
                    last = d
                    break
        if not parts:
            for t in self.last_topics:
                self.add_gap(t)
            return Reply("それについてはこれ以上知りません。調べておきます。" if ja else "That's all I know about it for now. I'll look it up.", 0.2, "generate", [], [], [])
        self.stats["continued"] += 1
        return Reply(" ".join(parts), 0.8, "recall", [last.source], ids, [])

    def _compose(self, text: str, hits, topics, ja: bool, question: bool, qtype: str) -> Reply:
        p = self.params
        if _MORE_RE.match(text.lower()):
            r = self._continuation(ja)
            if r is not None:
                return r
        note = self._guard_note
        if note is not None and not (hits and hits[0][0] >= p.answer_threshold):
            # 知らないことは知らないと言う (それでも知っていることがあれば添える)
            kind, subj, attr = note
            self.stats["honest_unknown"] += 1
            if self.gap_first(subj):            # 知らないと答えた話題を次に調べる (宿題の先頭へ)
                self.stats["unknown_queued"] += 1
            if kind == "attr":
                known = self.facts.lookup(subj)[:1]
                extra = ""
                if known:
                    r, o, _ = known[0]
                    extra = (f" {self.facts._render(subj, r, o, True)}" if ja else f" {self.facts._render(subj, r, o, False)}")
                msg = f"「{subj}の{attr}」はまだ知りません。調べておきます。{extra}" if ja else f"I don't know the {attr} of {subj} yet. I'll look it up.{extra}"
            else:
                msg = f"「{subj}」についてはまだ知りません。調べておきます。" if ja else f"I don't know about '{subj}' yet. I'll look it up."
            return Reply(msg.strip(), 0.1, "generate", [], [], [])
        if hits and hits[0][0] >= p.answer_threshold:
            conf, doc = hits[0]
            answer = doc.text
            nxt = self.kb.docs.get(doc.id + 1)
            same_next = nxt is not None and nxt.source == doc.source
            if same_next and _same_utterance(text, doc.text):
                answer = nxt.text
                doc = nxt
            elif same_next and len(doc.text) < 12 and doc.source in ("seed", "user"):
                answer = f"{doc.text} {nxt.text}"
            qphr = set(phrases(text))
            for c2, d2 in hits[1:3]:
                # 2 文目を足すのは、同じ出典で「隣接する文」か「質問の句を含む文」だけ (話題の混入を防ぐ)
                related = abs(d2.id - doc.id) <= 2 or (qphr and qphr & set(phrases(d2.text)))
                if d2.source == doc.source and doc.source not in ("seed", "chat") and related and c2 >= p.answer_threshold * 0.8 and len(answer) + len(d2.text) < 320:
                    answer = f"{answer} {d2.text}"
                    break
            self.stats["recall"] += 1
            return Reply(answer, round(conf, 3), "recall", [doc.source], [doc.id], [])
        if hits and hits[0][0] >= p.answer_threshold * 0.55:
            conf, doc = hits[0]
            prefix = "たぶんですが、" if ja else "I'm not sure, but: "
            suffix = " (もっと調べておきますね)" if ja else " (I'll look into it more.)"
            self.stats["guess"] += 1
            return Reply(prefix + doc.text + suffix, round(conf, 3), "guess", [doc.source], [doc.id], [])
        # 生成: 話題語を種に、見つかった知識の語を優先しながら LM で続きを作る
        seed_tokens: list[str] = []
        for t in topics:
            ts = tokenize(t)
            if ts and all(self.lm.knows(x) for x in ts):
                seed_tokens = ts
                break
        focus: set[str] = set()
        for _, d in hits[:3]:
            focus.update(tokenize(d.text))
        sentence = self.generate(seed_tokens, focus=focus, query=text, n_candidates=4)
        self.stats["generate"] += 1
        if len(sentence) < 4 or sentence == "".join(seed_tokens):
            if ja:
                msg = "まだよく知りません。" + (f"「{topics[0]}」について調べて学習しておきます。" if topics else "教えてもらえれば覚えます。")
            else:
                msg = "I don't know that yet." + (f" I'll go learn about '{topics[0]}'." if topics else " Tell me and I'll remember.")
            return Reply(msg, 0.05, "generate", [], [], [])
        tail = ("…と思います。" if ja and not sentence.endswith(("。", "！", "？", "!", "?")) else "")
        return Reply(sentence + tail, 0.15, "generate", [], [d.id for _, d in hits[:1]], [])

    # ------------------------------------------------------------ 生成 (接尾辞配列 + n-gram + 会話キャッシュ、自己一貫性)
    def _sample_sentence(self, seed_ids: list[int], focus_ids: set[int], max_len: int = 36, min_len: int = 4) -> list[int]:
        p = self.params
        out = [1] + list(seed_ids)  # BOS
        lm = self.lm
        rng = self.rng
        inv_t = 1.0 / max(p.temperature, 0.05)
        for _ in range(max_len):
            keys = lm._keys(out)
            # 候補集合: 接尾辞配列の最長一致の続き + n-gram の候補
            L, cont = self.suffix.continuations(out[1:], min_ctx=2) if len(self.suffix) else (0, None)
            cands: dict[int, float] = {}
            if cont:
                tot = sum(cont.values())
                for t, c in cont.items():
                    cands[t] = c / tot
            ng = None
            for n in range(len(keys) - 1, -1, -1):
                d = lm.ctx.get(keys[n])
                if d is None:
                    continue
                if type(d) is int:
                    if (d >> 20) >= 2:
                        ng = [d & 0xFFFFF]
                        break
                elif d[-1] >= (2 if n else 1):
                    ng = [t for t in d if t != -1]
                    if len(ng) > 48:
                        ng = sorted(ng, key=lambda t: d[t], reverse=True)[:48]
                    break
            for t in ng or ():
                cands.setdefault(t, 0.0)
            cands.pop(0, None)  # <unk> と <s> は生成しない
            cands.pop(1, None)
            if not cands:
                break
            # 混合: suffix_w × 最長一致分布 (一致が長いほど信頼) + (1-suffix_w) × n-gram + cache_w × キャッシュ
            sw = p.suffix_weight * min(1.0, L / 6.0) if cont else 0.0
            prev = out[-1]
            weights = []
            toks = list(cands)
            recent = out[-8:]
            for t in toks:
                pn = lm._prob_keys(keys, len(keys) - 1, t)
                pc = self.cache_lm.prob(prev, t) if p.cache_weight > 0 else 0.0
                pr = sw * cands[t] + (1 - sw) * ((1 - p.cache_weight) * pn + p.cache_weight * pc)
                if t == EOS and len(out) - 1 < min_len:
                    pr *= 0.05
                if t in recent and t != EOS:
                    pr *= 0.3
                if t in focus_ids:
                    pr *= 1.8
                weights.append(pr ** inv_t)
            tot = sum(weights)
            if tot <= 0:
                break
            r = rng.random() * tot
            acc = 0.0
            pick = toks[-1]
            for t, w in zip(toks, weights):
                acc += w
                if acc >= r:
                    pick = t
                    break
            if pick == EOS:
                break
            out.append(pick)
        return out[1:]

    def _score_candidate(self, ids: list[int], query_phr: set[str], focus_ids: set[int]) -> float:
        """自己一貫性: LM の平均対数確率 + 知識の語との重なり + 質問の句との重なり - 繰り返し。"""
        lm = self.lm
        toks = [1] + ids + [EOS]
        logp = 0.0
        for i in range(1, len(toks)):
            keys = lm._keys(toks[:i])
            logp += math.log(max(lm._prob_keys(keys, len(keys) - 1, toks[i]), 1e-9))
        avg = logp / max(len(toks) - 1, 1)
        words = [lm.words[i] for i in ids]
        text = "".join(words)
        overlap = sum(1 for t in ids if t in focus_ids) / max(len(ids), 1)
        qhit = sum(1 for ph in query_phr if ph in text) / max(len(query_phr), 1)
        rep_pen = 1.0 - len(set(ids)) / max(len(ids), 1)
        ends = 0.3 if text.endswith(("。", "！", "？", ".", "!", "?")) else 0.0
        return avg / 3.0 + 1.5 * overlap + 1.0 * qhit + ends - 2.0 * rep_pen

    def generate(self, seed_tokens: list[str], focus: set[str] | None = None, query: str = "", n_candidates: int = 4, max_len: int = 36) -> str:
        """候補を複数生成して最良を返す (self-consistency)。ニューラル LM が使える段階なら候補にも採点にも加える。"""
        lm = self.lm
        seed_ids = lm.ids(seed_tokens)
        focus_ids = {lm.vocab[t] for t in (focus or ()) if t in lm.vocab}
        qphr = set(phrases(query)) if query else set()
        cands: list[list[int]] = []
        for _ in range(max(1, n_candidates)):
            ids = self._sample_sentence(seed_ids, focus_ids, max_len=max_len)
            if len(ids) > len(seed_ids):
                cands.append(ids)
        if self.neural.ready:
            for _ in range(2):
                text_out = self.neural.continue_text("".join(seed_tokens), max_new=max_len)
                if text_out:
                    cands.append(seed_ids + lm.ids(tokenize(text_out)))
                    self.stats["neural_candidates"] += 1
        best, best_s = None, -1e9
        for ids in cands:
            sc = self._score_candidate(ids, qphr, focus_ids)
            if self.neural.model is not None:
                nlp = self.neural.score(detokenize(lm.words[i] for i in ids))
                if nlp is not None:
                    sc += 0.3 * (nlp / 3.0)  # ニューラル LM の平均対数尤度 (自然さ)
            if sc > best_s:
                best, best_s = ids, sc
        if best is None:
            return ""
        return detokenize(lm.words[i] for i in best).strip()

    def _command(self, text: str, ja: bool) -> Reply | None:
        low = text.lower()
        if low in ("👍", "good", "いいね", "正解", "そう", "yes", "合ってる", "あってる", "ok", "おk"):
            self._train_reranker(positive=True)
            if self._last_pair:
                u, b = self._last_pair
                self.dialogs.add(u, b, source="chat+", weight=3.0)
                if self.neural.model is not None:
                    self.neural.learn_turn(u, b, self._last_pair_ctx, weight=3.0, steps=2)  # 👍: 強く学ぶ
                    self.neural.feedback(True)
                else:
                    self._neural_pending_dialog.append((u, b, self._context_for(self.last_docs), 3.0))
            for d in self.last_docs:
                self.kb.feedback(d, +1.5)
                if self.last_question:
                    # この聞き方でこの文が正解: 質問の語を答えに結び付ける
                    self.kb.associate(d, self.last_question)
                    self.qa_log.append((self.last_question, d, +1))
            if self.last_mode == "guess":
                self.params.answer_threshold = max(0.2, self.params.answer_threshold - 0.02)
            self.stats["feedback_pos"] += 1
            return Reply("ありがとう、覚えておきます。" if ja else "Thanks, noted.", 1.0, "command", [], [], [])
        if low in ("👎", "bad", "違う", "ちがう", "wrong", "no", "間違い", "まちがい"):
            self._train_reranker(positive=False)
            if self._last_pair and self.neural.model is not None:
                u, b = self._last_pair
                self.neural.learn_turn(u, b, self._last_pair_ctx, weight=-1.0, steps=2)  # 👎: unlikelihood でその答えを出しにくく
                self.neural.feedback(False)
                self.stats["unlearned_turns"] += 1
            for d in self.last_docs:
                self.kb.feedback(d, -2.0)
                if self.last_question:
                    self.qa_log.append((self.last_question, d, -1))
            if self.last_mode == "recall":
                self.params.answer_threshold = min(0.9, self.params.answer_threshold + 0.03)
            self.stats["feedback_neg"] += 1
            for k in keywords(self.last_question, limit=2):
                self.add_gap(k)
            self.last_mode = "corrected"
            return Reply("ごめんなさい。正しい答えを教えてくれれば覚えます。" if ja else "Sorry. Tell me the right answer and I'll remember it.", 1.0, "command", [], [], [])
        for prefix in ("覚えて:", "覚えて：", "remember:", "learn:", "学習:", "学習："):
            if low.startswith(prefix):
                return self._teach(text[len(prefix):].strip(), ja)
        for prefix in ("調べて:", "調べて：", "search:", "lookup:"):
            if low.startswith(prefix):
                if self.cfg.web_enabled:
                    return None  # オンラインならエージェントがその場で調べて答える
                topic = text[len(prefix):].strip()
                self.gaps.appendleft(topic)
                self._topic_strategy[topic] = "gap"
                return Reply(f"「{topic}」を次に調べます。" if ja else f"I'll look up '{topic}' next.", 1.0, "command", [], [], [])
        # 👎 の直後の平叙文は「正しい答え」として扱い、直前の質問に結び付ける
        if self.last_mode == "corrected" and not is_question(text) and len(text) >= 6:
            r = self._teach(text, ja)
            self.last_mode = ""
            return r
        return None

    def _train_reranker(self, positive: bool) -> None:
        """直前の答えに使った候補を正例/負例、他の候補を逆側の弱い例として学習。"""
        used = set(self.last_docs)
        for doc_id, x in self._last_features.items():
            if doc_id in used:
                self.reranker.update(x, 1.0 if positive else 0.0)
            elif positive:
                self.reranker.update(x, 0.0)  # 選ばれなかった候補は「より悪かった」
        self.stats["reranker_updates"] += 1

    def _teach(self, body: str, ja: bool) -> Reply:
        n = 0
        first = None
        for s in split_sentences(body) or [body]:
            facts = extract_facts(s)
            d = self.kb.add(s, "user", quality=sentence_quality(s, bool(facts)))
            if d:
                d.score = 3.0
                for subj, rel, obj in facts:
                    if self.facts.add(subj, rel, obj, d.id):
                        self.stats["facts_learned"] += 1
                self.lm.learn(tokenize(s))
                n += 1
                first = first or d
        # 直前の質問への「答え」として教えられた時だけ結び付ける (👎 の直後、または話題語が重なる時)
        if first is not None and self.last_question:
            related = self.last_mode == "corrected" or bool(set(phrases(self.last_question)) & set(phrases(body)))
            if related:
                self.kb.associate(first.id, self.last_question)
                self.qa_log.append((self.last_question, first.id, +1))
        self.stats["taught"] += n
        return Reply(f"覚えました ({n} 文)。" if ja else f"Got it ({n} sentences).", 1.0, "command", [], [], [])

    # ------------------------------------------------------------ 自己評価と進化 (evolution.py に委譲)
    def evaluate(self, seed: int | None = None) -> dict:
        return self.evolution.evaluate(seed)

    def evolve_step(self) -> dict:
        return self.evolution.step()

    @property
    def generation(self) -> int:
        return self.evolution.generation

    @generation.setter
    def generation(self, v: int) -> None:
        self.evolution.generation = v

    @property
    def sigma(self) -> float:
        return self.evolution.sigma

    @sigma.setter
    def sigma(self, v: float) -> None:
        self.evolution.sigma = v

    @property
    def fitness(self):
        return self.evolution.fitness

    @fitness.setter
    def fitness(self, v) -> None:
        self.evolution.fitness = v

    @property
    def fitness_log(self) -> deque:
        return self.evolution.log

    @fitness_log.setter
    def fitness_log(self, v) -> None:
        self.evolution.log = deque(v, maxlen=200)

    # ------------------------------------------------------------ 整理 (人の睡眠中の記憶整理に相当)
    def consolidate(self) -> dict:
        """重複に近い知識の統合、使われない知識の減衰、関心の減衰、LM の軽い剪定。"""
        with self.lock:
            merged = 0
            # 同じ主語の事実が複数の文書から来ている時、古い方の文書を統合対象にする
            seen: dict[tuple[str, str], int] = {}
            victims: set[int] = set()
            for key, lst in list(self.facts.by_subject.items()):
                for rel, obj, doc_id in lst:
                    k = (key, rel)
                    prev = seen.get(k)
                    if prev is None:
                        seen[k] = doc_id
                        continue
                    a, b = self.kb.docs.get(prev), self.kb.docs.get(doc_id)
                    if a is None or b is None:
                        continue
                    # 内容がほぼ同じ (語の Jaccard ≥ 0.7) なら品質の低い方を消す
                    ta, tb = set(terms(a.text)), set(terms(b.text))
                    if ta and tb and len(ta & tb) / len(ta | tb) >= 0.7:
                        loser = a if (a.quality + a.score, a.hits) < (b.quality + b.score, b.hits) else b
                        if loser.source not in ("user", "seed"):
                            victims.add(loser.id)
            for doc_id in victims:
                self.kb.remove(doc_id)
                merged += 1
            # 使われない知識は少しずつ信頼度が下がり、関心も減衰する
            for d in self.kb.docs.values():
                if d.hits == 0 and d.score > -3.0 and d.source not in ("user", "seed"):
                    d.score -= 0.05
            for k in list(self.interest):
                self.interest[k] *= 0.8
                if self.interest[k] < 0.05:
                    del self.interest[k]
            pruned = self.lm.prune(min_count=2) if self.lm.estimated_bytes() > self.guard.budget * 0.4 else 0
            self.background_step(budget_docs=2000)
            self.semantic.refresh_sketches(limit=5000)
            if self._suffix_dirty:
                self.rebuild_suffix()
            self.stats["consolidations"] += 1
            self.stats["merged_docs"] += merged
            log.info("整理: 統合 %d 文, LM 剪定 %d", merged, pruned)
            return {"merged": merged, "pruned_lm": pruned}

    # ------------------------------------------------------------ メモリ
    def _maybe_enforce(self) -> None:
        self._enforce_counter += 1
        if self._enforce_counter % 25 == 0 or self.lm.estimated_bytes() + self.kb.estimated_bytes() > self.guard.budget:
            self.enforce_memory()

    def enforce_memory(self) -> dict:
        with self.lock:
            scale = 0.8 ** self._tighten
            budget = self.guard.budget * scale
            lm_budget = int(budget * 0.45)
            kb_budget = int(budget * 0.30)
            sem_budget = int(budget * 0.10)   # 残り 15% は接尾辞配列 (8 bytes/トークン) と余裕
            removed_lm = self.lm.shrink_to(lm_budget)
            removed_kb = self.kb.shrink_to(kb_budget, self.cfg.max_docs)
            self.semantic.shrink_to(sem_budget, keep_terms=self.interest)
            if removed_lm or removed_kb:
                self.guard.collect()
            live = self.lm.estimated_bytes() + self.kb.estimated_bytes() + self.semantic.estimated_bytes() + 8 * len(self.suffix)
            if self.guard.over_soft() and self._tighten < MAX_TIGHTEN and live > budget * 0.25:
                # 推定が甘かった: 予算を恒久的に締めてもう一度削る (締め過ぎないよう上限あり。
                # Python は解放したメモリを OS に返さないことが多いので RSS は下がらないことがある)
                self._tighten += 1
                removed_lm += self.lm.shrink_to(int(self.lm.estimated_bytes() * 0.7))
                removed_kb += self.kb.shrink_to(int(self.kb.estimated_bytes() * 0.7), self.cfg.max_docs)
                self.guard.collect()
                self.stats["memory_tightened"] += 1
            if removed_lm or removed_kb:
                self.stats["pruned_lm"] += removed_lm
                self.stats["pruned_kb"] += removed_kb
            # メモリが埋まってきたら、質の低い文は最初から取り込まない (選択的学習)
            fill = (self.lm.estimated_bytes() + self.kb.estimated_bytes() + self.semantic.estimated_bytes() + 8 * len(self.suffix)) / max(budget, 1)
            self.admission = 0.0 if fill < 0.6 else min(0.6, (fill - 0.6) * 1.5)
            return {"removed_lm": removed_lm, "removed_kb": removed_kb, "rss_mb": round(self.guard.usage() / MB, 1)}

    # ------------------------------------------------------------ 保存/読込
    def save(self, path: str | Path | None = None) -> Path:
        path = Path(path) if path else self.cfg.data_dir / SAVE_NAME
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock:
            state = {
                "version": 2,
                "lm": self.lm.state(),
                "kb": {"docs": [(d.id, d.text, d.source, d.added, d.score, d.quality, d.hits) for d in self.kb.docs.values()], "next_id": self.kb.next_id, "assoc": self.kb.assoc},
                "facts": self.facts.state(),
                "interest": self.interest,
                "semantic": self.semantic.state(),
                "reranker": self.reranker.state(),
                "dialog_holdout": list(self.dialog_holdout), "dialogs": self.dialogs.state(),
                # 入れ替わる取り置きも保存する。10 分ごとに再開する運用では、これが消えるたびに
                # 物差しが変わり、同じモデルの評価値が動いてしまう (実測: 再開直後に 0.506 → 0.446)。
                "fresh_holdout": [list(x) for x in self._fresh_holdout], "fresh_holdout_step": self._fresh_holdout_step,
                "fresh_shift": self._fresh_shift, "rag_docs": list(self._rag_docs), "rag_docs_step": self._rag_docs_step,
                "agent": self.agent.state(),
                "params": asdict(self.params),
                "generation": self.generation,
                "sigma": self.sigma,
                "fitness": self.fitness,
                "fitness_log": list(self.fitness_log),
                "holdout": self.holdout,
                "gaps": list(self.gaps),
                "explored": self.explored,
                "strategy_stats": self.strategy_stats,
                "qa_log": list(self.qa_log),
                "stats": dict(self.stats),
                "tighten": self._tighten,
                "created": self.created,
            }
            tmp = path.with_suffix(".tmp")
            with gzip.open(tmp, "wb", compresslevel=5) as f:
                pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, path)
            self.stats["saves"] += 1
            try:
                self.neural.save()
            except Exception as e:  # ニューラル LM の保存失敗で本体の保存を止めない
                log.warning("ニューラル LM 保存失敗: %s", e)
        return path

    def load(self, path: str | Path | None = None) -> bool:
        path = Path(path) if path else self.cfg.data_dir / SAVE_NAME
        if not path.exists():
            return False
        with gzip.open(path, "rb") as f:
            state = pickle.load(f)
        if state.get("version", 1) < 2:
            log.warning("古い保存形式 (v1) は読めません。新規に学習し直します: %s", path)
            return False
        with self.lock:
            self.lm = NGramLM.from_state(state["lm"])
            self.kb = KnowledgeBase()
            id_map: dict[int, int] = {}
            for row in state["kb"]["docs"]:
                id_, text, source, added, score = row[:5]
                d = self.kb.add(text, source)
                if d:
                    d.added, d.score = added, score
                    if len(row) >= 7:
                        d.quality, d.hits = row[5], row[6]
                    id_map[id_] = d.id
            self.facts = FactStore()
            for key, lst in state.get("facts", {}).get("by_subject", {}).items():
                for rel, obj, old_id in lst:
                    new_id = id_map.get(old_id)
                    if new_id is not None:
                        self.facts.add(key, rel, obj, new_id)
            for old_id, srcs in state.get("facts", {}).get("extra_sources", {}).items():
                new_id = id_map.get(int(old_id)) if not isinstance(old_id, int) else id_map.get(old_id)
                if new_id is not None:
                    self.facts.extra_sources[new_id] = set(srcs)
            self.kb.on_remove = self.facts.remove_doc
            self.facts.source_of = self._source_of
            self.interest = state.get("interest", {})
            self.semantic = SemanticSpace.from_state(state["semantic"]) if "semantic" in state else SemanticSpace()
            self.reranker = Reranker.from_state(state.get("reranker", {}))
            self.dialog_holdout = [tuple(x) for x in state.get("dialog_holdout", [])]
            self._fresh_holdout = [tuple(x) for x in state.get("fresh_holdout", [])]
            self._fresh_holdout_step = int(state.get("fresh_holdout_step", 0))
            self._fresh_shift = float(state.get("fresh_shift", 0.0))
            self._rag_docs = [int(x) for x in state.get("rag_docs", [])]
            self._rag_docs_step = int(state.get("rag_docs_step", 0))
            self.dialogs = DialogStore.from_state(state.get("dialogs", []), self.cfg.max_dialogs)
            self.agent.load_state(state.get("agent", {}))
            self._semantic_queue = deque(maxlen=50000)
            if not self.semantic.vec:
                self._semantic_queue.extend(self.kb.docs.keys())
            self._suffix_dirty = len(self.kb)  # 接尾辞配列は保存しない (再構築が速い)
            for old_id, extra in state["kb"].get("assoc", {}).items():
                new_id = id_map.get(old_id)
                if new_id is not None:
                    self.kb.associate(new_id, " ".join(extra))
            params = {k: v for k, v in state["params"].items() if k in Params.__dataclass_fields__}
            self.params = Params(**params)
            self._apply_params()
            self.generation = state["generation"]
            self.sigma = state.get("sigma", 1.0)
            self.fitness = state.get("fitness")
            self.fitness_log = deque(state.get("fitness_log", []), maxlen=200)
            self.holdout = state.get("holdout", [])
            self.gaps = deque(state.get("gaps", []), maxlen=200)
            self.explored = state.get("explored", {})
            self.strategy_stats = {s: list(state.get("strategy_stats", {}).get(s, [0, 0.0])) for s in STRATEGIES}
            self.qa_log = deque([(q, id_map.get(d, -1), s) for q, d, s in state.get("qa_log", [])], maxlen=200)
            self.stats = Counter(state.get("stats", {}))
            self._tighten = state.get("tighten", 0)
            self.created = state.get("created", time.time())
            self.enforce_memory()
        return True

    # ------------------------------------------------------------ 状態
    def describe(self) -> dict:
        with self.lock:
            return {
                "generation": self.generation,
                "fitness": self.fitness,
                "sigma": round(self.sigma, 3),
                "params": asdict(self.params),
                "lm": self.lm.stats(),
                "kb": self.kb.stats(),
                "facts": self.facts.stats(),
                "semantic": self.semantic.stats(),
                "suffix": self.suffix.stats(),
                "reranker": {"samples": self.reranker.samples, "w": {k: round(v, 2) for k, v in self.reranker.w.items()}},
                "dialogs": {"pairs": len(self.dialogs), "with_history": self.dialogs.with_history(), "by_source": self.dialogs.by_source()},
                "profile": dict(self.agent.profile),
                "neural": (self.neural.ensure_model({}) and self.neural.stats()) if (self.neural.available and self.neural.model is None and self.neural.path.exists()) else self.neural.stats(),
                "interest": sorted(self.interest.items(), key=lambda x: -x[1])[:8],
                "admission": round(self.admission, 2),
                "memory": self.guard.describe(),
                "holdout": len(self.holdout),
                "gaps": list(self.gaps)[:10],
                "explored": len(self.explored),
                "strategies": {k: {"tries": v[0], "avg_gain": round(v[1] / v[0], 2) if v[0] else None} for k, v in self.strategy_stats.items()},
                "qa_log": len(self.qa_log),
                "stats": dict(self.stats),
                "timers_ms": {k: round(v * 1000, 1) for k, v in self.timers.items()},
                "uptime_h": round((time.time() - self.created) / 3600, 2),
            }

    def describe_json(self) -> str:
        return json.dumps(self.describe(), ensure_ascii=False, indent=2)
