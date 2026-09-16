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
from collections import Counter, deque
from dataclasses import dataclass, asdict, replace
from pathlib import Path

from .config import Config
from .knowledge import KnowledgeBase, Doc
from .lm import NGramLM
from .memory import MemoryGuard, MB
from .tokenizer import (
    detokenize,
    is_phrase,
    is_question,
    keywords,
    normalize,
    split_sentences,
    terms,
    tokenize,
)

log = logging.getLogger("tinyai.brain")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SAVE_NAME = "brain.pkl.gz"
MAX_TIGHTEN = 6  # 予算を締める回数の上限 (0.8^6 ≈ 26%)
STRATEGIES = ("gap", "curiosity", "sparse", "seed")


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

    def mutate(self, rng: random.Random, max_order: int, sigma: float = 1.0) -> "Params":
        p = replace(self)
        which = rng.choice(["use_order", "discount", "k1", "b", "phrase_bonus", "expand_weight", "rerank_weight", "discount", "k1"])
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
        return p


@dataclass
class Reply:
    text: str
    confidence: float
    mode: str                 # recall / guess / generate / command
    sources: list
    doc_ids: list
    learned_topics: list


# ---------------------------------------------------------------- 質問タイプ
_QTYPE_PATTERNS = [
    ("when", re.compile(r"いつ|何年|何月|何日|何世紀|\bwhen\b|what year", re.I)),
    ("where", re.compile(r"どこ|何処|どちら|\bwhere\b", re.I)),
    ("howmany", re.compile(r"いくつ|いくら|何人|何個|何回|何歳|どれくらい|どのくらい|高さ|長さ|広さ|人口|距離|面積|重さ|速さ|how (many|much|tall|long|far|old|big|fast)", re.I)),
    ("who", re.compile(r"誰|だれ|\bwho\b", re.I)),
    ("why", re.compile(r"なぜ|何故|どうして|\bwhy\b", re.I)),
    ("definition", re.compile(r"とは|って何|ってなに|とは何|何ですか|なんですか|何のこと|意味|what is|what are|what's|define|explain|教えて", re.I)),
]
_YEAR_RE = re.compile(r"\d{3,4}\s*年|\d+\s*世紀|\b1\d{3}\b|\b20\d{2}\b|年代|月|日")
_PLACE_RE = re.compile(r"[都道府県市区町村国島湾山川洲]|に位置|にある|located|\bin\b")
_NUMBER_RE = re.compile(r"\d[\d,.]*\s*(m|km|メートル|キロ|人|個|km2|km²|平方|トン|kg|グラム|秒|分|時間|年|歳|倍|%|パーセント|円|ドル|億|万|千)")
_WHO_RE = re.compile(r"氏|さん|博士|教授|作家|学者|者|創業|設立|発明|開発|by\b|founded|invented")


def question_type(text: str) -> str:
    for name, pat in _QTYPE_PATTERNS:
        if pat.search(text):
            return name
    return "definition" if is_question(text) else "none"


def _rerank_bonus(qtype: str, doc_text: str, subject: str | None) -> float:
    """質問タイプに合う文に加点 (0..1)。"""
    b = 0.0
    if qtype == "definition":
        if subject:
            head = doc_text[: len(subject) + 4]
            if head.startswith(subject) or subject in head:
                b += 0.6
            if re.search(re.escape(subject) + r"\s*(とは|は|が|、|は、|とは、|\(|（)", doc_text):
                b += 0.4
        if re.search(r"とは|である|です|のこと|を指す|refers to|is a|is the|は、", doc_text):
            b += 0.2
    elif qtype == "when":
        b += 1.0 if _YEAR_RE.search(doc_text) else 0.0
    elif qtype == "where":
        b += 1.0 if _PLACE_RE.search(doc_text) else 0.0
    elif qtype == "howmany":
        b += 1.0 if _NUMBER_RE.search(doc_text) else (0.3 if re.search(r"\d", doc_text) else 0.0)
    elif qtype == "who":
        b += 0.8 if _WHO_RE.search(doc_text) else 0.0
    elif qtype == "why":
        b += 0.8 if re.search(r"ため|から|ので|理由|原因|because|due to", doc_text) else 0.0
    return min(b, 1.0)


def _same_utterance(a: str, b: str) -> bool:
    strip = lambda x: "".join(ch for ch in x.lower() if ch.isalnum())  # noqa: E731
    return strip(a) == strip(b)


def _cjk_ratio(text: str) -> float:
    if not text:
        return 0.0
    cjk = sum(1 for ch in text if "぀" <= ch <= "ヿ" or "一" <= ch <= "鿿")
    return cjk / len(text)


_MORE_RE = re.compile(r"^(もっと|詳しく|もっと詳しく|続けて|続き|他には|ほかには|それで|それから|more|tell me more|continue|go on|and\??)[。!！?？]*$", re.I)
_FOLLOWUP_RE = re.compile(r"^(それ|これ|あれ|そこ|そいつ|彼|彼女|it|that|this|they|he|she)")


class Brain:
    def __init__(self, config: Config | None = None, rng: random.Random | None = None):
        self.cfg = config or Config()
        self.rng = rng or random.Random(self.cfg.seed)
        self.guard = MemoryGuard(self.cfg.memory_mb, hard=self.cfg.hard_limit)
        self.lock = threading.RLock()
        self.lm = NGramLM(max_order=self.cfg.max_order)
        self.kb = KnowledgeBase()
        self.params = Params(use_order=min(3, self.cfg.max_order))
        self._apply_params()
        self.generation = 0
        self.fitness = None
        self.fitness_log: deque = deque(maxlen=200)
        self.sigma = 1.0                          # 変異幅 (適応)
        self.holdout: list[list[str]] = []
        self._holdout_counter = 0
        self.gaps: deque = deque(maxlen=200)      # 調べたい話題
        self.explored: dict[str, float] = {}      # 話題 -> 最終探索時刻
        self.strategy_stats: dict[str, list] = {s: [0, 0.0] for s in STRATEGIES}  # [試行, 収穫合計]
        self._topic_strategy: dict[str, str] = {}
        self.qa_log: deque = deque(maxlen=200)    # (質問, doc_id, +1/-1)
        self.stats: Counter = Counter()
        self.history: deque = deque(maxlen=20)
        self.last_docs: list[int] = []
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
        added = 0
        from_web = source.startswith("http")
        kb, lm = self.kb, self.lm
        for s in sentences:
            self._holdout_counter += 1
            if from_web and self._holdout_counter % 40 == 0:
                toks = tokenize(s)
                if len(toks) >= 4:
                    if len(self.holdout) < self.cfg.holdout_size:
                        self.holdout.append(toks)
                    else:
                        self.holdout[self.rng.randrange(len(self.holdout))] = toks
                    continue
            # 知識ベースが受け付けた文 (重複でもジャンクでもない) だけ LM に入れる
            if kb.add(s, source) is None:
                continue
            added += 1
            toks = tokenize(s)
            if len(toks) >= 2:
                lm.learn(toks)
            if added % 500 == 0:
                self._maybe_enforce()
        return added

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
    def add_gap(self, topic: str) -> None:
        topic = topic.strip()
        if len(topic) < 2 or topic in self.gaps:
            return
        if time.time() - self.explored.get(topic, 0) < 86400:
            return
        self.gaps.append(topic)

    def _pick_strategy(self) -> str:
        """収穫量に基づく UCB1 で探索戦略を選ぶ。"""
        total = sum(st[0] for st in self.strategy_stats.values()) + 1
        best, best_v = "seed", -1.0
        for name in ("curiosity", "sparse", "seed"):
            tries, gain = self.strategy_stats[name]
            if tries == 0:
                return name
            v = gain / tries + math.sqrt(2 * math.log(total) / tries)
            if v > best_v:
                best, best_v = name, v
        return best

    def _candidates(self, strategy: str, now: float) -> list[str]:
        if strategy == "curiosity":
            out = []
            for d in self.kb.random_docs(5, self.rng):
                for k in keywords(d.text, limit=3):
                    if is_phrase(k) and now - self.explored.get(k, 0) > 86400 * 3:
                        out.append(k)
            return out
        if strategy == "sparse":
            return [t for t in self.kb.sparse_terms(10, self.rng, min_len=3) if now - self.explored.get(t, 0) > 86400 * 3]
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
                rest = [s for s in ("curiosity", "sparse", "seed") if s not in tried]
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

            toks = tokenize(text)
            if toks:
                self.lm.learn(toks)  # 会話の文体を学ぶ
            question = is_question(text)
            new_doc = None
            if not question and len(text) >= 8:
                new_doc = self.kb.add(text, "chat")

            # 文脈: 話題語が無い発話は前の話題を引き継ぐ
            topics = [k for k in keywords(text, limit=3) if is_phrase(k)]
            query = text
            if not topics and self.last_topics and (question or _FOLLOWUP_RE.match(text.lower())):
                query = text + " " + " ".join(self.last_topics)
                topics = list(self.last_topics)
            qtype = question_type(text)
            hits = self._search(query, exclude_id=new_doc.id if new_doc else None, qtype=qtype, subject=topics[0] if topics else None)
            reply = self._compose(text, hits, topics, ja, question, qtype)
            for t in topics:
                if reply.confidence < 0.7:
                    self.add_gap(t)
            reply.learned_topics = [t for t in topics if t in self.gaps]
            self.last_docs = reply.doc_ids
            self.last_mode = reply.mode
            self.last_question = text
            if topics:
                self.last_topics = topics[:2]
            self.history.append(("ai", reply.text))
            self._maybe_enforce()
            return reply

    def _search(self, text: str, exclude_id: int | None = None, k: int = 5, qtype: str = "none", subject: str | None = None):
        p = self.params
        hits = self.kb.search(text, k=k + 1)
        conf = self._confidences(hits, text, exclude_id)
        # 確信が低ければ関連語でクエリを広げてもう一度
        if (not conf or conf[0][0] < p.answer_threshold) and p.expand_weight > 0:
            extra: dict[str, float] = {}
            for t in [t for t in terms(text) if is_phrase(t)][:3]:
                for u, w in self.kb.related_terms(t, k=3):
                    extra[u] = max(extra.get(u, 0.0), min(1.0, w) * p.expand_weight)
            if extra:
                hits2 = self.kb.search(text, k=k + 1, extra=extra)
                conf2 = self._confidences(hits2, text, exclude_id, penalty=0.9)
                seen = {d.id for _, d in conf}
                conf.extend(x for x in conf2 if x[1].id not in seen)
                self.stats["expanded"] += 1
        if qtype != "none" and p.rerank_weight > 0:
            conf = [(min(1.0, c + p.rerank_weight * _rerank_bonus(qtype, d.text, subject)), d) for c, d in conf]
        conf.sort(key=lambda x: -x[0])
        return conf[:k]

    def _confidences(self, hits, text: str, exclude_id: int | None, penalty: float = 1.0):
        out = []
        for score, doc in hits:
            if doc.id == exclude_id:
                continue
            cover = self.kb.coverage(doc.id, text)
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
        for i in range(1, 4):
            nxt = self.kb.docs.get(last.id + i)
            if nxt is None or nxt.source != last.source:
                break
            parts.append(nxt.text)
            ids.append(nxt.id)
            if sum(len(x) for x in parts) > 240:
                break
        if not parts:
            return Reply("それについてはこれ以上知りません。調べておきます。" if ja else "That's all I know about it for now. I'll look it up.", 0.2, "generate", [], [], [])
        self.stats["continued"] += 1
        return Reply(" ".join(parts), 0.8, "recall", [last.source], ids, [])

    def _compose(self, text: str, hits, topics, ja: bool, question: bool, qtype: str) -> Reply:
        p = self.params
        if _MORE_RE.match(text.lower()):
            r = self._continuation(ja)
            if r is not None:
                return r
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
            for c2, d2 in hits[1:3]:
                if d2.source == doc.source and doc.source not in ("seed", "chat") and c2 >= p.answer_threshold * 0.8 and len(answer) + len(d2.text) < 320:
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
        gen = self.lm.generate(seed_tokens, max_len=30, temperature=p.temperature, rng=self.rng, focus=focus)
        sentence = detokenize(gen).strip()
        self.stats["generate"] += 1
        if len(sentence) < 4 or sentence == "".join(seed_tokens):
            if ja:
                msg = "まだよく知りません。" + (f"「{topics[0]}」について調べて学習しておきます。" if topics else "教えてもらえれば覚えます。")
            else:
                msg = "I don't know that yet." + (f" I'll go learn about '{topics[0]}'." if topics else " Tell me and I'll remember.")
            return Reply(msg, 0.05, "generate", [], [], [])
        tail = ("…と思います。" if ja and not sentence.endswith(("。", "！", "？", "!", "?")) else "")
        return Reply(sentence + tail, 0.15, "generate", [], [d.id for _, d in hits[:1]], [])

    def _command(self, text: str, ja: bool) -> Reply | None:
        low = text.lower()
        if low in ("👍", "good", "いいね", "正解", "そう", "yes", "合ってる", "あってる", "ok", "おk"):
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

    def _teach(self, body: str, ja: bool) -> Reply:
        n = 0
        first = None
        for s in split_sentences(body) or [body]:
            d = self.kb.add(s, "user")
            if d:
                d.score = 3.0
                self.lm.learn(tokenize(s))
                n += 1
                first = first or d
        if first is not None and self.last_question:
            self.kb.associate(first.id, self.last_question)
            self.qa_log.append((self.last_question, first.id, +1))
        self.stats["taught"] += n
        return Reply(f"覚えました ({n} 文)。" if ja else f"Got it ({n} sentences).", 1.0, "command", [], [], [])

    # ------------------------------------------------------------ 自己評価と進化
    def _retrieval_selftest(self, rng: random.Random, n: int = 40) -> float:
        docs = self.kb.random_docs(n, rng)
        if not docs:
            return 0.0
        hit = 0
        for d in docs:
            ts = terms(d.text)
            if len(ts) < 2:
                continue
            q = [t for i, t in enumerate(ts) if i % 2 == 0]
            res = self.kb.search(" ".join(q), k=3)
            if any(x.id == d.id for _, x in res):
                hit += 1
        return hit / len(docs)

    def _qa_score(self) -> float:
        """ユーザーの 👍/👎 と教えた答えが、今のパラメータで再現できる割合 (-1..1)。"""
        if not self.qa_log:
            return 0.0
        s = 0.0
        n = 0
        for q, doc_id, sign in list(self.qa_log)[-60:]:
            if doc_id not in self.kb.docs:
                continue
            hits = self._search(q, qtype=question_type(q), subject=(keywords(q, limit=1) or [None])[0], k=3)
            top = hits[0][1].id == doc_id if hits else False
            s += sign if top else -sign * 0.5
            n += 1
        return s / n if n else 0.0

    def evaluate(self, seed: int | None = None) -> dict:
        rng = random.Random(seed if seed is not None else self.rng.random())
        ppl = self.lm.perplexity(self.holdout) if self.holdout else float("nan")
        hit = self._retrieval_selftest(rng)
        qa = self._qa_score()
        f = hit + 0.5 * qa - (0.1 * math.log(ppl) if ppl == ppl else 0.0)
        return {"perplexity": round(ppl, 2) if ppl == ppl else None, "retrieval_hit": round(hit, 3), "qa": round(qa, 3), "fitness": round(f, 4)}

    def evolve_step(self) -> dict:
        """パラメータを 1 つ変異させ、自己評価が上がれば採用する。変異幅は適応する。"""
        with self.lock:
            seed = self.rng.randrange(1 << 30)
            base = self.evaluate(seed)
            old = self.params
            cand = old.mutate(self.rng, self.lm.max_order, self.sigma)
            self.params = cand
            self._apply_params()
            new = self.evaluate(seed)
            accepted = new["fitness"] > base["fitness"] + 1e-6
            if accepted:
                self.generation += 1
                self.fitness = new
                self.sigma = min(3.0, self.sigma * 1.5)
                self.stats["evolutions_accepted"] += 1
            else:
                self.params = old
                self._apply_params()
                self.fitness = base
                self.sigma = max(0.2, self.sigma * 0.9)
            self.stats["evolutions_tried"] += 1
            rec = {"time": time.time(), "generation": self.generation, "accepted": accepted, "before": base, "after": new, "params": asdict(self.params), "sigma": round(self.sigma, 3)}
            self.fitness_log.append(rec)
            log.info("進化 gen=%d accepted=%s %s -> %s", self.generation, accepted, base, new)
            return rec

    # ------------------------------------------------------------ メモリ
    def _maybe_enforce(self) -> None:
        self._enforce_counter += 1
        if self._enforce_counter % 25 == 0 or self.lm.estimated_bytes() + self.kb.estimated_bytes() > self.guard.budget:
            self.enforce_memory()

    def enforce_memory(self) -> dict:
        with self.lock:
            scale = 0.8 ** self._tighten
            budget = self.guard.budget * scale
            lm_budget = int(budget * 0.55)
            kb_budget = int(budget * 0.35)
            removed_lm = self.lm.shrink_to(lm_budget)
            removed_kb = self.kb.shrink_to(kb_budget, self.cfg.max_docs)
            if removed_lm or removed_kb:
                self.guard.collect()
            live = self.lm.estimated_bytes() + self.kb.estimated_bytes()
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
            return {"removed_lm": removed_lm, "removed_kb": removed_kb, "rss_mb": round(self.guard.usage() / MB, 1)}

    # ------------------------------------------------------------ 保存/読込
    def save(self, path: str | Path | None = None) -> Path:
        path = Path(path) if path else self.cfg.data_dir / SAVE_NAME
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock:
            state = {
                "version": 2,
                "lm": self.lm.state(),
                "kb": {"docs": [(d.id, d.text, d.source, d.added, d.score) for d in self.kb.docs.values()], "next_id": self.kb.next_id, "assoc": self.kb.assoc},
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
            for id_, text, source, added, score in state["kb"]["docs"]:
                d = self.kb.add(text, source)
                if d:
                    d.added, d.score = added, score
                    id_map[id_] = d.id
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
                "memory": self.guard.describe(),
                "holdout": len(self.holdout),
                "gaps": list(self.gaps)[:10],
                "explored": len(self.explored),
                "strategies": {k: {"tries": v[0], "avg_gain": round(v[1] / v[0], 2) if v[0] else None} for k, v in self.strategy_stats.items()},
                "qa_log": len(self.qa_log),
                "stats": dict(self.stats),
                "uptime_h": round((time.time() - self.created) / 3600, 2),
            }

    def describe_json(self) -> str:
        return json.dumps(self.describe(), ensure_ascii=False, indent=2)
