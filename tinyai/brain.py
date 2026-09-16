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
from typing import Iterable

from .config import Config
from .brain_types import question_type, _rerank_bonus  # noqa: F401
from .evolution import Evolution, Params
from .facts import FactStore, extract_facts
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
        self.facts = FactStore()
        self.kb.on_remove = self.facts.remove_doc
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
        added = 0
        # 自己評価用の取り置きは、長い Web 文書からだけ (短い入力は 1 文が重要なので全部学ぶ)
        holdout_ok = source.startswith("http") and len(sentences) >= 40
        kb, lm = self.kb, self.lm
        for s in sentences:
            self._holdout_counter += 1
            if holdout_ok and self._holdout_counter % 40 == 0:
                toks = tokenize(s)
                if len(toks) >= 4:
                    if len(self.holdout) < self.cfg.holdout_size:
                        self.holdout.append(toks)
                    else:
                        self.holdout[self.rng.randrange(len(self.holdout))] = toks
                    continue
            # 知識ベースが受け付けた文 (重複でもジャンクでもない) だけ LM に入れる
            facts = extract_facts(s) if len(s) <= 200 else []
            q = sentence_quality(s, bool(facts))
            if q < self.admission and not facts and source not in ("user", "chat", "seed"):
                self.stats["skipped_low_quality"] += 1
                continue
            doc = kb.add(s, source, quality=q)
            if doc is None:
                continue
            added += 1
            for subj, rel, obj in facts:
                if self.facts.add(subj, rel, obj, doc.id):
                    self.stats["facts_learned"] += 1
            toks = tokenize(s)
            if len(toks) >= 2:
                lm.learn(toks)
            if added % 500 == 0:
                self._maybe_enforce()
        return added

    def learn_batch(self, batch, collector=None) -> int:
        """収集システムの 1 バッチ (複数ページ) を学習し、リンクをフロンティアへ、収穫を報告する。"""
        total = 0
        best: tuple[int, str, str] | None = None
        for src, text, anchors in batch.pages:
            n = self.learn_text(text, source=src)
            total += n
            if collector is not None:
                collector.report(batch.source or src, n)
                if anchors:
                    collector.push_links(anchors, depth=1, base_url=src)
            if n and (best is None or n > best[0]):
                best = (n, src, text)
            log.info("学習 [%s] %s <- %s (%d 文)", batch.kind, batch.topic, src, n)
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

            toks = tokenize(text)
            if toks:
                self.lm.learn(toks)  # 会話の文体を学ぶ
            question = is_question(text)
            new_doc = None
            if not question and len(text) >= 8:
                new_doc = self.kb.add(text, "chat")

            # 文脈: 話題語が無い発話は前の話題を引き継ぐ
            topics = [k for k in keywords(text, limit=3) if is_phrase(k)]
            self._bump_interest(topics)
            query = text
            if not topics and self.last_topics and (question or _FOLLOWUP_RE.match(text.lower())):
                query = text + " " + " ".join(self.last_topics)
                topics = list(self.last_topics)
            qtype = question_type(text)
            reply = self._answer_from_facts(text, ja)
            if reply is None:
                hits = self._search(query, exclude_id=new_doc.id if new_doc else None, qtype=qtype, subject=topics[0] if topics else None)
                reply = self._compose(text, hits, topics, ja, question, qtype)
            reply = self._attach_notices(reply, topics, ja)
            for t in topics:
                if reply.confidence < 0.7 and (not t.isascii() or len(t) >= 4):
                    self.add_gap(t)
            reply.learned_topics = [t for t in topics if t in self.gaps]
            self.last_docs = reply.doc_ids
            self._said.extend(reply.doc_ids)
            self.last_mode = reply.mode
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
        if first is not None and self.last_question:
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
            # メモリが埋まってきたら、質の低い文は最初から取り込まない (選択的学習)
            fill = (self.lm.estimated_bytes() + self.kb.estimated_bytes()) / max(budget, 1)
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
            self.kb.on_remove = self.facts.remove_doc
            self.interest = state.get("interest", {})
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
                "interest": sorted(self.interest.items(), key=lambda x: -x[1])[:8],
                "admission": round(self.admission, 2),
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
