"""Brain: 言語モデル + 知識ベース + 進化パラメータ + メモリ制御をまとめた中核。"""
from __future__ import annotations

import gzip
import json
import logging
import math
import os
import pickle
import random
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
    is_question,
    keywords,
    normalize,
    split_sentences,
    term_weight,
    terms,
    tokenize,
)

log = logging.getLogger("tinyai.brain")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SAVE_NAME = "brain.pkl.gz"
MAX_TIGHTEN = 6  # 予算を締める回数の上限 (0.8^6 ≈ 26%)


# ---------------------------------------------------------------- 進化するパラメータ
@dataclass
class Params:
    use_order: int = 3          # LM で使う n-gram 次数
    discount: float = 0.75      # 絶対ディスカウント
    k1: float = 1.4             # BM25
    b: float = 0.6
    phrase_bonus: float = 1.5
    answer_threshold: float = 0.45  # これ以上の確信度なら知識をそのまま答える
    temperature: float = 0.8

    def mutate(self, rng: random.Random, max_order: int) -> "Params":
        p = replace(self)
        which = rng.choice(["use_order", "discount", "k1", "b", "phrase_bonus", "discount", "k1"])
        if which == "use_order":
            p.use_order = max(2, min(max_order, p.use_order + rng.choice([-1, 1])))
        elif which == "discount":
            p.discount = min(0.98, max(0.3, p.discount + rng.gauss(0, 0.08)))
        elif which == "k1":
            p.k1 = min(3.0, max(0.5, p.k1 + rng.gauss(0, 0.2)))
        elif which == "b":
            p.b = min(1.0, max(0.0, p.b + rng.gauss(0, 0.1)))
        elif which == "phrase_bonus":
            p.phrase_bonus = min(4.0, max(1.0, p.phrase_bonus + rng.gauss(0, 0.3)))
        return p


@dataclass
class Reply:
    text: str
    confidence: float
    mode: str                 # recall / guess / generate / command
    sources: list
    doc_ids: list
    learned_topics: list


def _same_utterance(a: str, b: str) -> bool:
    strip = lambda x: "".join(ch for ch in x.lower() if ch.isalnum())  # noqa: E731
    return strip(a) == strip(b)


def _cjk_ratio(text: str) -> float:
    if not text:
        return 0.0
    cjk = sum(1 for ch in text if "぀" <= ch <= "ヿ" or "一" <= ch <= "鿿")
    return cjk / len(text)


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
        self.holdout: list[list[str]] = []
        self._holdout_counter = 0
        self.gaps: deque = deque(maxlen=200)     # 調べたい話題
        self.explored: dict[str, float] = {}     # 話題 -> 最終探索時刻
        self.stats: Counter = Counter()
        self.history: deque = deque(maxlen=20)
        self.last_docs: list[int] = []
        self.last_mode = ""
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
        """文に分けて LM と知識ベースに取り込む。取り込んだ文数を返す。
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
                # 上限に当たった: 半分に削ってから残りを諦める (次の文で再開できる)
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
        if True:
            for s in sentences:
                toks = tokenize(s)
                if not toks:
                    continue
                self._holdout_counter += 1
                # Web から得た文の一部は自己評価用に取り置く (学習には使わない)
                if len(toks) >= 4 and self._holdout_counter % 40 == 0 and source.startswith("http"):
                    if len(self.holdout) < self.cfg.holdout_size:
                        self.holdout.append(toks)
                    else:
                        self.holdout[self.rng.randrange(len(self.holdout))] = toks
                    continue
                if len(toks) >= 2:
                    self.lm.learn(toks)
                if self.kb.add(s, source):
                    added += 1
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
        """話題を検索して読み、学習する。学習文数を返す。"""
        total = 0
        for src, txt in fetcher.search_and_read(topic, langs=self.cfg.languages):
            n = self.learn_text(txt, source=src)
            total += n
            log.info("学習 %s <- %s (%d 文)", topic, src, n)
        with self.lock:
            self.explored[topic] = time.time()
            self.stats["topics_explored"] += 1
            self.stats["pages_read"] += 1 if total else 0
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

    def next_topic(self) -> str | None:
        with self.lock:
            now = time.time()
            while self.gaps:
                t = self.gaps.popleft()
                if now - self.explored.get(t, 0) > 86400:
                    return t
            r = self.rng.random()
            # 好奇心: 既知の文から話題語を拾って広げる
            if r < 0.5:
                for d in self.kb.random_docs(5, self.rng):
                    for k in keywords(d.text, limit=3):
                        if now - self.explored.get(k, 0) > 86400 * 3 and len(k) >= 2:
                            return k
            # 知識の薄い語
            if r < 0.8:
                for t in self.kb.sparse_terms(10, self.rng, min_len=3):
                    if now - self.explored.get(t, 0) > 86400 * 3:
                        return t
            cands = [t for t in self.seed_topics if now - self.explored.get(t, 0) > 86400 * 7]
            if cands:
                return self.rng.choice(cands)
            return self.rng.choice(self.seed_topics) if self.seed_topics else None

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

            hits = self._search(text, exclude_id=new_doc.id if new_doc else None)
            topics = keywords(text, limit=3)
            reply = self._compose(text, hits, topics, ja, question)
            for t in topics:
                if reply.confidence < 0.7:
                    self.add_gap(t)
            reply.learned_topics = [t for t in topics if t in self.gaps]
            self.last_docs = reply.doc_ids
            self.last_mode = reply.mode
            self.history.append(("ai", reply.text))
            self._maybe_enforce()
            return reply

    def _search(self, text: str, exclude_id: int | None = None, k: int = 5):
        hits = self.kb.search(text, k=k + 1)
        out = []
        q_terms = set(terms(text))
        total_w = sum(term_weight(t) for t in q_terms) or 1.0
        for score, doc in hits:
            if doc.id == exclude_id:
                continue
            matched = sum(term_weight(t) for t in q_terms if doc.id in self.kb.index.get(t, ()))
            cover = matched / total_w
            # 確信度: 語のカバー率とスコアの飽和値
            conf = 0.65 * cover + 0.35 * (1 - math.exp(-score / 6.0))
            out.append((conf, doc))
        out.sort(key=lambda x: -x[0])
        return out[:k]

    def _compose(self, text: str, hits, topics, ja: bool, question: bool) -> Reply:
        p = self.params
        if hits and hits[0][0] >= p.answer_threshold:
            conf, doc = hits[0]
            answer = doc.text
            nxt = self.kb.docs.get(doc.id + 1)
            same_next = nxt is not None and nxt.source == doc.source
            if same_next and _same_utterance(text, doc.text):
                # 「こんにちは」に「こんにちは!」と返すのではなく、続きの文で応える
                answer = nxt.text
                doc = nxt
            elif same_next and len(doc.text) < 12 and doc.source in ("seed", "user"):
                answer = f"{doc.text} {nxt.text}"
            # 同じ出典の続きの文を 1 つ足す
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
        # 生成: 話題語を種にして LM で続きを作る
        seed_tokens: list[str] = []
        for t in topics:
            ts = tokenize(t)
            if ts and all(self.lm.knows(x) for x in ts):
                seed_tokens = ts
                break
        gen = self.lm.generate(seed_tokens, max_len=30, temperature=p.temperature, rng=self.rng)
        sentence = detokenize(gen).strip()
        self.stats["generate"] += 1
        if len(sentence) < 4 or sentence == "".join(seed_tokens):
            if ja:
                msg = "まだよく知りません。" + (f"「{topics[0]}」について調べて学習しておきます。" if topics else "教えてもらえれば覚えます。")
            else:
                msg = "I don't know that yet." + (f" I'll go learn about '{topics[0]}'." if topics else " Tell me and I'll remember.")
            return Reply(msg, 0.05, "generate", [], [], [])
        tail = ("…と思います。" if ja and not sentence.endswith(("。", "！", "？")) else "")
        return Reply(sentence + tail, 0.15, "generate", [], [], [])

    def _command(self, text: str, ja: bool) -> Reply | None:
        low = text.lower()
        if low in ("👍", "good", "いいね", "正解", "そう", "yes", "合ってる", "あってる"):
            for d in self.last_docs:
                self.kb.feedback(d, +1.5)
            if self.last_mode == "guess":
                self.params.answer_threshold = max(0.2, self.params.answer_threshold - 0.02)
            self.stats["feedback_pos"] += 1
            return Reply("ありがとう、覚えておきます。" if ja else "Thanks, noted.", 1.0, "command", [], [], [])
        if low in ("👎", "bad", "違う", "ちがう", "wrong", "no", "間違い", "まちがい"):
            for d in self.last_docs:
                self.kb.feedback(d, -2.0)
            if self.last_mode == "recall":
                self.params.answer_threshold = min(0.9, self.params.answer_threshold + 0.03)
            self.stats["feedback_neg"] += 1
            # 直前の話題を調べ直す
            for role, t in reversed(self.history):
                if role == "user":
                    for k in keywords(t, limit=2):
                        self.add_gap(k)
                    break
            return Reply("ごめんなさい。正しい答えを教えてくれれば覚えます。" if ja else "Sorry. Tell me the right answer and I'll remember it.", 1.0, "command", [], [], [])
        for prefix in ("覚えて:", "覚えて：", "remember:", "learn:", "学習:", "学習："):
            if low.startswith(prefix):
                body = text[len(prefix):].strip()
                n = 0
                for s in split_sentences(body) or [body]:
                    d = self.kb.add(s, "user")
                    if d:
                        d.score = 3.0
                        self.lm.learn(tokenize(s))
                        n += 1
                self.stats["taught"] += n
                return Reply(f"覚えました ({n} 文)。" if ja else f"Got it ({n} sentences).", 1.0, "command", [], [], [])
        for prefix in ("調べて:", "調べて：", "search:", "lookup:"):
            if low.startswith(prefix):
                topic = text[len(prefix):].strip()
                self.gaps.appendleft(topic)
                return Reply(f"「{topic}」を次に調べます。" if ja else f"I'll look up '{topic}' next.", 1.0, "command", [], [], [])
        return None

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

    def evaluate(self, seed: int | None = None) -> dict:
        rng = random.Random(seed if seed is not None else self.rng.random())
        ppl = self.lm.perplexity(self.holdout) if self.holdout else float("nan")
        hit = self._retrieval_selftest(rng)
        f = hit - (0.1 * math.log(ppl) if ppl == ppl else 0.0)
        return {"perplexity": round(ppl, 2) if ppl == ppl else None, "retrieval_hit": round(hit, 3), "fitness": round(f, 4)}

    def evolve_step(self) -> dict:
        """パラメータを 1 つ変異させ、自己評価が上がれば採用する。"""
        with self.lock:
            seed = self.rng.randrange(1 << 30)
            base = self.evaluate(seed)
            old = self.params
            cand = old.mutate(self.rng, self.lm.max_order)
            self.params = cand
            self._apply_params()
            new = self.evaluate(seed)
            accepted = new["fitness"] > base["fitness"] + 1e-6
            if accepted:
                self.generation += 1
                self.fitness = new
                self.stats["evolutions_accepted"] += 1
            else:
                self.params = old
                self._apply_params()
                self.fitness = base
            self.stats["evolutions_tried"] += 1
            rec = {"time": time.time(), "generation": self.generation, "accepted": accepted, "before": base, "after": new, "params": asdict(self.params)}
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
                "version": 1,
                "lm": {"ctx": self.lm.ctx, "entries": self.lm.entries, "sentences": self.lm.sentences, "max_order": self.lm.max_order, "prune_threshold": self.lm.prune_threshold},
                "kb": {"docs": [(d.id, d.text, d.source, d.added, d.score) for d in self.kb.docs.values()], "next_id": self.kb.next_id},
                "params": asdict(self.params),
                "generation": self.generation,
                "fitness": self.fitness,
                "fitness_log": list(self.fitness_log),
                "holdout": self.holdout,
                "gaps": list(self.gaps),
                "explored": self.explored,
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
        with self.lock:
            lm = state["lm"]
            self.lm = NGramLM(max_order=lm["max_order"])
            self.lm.ctx = lm["ctx"]
            self.lm.entries = lm["entries"]
            self.lm.sentences = lm["sentences"]
            self.lm.prune_threshold = lm.get("prune_threshold", 1)
            self.kb = KnowledgeBase()
            for id_, text, source, added, score in state["kb"]["docs"]:
                d = self.kb.add(text, source)
                if d:
                    d.added, d.score = added, score
            self.params = Params(**state["params"])
            self._apply_params()
            self.generation = state["generation"]
            self.fitness = state.get("fitness")
            self.fitness_log = deque(state.get("fitness_log", []), maxlen=200)
            self.holdout = state.get("holdout", [])
            self.gaps = deque(state.get("gaps", []), maxlen=200)
            self.explored = state.get("explored", {})
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
                "params": asdict(self.params),
                "lm": self.lm.stats(),
                "kb": self.kb.stats(),
                "memory": self.guard.describe(),
                "holdout": len(self.holdout),
                "gaps": list(self.gaps)[:10],
                "explored": len(self.explored),
                "stats": dict(self.stats),
                "uptime_h": round((time.time() - self.created) / 3600, 2),
            }

    def describe_json(self) -> str:
        return json.dumps(self.describe(), ensure_ascii=False, indent=2)
