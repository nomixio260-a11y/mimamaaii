"""学習データの自動収集システム (プラグイン登録制)。

* ソースは `Source` (名前・言語・search(topic, n) -> pages) として登録し、
  成功率 × 新規性収穫 で健全性スコアを持つ。健全な順に試し、足りたら止める
* 広く多様に: Wikipedia (Core API / Action API)、Wiktionary (定義)、Wikinews (鮮度)、Wikibooks、
  Wikidata (説明文)、DuckDuckGo (一般 Web)、青空文庫 (文学)、Project Gutenberg (英語文学)、
  RSS/Atom フィード、Wikipedia ランダム記事、登録サイトの巡回、フロンティア (リンク追跡)
* 収穫は文数だけでなく「新しく覚えた語の数」(新規性) で測る → 既知の話題ばかり読まない
* URL 重複排除、robots.txt と 429 バックオフ、先読みワーカーで通信と学習を並列化
"""
from __future__ import annotations

import csv
import hashlib
import heapq
import io
import logging
import queue
import random
import re
import threading
import time
import urllib.parse
import zipfile
from pathlib import Path
from typing import Callable, Iterable

from .dialog import extract_quote_pairs
from .web import Fetcher, Page, html_to_text

log = logging.getLogger("tinyai.collector")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
Pages = list  # [(source_url, text, anchors)]


class Batch:
    """1 回の収集結果。pages は [(source_url, text, anchors)]、dialogs は [(発話, 応答)]。"""
    __slots__ = ("topic", "kind", "pages", "source", "elapsed", "dialogs")

    def __init__(self, topic: str, kind: str, pages: list, source: str, elapsed: float, dialogs: list | None = None):
        self.topic, self.kind, self.pages, self.source, self.elapsed = topic, kind, pages, source, elapsed
        self.dialogs = dialogs or []


class SourceHealth:
    __slots__ = ("tries", "ok", "gain", "novelty", "latency", "surprise", "surprise_n")

    def __init__(self):
        self.tries = 0
        self.ok = 0
        self.gain = 0.0
        self.novelty = 0.0
        self.latency = 0.0
        self.surprise = 0.0     # ニューラル LM から見た情報量 (平均トークン損失 nat) の移動平均
        self.surprise_n = 0

    def record(self, ok: bool, gain: float, latency: float) -> None:
        self.tries += 1
        if ok:
            self.ok += 1
        self.gain += gain
        self.latency = 0.8 * self.latency + 0.2 * latency if self.latency else latency

    @property
    def value(self) -> float:
        """データの学習価値 (0〜1): 驚きが 2〜6 nat のとき高い。低すぎる = 既知、高すぎる = ジャンク/別言語。"""
        if self.surprise_n == 0:
            return 0.5
        s = self.surprise
        if s < 1.0:
            return 0.2
        if s <= 6.0:
            return 1.0 if 2.0 <= s <= 5.0 else 0.7
        return 0.3

    @property
    def score(self) -> float:
        # 成功率 × (文の収穫 + 新規性) × 学習価値。未試行の源は楽観的な高い点 (探索) を与える
        if self.tries == 0 and self.gain == 0 and self.novelty == 0:
            return 0.6   # 未試行: 失敗続きの源 (≈0.25) より高く、実績のある源より低い
        t = max(self.tries, 1)
        return (self.ok + 1) / (self.tries + 2) * (self.gain / t + self.novelty / t + 0.5) * (0.5 + self.value)

    def to_dict(self) -> dict:
        t = max(self.tries, 1)
        return {"tries": self.tries, "ok": self.ok, "avg_gain": round(self.gain / t, 2), "avg_novelty": round(self.novelty / t, 2), "latency": round(self.latency, 2),
                "surprise": round(self.surprise, 2), "value": round(self.value, 2), "score": round(self.score, 3)}


class Source:
    """収集ソースの共通型。search(topic, n) は [(url, text, anchors)] を返す。"""

    name = "base"
    kind = "topic"       # topic (話題検索) / stream (話題に依らない供給)
    weight = 1.0         # 供給源としての選ばれやすさ

    def __init__(self, col: "Collector", lang: str = "ja"):
        self.col = col
        self.lang = lang
        self.f = col.fetcher

    def search(self, topic: str, n: int) -> Pages:
        return []

    def stream(self) -> Pages:
        """話題に依らず新しいページを供給する (ランダム記事、文学など)。"""
        return []


# ---------------------------------------------------------------- Wikimedia 系
class WikimediaCore(Source):
    name = "wikimedia"

    def search(self, topic, n):
        out = []
        for key in self.f.wikimedia_search(topic, lang=self.lang, limit=n):
            url = f"https://{self.lang}.wikipedia.org/wiki/{urllib.parse.quote(key)}"
            if not self.col.mark_seen(url):
                continue
            page = self.f.wikimedia_page(key, lang=self.lang)
            if page and len(page.text) > 200:
                out.append((url, page.text, page.anchors))
            if len(out) >= n:
                break
        return out


class MediaWikiAction(Source):
    """Action API を持つ Wikimedia プロジェクト共通 (wikipedia / wiktionary / wikinews / wikibooks)。"""
    project = "wikipedia"
    name = "wikipedia"
    min_chars = 200

    def _host(self) -> str:
        return f"{self.lang}.{self.project}.org"

    def search(self, topic, n):
        f = self.f
        q = urllib.parse.quote(topic)
        js = f.get_json(f"https://{self._host()}/w/api.php?action=query&list=search&srsearch={q}&format=json&srlimit={n}&utf8=1")
        out = []
        for hit in (js or {}).get("query", {}).get("search", []):
            title = hit["title"]
            url = f"https://{self._host()}/wiki/{urllib.parse.quote(title)}"
            if not self.col.mark_seen(url):
                continue
            t = urllib.parse.quote(title)
            js2 = f.get_json(f"https://{self._host()}/w/api.php?action=query&prop=extracts&explaintext=1&titles={t}&format=json&utf8=1&exchars=8000")
            txt = ""
            for page in (js2 or {}).get("query", {}).get("pages", {}).values():
                txt = re.sub(r"\n=+[^=\n]+=+\n", "\n", page.get("extract", ""))
            if len(txt) >= self.min_chars:
                out.append((url, txt, []))
            if len(out) >= n:
                break
        return out


class Wiktionary(MediaWikiAction):
    project = "wiktionary"
    name = "wiktionary"
    min_chars = 40

    def search(self, topic, n):
        # 辞書は見出し語そのものを引く。「X: 定義」を「Xとは、定義である。」に整える
        pages = super().search(topic, min(n, 1))
        out = []
        for url, txt, _ in pages:
            # 発音記号や語形変化の行を除き、語義らしい行だけ残す
            lines = [ln.strip() for ln in txt.splitlines()
                     if 6 <= len(ln.strip()) <= 200 and "IPA" not in ln and not ln.strip().startswith(("(", "（", "[", "＊", "*"))]
            body = "\n".join(lines[:12])
            # 見出し語の行 (「宇宙 (うちゅう)」) ではなく最初の語義らしい行を定義文にする
            defs = [ln for ln in lines if len(ln) >= 12 and not ln.startswith(topic) and "語源" not in ln[:3]]
            if body:
                head = f"{topic}とは、{defs[0].rstrip('。')}である。\n" if defs and self.lang == "ja" else ""
                out.append((url, head + body, []))
        return out


class Wikinews(MediaWikiAction):
    project = "wikinews"
    name = "wikinews"
    weight = 0.6


class Wikibooks(MediaWikiAction):
    project = "wikibooks"
    name = "wikibooks"
    weight = 0.5


class Wikidata(Source):
    name = "wikidata"

    def search(self, topic, n):
        q = urllib.parse.quote(topic)
        js = self.f.get_json(f"https://www.wikidata.org/w/api.php?action=wbsearchentities&search={q}&language={self.lang}&uselang={self.lang}&format=json&limit=5")
        lines = []
        for item in (js or {}).get("search", []):
            label = item.get("label", "").strip()
            desc = item.get("description", "").strip()
            if label and len(desc) > 3:
                lines.append(f"{label}とは、{desc}である。" if self.lang == "ja" else f"{label} is a {desc}.")
        if not lines:
            return []
        url = f"https://www.wikidata.org/wiki/Special:Search?search={q}"
        return [(url, "\n".join(lines), [])] if self.col.mark_seen(url) else []


class DuckDuckGo(Source):
    name = "duckduckgo"
    weight = 0.5

    def search(self, topic, n):
        out = []
        for url in self.f.duckduckgo_search(topic, limit=n + 2):
            if not self.col.mark_seen(url):
                continue
            page = self.f.get_page(url)
            if page and len(page.text) > 300:
                out.append((url, page.text, [(u, a) for u, a in page.anchors if u.startswith(("http", "/"))]))
            if len(out) >= n:
                break
        return out


class WikipediaRandom(Source):
    name = "random"
    kind = "stream"
    weight = 0.8

    def stream(self):
        js = self.f.get_json(f"https://{self.lang}.wikipedia.org/w/api.php?action=query&list=random&rnnamespace=0&rnlimit=1&format=json")
        titles = [r["title"] for r in (js or {}).get("query", {}).get("random", [])]
        if not titles:
            return []
        url = f"https://{self.lang}.wikipedia.org/wiki/{urllib.parse.quote(titles[0])}"
        if not self.col.mark_seen(url):
            return []
        page = self.f.wikimedia_page(titles[0], lang=self.lang)
        return [(url, page.text, page.anchors)] if page and len(page.text) > 200 else []


# ---------------------------------------------------------------- 文学 (文体の多様性)
class Aozora(Source):
    """青空文庫: 著作権切れの日本語文学。索引 (CSV zip) を 1 度だけ取り、ランダムな作品の XHTML を読む。"""
    name = "aozora"
    kind = "stream"
    weight = 0.7
    INDEX = "https://www.aozora.gr.jp/index_pages/list_person_all_extended_utf8.zip"

    def __init__(self, col, lang="ja"):
        super().__init__(col, lang)
        self._works: list[tuple[str, str, str]] | None = None  # (title, author, xhtml url)

    def _load_index(self) -> None:
        self._works = []
        data = self.f.get(self.INDEX, max_bytes=12 * 1024 * 1024) if self.f else None
        if not data:
            return
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                raw = z.read(z.namelist()[0]).decode("utf-8", "replace")
        except (zipfile.BadZipFile, IndexError):
            return
        rows = csv.reader(io.StringIO(raw))
        header = next(rows, None)
        if not header:
            return
        idx = {h.lstrip("﻿"): i for i, h in enumerate(header)}
        try:
            it, ia, iu, ic, ik = idx["作品名"], idx["姓"], idx["XHTML/HTMLファイルURL"], idx["作品著作権フラグ"], idx["文字遣い種別"]
        except KeyError:
            return
        for r in rows:
            if len(r) <= iu or r[ic] != "なし" or "新字新仮名" not in r[ik] or not r[iu].startswith("http"):
                continue
            self._works.append((r[it], r[ia], r[iu]))
        log.info("青空文庫の索引: %d 作品", len(self._works))

    def stream(self):
        if self._works is None:
            self._load_index()
        if not self._works:
            return []
        title, author, url = random.choice(self._works)
        if not self.col.mark_seen(url):
            return []
        data = self.f.get(url)
        if not data:
            return []
        html = data.decode("cp932", "replace") if b"Shift_JIS" in data[:600] or b"shift_jis" in data[:600] else data.decode("utf-8", "replace")
        html = re.sub(r"<rt>.*?</rt>|<rp>.*?</rp>", "", html, flags=re.S)  # ルビの読みを除く
        text = html_to_text(html)[0]
        if len(text) < 300:
            return []
        self.last_dialogs = extract_quote_pairs(text)  # 会話文の応酬 (会話データ)
        return [(url, f"『{title}』（{author}）\n" + text[:60000], [])]


class HuggingFaceDatasets(Source):
    """Hugging Face datasets-server (公開データセットの行を JSON で返す API) から対話/指示データを読む。
    data/datasets.txt に「dataset<TAB>config<TAB>split<TAB>形式」を書く。形式は
    conversations (from/value の配列) / instruction (instruction, input, output) / qa (question, answer) / text (text)。"""
    name = "hfdatasets"
    kind = "stream"
    weight = 1.5      # 会話・指示・読解データは希少なので少し優先
    PAGE = 100

    def __init__(self, col, lang="ja"):
        super().__init__(col, lang)
        self.offsets: dict[str, int] = {}
        self.last_dialogs: list = []

    def _specs(self) -> list[tuple[str, str, str, str]]:
        out = []
        for line in self.col._list_lines("datasets.txt"):
            parts = [x.strip() for x in line.split("\t")]
            if len(parts) >= 4 and not parts[0].startswith("#"):
                out.append((parts[0], parts[1], parts[2], parts[3]))
        return out

    @staticmethod
    def _pairs_from_row(row: dict, fmt: str) -> list[tuple]:
        """行から (発話, 応答) または (発話, 応答, 文脈) を作る。
        conversations: from/value or role/content の配列 (複数ターンは直前の応答を文脈にする)
        instruction: instruction, input, output / qa: question, answer / squad: question, context, answers.text[0]"""
        pairs = []
        if fmt == "conversations":
            conv = row.get("conversations") or row.get("messages") or []
            prev = None
            for m in conv:
                role = (m.get("from") or m.get("role") or "").lower()
                val = (m.get("value") or m.get("content") or "").strip()
                if role in ("human", "user", "prompter"):
                    prev = val
                elif role in ("gpt", "assistant", "bot") and prev:
                    last_bot = pairs[-1][1] if pairs else None
                    pairs.append((prev, val, last_bot[:200]) if last_bot else (prev, val))
                    prev = None
        elif fmt == "instruction":
            q = (row.get("instruction") or "").strip()
            inp = (row.get("input") or "").strip()
            a = (row.get("output") or row.get("response") or "").strip()
            if q and a:
                pairs.append((f"{q}\n{inp}" if inp else q, a))
        elif fmt == "qa":
            q = (row.get("question") or row.get("title") or "").strip()
            a = (row.get("answer") or row.get("answers") or "").strip() if isinstance(row.get("answer") or row.get("answers"), str) else ""
            if q and a:
                pairs.append((q, a))
        elif fmt == "preference":   # 選好データ: 採用された応答は正例、不採用の応答は負例 (unlikelihood)
            conv = row.get("conversations") or []
            last_user = ""
            for m in conv:
                if (m.get("from") or m.get("role") or "").lower() in ("human", "user", "prompter"):
                    last_user = (m.get("value") or m.get("content") or "").strip()
            chosen = (row.get("chosen") or "").strip()
            rejected = (row.get("rejected") or "").strip()
            if last_user and chosen:
                pairs.append((last_user, chosen))
                if rejected and rejected != chosen:
                    pairs.append((last_user, rejected, None, -1.0))
        elif fmt == "squad":   # 読解: 文脈の中から答える練習 (RAG と同じ形)
            q = (row.get("question") or "").strip()
            ctx = (row.get("context") or "").strip()
            ans = row.get("answers") or {}
            texts = ans.get("text") if isinstance(ans, dict) else ans
            a = (texts[0] if isinstance(texts, list) and texts else texts if isinstance(texts, str) else "").strip()
            if q and a and ctx:
                # 答えを含む文を中心に文脈を切り出す (最大 240 文字)
                i = ctx.find(a)
                lo = max(0, i - 100) if i >= 0 else 0
                pairs.append((q, a + ("" if a.endswith(("。", ".")) else "。"), ctx[lo : lo + 240]))
        return pairs

    def stream(self):
        specs = self._specs()
        if not specs:
            return []
        ds, cfg, split, fmt = random.choice(specs)
        key = f"{ds}/{cfg}/{split}"
        if key not in self.offsets:
            # 起動ごとに同じ先頭の行を読まないよう、総行数を聞いてランダムな位置から始める
            js0 = self.f.get_json(f"https://datasets-server.huggingface.co/rows?dataset={urllib.parse.quote(ds, safe='')}&config={urllib.parse.quote(cfg)}&split={split}&offset=0&length=1")
            total = int((js0 or {}).get("num_rows_total") or 0)
            self.offsets[key] = random.randrange(0, total - self.PAGE) if total > self.PAGE * 2 else 0
        off = self.offsets.get(key, 0)
        url = (f"https://datasets-server.huggingface.co/rows?dataset={urllib.parse.quote(ds, safe='')}&config={urllib.parse.quote(cfg)}"
               f"&split={split}&offset={off}&length={self.PAGE}")
        js = self.f.get_json(url)
        rows = (js or {}).get("rows", [])
        if not rows:
            self.offsets[key] = 0  # 末尾まで来たら最初から
            return []
        self.offsets[key] = off + len(rows)
        pairs = []
        texts = []
        for r in rows:
            row = r.get("row", {})
            if fmt == "text":
                t = (row.get("text") or "").strip()
                if len(t) > 40:
                    texts.append(t)
                continue
            pairs.extend(self._pairs_from_row(row, fmt))
        self.last_dialogs = pairs
        # 応答文は知識としても学ぶ (説明文であることが多い)
        body = "\n".join(texts) + "\n" + "\n".join(p[1] for p in pairs if len(p[1]) >= 20 and not (len(p) > 3 and p[3] < 0)) + "\n" + "\n".join(p[2] for p in pairs if len(p) > 2 and p[2] and fmt == "squad")
        src = f"hf:{ds}#{off}"
        return [(src, body, [])] if len(body) > 40 else []


class StackExchange(Source):
    """Stack Exchange API: 質問 → 採用回答 のペア (CC BY-SA)。認証なしで 1 日 300 リクエスト。"""
    name = "stackexchange"
    kind = "stream"
    weight = 0.6

    def __init__(self, col, lang="ja"):
        super().__init__(col, lang)
        self.page = 1
        self.last_dialogs: list = []

    def stream(self):
        site = "ja.stackoverflow" if self.lang == "ja" else "stackoverflow"
        url = (f"https://api.stackexchange.com/2.3/questions?order=desc&sort=votes&site={site}&pagesize=20&page={self.page}"
               f"&filter=withbody")
        js = self.f.get_json(url)
        items = (js or {}).get("items", [])
        if not items:
            self.page = 1
            return []
        self.page += 1
        qs = [q for q in items if q.get("accepted_answer_id")]
        if not qs:
            return []
        ids = ";".join(str(q["accepted_answer_id"]) for q in qs[:10])
        ans = self.f.get_json(f"https://api.stackexchange.com/2.3/answers/{ids}?site={site}&filter=withbody")
        body_by_id = {a["answer_id"]: a.get("body", "") for a in (ans or {}).get("items", [])}
        pairs = []
        texts = []
        for q in qs[:10]:
            a_html = body_by_id.get(q["accepted_answer_id"])
            if not a_html:
                continue
            q_text = html_to_text(q.get("title", "") + "\n" + q.get("body", ""))[0]
            a_text = html_to_text(a_html)[0]
            if q_text and a_text:
                pairs.append((q_text[:300], a_text[:600]))
                texts.append(a_text)
        self.last_dialogs = pairs
        src = f"stackexchange:{site}#p{self.page - 1}"
        return [(src, "\n".join(texts), [])] if texts else []


class Gutenberg(Source):
    """Project Gutenberg: 英語の公有文学。ランダムな ID の平文を読む (無い ID は飛ばす)。"""
    name = "gutenberg"
    kind = "stream"
    weight = 0.3

    def stream(self):
        if self.lang != "en":
            return []
        for _ in range(3):
            bid = random.randint(1, 70000)
            url = f"https://www.gutenberg.org/cache/epub/{bid}/pg{bid}.txt"
            if not self.col.mark_seen(url):
                continue
            data = self.f.get(url)
            if not data:
                continue
            text = data.decode("utf-8", "replace")
            m = re.search(r"\*\*\* START OF [^\n]*\*\*\*(.*?)\*\*\* END OF", text, re.S)
            body = (m.group(1) if m else text)[:80000]
            if len(body) > 1000:
                return [(url, body, [])]
        return []


# ---------------------------------------------------------------- 収集システム
class Collector:
    def __init__(self, fetcher: Fetcher | None, data_dir: Path, languages: Iterable[str] = ("ja", "en"), interest: Callable[[str], float] | None = None, prefetch: int = 3, workers: int = 2):
        self.fetcher = fetcher
        self.data_dir = Path(data_dir)
        self.languages = tuple(languages)
        self.interest = interest or (lambda text: 0.0)
        self.health: dict[str, SourceHealth] = {}
        self.frontier: list[tuple[float, int, str, str, int]] = []
        self._seq = 0
        self.seen: set[bytes] = set()
        self.feed_last: dict[str, float] = {}
        self.ready: queue.Queue = queue.Queue(maxsize=max(1, prefetch))
        self.topic_fn: Callable[[], str | None] | None = None
        self._stop = threading.Event()
        self._poke = threading.Event()
        self._threads: list[threading.Thread] = []
        self.workers = workers
        self.lock = threading.RLock()  # pop_link -> mark_seen で再入する
        self.stats = {"batches": 0, "pages": 0, "frontier_reads": 0, "feed_items": 0, "stream_reads": 0}
        self.sources: dict[str, list[Source]] = {}   # lang -> 話題ソース
        self.streams: list[Source] = []               # 話題に依らない供給源
        if fetcher is not None:
            for lang in self.languages:
                self.sources[lang] = [WikimediaCore(self, lang), MediaWikiAction(self, lang), Wiktionary(self, lang), Wikidata(self, lang), Wikinews(self, lang), Wikibooks(self, lang), DuckDuckGo(self, lang)]
                self.streams.append(WikipediaRandom(self, lang))
                self.streams.append(Aozora(self, lang) if lang == "ja" else Gutenberg(self, lang))
                self.streams.append(HuggingFaceDatasets(self, lang))
                self.streams.append(StackExchange(self, lang))

    def register(self, source: Source, lang: str | None = None) -> None:
        """独自ソースの追加 (kind='stream' なら供給源、それ以外は話題ソース)。"""
        if source.kind == "stream":
            self.streams.append(source)
        else:
            self.sources.setdefault(lang or source.lang, []).append(source)

    # ------------------------------------------------------------ 補助
    def _h(self, name: str) -> SourceHealth:
        h = self.health.get(name)
        if h is None:
            h = self.health[name] = SourceHealth()
        return h

    def mark_seen(self, url: str) -> bool:
        """初見なら True。"""
        key = hashlib.blake2b(url.encode("utf-8"), digest_size=8).digest()
        with self.lock:
            if key in self.seen:
                return False
            if len(self.seen) > 60000:
                self.seen.clear()
            self.seen.add(key)
            return True

    _mark_seen = mark_seen

    def report(self, source: str, learned: int, novelty: int = 0, surprise: float | None = None) -> None:
        """学習側からの収穫報告 (文数、新しい語の数、ニューラル LM の驚き = 情報量)。"""
        h = self._h(source)
        h.gain += min(learned, 400) / 100.0
        h.novelty += min(novelty, 2000) / 200.0
        if surprise is not None:
            h.surprise = surprise if h.surprise_n == 0 else 0.8 * h.surprise + 0.2 * surprise
            h.surprise_n += 1

    # ------------------------------------------------------------ フロンティア
    def push_links(self, anchors: Iterable[tuple[str, str]], depth: int = 1, base_url: str = "") -> int:
        n = 0
        with self.lock:
            for link, anchor in anchors:
                if depth > 3 or not anchor or len(anchor) > 60:
                    continue
                if link.startswith("wiki:"):
                    target = link
                else:
                    target = urllib.parse.urljoin(base_url, link.split("#", 1)[0])
                    if not target.startswith(("http://", "https://")):
                        continue
                pri = self.interest(anchor) - 0.3 * depth
                if pri <= -0.9:
                    continue
                self._seq += 1
                heapq.heappush(self.frontier, (-pri, self._seq, target, anchor, depth))
                n += 1
            if len(self.frontier) > 800:
                self.frontier = heapq.nsmallest(500, self.frontier)
                heapq.heapify(self.frontier)
        return n

    def pop_link(self) -> tuple[str, str, int] | None:
        with self.lock:
            while self.frontier:
                _, _, target, anchor, depth = heapq.heappop(self.frontier)
                if self.mark_seen(target):
                    return target, anchor, depth
        return None

    # ------------------------------------------------------------ 収集
    def collect(self, topic: str, max_pages: int = 3) -> Batch:
        """話題を検索して読む。健全なソースから順に試し、足りた時点で止める。"""
        t0 = time.time()
        if self.fetcher is None:
            return Batch(topic, "topic", [], "", 0.0)
        langs = ([l for l in self.languages if l == "ja"] or list(self.languages)[:1]) if not topic.isascii() else list(self.languages)
        pages: list = []
        used = ""
        for lang in langs:
            srcs = sorted(self.sources.get(lang, []), key=lambda s: -(self._h(f"{s.name}:{lang}").score * s.weight))
            for src in srcs:
                if pages and src.name in ("duckduckgo", "wikibooks", "wikinews"):
                    continue  # 補助ソースは主ソースが空振りした時だけ
                if src.name in ("wikidata", "wiktionary") and len(pages) >= max_pages - 1:
                    continue
                name = f"{src.name}:{lang}"
                st = time.time()
                try:
                    got = src.search(topic, max_pages - len(pages))
                except Exception as e:  # ソース単位の失敗は記録して次へ
                    log.info("ソース失敗 %s: %s", name, e)
                    got = []
                self._h(name).record(bool(got), 0.0, time.time() - st)
                if got:
                    pages.extend(got)
                    used = used or name
                if len(pages) >= max_pages:
                    break
            if len(pages) >= max_pages:
                break
        self.stats["batches"] += 1
        self.stats["pages"] += len(pages)
        return Batch(topic, "topic", pages, used, time.time() - t0)

    def collect_stream(self) -> Batch | None:
        """話題に依らない供給源 (ランダム記事・文学) から 1 つ。健全性 × 重み で選ぶ。"""
        if not self.streams:
            return None
        primary = self.languages[0] if self.languages else "ja"
        # 第 1 言語を優先 (小さなモデルの容量を分散させない)。第 2 言語以降は半分の重み
        weights = [max(0.05, self._h(f"{s.name}:{s.lang}").score * s.weight * (1.0 if s.lang == primary else 0.4)) for s in self.streams]
        src = random.choices(self.streams, weights=weights)[0]
        name = f"{src.name}:{src.lang}"
        t0 = time.time()
        try:
            pages = src.stream()
        except Exception as e:
            log.info("供給源失敗 %s: %s", name, e)
            pages = []
        self._h(name).record(bool(pages), 0.0, time.time() - t0)
        self.stats["stream_reads"] += 1
        dialogs = getattr(src, "last_dialogs", None) or []
        if hasattr(src, "last_dialogs"):
            src.last_dialogs = []
        if not pages and not dialogs:
            return None
        for url, _, anchors in pages:
            if anchors:
                self.push_links(anchors, 1, base_url=url)
        self.stats["dialogs"] = self.stats.get("dialogs", 0) + len(dialogs)
        return Batch(pages[0][0] if pages else name, "stream", pages, name, time.time() - t0, dialogs=dialogs)

    def random_article(self, lang: str = "ja") -> Batch | None:  # 互換
        return self.collect_stream()

    def collect_link(self) -> Batch | None:
        """フロンティアから 1 ページ読む (検索なし)。"""
        if self.fetcher is None:
            return None
        item = self.pop_link()
        if item is None:
            return None
        target, anchor, depth = item
        t0 = time.time()
        pages = []
        if target.startswith("wiki:"):
            _, lang, key = target.split(":", 2)
            page = self.fetcher.wikimedia_page(key, lang=lang)
            src = f"https://{lang}.wikipedia.org/wiki/{urllib.parse.quote(key)}"
        else:
            page = self.fetcher.get_page(target)
            src = target
        if page and len(page.text) > 200:
            pages.append((src, page.text, page.anchors))
            self.push_links(page.anchors, depth + 1, base_url=src)
        self._h("frontier").record(bool(pages), 0.0, time.time() - t0)
        self.stats["frontier_reads"] += 1
        return Batch(anchor, "link", pages, "frontier", time.time() - t0)

    # ------------------------------------------------------------ フィード / サイト
    def _list_lines(self, name: str) -> list[str]:
        out: list[str] = []
        for p in (DATA_DIR / name, self.data_dir / name):
            try:
                out += [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")]
            except OSError:
                pass
        return list(dict.fromkeys(out))

    def _list_file(self, name: str) -> list[str]:
        return [ln for ln in self._list_lines(name) if ln.startswith("http")]

    def collect_feeds(self, min_interval: float = 1800.0, max_items: int = 5) -> Batch | None:
        if self.fetcher is None:
            return None
        now = time.time()
        for url in self._list_file("feeds.txt"):
            if now - self.feed_last.get(url, 0.0) < min_interval:
                continue
            self.feed_last[url] = now
            t0 = time.time()
            items = self.fetcher.get_feed(url)
            pages = []
            for title, link, summary in items:
                if link and not self.mark_seen(link):
                    continue
                text = f"{title}。{summary}" if title and summary and not summary.startswith(title) else (summary or title)
                if len(text) > 40:
                    pages.append((link or url, text, []))
                if link and len(pages) <= 2 and self.interest(title) > 0.3:
                    page = self.fetcher.get_page(link)
                    if page and len(page.text) > 300:
                        pages.append((link, page.text, []))
                if len(pages) >= max_items:
                    break
            self._h("feed").record(bool(pages), 0.0, time.time() - t0)
            self.stats["feed_items"] += len(pages)
            return Batch(url, "feed", pages, "feed", time.time() - t0)
        return None

    def collect_site(self, index: int) -> Batch | None:
        if self.fetcher is None:
            return None
        sites = self._list_file("sources.txt")
        if not sites:
            return None
        url = sites[index % len(sites)]
        t0 = time.time()
        pages = []
        page = self.fetcher.get_page(url) if self.mark_seen(url) or index % (4 * len(sites)) == 0 else None
        if page and len(page.text) > 200:
            pages.append((url, page.text, []))
            host = urllib.parse.urlsplit(url).netloc
            same = [(u, a) for u, a in page.anchors if urllib.parse.urlsplit(urllib.parse.urljoin(url, u)).netloc == host]
            self.push_links(same, 1, base_url=url)
        self._h("site").record(bool(pages), 0.0, time.time() - t0)
        return Batch(url, "site", pages, "site", time.time() - t0)

    # ------------------------------------------------------------ 先読み
    def start_prefetch(self, topic_fn: Callable[[], str | None]) -> None:
        if self.fetcher is None or self._threads:
            return
        self.topic_fn = topic_fn
        for w in range(max(1, self.workers)):
            th = threading.Thread(target=self._prefetch_loop, args=(w,), name=f"tinyai-prefetch-{w}", daemon=True)
            th.start()
            self._threads.append(th)

    def poke(self) -> None:
        self._poke.set()

    def stop(self) -> None:
        self._stop.set()
        self._poke.set()
        for th in self._threads:
            th.join(timeout=15)

    def _prefetch_loop(self, worker: int = 0) -> None:
        turn = worker
        while not self._stop.is_set():
            try:
                turn += 1
                batch = None
                if turn % 5 == 0:
                    batch = self.collect_feeds()
                if batch is None and turn % 4 == 0:
                    batch = self.collect_stream()      # ランダム記事・文学 (多様性)
                if batch is None and (turn % 3 == 0 or worker % 2 == 1):
                    batch = self.collect_link()        # 奇数ワーカーはフロンティア優先
                if batch is None and self.topic_fn is not None:
                    topic = self.topic_fn()
                    batch = self.collect(topic) if topic else None
                if batch is None:
                    batch = self.collect_link() or self.collect_stream()
                if batch is None or not batch.pages:
                    self._poke.wait(10.0)
                    self._poke.clear()
                    continue
                while not self._stop.is_set():
                    try:
                        self.ready.put(batch, timeout=1.0)
                        break
                    except queue.Full:
                        continue
            except Exception as e:
                log.warning("先読み失敗: %s", e)
                self._poke.wait(5.0)
                self._poke.clear()

    def next_ready(self, timeout: float = 0.0) -> Batch | None:
        try:
            return self.ready.get(timeout=timeout) if timeout > 0 else self.ready.get_nowait()
        except queue.Empty:
            return None

    def describe(self) -> dict:
        return {
            "sources": {k: v.to_dict() for k, v in self.health.items()},
            "registered": [f"{s.name}:{s.lang}" for lst in self.sources.values() for s in lst] + [f"{s.name}:{s.lang}" for s in self.streams],
            "frontier": len(self.frontier),
            "seen": len(self.seen),
            "ready": self.ready.qsize(),
            "prefetch_alive": sum(1 for th in self._threads if th.is_alive()),
            **self.stats,
        }
