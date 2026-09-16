"""学習データの自動収集システム。

* ソース (Wikimedia Core API / Wikipedia Action API / DuckDuckGo / フィード / 登録サイト) ごとに
  成功率・収穫量・遅延を記録し、健全なソースから順に使う
* 読んだページのリンク (アンカー文字列付き) を「フロンティア」に積み、
  会話の関心・知識の薄さに応じた優先度で次に読むページを選ぶ (検索 1 回分の通信を節約)
* RSS/Atom フィードから新着記事を取り込む (鮮度のある知識)
* URL の重複取得を避ける
* 先読みスレッドがネットワーク待ちを学習と並列化し、学習側は取得済みのページを即座に消費できる
"""
from __future__ import annotations

import hashlib
import heapq
import logging
import queue
import threading
import time
import urllib.parse
from pathlib import Path
from typing import Callable, Iterable

from .web import Fetcher, Page

log = logging.getLogger("tinyai.collector")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


class Batch:
    """1 回の収集結果。pages は [(source_url, text, anchors)]。"""
    __slots__ = ("topic", "kind", "pages", "source", "elapsed")

    def __init__(self, topic: str, kind: str, pages: list, source: str, elapsed: float):
        self.topic, self.kind, self.pages, self.source, self.elapsed = topic, kind, pages, source, elapsed


class SourceHealth:
    __slots__ = ("tries", "ok", "gain", "latency")

    def __init__(self):
        self.tries = 0
        self.ok = 0
        self.gain = 0.0
        self.latency = 0.0

    def record(self, ok: bool, gain: float, latency: float) -> None:
        self.tries += 1
        if ok:
            self.ok += 1
        self.gain += gain
        self.latency = 0.8 * self.latency + 0.2 * latency if self.latency else latency

    @property
    def score(self) -> float:
        # 成功率 × 平均収穫 (試行が少ないうちは楽観的)
        return (self.ok + 1) / (self.tries + 2) * (self.gain / max(self.tries, 1) + 0.5)

    def to_dict(self) -> dict:
        return {"tries": self.tries, "ok": self.ok, "avg_gain": round(self.gain / max(self.tries, 1), 2), "latency": round(self.latency, 2), "score": round(self.score, 3)}


class Collector:
    def __init__(self, fetcher: Fetcher | None, data_dir: Path, languages: Iterable[str] = ("ja", "en"), interest: Callable[[str], float] | None = None, prefetch: int = 3):
        self.fetcher = fetcher
        self.data_dir = Path(data_dir)
        self.languages = tuple(languages)
        self.interest = interest or (lambda text: 0.0)
        self.health: dict[str, SourceHealth] = {}
        self.frontier: list[tuple[float, int, str, str, int]] = []  # (-priority, seq, link, anchor, depth)
        self._seq = 0
        self.seen: set[str] = set()
        self.feed_last: dict[str, float] = {}
        self.ready: queue.Queue = queue.Queue(maxsize=max(1, prefetch))
        self.topic_fn: Callable[[], str | None] | None = None
        self._stop = threading.Event()
        self._poke = threading.Event()
        self._thread: threading.Thread | None = None
        self.lock = threading.RLock()  # pop_link -> _mark_seen で再入する
        self.stats = {"batches": 0, "pages": 0, "frontier_reads": 0, "feed_items": 0}

    # ------------------------------------------------------------ 補助
    def _h(self, name: str) -> SourceHealth:
        h = self.health.get(name)
        if h is None:
            h = self.health[name] = SourceHealth()
        return h

    def _mark_seen(self, url: str) -> bool:
        """初見なら True。"""
        key = hashlib.blake2b(url.encode("utf-8"), digest_size=8).digest()
        with self.lock:
            if key in self.seen:
                return False
            if len(self.seen) > 30000:
                self.seen.clear()
            self.seen.add(key)
            return True

    def report(self, source: str, learned: int) -> None:
        """学習側からの収穫報告 (文数)。ソースのスコアに反映する。"""
        h = self._h(source)
        h.gain += min(learned, 400) / 100.0

    # ------------------------------------------------------------ フロンティア
    def push_links(self, anchors: Iterable[tuple[str, str]], depth: int = 1, base_url: str = "") -> int:
        """ページ内リンクをフロンティアへ。優先度 = 関心 + 新規性 - 深さ。"""
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
            if len(self.frontier) > 600:
                self.frontier = heapq.nsmallest(400, self.frontier)
                heapq.heapify(self.frontier)
        return n

    def pop_link(self) -> tuple[str, str, int] | None:
        with self.lock:
            while self.frontier:
                _, _, target, anchor, depth = heapq.heappop(self.frontier)
                if self._mark_seen(target):
                    return target, anchor, depth
        return None

    # ------------------------------------------------------------ ソース
    def _wikimedia(self, topic: str, lang: str, max_pages: int) -> list[tuple[str, str, list]]:
        f = self.fetcher
        out = []
        for key in f.wikimedia_search(topic, lang=lang, limit=max_pages):
            url = f"https://{lang}.wikipedia.org/wiki/{urllib.parse.quote(key)}"
            if not self._mark_seen(url):
                continue
            page = f.wikimedia_page(key, lang=lang)
            if page and len(page.text) > 200:
                out.append((url, page.text, page.anchors))
            if len(out) >= max_pages:
                break
        return out

    def _wikipedia_action(self, topic: str, lang: str, max_pages: int) -> list[tuple[str, str, list]]:
        f = self.fetcher
        out = []
        for title in f.wikipedia_search(topic, lang=lang, limit=max_pages):
            url = f"https://{lang}.wikipedia.org/wiki/{urllib.parse.quote(title)}"
            if not self._mark_seen(url):
                continue
            txt = f.wikipedia_extract(title, lang=lang)
            if txt and len(txt) > 200:
                out.append((url, txt, []))
        return out

    def _duckduckgo(self, topic: str, lang: str, max_pages: int) -> list[tuple[str, str, list]]:
        f = self.fetcher
        out = []
        for url in f.duckduckgo_search(topic, limit=max_pages + 2):
            if not self._mark_seen(url):
                continue
            page = f.get_page(url)
            if page and len(page.text) > 300:
                out.append((url, page.text, [(u, a) for u, a in page.anchors if u.startswith(("http", "/"))]))
            if len(out) >= max_pages:
                break
        return out

    def _ordered_sources(self, lang: str) -> list[tuple[str, Callable]]:
        cands = [
            (f"wikimedia:{lang}", lambda t, n, lang=lang: self._wikimedia(t, lang, n)),
            (f"wikipedia:{lang}", lambda t, n, lang=lang: self._wikipedia_action(t, lang, n)),
            ("duckduckgo", lambda t, n: self._duckduckgo(t, lang, n)),
        ]
        cands.sort(key=lambda x: -self._h(x[0]).score)
        return cands

    def collect(self, topic: str, max_pages: int = 3) -> Batch:
        """話題を検索して読む。健全なソースから順に試し、足りた時点で止める。"""
        t0 = time.time()
        if self.fetcher is None:
            return Batch(topic, "topic", [], "", 0.0)
        langs = [l for l in self.languages if l == "ja"] or list(self.languages)[:1] if not topic.isascii() else list(self.languages)
        pages: list = []
        used = ""
        for lang in langs:
            for name, fn in self._ordered_sources(lang):
                if name == "duckduckgo" and pages:
                    continue
                st = time.time()
                try:
                    got = fn(topic, max_pages - len(pages))
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
    def _list_file(self, name: str) -> list[str]:
        out: list[str] = []
        for p in (DATA_DIR / name, self.data_dir / name):
            try:
                out += [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip().startswith("http")]
            except OSError:
                pass
        return list(dict.fromkeys(out))

    def collect_feeds(self, min_interval: float = 1800.0, max_items: int = 5) -> Batch | None:
        """更新間隔を過ぎたフィードを 1 つ読み、新着記事の要約 (と本文) を返す。"""
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
                if link and not self._mark_seen(link):
                    continue
                text = f"{title}。{summary}" if title and summary and not summary.startswith(title) else (summary or title)
                if len(text) > 40:
                    pages.append((link or url, text, []))
                if link and len(pages) <= 2 and self.interest(title) > 0.3:
                    page = self.fetcher.get_page(link)  # 関心の高い記事は本文も読む
                    if page and len(page.text) > 300:
                        pages.append((link, page.text, []))
                if len(pages) >= max_items:
                    break
            self._h("feed").record(bool(pages), 0.0, time.time() - t0)
            self.stats["feed_items"] += len(pages)
            return Batch(url, "feed", pages, "feed", time.time() - t0)
        return None

    def collect_site(self, index: int) -> Batch | None:
        """sources.txt のサイトを順番に巡回 (リンクはフロンティアへ)。"""
        if self.fetcher is None:
            return None
        sites = self._list_file("sources.txt")
        if not sites:
            return None
        url = sites[index % len(sites)]
        t0 = time.time()
        pages = []
        page = self.fetcher.get_page(url) if self._mark_seen(url) or index % (4 * len(sites)) == 0 else None
        if page and len(page.text) > 200:
            pages.append((url, page.text, []))
            host = urllib.parse.urlsplit(url).netloc
            same = [(u, a) for u, a in page.anchors if urllib.parse.urlsplit(urllib.parse.urljoin(url, u)).netloc == host]
            self.push_links(same, 1, base_url=url)
        self._h("site").record(bool(pages), 0.0, time.time() - t0)
        return Batch(url, "site", pages, "site", time.time() - t0)

    # ------------------------------------------------------------ 先読み
    def start_prefetch(self, topic_fn: Callable[[], str | None]) -> None:
        if self.fetcher is None or self._thread is not None:
            return
        self.topic_fn = topic_fn
        self._thread = threading.Thread(target=self._prefetch_loop, name="tinyai-prefetch", daemon=True)
        self._thread.start()

    def poke(self) -> None:
        self._poke.set()

    def stop(self) -> None:
        self._stop.set()
        self._poke.set()
        if self._thread is not None:
            self._thread.join(timeout=15)

    def _prefetch_loop(self) -> None:
        turn = 0
        while not self._stop.is_set():
            try:
                turn += 1
                batch = None
                if turn % 5 == 0:
                    batch = self.collect_feeds()
                if batch is None and turn % 3 == 0:
                    batch = self.collect_link()
                if batch is None and self.topic_fn is not None:
                    topic = self.topic_fn()
                    batch = self.collect(topic) if topic else None
                if batch is None:
                    batch = self.collect_link()
                if batch is None or not batch.pages:
                    # 何も取れなかった: 少し待つ (poke で即再開)
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
            "frontier": len(self.frontier),
            "seen": len(self.seen),
            "ready": self.ready.qsize(),
            "prefetch_alive": bool(self._thread and self._thread.is_alive()),
            **self.stats,
        }
