"""自律学習ループ (バックグラウンドスレッド)。

1 サイクル = 話題を選ぶ -> Web で探索して読む -> 学習 -> (数回に 1 回) 進化
           -> メモリ整理 -> 保存 -> 休止。
data_dir/inbox/ に置かれたテキスト/HTML も自動で取り込む。
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

from .brain import Brain
from .web import Fetcher

log = logging.getLogger("tinyai.evolve")


class Evolver(threading.Thread):
    def __init__(self, brain: Brain, fetcher: Fetcher | None = None, interval: float | None = None, max_cycles: int | None = None, max_seconds: float | None = None):
        super().__init__(name="tinyai-evolver", daemon=True)
        self.brain = brain
        cfg = brain.cfg
        self.fetcher = fetcher if fetcher is not None else (Fetcher(cfg.user_agent, cfg.fetch_timeout, cfg.max_page_bytes) if cfg.web_enabled else None)
        self.interval = cfg.evolve_interval if interval is None else interval
        self.max_cycles = max_cycles
        self.max_seconds = max_seconds
        self.stop_event = threading.Event()
        self.cycles = 0
        self.last_topic: str | None = None
        self.last_learned = 0
        self.started_at = 0.0
        self.errors = 0
        self._crawl_seen: set = set()

    # ------------------------------------------------------------ 制御
    def stop(self, wait: bool = True) -> None:
        self.stop_event.set()
        if wait and self.is_alive():
            self.join(timeout=30)

    def run(self) -> None:
        self.started_at = time.time()
        log.info("自律学習を開始 (interval=%.0fs, web=%s)", self.interval, bool(self.fetcher))
        while not self.stop_event.is_set():
            try:
                self.cycle()
            except MemoryError:
                log.warning("MemoryError: 強制プルーニング")
                self.brain.lm.shrink_to(self.brain.lm.estimated_bytes() // 2)
                self.brain.kb.shrink_to(self.brain.kb.estimated_bytes() // 2)
                self.brain.guard.collect()
            except Exception as e:  # 1 サイクルの失敗でループを止めない
                self.errors += 1
                log.warning("サイクル失敗: %s", e, exc_info=log.isEnabledFor(logging.DEBUG))
            if self.max_cycles is not None and self.cycles >= self.max_cycles:
                break
            if self.max_seconds is not None and time.time() - self.started_at >= self.max_seconds:
                break
            self.stop_event.wait(self.interval)
        try:
            self.brain.save()
        except Exception as e:
            log.warning("保存失敗: %s", e)
        log.info("自律学習を終了 (cycles=%d)", self.cycles)

    # ------------------------------------------------------------ 1 サイクル
    def cycle(self) -> dict:
        brain = self.brain
        cfg = brain.cfg
        self.cycles += 1
        learned = self.ingest_inbox()
        topic = None
        if self.fetcher is not None:
            sites = self.sites()
            if sites and self.cycles % 4 == 0:
                # data_dir/sources.txt に書かれたサイトを巡回する
                url = sites[(self.cycles // 4) % len(sites)]
                topic = f"site:{url}"
                for src, txt in self.fetcher.crawl_site(url, max_pages=2, seen=self._crawl_seen):
                    learned += brain.learn_text(txt, source=src)
                if len(self._crawl_seen) > 5000:
                    self._crawl_seen.clear()
            else:
                topic = brain.next_topic()
                if topic:
                    learned += brain.learn_from_web(topic, self.fetcher)
        self.last_topic, self.last_learned = topic, learned
        evolved = None
        if self.cycles % cfg.evolve_every == 0 and len(brain.kb) >= 20:
            evolved = brain.evolve_step()
        mem = brain.enforce_memory()
        if self.cycles % cfg.save_every == 0:
            brain.save()
        rec = {"cycle": self.cycles, "topic": topic, "learned": learned, "evolved": evolved and evolved["accepted"], "rss_mb": mem["rss_mb"]}
        log.info("cycle %(cycle)d topic=%(topic)s learned=%(learned)d evolved=%(evolved)s rss=%(rss_mb)sMB", rec)
        return rec

    def sites(self) -> list[str]:
        p = Path(self.brain.cfg.data_dir) / "sources.txt"
        try:
            return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip().startswith("http")]
        except OSError:
            return []

    def ingest_inbox(self) -> int:
        inbox = Path(self.brain.cfg.data_dir) / "inbox"
        if not inbox.is_dir():
            return 0
        done = inbox.parent / "learned"
        total = 0
        for p in sorted(inbox.iterdir()):
            if not p.is_file() or p.suffix.lower() not in (".txt", ".md", ".html", ".htm", ".json", ".csv"):
                continue
            try:
                n = self.brain.learn_file(p)
                total += n
                done.mkdir(parents=True, exist_ok=True)
                p.rename(done / p.name)
                log.info("inbox 取込 %s (%d 文)", p.name, n)
            except OSError as e:
                log.warning("inbox 取込失敗 %s: %s", p, e)
        return total

    def describe(self) -> dict:
        return {
            "alive": self.is_alive(),
            "cycles": self.cycles,
            "last_topic": self.last_topic,
            "last_learned": self.last_learned,
            "errors": self.errors,
            "fetched": getattr(self.fetcher, "fetched", 0),
            "failed": getattr(self.fetcher, "failed", 0),
        }
