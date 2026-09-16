"""自律学習ループ (バックグラウンドスレッド)。

人がリアルタイムに学ぶように振る舞う:
  * 会話で分からない話題が出た瞬間に起きて (イベント駆動)、その話題を最優先で調べる
  * 収集システムの先読みスレッドがページを用意しているので、学習は待たずに進む
  * 数サイクルごとにパラメータの進化、一定周期で記憶の整理 (consolidate) と保存
  * data_dir/inbox/ のファイルは毎サイクル取り込む
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

from .brain import Brain
from .collector import Collector
from .dumps import iter_archive_texts, learn_wiki_dump
from .web import Fetcher

log = logging.getLogger("tinyai.evolve")


class Evolver(threading.Thread):
    def __init__(self, brain: Brain, fetcher: Fetcher | None = None, interval: float | None = None, max_cycles: int | None = None, max_seconds: float | None = None, collector: Collector | None = None):
        super().__init__(name="tinyai-evolver", daemon=True)
        self.brain = brain
        cfg = brain.cfg
        if fetcher is None and cfg.web_enabled and collector is None:
            fetcher = Fetcher(cfg.user_agent, cfg.fetch_timeout, cfg.max_page_bytes)
        self.fetcher = fetcher
        self.collector = collector if collector is not None else Collector(
            fetcher, cfg.data_dir, cfg.languages, interest=brain.interest_score, prefetch=cfg.prefetch_depth, workers=cfg.prefetch_workers
        )
        self.interval = cfg.evolve_interval if interval is None else interval
        self.max_cycles = max_cycles
        self.max_seconds = max_seconds
        self.stop_event = threading.Event()
        self.wake = threading.Event()
        self.cycles = 0
        self.last_topic: str | None = None
        self.last_learned = 0
        self.started_at = 0.0
        self.errors = 0
        self._site_index = 0
        self._recent_gain: list[int] = []   # 直近サイクルの収穫 (適応的な間隔に使う)
        brain.on_gap = self._on_gap

    # ------------------------------------------------------------ 制御
    def _on_gap(self, topic: str) -> None:
        """会話で分からない話題が出た: すぐ起きる。"""
        self.wake.set()
        self.collector.poke()

    def stop(self, wait: bool = True) -> None:
        self.stop_event.set()
        self.wake.set()
        self.collector.stop()
        if wait and self.is_alive():
            self.join(timeout=30)

    def run(self) -> None:
        self.started_at = time.time()
        log.info("自律学習を開始 (interval=%.0fs, web=%s)", self.interval, self.collector.fetcher is not None)
        if self.collector.fetcher is not None:
            self.collector.start_prefetch(self.brain.next_topic)
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
            # 調査キューに話題があれば休まない。無ければ interval 待つが、会話で起こされたら即再開。
            # 収穫が続けば間隔を縮め (最短 1/4)、空振りが続けば伸ばす (最長 4 倍) = 処理コストの適応
            if not self.brain.gaps:
                self.wake.wait(self.adaptive_interval())
            self.wake.clear()
        self.collector.stop()
        try:
            self.brain.save()
        except Exception as e:
            log.warning("保存失敗: %s", e)
        log.info("自律学習を終了 (cycles=%d)", self.cycles)

    def adaptive_interval(self) -> float:
        if not self._recent_gain:
            return self.interval
        recent = self._recent_gain[-5:]
        avg = sum(recent) / len(recent)
        if avg >= 100:
            return self.interval * 0.25
        if avg >= 20:
            return self.interval * 0.5
        if avg == 0:
            return min(self.interval * 4, 600.0)
        return self.interval

    # ------------------------------------------------------------ 1 サイクル
    def cycle(self) -> dict:
        brain = self.brain
        cfg = brain.cfg
        col = self.collector
        self.cycles += 1
        learned = self.ingest_inbox()
        topic = None
        if col.fetcher is not None:
            batch = None
            if brain.gaps:
                # 会話由来の話題は先読みを待たずに同期で取りに行く (リアルタイム反映)
                topic = brain.next_topic()
                if topic:
                    batch = col.collect(topic)
            if batch is None and self.cycles % 6 == 0:
                batch = col.collect_site(self._site_index)
                self._site_index += 1
            if batch is None:
                batch = col.next_ready(timeout=1.0)
            if batch is None:
                topic = brain.next_topic()
                if topic:
                    batch = col.collect(topic)
            if batch is not None:
                topic = batch.topic
                learned += brain.learn_batch(batch, col)
        self.last_topic, self.last_learned = topic, learned
        self._recent_gain.append(learned)
        if len(self._recent_gain) > 50:
            del self._recent_gain[:25]
        brain.background_step(budget_docs=400)  # 意味ベクトル・接尾辞配列 (後回しの学習)
        brain.neural_step(steps=4, budget_seconds=cfg.neural_seconds_per_cycle)  # ニューラル LM の継続学習
        evolved = None
        if self.cycles % cfg.evolve_every == 0 and len(brain.kb) >= 20:
            evolved = brain.evolve_step()
        if self.cycles % (cfg.evolve_every * 5) == 0:
            brain.consolidate()
        mem = brain.enforce_memory()
        if self.cycles % cfg.save_every == 0:
            brain.save()
        rec = {"cycle": self.cycles, "topic": topic, "learned": learned, "evolved": evolved and evolved["accepted"], "rss_mb": mem["rss_mb"]}
        log.info("cycle %(cycle)d topic=%(topic)s learned=%(learned)d evolved=%(evolved)s rss=%(rss_mb)sMB", rec)
        return rec

    def ingest_inbox(self) -> int:
        inbox = Path(self.brain.cfg.data_dir) / "inbox"
        if not inbox.is_dir():
            return 0
        done = inbox.parent / "learned"
        total = 0
        for p in sorted(inbox.iterdir()):
            name = p.name.lower()
            if not p.is_file():
                continue
            try:
                if name.endswith((".xml", ".xml.bz2", ".xml.gz")) and "wiki" in name:
                    _, n = learn_wiki_dump(self.brain, p)
                elif name.endswith((".zip", ".gz", ".bz2", ".xz")):
                    n = sum(self.brain.learn_text(text, source=f"file:{nm}") for nm, text in iter_archive_texts(p))
                elif p.suffix.lower() in (".txt", ".md", ".html", ".htm", ".json", ".csv"):
                    n = self.brain.learn_file(p)
                else:
                    continue
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
            "interval": round(self.adaptive_interval(), 1),
            "fetched": getattr(self.collector.fetcher, "fetched", 0),
            "failed": getattr(self.collector.fetcher, "failed", 0),
            "collector": self.collector.describe(),
        }
