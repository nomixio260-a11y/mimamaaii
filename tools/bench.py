"""学習スループット / メモリ / 検索速度のベンチマーク。

    python tools/bench.py corpus.txt [--memory 256]
"""
from __future__ import annotations

import argparse
import random
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tinyai import Brain, Config  # noqa: E402
from tinyai.memory import rss_bytes  # noqa: E402
from tinyai.tokenizer import keywords  # noqa: E402

MB = 1024 * 1024


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--memory", type=int, default=256)
    ap.add_argument("--queries", type=int, default=300)
    args = ap.parse_args()
    text = Path(args.corpus).read_text(encoding="utf-8", errors="replace")
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Config(data_dir=Path(tmp), memory_mb=args.memory, hard_limit=False, web_enabled=False, seed=1)
        b = Brain(cfg)
        b.bootstrap()
        rss0 = rss_bytes()
        t0 = time.perf_counter()
        n = b.learn_text(text, source="https://bench/corpus")
        t1 = time.perf_counter()
        rss1 = rss_bytes()
        docs = list(b.kb.docs.values())
        rng = random.Random(0)
        qs = []
        for d in rng.sample(docs, min(args.queries, len(docs))):
            ks = keywords(d.text, limit=2)
            if ks:
                qs.append(("".join(ks) + "とは？", d.id))
        t2 = time.perf_counter()
        hit = 0
        for q, did in qs:
            r = b.reply(q)
            if did in r.doc_ids:
                hit += 1
        t3 = time.perf_counter()
        ev = b.evaluate(seed=1)
        t4 = time.perf_counter()
        for _ in range(3):
            b.evolve_step()
        t5 = time.perf_counter()
        print(f"学習: {n} 文 / {t1 - t0:.2f}s = {n / (t1 - t0):.0f} 文/s")
        print(f"メモリ増分: {(rss1 - rss0) / MB:.1f}MB  (1 文あたり {(rss1 - rss0) / max(n, 1):.0f} bytes)  推定 LM {b.lm.estimated_bytes() / MB:.1f}MB KB {b.kb.estimated_bytes() / MB:.1f}MB")
        print(f"LM: {b.lm.stats()}")
        print(f"KB: {b.kb.stats()}")
        print(f"応答: {len(qs)} 問 / {t3 - t2:.2f}s = {(t3 - t2) / max(len(qs), 1) * 1000:.1f} ms/問  自問自答の的中 {hit / max(len(qs), 1):.3f}")
        print(f"評価: {ev} ({t4 - t3:.2f}s)  進化 3 回 {t5 - t4:.2f}s")
        p = b.save()
        print(f"保存: {p.stat().st_size / MB:.2f}MB")


if __name__ == "__main__":
    main()
