"""メモリ上限の監視と強制。

2 段構え:
  1. ソフト上限: /proc/self/statm から RSS を読み、上限の 85% を超えたら
     Brain にプルーニングを要求する。
  2. ハード上限: RLIMIT_DATA (無ければ RLIMIT_AS) を設定し、暴走しても
     OS が MemoryError を発生させる。
"""
from __future__ import annotations

import gc
import os
import sys

MB = 1024 * 1024


def rss_bytes() -> int:
    """現在の常駐メモリ (bytes)。取れなければ 0。"""
    try:
        with open("/proc/self/statm") as f:
            parts = f.read().split()
        return int(parts[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        pass
    try:
        import resource

        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return ru * (1 if sys.platform == "darwin" else 1024)
    except Exception:
        return 0


class MemoryGuard:
    def __init__(self, limit_mb: int, hard: bool = True, soft_ratio: float = 0.85):
        self.limit = int(limit_mb) * MB
        self.soft = int(self.limit * soft_ratio)
        self.baseline = rss_bytes()
        self.hard_applied = None
        if hard:
            self.hard_applied = self._apply_hard_limit()

    def _apply_hard_limit(self) -> str | None:
        try:
            import resource
        except ImportError:
            return None
        # RLIMIT_DATA は Linux 4.7+ でヒープ + 匿名 mmap を含むので RSS に近い。
        # スレッドスタック等のために余裕を持たせる。
        headroom = max(96 * MB, self.limit // 2)
        for name in ("RLIMIT_DATA", "RLIMIT_AS"):
            lim = getattr(resource, name, None)
            if lim is None:
                continue
            target = self.limit + headroom if name == "RLIMIT_DATA" else self.limit * 4 + 512 * MB
            try:
                soft, hard = resource.getrlimit(lim)
                if hard != resource.RLIM_INFINITY and hard < target:
                    target = hard
                resource.setrlimit(lim, (target, hard))
                return f"{name}={target // MB}MB"
            except (ValueError, OSError):
                continue
        return None

    # 学習に使ってよい残り予算 (bytes)。ベースライン (インタプリタ自身) を差し引く。
    @property
    def budget(self) -> int:
        return max(16 * MB, self.soft - self.baseline)

    def usage(self) -> int:
        return rss_bytes()

    def over_soft(self) -> bool:
        return rss_bytes() > self.soft

    def pressure(self) -> float:
        """0.0 (余裕) .. 1.0+ (ソフト上限超過)。"""
        return rss_bytes() / self.soft if self.soft else 0.0

    @staticmethod
    def collect() -> None:
        gc.collect()

    def describe(self) -> dict:
        return {
            "limit_mb": self.limit // MB,
            "soft_mb": self.soft // MB,
            "rss_mb": round(rss_bytes() / MB, 1),
            "hard_limit": self.hard_applied,
        }
