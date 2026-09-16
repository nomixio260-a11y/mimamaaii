"""設定。環境変数 TINYAI_* または CLI 引数で上書きできる。"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


@dataclass
class Config:
    # 保存先ディレクトリ
    data_dir: Path = field(default_factory=lambda: Path(os.environ.get("TINYAI_DATA", "~/.tinyai")).expanduser())
    # プロセス全体のメモリ上限 (MB)。RSS がこれを超えないよう自動プルーニングする。
    memory_mb: int = field(default_factory=lambda: _env_int("TINYAI_MEMORY_MB", 256))
    # OS レベルの強制上限 (RLIMIT) も掛けるか
    hard_limit: bool = True
    # 言語モデルの最大 n-gram 次数 (これより高い次数は保存しない)
    max_order: int = 4
    # 知識ベースに保持する最大文数 (メモリ予算とどちらか厳しい方)
    max_docs: int = 60000
    # Web 探索
    web_enabled: bool = True
    languages: tuple = ("ja", "en")
    fetch_timeout: float = 12.0
    max_page_bytes: int = 400_000
    user_agent: str = "tinyai/0.1 (+https://github.com/nomixio260-a11y/mimamaaii; self-learning toy bot)"
    # 自律学習ループ
    evolve_interval: float = 20.0      # 秒。1 サイクルごとの休止
    evolve_every: int = 3              # 何サイクルごとにパラメータ進化を試すか
    holdout_size: int = 300            # 自己評価用に取り置く文の数
    save_every: int = 5                # 何サイクルごとに保存するか
    seed: int | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["data_dir"] = str(self.data_dir)
        return d
