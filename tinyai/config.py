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
    memory_mb: int = field(default_factory=lambda: _env_int("TINYAI_MEMORY_MB", 500))
    # OS レベルの強制上限 (RLIMIT) も掛けるか
    hard_limit: bool = True
    # 言語モデルの最大 n-gram 次数 (これより高い次数は保存しない)。
    # tools/experiment.py の結果: 数千文規模では 4 次は 3 次とパープレキシティが同じで、
    # エントリ数 1.8 倍・学習時間 1.6 倍。大規模コーパスなら 4 に上げる。
    max_order: int = _env_int("TINYAI_MAX_ORDER", 3)
    # 知識ベースに保持する最大文数 (0 ならメモリ上限から自動で決める: 1MB あたり 400 文)。
    # 実際にはメモリ予算とどちらか厳しい方が効く
    max_docs: int = 0
    # Web 探索
    web_enabled: bool = True
    languages: tuple = ("ja", "en")
    fetch_timeout: float = 12.0
    max_page_bytes: int = 400_000
    user_agent: str = "tinyai/0.1 (+https://github.com/nomixio260-a11y/mimamaaii; self-learning toy bot)"
    # 自律学習ループ
    evolve_interval: float = 20.0      # 秒。1 サイクルごとの休止
    prefetch_workers: int = field(default_factory=lambda: _env_int("TINYAI_WORKERS", 2))  # 先読みスレッド数
    prefetch_depth: int = 4            # 先読みして貯めておくバッチ数
    neural_seconds_per_cycle: float = 2.0  # 自律ループ 1 サイクルあたりニューラル LM の学習に使う秒数 (numpy がある時)
    neural_size: str = field(default_factory=lambda: os.environ.get("TINYAI_NEURAL", "auto"))  # auto / small / base / large / xl (auto はメモリ上限から)
    neural_dropout: float = field(default_factory=lambda: float(os.environ.get("TINYAI_NEURAL_DROPOUT", "0.1")))  # 学習時のドロップアウト率 (過学習の抑制)
    neural_rethink: bool = True        # ニューラル応答の 2 段階生成 (下書きの語で再検索してから答える)
    neural_first: bool = True          # 学習が進んだら (ppl 基準) ニューラル生成を応答の主経路にする
    neural_only: bool = True           # 準備が整ったら応答は常にニューラル生成 (検索は文脈の供給に回る、事実の即答も使わない)
    tools: bool = False                # 計算・日付・単位換算などの道具を使う (既定オフ: 応答はネットワークが担う)
    online_learning: bool = True       # 各ターンの直後にその対話で勾配更新する
    background_training: bool = True   # chat / serve 中も裏で学習スレッドを回す
    neural_override_conf: float = 0.8  # 検索の確信度がこれ未満ならニューラル生成を優先 (これ以上は正確な知識文を返す)
    cite: bool = True                  # Web 由来の答えに出典 (ホスト名) を添える
    neural_workers: int = field(default_factory=lambda: _env_int("TINYAI_NEURAL_WORKERS", max(1, (os.cpu_count() or 2) - 1)))  # train コマンドのデータ並列数
    evolve_every: int = 3              # 何サイクルごとにパラメータ進化を試すか
    holdout_size: int = 300            # 自己評価用に取り置く文の数
    save_every: int = 5                # 何サイクルごとに保存するか
    seed: int | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["data_dir"] = str(self.data_dir)
        return d


    def __post_init__(self):
        if not self.max_docs:
            # メモリ 1MB あたり 400 文 (500MB なら 20 万文)。知識ベースの取り分は予算の 30%
            self.max_docs = max(20000, int(self.memory_mb) * 400)
