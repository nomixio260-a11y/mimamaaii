"""tinyai - 超小型・自己進化型の会話AI (標準ライブラリのみ)。

外部依存なし。n-gram 言語モデル + BM25 検索メモリ + 自律クローラで、
会話・学習・Web 探索を繰り返しながら賢くなる。メモリ使用量は
設定した上限 (既定 256MB) に収まるよう常時プルーニングされる。
"""

__version__ = "0.1.0"

from .config import Config  # noqa: F401
from .brain import Brain  # noqa: F401
