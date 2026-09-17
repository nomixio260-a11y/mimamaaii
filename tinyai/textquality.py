"""学習に入れてよい地の文かどうかの判定 (軽い規則だけ)。

収集したページの本文をそのままコーパスに入れると、目次・ナビゲーション・表・数字の羅列・
別言語の断片が 3 割ほど混ざる (実測: 標本 200 件中 64 件)。小さなモデルほど低品質な文の影響が
大きいので、句読点・数字・文字種の割合という安い指標で落とす。意味の判定はしない (できない)。
"""
from __future__ import annotations

_BAD_WORDS = ("nan", "none", "null", "undefined")


def _ratios(text: str) -> tuple[float, float, float]:
    n = max(len(text), 1)
    ja = sum(1 for c in text if "぀" <= c <= "ヿ" or "一" <= c <= "鿿")
    en = sum(1 for c in text if c.isascii() and c.isalpha())
    dig = sum(1 for c in text if c.isdigit())
    return ja / n, en / n, dig / n


def good_prose(text: str, min_len: int = 40) -> bool:
    """地の文として学習に使えるか。文末記号が一定以上あり、数字と記号に偏っていないこと。"""
    t = text.strip()
    if len(t) < min_len:
        return False
    if t.count("\ufffd") > len(t) * 0.002:     # 文字化け (復号に失敗したページ)
        return False
    marks = t.count("。") + t.count("！") + t.count("？") + t.count(".") + t.count("!") + t.count("?")
    if marks < max(1, len(t) // 200):          # 200 字に 1 つも文末が無ければ目次・表の類
        return False
    ja, en, dig = _ratios(t)
    if dig > 0.25:                              # 数字だらけ (表・統計の羅列)
        return False
    if ja < 0.15 and en < 0.40:                 # どの言語の地の文とも言えない
        return False
    if ja >= 0.15:
        # 日本語の文なのに読点がまったく無い長文は、体言の羅列であることが多い
        if len(t) > 120 and t.count("、") == 0 and marks < 2:
            return False
    return True


def strip_broken(text: str) -> str:
    """文字化けの記号 (U+FFFD) を落とす。JSON に出すと読めない文字として残るため。"""
    return text.replace("\ufffd", "")


def clean_field(value) -> str:
    """データセットの欠損値が文字列として流れてくるのを防ぐ ("nan" などが本文に混ざる)。"""
    if value is None:
        return ""
    t = strip_broken(str(value)).strip()
    return "" if t.lower() in _BAD_WORDS else t
