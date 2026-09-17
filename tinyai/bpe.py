"""サブワードトークナイザ (頻度ベース、WordPiece 風の最長一致)。

* 文字 (出現 2 回以上) と、頻度の高い 2〜4 文字の連続 (漢字・かな・英字の連続) を語彙にする
* 符号化は左から最長一致 (最大 4 文字)、語彙に無い文字は <unk> ではなく 1 文字ずつ登録済みなら必ず通る
* 学習は n-gram の数え上げだけなので、数 MB のテキストでも数秒で終わる
* BPE のように併合履歴を持たないので、語彙は「保存されたリスト」そのもの
"""
from __future__ import annotations

import json
import re
import string
import unicodedata
from collections import Counter
from pathlib import Path

SPECIALS = ["<pad>", "<unk>", "<bos>", "<eos>", "<usr>", "<bot>", "<ctx>", "<sep>"]
PAD, UNK, BOS, EOS, USR, BOT, CTX, SEP = range(8)
SPACE = "▁"  # 英単語の前の空白を表す
# どのコーパスでも必ず入れる基本文字 (ASCII 印字可能文字 + 日本語の句読点・記号)
BASE_CHARS = [c for c in string.ascii_lowercase + string.digits + string.punctuation] + list("。、・「」『』（）！？〜ー…") + [SPACE]

_RUN_RE = re.compile(r"[A-Za-z0-9]+|[぀-ゟ]+|[゠-ヿ]+|[一-鿿㐀-䶿]+|\S")


class SubwordTokenizer:
    MAXLEN = 4

    def __init__(self, tokens: list[str] | None = None):
        self.tokens: list[str] = list(SPECIALS) + [t for t in (tokens or []) if t not in SPECIALS]
        self.index: dict[str, int] = {t: i for i, t in enumerate(self.tokens)}
        self._by_first: dict[str, list[str]] = {}
        self._rebuild()

    def _rebuild(self) -> None:
        self._by_first = {}
        for t in self.tokens[len(SPECIALS):]:
            self._by_first.setdefault(t[0], []).append(t)
        for k, lst in self._by_first.items():
            lst.sort(key=len, reverse=True)

    # ------------------------------------------------------------ 学習
    @staticmethod
    def normalize(text: str) -> str:
        return unicodedata.normalize("NFKC", text).lower()

    @classmethod
    def train(cls, texts, size: int = 6000, min_count: int = 3) -> "SubwordTokenizer":
        chars: Counter = Counter()
        grams: Counter = Counter()
        for text in texts:
            text = cls.normalize(text)
            for m in _RUN_RE.finditer(text):
                run = m.group(0)
                chars.update(run)
                n = len(run)
                if n < 2:
                    continue
                # 英数字の連続は語そのもの (≤ 12 文字) も候補に
                if run[0].isascii() and n <= 12:
                    grams[run] += 1
                for L in (2, 3, 4):
                    for i in range(n - L + 1):
                        grams[run[i : i + L]] += 1
        vocab = list(dict.fromkeys(BASE_CHARS + [c for c, n in chars.items() if n >= 2 or not c.isascii()]))
        # 長い単位ほど 1 トークンで多くの文字を表せるので (長さ-1) × 頻度で選ぶ
        ranked = sorted(((n * (len(g) - 1), g) for g, n in grams.items() if n >= min_count), reverse=True)
        room = max(0, size - len(SPECIALS) - len(vocab))
        vocab += [g for _, g in ranked[:room]]
        return cls(vocab)

    # ------------------------------------------------------------ 符号化
    def encode(self, text: str, max_tokens: int | None = None) -> list[int]:
        text = self.normalize(text)
        out: list[int] = []
        idx = self.index
        by_first = self._by_first
        space_id = idx.get(SPACE)
        i = 0
        n = len(text)
        while i < n:
            ch = text[i]
            if ch.isspace():
                i += 1
                # 英数字の語の前の空白は ▁ で残す (復号で空白に戻す)
                if space_id is not None and i < n and text[i].isascii() and text[i].isalnum() and out and out[-1] >= len(SPECIALS):
                    out.append(space_id)
                continue
            cands = by_first.get(ch)
            hit = None
            if cands:
                for t in cands:
                    if text.startswith(t, i):
                        hit = t
                        break
            if hit is None:
                out.append(UNK)
                i += 1
            else:
                out.append(idx[hit])
                i += len(hit)
            if max_tokens is not None and len(out) >= max_tokens:
                break
        return out

    def decode(self, ids) -> str:
        parts = []
        for i in ids:
            if i < len(SPECIALS):
                continue
            t = self.tokens[i]
            parts.append(" " if t == SPACE else t)
        return "".join(parts)

    def __len__(self) -> int:
        return len(self.tokens)

    def add_tokens(self, new: list[str]) -> int:
        """語彙に新しい単位を追加 (モデル側の add_tokens と同期して呼ぶ)。追加数を返す。"""
        added = 0
        for t in new:
            if t and t not in self.index and len(t) <= self.MAXLEN * 3:
                self.index[t] = len(self.tokens)
                self.tokens.append(t)
                added += 1
        if added:
            self._rebuild()
        return added

    def frequent_new_units(self, texts, top: int = 200, min_count: int = 5) -> list[str]:
        """新しいテキストに頻出するが語彙に無い 2〜4 文字の連続 (語彙の進化用)。

        実際に符号化した時に何トークン減るかで順位を付ける。単に「文字数 - 1」で数えると、
        既にうまく分割できている英単語の断片 (velo, elop, dev …) が上位を占めてしまい、
        1 トークンあたりの文字数 (実測 1.45 文字) が伸びない。日本語は 1 文字ずつに割れやすく、
        まとめた時の削減が大きいので、実測の削減量で並べれば自然に日本語の語が上に来る。"""
        grams: Counter = Counter()
        for text in texts:
            text = self.normalize(text)
            for m in _RUN_RE.finditer(text):
                run = m.group(0)
                for L in (2, 3, 4):
                    for i in range(len(run) - L + 1):
                        g = run[i : i + L]
                        if g not in self.index:
                            grams[g] += 1
        scored = []
        for g, n in grams.items():
            if n < min_count:
                continue
            saved = len(self._encode_unit(g)) - 1      # 今は何トークンか → 1 トークンになると何個減るか
            if saved <= 0:
                continue
            scored.append((n * saved, g))
        scored.sort(reverse=True)
        return [g for _, g in scored[:top]]

    def _encode_unit(self, unit: str) -> list:
        """1 つの連続を今の語彙で符号化した時のトークン列 (長さだけ使う)。"""
        return self.encode(unit, max_tokens=len(unit) + 2)

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.tokens, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "SubwordTokenizer":
        toks = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls([t for t in toks if t not in SPECIALS])
