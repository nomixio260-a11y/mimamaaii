"""日本語/英語混在テキストのための軽量トークナイザ。

形態素解析器を使わず、
  * ラテン文字/数字の連続 -> 1 語
  * CJK (かな/漢字/ハングル) -> 1 文字ずつ (言語モデル用)
  * それ以外の記号 -> 1 文字
に分割する。検索用の「語」は CJK 連続部分の文字 bigram とラテン語で作る。
"""
from __future__ import annotations

import re
import unicodedata
from typing import Iterable

SENT_END = "</s>"
SENT_START = "<s>"

_WORD_RE = re.compile(
    r"[A-Za-z0-9_'’]+"             # ラテン語・数字
    r"|[぀-ヿㇰ-ㇿ]"  # ひらがな・カタカナ
    r"|[㐀-䶿一-鿿々〆]"  # 漢字・々
    r"|[가-힯]"             # ハングル
    r"|[^\sA-Za-z0-9_'’぀-ヿㇰ-ㇿ㐀-䶿一-鿿々〆가-힯]"
)
# 文末: 。！？ の後 (ただし閉じ括弧・引用符が続けばその後)、英語のピリオド + 空白 + 大文字、改行
_SENT_SPLIT_RE = re.compile(
    r"(?<=[。！？!?])(?![」』）)】])\s*"                       # 。！？ の後 (閉じ括弧が続く時は区切らない)
    r"|(?<=\.)(?<!Mr\.)(?<!Mrs\.)(?<!Dr\.)(?<!Ms\.)(?<!St\.)(?<!No\.)(?<!vs\.)(?<!etc\.)(?<!Jr\.)(?<!Prof\.)(?<!Inc\.)(?<!Ltd\.)\s+(?=[A-Z0-9])"  # 英語のピリオド (略語を除く)
    r"|\n+"
)
_SPACE_RE = re.compile(r"[ \t　]+")

_KANA = "぀-ヿㇰ-ㇿ"
_KANJI = "㐀-䶿一-鿿々〆"
_HANGUL = "가-힯"
_CJK_RUN_RE = re.compile(rf"[{_KANA}{_KANJI}{_HANGUL}]+")
_KANJI_KATA_RUN_RE = re.compile(rf"[{_KANJI}゠-ヿ]+")
_LATIN_RE = re.compile(r"[a-z0-9][a-z0-9_'’]*")

STOPWORDS_EN = set(
    "a an the of to in on at for and or but is are was were be been being am do does did "
    "i you he she it we they me him her us them my your his its our their this that these those "
    "whom whose not no yes so if then than too very can could "
    "will would shall should may might must have has had with without from by as about into over "
    "up down out off just also there here".split()
)
# 単独では意味の薄いかな
STOPWORDS_KANA = set("はがのをにへとでもやかなねよねぇーっぁぃぅぇぉゃゅょわ")


def normalize(text: str) -> str:
    if not unicodedata.is_normalized("NFKC", text):
        text = unicodedata.normalize("NFKC", text)
    if "\r" in text:
        text = text.replace("\r", "\n")
    if "  " in text or "\t" in text or "\u3000" in text:
        text = _SPACE_RE.sub(" ", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    text = normalize(text)
    out = []
    for s in _SENT_SPLIT_RE.split(text):
        s = s.strip()
        if len(s) >= 4:
            # 「」で始まり閉じない断片は前の文に付ける (会話の引用が途中で切れるのを防ぐ)
            if out and s[0] in "」』）)" :
                out[-1] += s
            else:
                out.append(s)
    return out


def tokenize(text: str) -> list[str]:
    """言語モデル用トークン列 (小文字化)。"""
    text = normalize(text).lower()
    return _WORD_RE.findall(text)


def detokenize(tokens: Iterable[str]) -> str:
    """トークン列を自然な文字列に戻す。ラテン語間のみ空白を入れる。"""
    out: list[str] = []
    prev_latin = False
    for t in tokens:
        if t in (SENT_START, SENT_END):
            continue
        is_latin = bool(_LATIN_RE.fullmatch(t))
        if out and is_latin and prev_latin:
            out.append(" ")
        elif out and is_latin and out[-1] in ",.;:!?":
            out.append(" ")
        out.append(t)
        prev_latin = is_latin
    return "".join(out)


def is_question(text: str) -> bool:
    t = normalize(text)
    if not t:
        return False
    if t.endswith(("?", "？")):
        return True
    tail = t[-4:]
    if any(tail.endswith(x) for x in ("か", "かな", "の", "何", "なに", "誰", "どこ", "いつ", "なぜ", "どう", "どれ", "教えて", "って")):
        return True
    low = t.lower()
    return low.split(" ", 1)[0] in {"what", "who", "when", "where", "why", "how", "which", "is", "are", "do", "does", "can", "tell"}


def terms(text: str) -> list[str]:
    """検索インデックス用の語。ラテン語 + CJK 文字 bigram + 漢字/カタカナ連続語。"""
    text = normalize(text).lower()
    out: list[str] = []
    for w in _LATIN_RE.findall(text):
        if len(w) >= 2:  # ストップワードも残す (重みは term_weight で下げる)
            out.append(w)
    for run in _CJK_RUN_RE.findall(text):
        if len(run) == 1:
            if run not in STOPWORDS_KANA:
                out.append(run)
            continue
        for i in range(len(run) - 1):
            bg = run[i : i + 2]
            if bg[0] in STOPWORDS_KANA and bg[1] in STOPWORDS_KANA:
                continue
            out.append(bg)
    for run in _KANJI_KATA_RUN_RE.findall(text):
        if 2 <= len(run) <= 12:
            out.append(run)
    if not out:
        out = [ch for ch in text if ch.isalnum()]
    return out


def term_weight(t: str) -> float:
    """検索語の情報量の目安。かなだけの語は弱い。"""
    if all("\u3040" <= ch <= "\u30ff" for ch in t) and len(t) <= 2:
        return 0.4
    if t in STOPWORDS_EN:
        return 0.3
    return 1.0


_PHRASE_RE = re.compile(rf"[{_KANJI}\u30a0-\u30ff]{{2,}}|[a-z][a-z0-9_'’]{{2,}}")


def is_phrase(t: str) -> bool:
    """「話題語」として意味を持つ語か (漢字/カタカナ 2 文字以上の連続、または 3 文字以上のラテン語)。"""
    return bool(_PHRASE_RE.fullmatch(t)) and t not in STOPWORDS_EN


def phrases(text: str) -> list[str]:
    """文中の句 (漢字/カタカナ 2〜12 文字の連続、3 文字以上の英単語) を出現順に。意味ベクトル用。"""
    text = normalize(text)
    out: list[str] = []
    for m in _PHRASE_SEQ_RE.finditer(text.lower()):
        w = m.group(0)
        if w.isascii():
            if len(w) >= 3 and w not in STOPWORDS_EN and not w.isdigit():
                out.append(w)
        elif 2 <= len(w) <= 12:
            out.append(w)
    return out


_PHRASE_SEQ_RE = re.compile(rf"[{_KANJI}\u30a0-\u30ff]+|[a-z][a-z0-9_'’]*")


class SentenceInfo:
    """1 文の解析結果をまとめて持つ (学習の各段階で同じ正規表現を何度も走らせないため)。"""
    __slots__ = ("text", "tokens", "terms", "phrases", "content_key")

    def __init__(self, text: str, tokens: list[str], terms_: list[str], phrases_: list[str], content_key: int | None = None):
        self.text, self.tokens, self.terms, self.phrases, self.content_key = text, tokens, terms_, phrases_, content_key


def analyze(text: str) -> SentenceInfo:
    """正規化済みの 1 文から、LM トークン・検索語・句を 1 回の走査で作る。"""
    low = text.lower()
    tokens = _WORD_RE.findall(low)
    terms_: list[str] = []
    phrases_: list[str] = []
    strong: set[str] = set()  # 情報量 1.0 の語 (近似重複の鍵に使う)
    for w in _LATIN_RE.findall(low):
        if len(w) >= 2:
            terms_.append(w)
            if w not in STOPWORDS_EN:
                strong.add(w)
        if len(w) >= 3 and w not in STOPWORDS_EN and not w.isdigit():
            phrases_.append(w)
    for run in _CJK_RUN_RE.findall(low):
        if len(run) == 1:
            if run not in STOPWORDS_KANA:
                terms_.append(run)
                if not ("\u3040" <= run <= "\u30ff"):
                    strong.add(run)
            continue
        for i in range(len(run) - 1):
            bg = run[i : i + 2]
            if bg[0] in STOPWORDS_KANA and bg[1] in STOPWORDS_KANA:
                continue
            terms_.append(bg)
            # かなだけの bigram は弱い (term_weight 0.4)。漢字/カタカナを含めば強い
            if not (("\u3040" <= bg[0] <= "\u30ff") and ("\u3040" <= bg[1] <= "\u30ff")):
                strong.add(bg)
    for run in _KANJI_KATA_RUN_RE.findall(low):
        n = len(run)
        if 2 <= n <= 12:
            terms_.append(run)
            phrases_.append(run)
            strong.add(run)
    if not terms_:
        terms_ = [ch for ch in low if ch.isalnum()]
    key = hash(tuple(sorted(strong))) if len(strong) >= 4 else None
    return SentenceInfo(text, tokens, terms_, phrases_, key)


def keywords(text: str, limit: int = 6) -> list[str]:
    """人間に見せたり検索クエリに使う「話題語」。漢字/カタカナ連続語とラテン語を優先。"""
    text = normalize(text)
    cands: list[str] = []
    for run in _KANJI_KATA_RUN_RE.findall(text):
        if 2 <= len(run) <= 12:
            cands.append(run)
    for w in _LATIN_RE.findall(text.lower()):
        if len(w) >= 3 and w not in STOPWORDS_EN and not w.isdigit():
            cands.append(w)
    seen = set()
    out = []
    for c in sorted(cands, key=len, reverse=True):
        if c not in seen:
            seen.add(c)
            out.append(c)
        if len(out) >= limit:
            break
    return out
