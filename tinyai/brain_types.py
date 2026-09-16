"""質問タイプの判定とタイプ別のリランク (brain と evolution の両方から使う)。"""
from __future__ import annotations

import re

from .tokenizer import is_question


# ---------------------------------------------------------------- 質問タイプ
_QTYPE_PATTERNS = [
    ("when", re.compile(r"いつ|何年|何月|何日|何世紀|\bwhen\b|what year", re.I)),
    ("where", re.compile(r"どこ|何処|どちら|\bwhere\b", re.I)),
    ("howmany", re.compile(r"いくつ|いくら|何人|何個|何回|何歳|どれくらい|どのくらい|高さ|長さ|広さ|人口|距離|面積|重さ|速さ|how (many|much|tall|long|far|old|big|fast)", re.I)),
    ("who", re.compile(r"誰|だれ|\bwho\b", re.I)),
    ("why", re.compile(r"なぜ|何故|どうして|\bwhy\b", re.I)),
    ("definition", re.compile(r"とは|って何|ってなに|とは何|何ですか|なんですか|何のこと|意味|what is|what are|what's|define|explain|教えて", re.I)),
]
_YEAR_RE = re.compile(r"\d{3,4}\s*年|\d+\s*世紀|\b1\d{3}\b|\b20\d{2}\b|年代|月|日")
_PLACE_RE = re.compile(r"[都道府県市区町村国島湾山川洲]|に位置|にある|located|\bin\b")
_NUMBER_RE = re.compile(r"\d[\d,.]*\s*(m|km|メートル|キロ|人|個|km2|km²|平方|トン|kg|グラム|秒|分|時間|年|歳|倍|%|パーセント|円|ドル|億|万|千)")
_WHO_RE = re.compile(r"氏|さん|博士|教授|作家|学者|者|創業|設立|発明|開発|by\b|founded|invented")


def question_type(text: str) -> str:
    for name, pat in _QTYPE_PATTERNS:
        if pat.search(text):
            return name
    return "definition" if is_question(text) else "none"


def _rerank_bonus(qtype: str, doc_text: str, subject: str | None) -> float:
    """質問タイプに合う文に加点 (0..1)。"""
    b = 0.0
    if qtype == "definition":
        if subject:
            head = doc_text[: len(subject) + 4]
            if head.startswith(subject) or subject in head:
                b += 0.6
            if re.search(re.escape(subject) + r"\s*(とは|は|が|、|は、|とは、|\(|（)", doc_text):
                b += 0.4
        if re.search(r"とは|である|です|のこと|を指す|refers to|is a|is the|は、", doc_text):
            b += 0.2
    elif qtype == "when":
        b += 1.0 if _YEAR_RE.search(doc_text) else 0.0
    elif qtype == "where":
        b += 1.0 if _PLACE_RE.search(doc_text) else 0.0
    elif qtype == "howmany":
        b += 1.0 if _NUMBER_RE.search(doc_text) else (0.3 if re.search(r"\d", doc_text) else 0.0)
    elif qtype == "who":
        b += 0.8 if _WHO_RE.search(doc_text) else 0.0
    elif qtype == "why":
        b += 0.8 if re.search(r"ため|から|ので|理由|原因|because|due to", doc_text) else 0.0
    return min(b, 1.0)


