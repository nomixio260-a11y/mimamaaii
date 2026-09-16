"""MediaWiki のウィキテキストを平文にする軽量クリーナ (ダンプ取り込み用)。

完全な構文解析はしない。テンプレート・表・参照・ファイル・カテゴリ・見出し記号を落とし、
リンクは表示文字だけ残す。数 GB のダンプを流すので正規表現は少数・単純に。
"""
from __future__ import annotations

import html
import re

_REF_RE = re.compile(r"<ref[^>/]*/>|<ref[^>]*>.*?</ref>", re.S)
_COMMENT_RE = re.compile(r"<!--.*?-->", re.S)
_TAG_RE = re.compile(r"<[^>]+>")
_FILE_RE = re.compile(r"\[\[(?:ファイル|File|Image|画像|Category|カテゴリ):[^\]]*\]\]", re.I)
_LINK_RE = re.compile(r"\[\[([^\]|]*)(?:\|([^\]]*))?\]\]")
_EXTLINK_RE = re.compile(r"\[https?://[^\s\]]+\s*([^\]]*)\]")
_HEADING_RE = re.compile(r"^=+\s*(.*?)\s*=+\s*$", re.M)
_TABLE_RE = re.compile(r"\{\|.*?\|\}", re.S)
_BOLD_RE = re.compile(r"'{2,5}")
_LIST_RE = re.compile(r"^[*#:;]+\s*", re.M)


def _strip_templates(text: str) -> str:
    """{{ ... }} を入れ子込みで除去 (1 パス)。"""
    out = []
    depth = 0
    i = 0
    n = len(text)
    while i < n:
        if text.startswith("{{", i):
            depth += 1
            i += 2
            continue
        if text.startswith("}}", i) and depth:
            depth -= 1
            i += 2
            continue
        if not depth:
            out.append(text[i])
        i += 1
    return "".join(out)


def wikitext_to_text(src: str, max_chars: int = 60000) -> str:
    src = src[:max_chars]
    src = _COMMENT_RE.sub("", src)
    src = _REF_RE.sub("", src)
    src = _TABLE_RE.sub("", src)
    src = _strip_templates(src)
    src = _FILE_RE.sub("", src)
    src = _LINK_RE.sub(lambda m: m.group(2) if m.group(2) is not None else m.group(1), src)
    src = _EXTLINK_RE.sub(lambda m: m.group(1), src)
    src = _HEADING_RE.sub("", src)
    src = _TAG_RE.sub("", src)
    src = _BOLD_RE.sub("", src)
    src = _LIST_RE.sub("", src)
    src = html.unescape(src)
    lines = [ln.strip() for ln in src.split("\n")]
    return "\n".join(ln for ln in lines if len(ln) >= 8)
