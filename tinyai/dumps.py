"""大量データのストリーミング取り込み (メモリ一定)。

* Wikipedia などの XML ダンプ (.xml / .xml.bz2 / .xml.gz): ページごとに本文を取り出して学習
* テキストの書庫 (.zip / .gz / .bz2 / .xz): 中のテキストを順に学習
* ディレクトリ: 再帰的に走査

どれも 1 ページ (または 1 ファイル) ずつ処理するので、数 GB のダンプでもメモリは増えない。
学習側の admission (品質しきい値) とメモリ制御が、上限内で「良い文だけ残る」ように働く。
"""
from __future__ import annotations

import bz2
import gzip
import logging
import lzma
import re
import zipfile
from pathlib import Path
from typing import Callable, Iterator

from .wikitext import wikitext_to_text

log = logging.getLogger("tinyai.dumps")

_TITLE_RE = re.compile(r"<title>(.*?)</title>")
_NS_RE = re.compile(r"<ns>(\d+)</ns>")
_TEXT_START_RE = re.compile(r"<text[^>]*>")


def _open_any(path: Path):
    name = path.name.lower()
    if name.endswith(".bz2"):
        return bz2.open(path, "rt", encoding="utf-8", errors="replace")
    if name.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    if name.endswith(".xz"):
        return lzma.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def iter_wiki_pages(path: Path, max_pages: int | None = None) -> Iterator[tuple[str, str]]:
    """MediaWiki XML ダンプから (title, wikitext) を順に返す。名前空間 0 (記事) だけ。"""
    n = 0
    with _open_any(path) as f:
        title = ""
        ns = 0
        buf: list[str] | None = None
        for line in f:
            if buf is None:
                # title / ns / text が同じ行にあっても拾えるように順に見る
                m = _TITLE_RE.search(line)
                if m:
                    title = m.group(1)
                    ns = 0
                m = _NS_RE.search(line)
                if m:
                    ns = int(m.group(1))
                m = _TEXT_START_RE.search(line)
                if m and ns == 0:
                    rest = line[m.end():]
                    if "</text>" in rest:
                        yield title, rest.split("</text>", 1)[0]
                        n += 1
                    else:
                        buf = [rest]
            else:
                if "</text>" in line:
                    buf.append(line.split("</text>", 1)[0])
                    yield title, "".join(buf)
                    n += 1
                    buf = None
                else:
                    buf.append(line)
            if max_pages is not None and n >= max_pages:
                return


def _is_redirect(wikitext: str) -> bool:
    head = wikitext[:40].lstrip().lower()
    return head.startswith("#redirect") or head.startswith("#転送")


def learn_wiki_dump(brain, path: str | Path, max_pages: int | None = None, progress: Callable[[int, int], None] | None = None) -> tuple[int, int]:
    """ダンプを流して学習。(ページ数, 文数) を返す。"""
    pages = sents = 0
    for title, wt in iter_wiki_pages(Path(path), max_pages):
        if _is_redirect(wt) or len(wt) < 60:
            continue
        text = wikitext_to_text(wt)
        if len(text) < 40:
            continue
        sents += brain.learn_text(text, source=f"dump:{title[:60]}")
        pages += 1
        if progress and pages % 200 == 0:
            progress(pages, sents)
    return pages, sents


def iter_archive_texts(path: Path, max_bytes_each: int = 8 * 1024 * 1024) -> Iterator[tuple[str, str]]:
    """zip / gz / bz2 / xz / 平文の中のテキストを (名前, 本文) で順に返す。"""
    name = path.name.lower()
    if name.endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            for info in z.infolist():
                if info.is_dir() or info.file_size > max_bytes_each:
                    continue
                if not info.filename.lower().endswith((".txt", ".md", ".html", ".htm", ".csv", ".json", ".xml")):
                    continue
                raw = z.read(info)
                text = _decode(raw)
                if info.filename.lower().endswith((".html", ".htm")):
                    from .web import html_to_text

                    text = html_to_text(text)[0]
                yield f"{path.name}/{info.filename}", text
        return
    if name.endswith((".gz", ".bz2", ".xz")):
        with _open_any(path) as f:
            yield path.name, f.read(max_bytes_each)
        return
    yield path.name, path.read_bytes()[:max_bytes_each].decode("utf-8", "replace")


def _decode(raw: bytes) -> str:
    for enc in ("utf-8", "cp932", "euc_jp"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", "replace")
