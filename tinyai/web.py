"""Web 取得・検索 (標準ライブラリのみ)。

* robots.txt を尊重し、1 ホストあたりの取得間隔を空ける
* HTML -> 本文テキスト抽出 (script/style/nav などを除外)
* 検索は Wikipedia API (ja/en) と DuckDuckGo HTML 版を使う
"""
from __future__ import annotations

import gzip
import json
import logging
import os
import re
import ssl
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import urllib.robotparser
from html.parser import HTMLParser

log = logging.getLogger("tinyai.web")

_SKIP_TAGS = {"script", "style", "noscript", "nav", "header", "footer", "aside", "form", "svg", "iframe", "template"}
_BLOCK_TAGS = {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "td", "th", "section", "article", "blockquote", "pre", "dd", "dt"}


# Wikipedia などでよくある「本文ではない」ブロックの class/role
_SKIP_CLASS_RE = re.compile(
    r"hatnote|ambox|mbox|navbox|infobox|sidebar|reference|reflist|catlinks|mw-editsection|toc\b|metadata|noprint|"
    r"cookie|banner|advert|breadcrumb|sitesub|printfooter|footer|menu|comment",
    re.I,
)


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.links: list[str] = []
        self.title = ""
        self._skip = 0
        self._skip_stack: list[str] = []
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP_TAGS:
            self._skip += 1
            self._skip_stack.append(tag)
        elif tag == "title":
            self._in_title = True
        elif tag == "a":
            for k, v in attrs:
                if k == "href" and v:
                    self.links.append(v)
        if tag in ("div", "table", "span", "ul", "ol", "section", "aside", "p") and not self._skip:
            for k, v in attrs:
                if k in ("class", "role", "id") and v and (_SKIP_CLASS_RE.search(v) or v == "note"):
                    self._skip += 1
                    self._skip_stack.append(tag)
                    break
        if tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if self._skip and self._skip_stack and self._skip_stack[-1] == tag:
            self._skip_stack.pop()
            self._skip -= 1
        elif tag == "title":
            self._in_title = False
        if tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        if not self._skip:
            self.parts.append(data)


def html_to_text(html: str) -> tuple[str, list[str], str]:
    """(本文, リンク一覧, タイトル) を返す。"""
    p = _TextExtractor()
    try:
        p.feed(html)
        p.close()
    except Exception:  # 壊れた HTML でも部分結果を返す
        pass
    text = "".join(p.parts)
    text = re.sub(r"[ \t　]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    lines = [ln.strip() for ln in text.split("\n")]
    # 短すぎる行 (メニュー等) は捨てる
    lines = [ln for ln in lines if len(ln) >= 20 or ln.endswith(("。", ".", "!", "?", "！", "？"))]
    return "\n".join(lines), p.links, p.title.strip()


class Fetcher:
    def __init__(self, user_agent: str, timeout: float = 12.0, max_bytes: int = 400_000, per_host_delay: float = 2.0):
        self.user_agent = user_agent
        self.timeout = timeout
        self.max_bytes = max_bytes
        self.per_host_delay = per_host_delay
        self._robots: dict[str, urllib.robotparser.RobotFileParser | None] = {}
        self._last_hit: dict[str, float] = {}
        self._cooldown: dict[str, float] = {}   # host -> この時刻まで叩かない (429/503 対策)
        self._backoff: dict[str, float] = {}    # host -> 現在のバックオフ秒
        self._lock = threading.Lock()
        self._ssl = ssl.create_default_context()
        ca = os.environ.get("SSL_CERT_FILE") or os.environ.get("REQUESTS_CA_BUNDLE")
        if ca and os.path.exists(ca):
            try:
                self._ssl.load_verify_locations(ca)
            except ssl.SSLError:
                pass
        self.fetched = 0
        self.failed = 0

    # ------------------------------------------------------------ robots
    def _allowed(self, url: str) -> bool:
        parts = urllib.parse.urlsplit(url)
        host = parts.netloc
        # MediaWiki API はクローラ向け robots.txt の対象外 (API 利用規約に従い UA と間隔を守る)
        if parts.path == "/w/api.php" and host.endswith((".wikipedia.org", ".wikimedia.org")):
            return True
        rp = self._robots.get(host, "unset")
        if rp == "unset":
            rp = urllib.robotparser.RobotFileParser()
            robots_url = f"{parts.scheme}://{host}/robots.txt"
            try:
                raw = self._raw_get(robots_url, limit=200_000)
                rp.parse(raw.decode("utf-8", "replace").splitlines())
            except Exception:
                rp = None  # 取れなければ許可扱い
            self._robots[host] = rp
        if rp is None:
            return True
        try:
            return rp.can_fetch(self.user_agent, url)
        except Exception:
            return True

    def _throttle(self, host: str) -> None:
        with self._lock:
            last = self._last_hit.get(host, 0.0)
            wait = self.per_host_delay - (time.time() - last)
            if wait > 0:
                time.sleep(wait)
            self._last_hit[host] = time.time()

    def _raw_get(self, url: str, limit: int | None = None) -> bytes:
        req = urllib.request.Request(url, headers={"User-Agent": self.user_agent, "Accept-Encoding": "gzip", "Accept-Language": "ja,en;q=0.7"})
        with urllib.request.urlopen(req, timeout=self.timeout, context=self._ssl) as r:
            data = r.read((limit or self.max_bytes) + 1)
            if r.headers.get("Content-Encoding") == "gzip":
                try:
                    data = gzip.decompress(data)
                except OSError:
                    pass
            return data[: (limit or self.max_bytes)]

    # ------------------------------------------------------------ 公開 API
    def get(self, url: str) -> bytes | None:
        if not url.startswith(("http://", "https://")):
            return None
        host = urllib.parse.urlsplit(url).netloc
        until = self._cooldown.get(host, 0.0)
        if until > time.time():
            log.info("クールダウン中 (%ds) %s", until - time.time(), host)
            return None
        if not self._allowed(url):
            log.info("robots.txt により拒否: %s", url)
            return None
        self._throttle(host)
        try:
            data = self._raw_get(url)
            self.fetched += 1
            self._backoff.pop(host, None)
            return data
        except urllib.error.HTTPError as e:
            self.failed += 1
            if e.code in (429, 503):
                retry = e.headers.get("Retry-After") if e.headers else None
                try:
                    wait = float(retry) if retry else 0.0
                except ValueError:
                    wait = 0.0
                back = self._backoff.get(host, 30.0)
                wait = max(wait, back)
                self._backoff[host] = min(back * 2, 3600.0)
                self._cooldown[host] = time.time() + wait
                log.info("レート制限 %s: %ds 待機", host, wait)
            else:
                log.info("取得失敗 %s: %s", url, e)
            return None
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
            self.failed += 1
            log.info("取得失敗 %s: %s", url, e)
            return None

    def get_json(self, url: str):
        data = self.get(url)
        if data is None:
            return None
        try:
            return json.loads(data.decode("utf-8", "replace"))
        except ValueError:
            return None

    def get_text(self, url: str) -> tuple[str, list[str], str] | None:
        data = self.get(url)
        if data is None:
            return None
        html = data.decode("utf-8", "replace")
        return html_to_text(html)

    # ------------------------------------------------------------ 検索
    def wikipedia_search(self, query: str, lang: str = "ja", limit: int = 3) -> list[str]:
        q = urllib.parse.quote(query)
        url = f"https://{lang}.wikipedia.org/w/api.php?action=query&list=search&srsearch={q}&format=json&srlimit={limit}&utf8=1"
        js = self.get_json(url)
        if not js:
            return []
        return [hit["title"] for hit in js.get("query", {}).get("search", [])]

    def wikipedia_extract(self, title: str, lang: str = "ja", max_chars: int = 6000) -> str:
        t = urllib.parse.quote(title)
        url = (
            f"https://{lang}.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext=1&exintro=0"
            f"&titles={t}&format=json&utf8=1&exchars={max_chars}"
        )
        js = self.get_json(url)
        if not js:
            return ""
        pages = js.get("query", {}).get("pages", {})
        for page in pages.values():
            txt = page.get("extract", "")
            # 見出し (== xx ==) を除去
            txt = re.sub(r"\n=+[^=\n]+=+\n", "\n", txt)
            return txt
        return ""

    # api.wikimedia.org は wikipedia.org 本体とは別ホストで、レート制限も別に掛かる
    def wikimedia_search(self, query: str, lang: str = "ja", limit: int = 3) -> list[str]:
        q = urllib.parse.quote(query)
        js = self.get_json(f"https://api.wikimedia.org/core/v1/wikipedia/{lang}/search/page?q={q}&limit={limit}")
        if not js:
            return []
        return [p["key"] for p in js.get("pages", []) if p.get("key")]

    def wikimedia_page_text(self, key: str, lang: str = "ja") -> str:
        k = urllib.parse.quote(key)
        data = self.get(f"https://api.wikimedia.org/core/v1/wikipedia/{lang}/page/{k}/html")
        if not data:
            return ""
        text, _, _ = html_to_text(data.decode("utf-8", "replace"))
        # 出典番号 [1] や編集リンクを除く
        text = re.sub(r"\[\d+\]|\[編集\]|\[edit\]", "", text)
        return text

    def crawl_site(self, start_url: str, max_pages: int = 3, seen: set | None = None) -> list[tuple[str, str]]:
        """同一ドメイン内でリンクをたどり、最大 max_pages ページの本文を返す。"""
        seen = seen if seen is not None else set()
        host = urllib.parse.urlsplit(start_url).netloc
        queue = [start_url]
        out: list[tuple[str, str]] = []
        while queue and len(out) < max_pages:
            url = queue.pop(0)
            if url in seen:
                continue
            seen.add(url)
            res = self.get_text(url)
            if not res:
                continue
            text, links, _ = res
            if len(text) > 200:
                out.append((url, text))
            for href in links:
                full = urllib.parse.urljoin(url, href.split("#", 1)[0])
                p = urllib.parse.urlsplit(full)
                if p.netloc == host and p.scheme in ("http", "https") and full not in seen and len(queue) < 50:
                    if not re.search(r"\.(png|jpe?g|gif|svg|pdf|zip|css|js|ico|mp[34]|webp)$", p.path, re.I):
                        queue.append(full)
        return out

    def duckduckgo_search(self, query: str, limit: int = 5) -> list[str]:
        q = urllib.parse.quote_plus(query)
        res = self.get_text(f"https://html.duckduckgo.com/html/?q={q}")
        if not res:
            return []
        _, links, _ = res
        out: list[str] = []
        for href in links:
            # 結果リンクは /l/?uddg=<encoded url> 形式
            if "uddg=" in href:
                qs = urllib.parse.parse_qs(urllib.parse.urlsplit(href).query)
                target = qs.get("uddg", [""])[0]
            elif href.startswith("http") and "duckduckgo.com" not in href:
                target = href
            else:
                continue
            if target and target not in out:
                out.append(target)
            if len(out) >= limit:
                break
        return out

    def search_and_read(self, query: str, langs=("ja", "en"), max_pages: int = 3) -> list[tuple[str, str]]:
        """クエリで探索し [(source, text), ...] を返す。Wikipedia を優先し、
        足りなければ DuckDuckGo の結果ページを読む。"""
        out: list[tuple[str, str]] = []
        # 日本語のクエリは日本語版だけ、ASCII のクエリは指定言語すべてを見る
        if not query.isascii():
            langs = [l for l in langs if l == "ja"] or list(langs)[:1]
        for lang in langs:
            # 1) Wikimedia Core API  2) Wikipedia Action API (片方が制限されても片方で読める)
            keys = self.wikimedia_search(query, lang=lang, limit=2)
            for key in keys:
                txt = self.wikimedia_page_text(key, lang=lang)
                if txt and len(txt) > 200:
                    out.append((f"https://{lang}.wikipedia.org/wiki/{urllib.parse.quote(key)}", txt))
                if len(out) >= max_pages:
                    return out
            if not keys:
                for title in self.wikipedia_search(query, lang=lang, limit=2):
                    txt = self.wikipedia_extract(title, lang=lang)
                    if txt and len(txt) > 200:
                        out.append((f"https://{lang}.wikipedia.org/wiki/{urllib.parse.quote(title)}", txt))
                    if len(out) >= max_pages:
                        return out
        if len(out) < max_pages:
            for url in self.duckduckgo_search(query, limit=max_pages):
                res = self.get_text(url)
                if res and len(res[0]) > 300:
                    out.append((url, res[0]))
                if len(out) >= max_pages:
                    break
        return out
