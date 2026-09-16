"""エージェント層: 意図を理解し、道具 (計算・日付・単位換算・比較・列挙・要約・その場での調査・
利用者プロファイル) を正確に使い、指示 (箇条書き・一言で・N 文字以内) に合わせて答えを整える。

ChatGPT 型の対話体験の多くは、モデル本体ではなくこの層が担う:
  * 計算は言語モデルに推測させず、安全な式評価器で正確に解く
  * 「今日は何日」「3 日後は何曜日」は datetime で解く
  * 「A と B の違い」「どちらが高い」は事実ストアの数値を比べる
  * 「X の例を 3 つ」は知識文の列挙を抽出する
  * 「要約して」は URL / 貼り付けた文章を抽出型要約する
  * 「調べて」/ 未知の話題は、その場で収集して学んでから答える (時間上限つき)
  * 「私の名前は◯◯」は利用者プロファイルとして記憶し、以後の会話で使う
"""
from __future__ import annotations

import ast
import datetime as _dt
import logging
import math
import operator
import re
import time
from collections import Counter

from .tokenizer import is_phrase, keywords, normalize, phrases, split_sentences, terms

log = logging.getLogger("tinyai.agent")

# ---------------------------------------------------------------- 計算
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.Mod: operator.mod, ast.FloorDiv: operator.floordiv, ast.USub: operator.neg, ast.UAdd: operator.pos}
_FUNCS = {"sqrt": math.sqrt, "abs": abs, "round": round, "sin": math.sin, "cos": math.cos, "tan": math.tan, "log": math.log, "log10": math.log10, "exp": math.exp, "floor": math.floor, "ceil": math.ceil}
_CONSTS = {"pi": math.pi, "e": math.e, "π": math.pi}
_EXPR_CHARS = re.compile(r"[0-9０-９+\-−*×÷/^().,%√ 　]+")
_CALC_HINT = re.compile(r"[+\-−*×÷/^√%]|平方根|の(?:二乗|2乗|三乗|3乗)|乗|計算|足す|引く|掛け|割")


def _safe_eval(node):
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        a, b = _safe_eval(node.left), _safe_eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(b) > 1000:
            raise ValueError("指数が大きすぎます")
        return _OPS[type(node.op)](a, b)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS:
        return _FUNCS[node.func.id](*[_safe_eval(a) for a in node.args])
    if isinstance(node, ast.Name) and node.id in _CONSTS:
        return _CONSTS[node.id]
    raise ValueError("許可されていない式")


def calculate(text: str) -> tuple[str, float] | None:
    """文中の算術式を取り出して評価。(整形した式, 値) または None。"""
    t = normalize(text)
    t = t.replace("×", "*").replace("÷", "/").replace("−", "-").replace("^", "**").replace("、", ",")
    t = re.sub(r"(\d+(?:\.\d+)?)\s*の\s*平方根", r"sqrt(\1)", t)
    t = re.sub(r"√\s*(\d+(?:\.\d+)?)", r"sqrt(\1)", t)
    t = re.sub(r"(\d+(?:\.\d+)?)\s*の\s*(\d+)\s*乗", r"(\1)**\2", t)
    t = re.sub(r"(\d+(?:\.\d+)?)\s*(?:足す|プラス|たす)\s*(\d+(?:\.\d+)?)", r"\1+\2", t)
    t = re.sub(r"(\d+(?:\.\d+)?)\s*(?:引く|マイナス|ひく)\s*(\d+(?:\.\d+)?)", r"\1-\2", t)
    t = re.sub(r"(\d+(?:\.\d+)?)\s*(?:掛ける|かける|倍)\s*(\d+(?:\.\d+)?)", r"\1*\2", t)
    t = re.sub(r"(\d+(?:\.\d+)?)\s*(?:割る|わる)\s*(\d+(?:\.\d+)?)", r"\1/\2", t)
    t = re.sub(r"(\d+(?:\.\d+)?)\s*%\s*(?:の|of)\s*(\d+(?:\.\d+)?)", r"(\1/100)*\2", t)
    t = re.sub(r"(\d+(?:\.\d+)?)\s*[円個人本枚台]?\s*の\s*(\d+(?:\.\d+)?)\s*%", r"(\1)*(\2/100)", t)
    m = re.search(r"[-(]?\s*(?:sqrt\()?\d[\d.,]*[\d.,()*+\-/ %sqrt]*", t)
    if not m:
        return None
    expr = m.group(0).replace(",", "").replace(" ", "").rstrip("=.")
    expr = re.sub(r"[+\-*/(]+$", "", expr)
    if not re.search(r"[+\-*/]|\*\*|sqrt", expr) or not re.search(r"\d", expr):
        return None
    try:
        val = _safe_eval(ast.parse(expr, mode="eval"))
    except (ValueError, SyntaxError, ZeroDivisionError, OverflowError, TypeError):
        return None
    shown = expr.replace("**", "^").replace("*", "×").replace("/", "÷")
    return shown, val


def _fmt_num(v: float) -> str:
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    if isinstance(v, int):
        return f"{v:,}"
    return f"{v:,.6g}"


# ---------------------------------------------------------------- 日付
_WEEKDAYS_JA = "月火水木金土日"
_DATE_HINT = re.compile(r"今日|きょう|明日|あした|昨日|きのう|今|何日|何曜日|曜日|日付|何時|何年|来週|来月|来年|日後|週間後|か月後|ヶ月後|年後|日前|west|today|date|time")


def answer_datetime(text: str, now: _dt.datetime | None = None) -> str | None:
    t = normalize(text)
    if not _DATE_HINT.search(t):
        return None
    now = now or _dt.datetime.now()
    ja = not t.isascii()

    def fmt(d: _dt.datetime) -> str:
        return f"{d.year}年{d.month}月{d.day}日（{_WEEKDAYS_JA[d.weekday()]}曜日）" if ja else d.strftime("%Y-%m-%d (%a)")

    m = re.search(r"(\d+)\s*(日後|週間後|か月後|ヶ月後|年後|日前)", t)
    if m:
        n = int(m.group(1))
        unit = m.group(2)
        if unit == "日後":
            d = now + _dt.timedelta(days=n)
        elif unit == "日前":
            d = now - _dt.timedelta(days=n)
        elif unit == "週間後":
            d = now + _dt.timedelta(weeks=n)
        elif unit in ("か月後", "ヶ月後"):
            mo = now.month - 1 + n
            d = now.replace(year=now.year + mo // 12, month=mo % 12 + 1, day=min(now.day, 28))
        else:
            d = now.replace(year=now.year + n)
        return f"{n}{unit}は {fmt(d)} です。"
    if re.search(r"何時|時刻|time", t):
        return f"今は {now.strftime('%H:%M')} です。" if ja else f"It is {now.strftime('%H:%M')}."
    if re.search(r"明日|あした", t):
        return f"明日は {fmt(now + _dt.timedelta(days=1))} です。"
    if re.search(r"昨日|きのう", t):
        return f"昨日は {fmt(now - _dt.timedelta(days=1))} でした。"
    if re.search(r"来週", t):
        return f"来週の今日は {fmt(now + _dt.timedelta(days=7))} です。"
    if re.search(r"今日|きょう|何曜日|何日|日付|today|date|何年", t):
        return f"今日は {fmt(now)} です。" if ja else f"Today is {fmt(now)}."
    return None


# ---------------------------------------------------------------- 単位換算
_UNITS = {
    "km": ("length", 1000.0), "キロ": ("length", 1000.0), "キロメートル": ("length", 1000.0), "m": ("length", 1.0), "メートル": ("length", 1.0),
    "cm": ("length", 0.01), "センチ": ("length", 0.01), "mm": ("length", 0.001), "ミリ": ("length", 0.001),
    "マイル": ("length", 1609.344), "mile": ("length", 1609.344), "miles": ("length", 1609.344), "フィート": ("length", 0.3048), "ft": ("length", 0.3048),
    "インチ": ("length", 0.0254), "inch": ("length", 0.0254), "ヤード": ("length", 0.9144),
    "kg": ("mass", 1.0), "キログラム": ("mass", 1.0), "g": ("mass", 0.001), "グラム": ("mass", 0.001), "ポンド": ("mass", 0.45359237), "lb": ("mass", 0.45359237),
    "オンス": ("mass", 0.028349523), "トン": ("mass", 1000.0), "t": ("mass", 1000.0),
    "l": ("volume", 1.0), "リットル": ("volume", 1.0), "ml": ("volume", 0.001), "ガロン": ("volume", 3.785411784),
    "℃": ("temp", 0), "°c": ("temp", 0), "度": ("temp", 0), "摂氏": ("temp", 0), "°f": ("temp", 1), "華氏": ("temp", 1), "f": ("temp", 1),
    "時間": ("time", 3600.0), "分": ("time", 60.0), "秒": ("time", 1.0), "日": ("time", 86400.0),
}
_UNIT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(" + "|".join(sorted(map(re.escape, _UNITS), key=len, reverse=True)) + r")\s*(?:を|は|って|=)?\s*(?:何|なん)?\s*(" + "|".join(sorted(map(re.escape, _UNITS), key=len, reverse=True)) + r")\s*(?:に|で|だと|に直す|に換算|に変換|\?|？|か)", re.I)


def convert_units(text: str) -> str | None:
    t = normalize(text)
    m = _UNIT_RE.search(t.lower())
    if not m:
        return None
    val, u1, u2 = float(m.group(1)), m.group(2), m.group(3)
    k1, k2 = _UNITS.get(u1), _UNITS.get(u2)
    if not k1 or not k2 or k1[0] != k2[0]:
        return None
    if k1[0] == "temp":
        if k1[1] == 0 and k2[1] == 1:
            res = val * 9 / 5 + 32
        elif k1[1] == 1 and k2[1] == 0:
            res = (val - 32) * 5 / 9
        else:
            return None
    else:
        res = val * k1[1] / k2[1]
    return f"{_fmt_num(val)} {u1} は約 {_fmt_num(round(res, 4))} {u2} です。"


# ---------------------------------------------------------------- 比較
_CMP_RE = re.compile(r"^(.+?)と(.+?)(?:は|では|だと)?(?:どちら|どっち)が(高い|大きい|多い|長い|広い|古い|新しい|速い|重い|深い)")
_DIFF_RE = re.compile(r"^(.+?)と(.+?)の(違い|差|比較)")
_CMP_ATTR = {"高い": ("高さ", "標高", "height"), "大きい": ("面積", "大きさ", "直径", "人口", "area"), "多い": ("人口", "数", "population"), "長い": ("長さ", "全長", "length"),
             "広い": ("面積", "広さ", "area"), "古い": ("設立", "創業", "完成", "event"), "新しい": ("設立", "創業", "完成", "event"), "速い": ("速さ", "速度", "speed"), "重い": ("重さ", "質量", "weight"), "深い": ("深さ", "水深", "depth")}
_NUM_RE = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(億|万|千)?")


def _number_of(obj: str) -> float | None:
    m = _NUM_RE.search(obj.replace(" ", ""))
    if not m:
        return None
    v = float(m.group(1).replace(",", ""))
    mult = {"億": 1e8, "万": 1e4, "千": 1e3}.get(m.group(2), 1.0)
    # 「1 億 2500 万」 のような連結
    rest = obj[m.end():]
    m2 = _NUM_RE.search(rest.replace(" ", "")) if m.group(2) else None
    v = v * mult
    if m2 and m2.group(2):
        v += float(m2.group(1).replace(",", "")) * {"億": 1e8, "万": 1e4, "千": 1e3}[m2.group(2)]
    return v


# ---------------------------------------------------------------- 列挙・要約
_LIST_RE = re.compile(r"^(.+?)の(例|種類|代表例|具体例|一覧|リスト)を?\s*(\d+)?\s*(?:つ|個|件)?\s*(?:ほど|くらい)?(挙げて|あげて|教えて|出して|列挙して|リストアップして)")
_ENUM_SPLIT = re.compile(r"[、,・]|や|および|または|など")


def extract_items(sentences: list[str], topic: str, limit: int) -> list[str]:
    items: list[str] = []
    seen = set()
    for s in sentences:
        # 「A、B、C など」 の並びを拾う
        for m in re.finditer(r"([^。、,]{1,40}(?:[、,][^。、,]{1,40}){1,8})(?:など|等|といった|が挙げられる|がある)", s):
            for j, part in enumerate(re.split(r"[、,]", m.group(1))):
                part = re.sub(r"(など|等|といった|が挙げられる|がある)$", "", part.strip("「」 "))
                if j == 0:
                    part = re.sub(r"^.*?(?:には|としては|として|例えば|たとえば|は|が)", "", part)  # 「手法には決定木」→「決定木」
                if 1 < len(part) <= 30 and part not in seen and topic not in part:
                    seen.add(part)
                    items.append(part)
                    if len(items) >= limit:
                        return items
    return items


def extractive_summary(text: str, max_sentences: int = 3) -> str:
    sents = split_sentences(text)
    if len(sents) <= max_sentences:
        return " ".join(sents)
    freq = Counter()
    for s in sents:
        freq.update(t for t in set(terms(s)) if is_phrase(t))
    scored = []
    for i, s in enumerate(sents):
        ph = [t for t in set(terms(s)) if is_phrase(t)]
        score = sum(freq[t] for t in ph) / (len(ph) + 2)
        score *= 1.0 + 0.5 * (1.0 - i / len(sents))  # 先頭を優先
        if 20 <= len(s) <= 200:
            score *= 1.2
        scored.append((score, i, s))
    top = sorted(scored, reverse=True)[:max_sentences]
    return " ".join(s for _, _, s in sorted(top, key=lambda x: x[1]))


# ---------------------------------------------------------------- 利用者プロファイル
_PROFILE_SET = [
    ("名前", re.compile(r"^(?:私|わたし|僕|ぼく|俺|おれ|自分)の名前は\s*(.+?)(?:です|だ|といいます|と言います|。|$)")),
    ("名前", re.compile(r"^(?:私|わたし|僕|ぼく|俺|おれ)は\s*(.{1,20}?)(?:です|といいます|と言います)[。!！]*$")),
    ("好きなもの", re.compile(r"^(?:私|わたし|僕|ぼく|俺|おれ)は\s*(.+?)が(?:大)?好き")),
    ("趣味", re.compile(r"^(?:私|わたし|僕|ぼく|俺|おれ)の趣味は\s*(.+?)(?:です|だ|。|$)")),
    ("住んでいる場所", re.compile(r"^(?:私|わたし|僕|ぼく|俺|おれ)は\s*(.+?)に住んで")),
    ("仕事", re.compile(r"^(?:私|わたし|僕|ぼく|俺|おれ)(?:は|の仕事は|の職業は)\s*(.+?)(?:をしています|です|だ|。|$)")),
]
_PROFILE_ASK = re.compile(r"^(?:私|わたし|僕|ぼく|俺|おれ)の(名前|好きなもの|好きなの|趣味|住んでいる場所|住所|仕事|職業)(?:は|って|を覚えてる|を覚えている)?[?？。]*$")
_PROFILE_KEY = {"好きなの": "好きなもの", "住所": "住んでいる場所", "職業": "仕事"}

# ---------------------------------------------------------------- 指示 (書式)
_FMT_BULLETS = re.compile(r"箇条書き|リストで|list form|bullet")
_FMT_SHORT = re.compile(r"一言で|ひとことで|短く|簡潔に|手短に|briefly|in one sentence")
_FMT_LIMIT = re.compile(r"(\d+)\s*(?:文字|字)\s*(?:以内|まで|で)")
_FMT_STRIP = re.compile(r"(?:、)?\s*(?:箇条書きで|リストで|一言で|ひとことで|短く|簡潔に|手短に|\d+\s*(?:文字|字)\s*(?:以内|まで|で))\s*")


def apply_format(text: str, instruction: str) -> str:
    out = text
    if _FMT_BULLETS.search(instruction):
        sents = split_sentences(out)
        if len(sents) >= 2:
            out = "\n".join(f"・{s}" for s in sents)
    elif _FMT_SHORT.search(instruction):
        sents = split_sentences(out)
        if sents:
            out = sents[0]
    m = _FMT_LIMIT.search(instruction)
    if m:
        n = int(m.group(1))
        if len(out) > n:
            cut = out[:n]
            k = max(cut.rfind("。"), cut.rfind("、"))
            out = (cut[: k + 1] if k >= n // 2 else cut).rstrip("、") + ("" if out[:n].endswith("。") else "…")
    return out


def strip_format(instruction: str) -> str:
    out = _FMT_STRIP.sub("", instruction).strip()
    out = out.replace("をで", "を").replace("を説明", "について説明").replace("を教えて", "について教えて")
    out = re.sub(r"[をで]$", "", out).strip()
    return out or instruction


# ---------------------------------------------------------------- エージェント
class Agent:
    def __init__(self, brain):
        self.brain = brain
        self.profile: dict[str, str] = {}
        self._fetcher = None
        self._collector = None

    # ------------------------------------------------------------ 道具の準備
    def _get_collector(self):
        if self._collector is None and self.brain.cfg.web_enabled:
            from .collector import Collector
            from .web import Fetcher

            cfg = self.brain.cfg
            self._fetcher = Fetcher(cfg.user_agent, cfg.fetch_timeout, cfg.max_page_bytes)
            self._collector = Collector(self._fetcher, cfg.data_dir, cfg.languages, interest=self.brain.interest_score)
        return self._collector

    # ------------------------------------------------------------ 入口
    def handle(self, text: str, ja: bool):
        """道具で答えられれば Reply を返す。None なら通常の経路へ。"""
        from .brain import Reply

        t = normalize(text)
        # 利用者プロファイルの参照 (質問) を先に、登録 (平叙文) を後に
        m = _PROFILE_ASK.match(t)
        if m:
            key = _PROFILE_KEY.get(m.group(1), m.group(1))
            v = self.profile.get(key)
            msg = f"あなたの{key}は「{v}」です。" if v else f"あなたの{key}はまだ聞いていません。教えてくれれば覚えます。"
            return Reply(msg, 1.0 if v else 0.3, "tool:profile", [], [], [])
        for key, pat in _PROFILE_SET:
            m = pat.match(t)
            if m and len(m.group(1)) <= 40 and not re.search(r"[?？]", m.group(1)):
                self.profile[key] = m.group(1).strip()
                self.brain.stats["profile_set"] += 1
                name = self.profile.get("名前")
                ack = f"覚えました。{name}さんの{key}は「{self.profile[key]}」ですね。" if name and key != "名前" else f"覚えました。{key}は「{self.profile[key]}」ですね。"
                return Reply(ack, 1.0, "tool:profile", [], [], [])
        # 日付・時刻
        if not re.search(r"とは|って何|について", t):
            d = answer_datetime(t)
            if d:
                return Reply(d, 1.0, "tool:datetime", [], [], [])
        # 単位換算
        u = convert_units(t)
        if u:
            return Reply(u, 1.0, "tool:units", [], [], [])
        # 計算
        if _CALC_HINT.search(t) and re.search(r"\d", t) and not re.search(r"年|月|日|とは|について", t):
            c = calculate(t)
            if c:
                shown, val = c
                return Reply(f"{shown} = {_fmt_num(val)}", 1.0, "tool:calc", [], [], [])
        # 比較
        r = self._compare(t, ja)
        if r is not None:
            return r
        # 列挙
        r = self._list(t)
        if r is not None:
            return r
        # 要約 (URL / 貼り付けた文章)
        r = self._summarize(t, ja)
        if r is not None:
            return r
        # その場で調べる
        if re.match(r"^(?:調べて|検索して|search|look up)[:：]?\s*(.+)$", t) or re.search(r"(?:を|について)(?:今)?(?:調べて|検索して|調査して)(?:答えて|教えて)?[。!！?？]*$", t):
            return self._research(t, ja)
        return None

    # ------------------------------------------------------------ 比較
    def _compare(self, t: str, ja: bool):
        from .brain import Reply

        m = _CMP_RE.match(t)
        if m:
            a, b, adj = m.group(1).strip(), m.group(2).strip(), m.group(3)
            attrs = _CMP_ATTR[adj]
            va = vb = None
            fa = fb = None
            for attr in attrs:
                fa = fa or (self.brain.facts.lookup(a, attr) or [None])[0]
                fb = fb or (self.brain.facts.lookup(b, attr) or [None])[0]
            if fa and fb:
                va, vb = _number_of(fa[1]), _number_of(fb[1])
            if va is not None and vb is not None:
                if adj in ("古い",):
                    winner = a if va < vb else b
                elif adj == "新しい":
                    winner = a if va > vb else b
                else:
                    winner = a if va > vb else b
                msg = f"{winner}の方が{adj}です。{a}は{fa[1]}、{b}は{fb[1]}です。"
                return Reply(msg, 0.9, "tool:compare", [], [fa[2], fb[2]], [])
            missing = [x for x, f in ((a, fa), (b, fb)) if not f]
            if missing:
                for x in missing:
                    self.brain.add_gap(x)
                return Reply(f"「{'」と「'.join(missing)}」の{attrs[0]}はまだ知りません。調べておきます。", 0.2, "tool:compare", [], [], [])
        m = _DIFF_RE.match(t)
        if m:
            a, b = m.group(1).strip(), m.group(2).strip()
            parts = []
            ids = []
            for x in (a, b):
                facts = self.brain.facts.lookup(x)
                if facts:
                    rel, obj, doc_id = facts[0]
                    parts.append(self.brain.facts._render(x, rel, obj, True))
                    ids.append(doc_id)
                else:
                    hits = self.brain._search(f"{x}とは", k=1)
                    if hits and hits[0][0] >= self.brain.params.answer_threshold:
                        parts.append(hits[0][1].text)
                        ids.append(hits[0][1].id)
                    else:
                        self.brain.add_gap(x)
            if len(parts) == 2:
                return Reply(f"{a}: {parts[0]}\n{b}: {parts[1]}", 0.8, "tool:compare", [], ids, [])
            if parts:
                return Reply(f"片方しか知りません。{parts[0]} もう一方は調べておきます。", 0.3, "tool:compare", [], ids, [])
        return None

    # ------------------------------------------------------------ 列挙
    def _list(self, t: str):
        from .brain import Reply

        m = _LIST_RE.match(t)
        if not m:
            return None
        topic = m.group(1).strip()
        limit = int(m.group(3)) if m.group(3) else 3
        hits = self.brain.kb.search(topic, k=12)
        sents = [d.text for _, d in hits]
        items = extract_items(sents, topic, limit)
        if not items:
            self.brain.add_gap(topic)
            return Reply(f"「{topic}」の例はまだ十分に知りません。調べておきます。", 0.2, "tool:list", [], [], [])
        body = "\n".join(f"・{x}" for x in items)
        return Reply(f"{topic}の例:\n{body}", 0.7, "tool:list", [d.source for _, d in hits[:2]], [d.id for _, d in hits[:2]], [])

    # ------------------------------------------------------------ 要約
    def _summarize(self, t: str, ja: bool):
        from .brain import Reply

        if not re.search(r"要約|まとめて|summar", t):
            return None
        m = re.search(r"(https?://\S+)", t)
        if m:
            col = self._get_collector()
            if col is None:
                return Reply("オフラインなので URL は読めません。", 0.2, "tool:summarize", [], [], [])
            page = self._fetcher.get_page(m.group(1))
            if not page or len(page.text) < 100:
                return Reply("そのページは読めませんでした。", 0.2, "tool:summarize", [], [], [])
            self.brain.learn_text(page.text, source=m.group(1))
            summary = extractive_summary(page.text, 3)
            return Reply(f"要約: {summary}", 0.8, "tool:summarize", [m.group(1)], [], [])
        body = re.sub(r"(?:以下|次|これ|下記)?(?:の文章|の文|の内容)?を?要約して[:：]?|(?:を)?まとめて[:：]?", "", t).strip()
        if len(body) >= 120:
            return Reply(f"要約: {extractive_summary(body, 3)}", 0.8, "tool:summarize", [], [], [])
        # 直前の話題の要約
        if self.brain.last_topics:
            s = self.brain.summarize(self.brain.last_topics[0], ja=ja)
            if s:
                return Reply(s[0], 0.7, "summary", list(dict.fromkeys(s[2])), s[1], [])
        return None

    # ------------------------------------------------------------ その場で調べる
    def _research(self, t: str, ja: bool, budget_s: float = 20.0):
        from .brain import Reply

        m = re.match(r"^(?:調べて|検索して|search|look up)[:：]?\s*(.+)$", t)
        topic = m.group(1) if m else re.sub(r"(?:を|について)(?:今)?(?:調べて|検索して|調査して)(?:答えて|教えて)?[。!！?？]*$", "", t)
        topic = topic.strip(" 。?？")
        col = self._get_collector()
        if col is None:
            self.brain.add_gap(topic)
            return Reply(f"オフラインなので今は調べられません。「{topic}」は後で調べます。", 0.2, "tool:research", [], [], [])
        t0 = time.time()
        batch = col.collect(topic, max_pages=2)
        learned = self.brain.learn_batch(batch, col) if batch.pages else 0
        self.brain.background_step(budget_docs=200)
        self.brain.stats["research_turns"] += 1
        if learned == 0:
            self.brain.add_gap(topic)
            return Reply(f"「{topic}」について今すぐ見つけられませんでした。引き続き調べます。", 0.2, "tool:research", [], [], [])
        # 学んだ直後に答える (事実 → 要約 → 検索)
        ans = self.brain.facts.answer(f"{topic}とは？")
        if ans:
            return Reply(ans[0], 0.85, "tool:research", [self.brain.kb.docs[ans[1]].source], [ans[1]], [])
        s = self.brain.summarize(topic, ja=ja)
        if s:
            return Reply(s[0], 0.75, "tool:research", list(dict.fromkeys(s[2])), s[1], [])
        hits = self.brain._search(topic, k=2)
        if hits:
            return Reply(hits[0][1].text, 0.6, "tool:research", [hits[0][1].source], [hits[0][1].id], [])
        return Reply(f"「{topic}」について {learned} 文を学びましたが、うまく要約できませんでした。", 0.3, "tool:research", [], [], [])

    # ------------------------------------------------------------ 状態
    def state(self) -> dict:
        return {"profile": dict(self.profile)}

    def load_state(self, st: dict) -> None:
        self.profile = dict(st.get("profile", {}))
