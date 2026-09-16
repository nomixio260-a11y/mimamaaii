"""文からの事実 (主語 - 関係 - 目的語) の抽出と記憶。

形態素解析器なしの軽量パターン抽出:
  日本語: 「X とは Y」「X は Y である/です」「X の Y は Z」「X が Y に Z した」
  英語  : "X is a Y", "X was Y in Z", "X's Y is Z", "the Y of X is Z"

抽出した事実は subject -> [(relation, object, doc_id), ...] で持ち、
「X の Y は?」「X とは?」に検索を経ずに直接答えるために使う。
文書が追い出されたら、その文書由来の事実も消える。
"""
from __future__ import annotations

import re
from collections import defaultdict

from .tokenizer import is_phrase, keywords, normalize

# 主語として使える名詞句: 漢字/カタカナ/英数字の連続 (2〜24 文字)
_SUBJ = r"([A-Za-z0-9][A-Za-z0-9 .'’\-]{1,30}|[㐀-䶿一-鿿゠-ヿA-Za-z0-9々]{2,24})"
_OBJ = r"(.{2,80}?)"
# 主語の前に許す短い前置き (「また、」「一般に」など)
_PRE = r"^(?:[^、。]{0,8}[、]\s*)?"
# 事実らしい文末 (これに当てはまらない文は正規表現を走らせない)
_JA_TAIL_RE = re.compile(r"(である|です|だ|を指す|をいう|と呼ばれる|に位置する|にある|に属する|された|した|生まれた|設立|創業|発明|完成|開業|公開|になる|となる)[。.!?]?$")
_EN_TAIL_RE = re.compile(r"\b(is|are|was|were)\b", re.I)
# 属性名: 漢字/カタカナを 1 文字以上含む 1〜8 文字 (「高さ」「首都」「人口」「創業者」)
_ATTR = r"(?:[\u3040-\u309f]{0,2}[\u3400-\u4dbf\u4e00-\u9fff\u30a0-\u30ff]+[\u3040-\u309f]{0,2}){1,3}"

_JA_PATTERNS = [
    # X とは、Y である / のことである / を指す / です
    ("definition", re.compile(_PRE + _SUBJ + r"\s*とは、?\s*" + _OBJ + r"\s*(?:のこと)?(?:である|です|を指す|をいう|と呼ばれる)[。.]?$")),
    # X の Y は Z である/です  (Y は 2〜8 文字の名詞)
    ("attr", re.compile(_PRE + _SUBJ + r"\s*の\s*(" + _ATTR + r")\s*は、?\s*" + _OBJ + r"\s*(?:である|です|だ|になる|となる)?[。.]?$")),
    # X は Y である / です
    ("is", re.compile(_PRE + _SUBJ + r"\s*は、?\s*" + _OBJ + r"\s*(?:である|です|だ)[。.]?$")),
    # X は Y に位置する / にある
    ("location", re.compile(_PRE + _SUBJ + r"\s*は、?\s*" + _OBJ + r"\s*(?:に位置する|にある|に属する)[。.]?$")),
    # X は Y に Z された/した (設立・誕生・発明 など)
    ("event", re.compile(_PRE + _SUBJ + r"\s*は、?\s*(\d{3,4}\s*年[^、。]{0,12}?)\s*に\s*(.{1,30}?(?:された|した|生まれた|設立|創業|発明|完成|開業|公開))[。.]?$")),
]
_EN_PATTERNS = [
    ("definition", re.compile(r"^(?:the\s+)?" + _SUBJ + r"\s+(?:is|are|was|were)\s+(?:an?|the)\s+" + _OBJ + r"[.]?$", re.I)),
    ("attr", re.compile(r"^(?:the\s+)?([a-z][a-z ]{1,20}?)\s+of\s+(?:the\s+)?" + _SUBJ + r"\s+(?:is|are|was|were)\s+" + _OBJ + r"[.]?$", re.I)),
    ("attr2", re.compile(r"^" + _SUBJ + r"'s\s+([a-z][a-z ]{1,20}?)\s+(?:is|are|was|were)\s+" + _OBJ + r"[.]?$", re.I)),
    ("event", re.compile(r"^" + _SUBJ + r"\s+(?:was|were)\s+(founded|born|built|established|invented|created|released|discovered)\s+(?:in|on)\s+" + _OBJ + r"[.]?$", re.I)),
]

# 「X の Y は?」 型の質問
_JA_ATTR_Q = re.compile(r"^" + _SUBJ + r"\s*の\s*(" + _ATTR + r")\s*(?:は|って|を教えて|は何|はどこ|はいつ|はどれくらい|はいくつ)?[?？]?\s*(?:何|なに|どこ|いつ|誰|だれ|どれくらい|いくつ|教えて)?(?:ですか|でしょうか|か)?[?？。]*$")
# 「X はいつ〜?」「X はどこ?」「X は誰?」
_JA_WH_Q = re.compile(r"^" + _SUBJ + r"\s*(?:は|が|って)\s*(?:一体)?(いつ|どこ|誰|だれ|何処)")
_JA_DEF_Q = re.compile(r"^" + _SUBJ + r"\s*(?:とは|って|とは何|って何|ってなに|とはなに|について教えて|を教えて|とは何ですか|って何ですか)[?？。]*$")
_EN_ATTR_Q = re.compile(r"^(?:what|when|where|who)\s+(?:is|are|was|were)\s+(?:the\s+)?([a-z][a-z ]{1,20}?)\s+of\s+(?:the\s+)?" + _SUBJ + r"\s*[?]?$", re.I)
_EN_DEF_Q = re.compile(r"^(?:what|who)\s+(?:is|are|was|were)\s+(?:an?\s+|the\s+)?" + _SUBJ + r"\s*[?]?$", re.I)

_BAD_SUBJ = re.compile(r"^(これ|それ|あれ|この|その|あの|ここ|そこ|以下|以上|一方|また|なお|ただし|しかし|そして|また|同|各|本項|本記事|上記|下記|前者|後者|this|that|these|those|it|there|here|he|she|they|we|you|i)$", re.I)
_RELATION_ALIASES = {
    "definition": ("とは", "定義", "意味", "what", "definition"),
    "is": ("とは", "何", "what"),
    "location": ("場所", "所在地", "位置", "どこ", "where", "本社", "所在"),
    "event": ("いつ", "年", "設立", "誕生", "創業", "when", "founded", "born", "built", "created", "established"),
}
# 同じ意味の属性名 (英日・同義語)。lookup で相互に引ける
_ATTR_SYNONYMS = [
    ("creator", "author", "作者", "開発者", "創始者", "考案者", "設計者", "著者", "作った人"),
    ("founder", "創業者", "創設者", "設立者"),
    ("capital", "首都"),
    ("height", "高さ", "標高", "身長"),
    ("population", "人口"),
    ("area", "面積", "広さ"),
    ("length", "長さ", "全長"),
    ("president", "ceo", "社長", "代表", "代表者"),
    ("headquarters", "本社", "本部", "所在地"),
    ("birthday", "誕生日", "生年月日"),
    ("name", "名前", "名称"),
    ("advantage", "利点", "長所", "メリット"),
    ("purpose", "目的", "用途"),
]
_SYNONYM_OF: dict[str, frozenset] = {}
for _grp in _ATTR_SYNONYMS:
    _fs = frozenset(_grp)
    for _w in _grp:
        _SYNONYM_OF[_w] = _fs


def _clean(s: str) -> str:
    s = s.strip(" 、,。.:：「」『』()（）")
    return s


def extract_facts(sentence: str) -> list[tuple[str, str, str]]:
    """(subject, relation, object) のリスト。無ければ空。"""
    s = normalize(sentence)
    if len(s) < 6 or len(s) > 200:
        return []
    out: list[tuple[str, str, str]] = []
    ascii_ = s.isascii()
    if ascii_:
        if not _EN_TAIL_RE.search(s):
            return []
        patterns = _EN_PATTERNS
    else:
        if not _JA_TAIL_RE.search(s[-14:]):
            return []
        patterns = _JA_PATTERNS
    for kind, pat in patterns:
        m = pat.match(s)
        if not m:
            continue
        g = [(_clean(x) if x else "") for x in m.groups()]
        if kind == "attr" and not ascii_:
            subj, rel, obj = g[0], g[1], g[2]
            m2 = re.search(r"(に位置する|にある|にあります|に属する)$", obj)
            if m2:
                obj = obj[: m2.start()]
                rel = rel if rel not in ("場所", "所在地", "位置") else "location"
        elif kind == "attr" and ascii_:
            rel, subj, obj = g[0].lower(), g[1], g[2]
        elif kind == "attr2":
            subj, rel, obj = g[0], g[1].lower(), g[2]
        elif kind == "event":
            if ascii_:
                subj, rel, obj = g[0], g[1].lower(), g[2]
            else:
                subj, rel, obj = g[0], "event", f"{g[1]}に{g[2]}"
        else:
            subj, rel, obj = g[0], kind, g[1]
        subj = subj.strip()
        if not subj or _BAD_SUBJ.match(subj) or len(obj) < 2 or len(obj) > 80:
            continue
        # 直前が助詞なら「彼の名前は」のように別の語の属性なので捨てる
        if not ascii_ and m.start(1) > 0 and s[m.start(1) - 1] in "のがはをにでと":
            continue
        # 主語は話題語として意味のあるものだけ
        if not (is_phrase(subj.lower()) or any(is_phrase(k) for k in keywords(subj, limit=2))):
            continue
        out.append((subj, rel, obj))
        break  # 1 文 1 事実
    return out


def parse_question(text: str) -> tuple[str, str] | None:
    """質問を (subject, relation) に分解。relation は属性名または 'definition'。"""
    t = normalize(text)
    if t.isascii():
        m = _EN_ATTR_Q.match(t)
        if m:
            return _clean(m.group(2)), m.group(1).lower().strip()
        m = _EN_DEF_Q.match(t)
        if m:
            return _clean(m.group(1)), "definition"
        return None
    m = _JA_ATTR_Q.match(t)
    if m:
        subj, rel = _clean(m.group(1)), re.sub(r"(は誰|はいつ|はどこ|は何|はなに|は|って|を)$", "", m.group(2))
        if "の" in rel:  # 「AのBのC」: 最後の C が属性、手前は主語の連鎖
            parts = [x for x in rel.split("の") if x]
            if len(parts) >= 2:
                subj = subj + "の" + "の".join(parts[:-1])
                rel = parts[-1]
        if rel and not _BAD_SUBJ.match(subj):
            return subj, rel
    m = _JA_WH_Q.match(t)
    if m and not _BAD_SUBJ.match(m.group(1)):
        wh = m.group(2)
        return _clean(m.group(1)), {"いつ": "event", "どこ": "location", "何処": "location", "誰": "who", "だれ": "who"}[wh]
    m = _JA_DEF_Q.match(t)
    if m and not _BAD_SUBJ.match(m.group(1)):
        return _clean(m.group(1)), "definition"
    return None


class FactStore:
    """subject -> [(relation, object, doc_id)] の小さな記憶。件数上限あり。"""

    def __init__(self, max_facts: int = 50000):
        self.by_subject: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
        self.by_doc: dict[int, list[str]] = defaultdict(list)  # doc_id -> subjects (削除用)
        self.count = 0
        self.max_facts = max_facts

    @staticmethod
    def _key(subject: str) -> str:
        return subject.lower().replace(" ", "")

    def add(self, subject: str, relation: str, obj: str, doc_id: int) -> bool:
        key = self._key(subject)
        lst = self.by_subject[key]
        for r, o, d in lst:
            if r == relation and o == obj and d == doc_id:
                return False  # 同じ文書からの重複だけ排除 (別文書の同じ事実は consolidate の材料)
        if self.count >= self.max_facts:
            return False
        lst.append((relation, obj, doc_id))
        self.by_doc[doc_id].append(key)
        self.count += 1
        return True

    def add_from_sentence(self, sentence: str, doc_id: int) -> int:
        n = 0
        for subj, rel, obj in extract_facts(sentence):
            if self.add(subj, rel, obj, doc_id):
                n += 1
        return n

    def remove_doc(self, doc_id: int) -> int:
        keys = self.by_doc.pop(doc_id, None)
        if not keys:
            return 0
        removed = 0
        for key in keys:
            lst = self.by_subject.get(key)
            if not lst:
                continue
            keep = [f for f in lst if f[2] != doc_id]
            removed += len(lst) - len(keep)
            if keep:
                self.by_subject[key] = keep
            else:
                del self.by_subject[key]
        self.count -= removed
        return removed

    def lookup(self, subject: str, relation: str | None = None) -> list[tuple[str, str, int]]:
        key = self._key(subject)
        lst = self.by_subject.get(key)
        if not lst:
            return []  # 主語は完全一致のみ (部分一致は別の話題を混同しやすい)
        if relation is None:
            return list(lst)
        rel = relation.lower()
        out = [f for f in lst if f[0] == rel]
        if out:
            return out
        aliases = {r for r, names in _RELATION_ALIASES.items() if rel in names}
        aliases |= _SYNONYM_OF.get(rel, frozenset())
        out = [f for f in lst if f[0] in aliases]
        if out:
            return out
        # 属性名が関係名の一部 (「高さ」 と 「最高高さ」)
        return [f for f in lst if len(rel) >= 2 and (rel in f[0] or f[0] in rel)]

    def resolve_chain(self, subject: str, relations: list[str]) -> tuple[str, list[tuple[str, str, str, int]]] | None:
        """subject に relations を順に適用して辿る (多段推論)。途中経過も返す。"""
        cur = subject
        trail = []
        for rel in relations:
            facts = self.lookup(cur, rel)
            if not facts:
                return None
            r, obj, doc_id = facts[0]
            trail.append((cur, r, obj, doc_id))
            cur = _clean(obj)
        return cur, trail

    def answer(self, question: str) -> tuple[str, int] | None:
        """質問に事実で答えられれば (文, doc_id)。「AのBのCは?」は連鎖で辿る。"""
        parsed = parse_question(question)
        if not parsed:
            return None
        subj, rel = parsed
        ja = not question.isascii()
        # 多段: 主語が「A の B」 の形なら A→B を先に解決する
        if ja and "の" in subj:
            head, *rest = subj.split("の")
            if head and all(rest):
                chain = self.resolve_chain(head, rest)
                if chain is not None:
                    mid, trail = chain
                    facts = self.lookup(mid, rel)
                    if facts:
                        r, obj, doc_id = facts[0]
                        steps = "、".join(f"{a}の{b}は{c}" for a, b, c, _ in trail)
                        return f"{steps}なので、{self._render(mid, r, obj, ja)}", doc_id
        facts = self.lookup(subj, rel)
        if not facts and rel == "definition":
            # 定義そのものが無くても、知っている事実を 2 つまで並べて答える
            facts = self.lookup(subj)[:2]
            if not facts:
                return None
            parts = [self._render(subj, r, o, ja) for r, o, _ in facts]
            return " ".join(parts), facts[0][2]
        if not facts:
            return None
        relation, obj, doc_id = facts[0]
        # 質問の言語と保存した関係名の言語が違えば、質問側の語で表現する (creator ↔ 作者)
        if relation not in ("definition", "is", "location", "event") and relation.isascii() != rel.isascii():
            relation = rel
        return self._render(subj, relation, obj, ja), doc_id

    @staticmethod
    def _render(subj: str, relation: str, obj: str, ja: bool) -> str:
        if relation in ("definition", "is"):
            text = f"{subj}とは、{obj}です。" if ja else f"{subj} is {obj}."
        elif relation == "location":
            text = f"{subj}は{obj}にあります。" if ja else f"{subj} is in {obj}."
        elif relation == "event":
            text = f"{subj}は{obj}。" if ja else f"{subj} was {obj}."
        else:
            text = f"{subj}の{relation}は{obj}です。" if ja else f"The {relation} of {subj} is {obj}."
        return text

    def state(self) -> dict:
        return {"by_subject": dict(self.by_subject), "count": self.count}

    @classmethod
    def from_state(cls, st: dict, max_facts: int = 50000) -> "FactStore":
        fs = cls(max_facts)
        for key, lst in st.get("by_subject", {}).items():
            fs.by_subject[key] = list(lst)
            for _, _, doc_id in lst:
                fs.by_doc[doc_id].append(key)
        fs.count = sum(len(v) for v in fs.by_subject.values())
        return fs

    def stats(self) -> dict:
        return {"facts": self.count, "subjects": len(self.by_subject)}
