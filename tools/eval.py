"""総合評価: 会話評価セット (data/eval/*.tsv) + 検索 (MRR / Recall@3) + LM (ppl) + 生成 (多様性・接地率・繰り返し)
+ 遅延 (p50 / p95) + メモリ。結果は eval/results.jsonl に追記し、前回と比較する。

    python tools/eval.py                      # 全部
    python tools/eval.py --verbose            # 失敗と各問の出力を表示
    python tools/eval.py --paraphrase         # 言い換えを自動生成して頑健性も測る
    python tools/eval.py --corpus corpus.txt  # 検索/LM/生成の指標をこのコーパスで測る
    python tools/eval.py --compare            # 前回の結果との差分を表示
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tinyai import Brain, Config  # noqa: E402
from tinyai.memory import rss_bytes  # noqa: E402
from tinyai.tokenizer import is_phrase, keywords, phrases, split_sentences, tokenize  # noqa: E402

EVAL_DIR = ROOT / "data" / "eval"
RESULTS = ROOT / "eval" / "results.jsonl"
PARAPHRASES = ["{x}とは？", "{x}って何？", "{x}を教えて", "{x}について教えて"]


def dialog_state(b: Brain) -> dict:
    return {"history": list(b.history), "last_docs": list(b.last_docs), "last_mode": b.last_mode, "last_question": b.last_question,
            "last_topics": list(b.last_topics), "said": list(b._said)}


def restore_dialog(b: Brain, st: dict) -> None:
    b.history.clear()
    b.history.extend(st["history"])
    b.last_docs = list(st["last_docs"])
    b.last_mode = st["last_mode"]
    b.last_question = st["last_question"]
    b.last_topics = list(st["last_topics"])
    b._said.clear()
    b._said.extend(st["said"])


# ---------------------------------------------------------------- 会話評価
def parse_expect(spec: str) -> dict:
    out = {"any": [], "not": [], "regex": [], "modes": None, "conf": []}
    for part in spec.split("|") if spec else []:
        part = part.strip()
        if not part:
            continue
        if part.startswith("!"):
            out["not"].append(part[1:])
        elif part.startswith("~"):
            out["regex"].append(part[1:])
        elif part.startswith("mode:"):
            out["modes"] = set(part[5:].split(","))
        elif part.startswith("conf"):
            m = re.match(r"conf\s*(<=|>=|<|>)\s*([0-9.]+)", part)
            if m:
                out["conf"].append((m.group(1), float(m.group(2))))
        else:
            out["any"].append(part)
    return out


def judge(reply, expect: dict) -> tuple[bool, str]:
    text = reply.text.lower()
    reasons = []
    if expect["any"] and not any(a.lower() in text for a in expect["any"]):
        reasons.append("期待語なし")
    for n in expect["not"]:
        if n.lower() in text:
            reasons.append(f"禁止語 {n}")
    for rx in expect["regex"]:
        if not re.search(rx, reply.text):
            reasons.append(f"正規表現 {rx}")
    if expect["modes"] and reply.mode not in expect["modes"]:
        reasons.append(f"mode={reply.mode}")
    for op, v in expect["conf"]:
        c = reply.confidence
        ok = {"<": c < v, "<=": c <= v, ">": c > v, ">=": c >= v}[op]
        if not ok:
            reasons.append(f"conf={c}")
    if not (expect["any"] or expect["not"] or expect["regex"] or expect["modes"] or expect["conf"]):
        # 何も期待しない = 「知らない」と正直に言うこと
        if reply.confidence >= 0.3 and reply.mode not in ("generate",):
            reasons.append(f"知らないはずが conf={reply.confidence} mode={reply.mode}")
    return (not reasons), ", ".join(reasons)


def load_sets(paths) -> list[tuple[str, list]]:
    sets = []
    for path in paths:
        rows = []
        for ln in path.read_text(encoding="utf-8").splitlines():
            if not ln.strip() or ln.startswith("#"):
                continue
            parts = ln.split("\t") + ["", "", ""]
            q, expect, teach, cat = parts[0], parts[1], parts[2].replace("\\n", "\n"), parts[3] or path.stem
            rows.append((q, expect, teach, cat))
        sets.append((path.stem, rows))
    return sets


def make_brain(tmp: str) -> Brain:
    cfg = Config(data_dir=Path(tmp), memory_mb=256, hard_limit=False, web_enabled=False, seed=1, tools=True)
    b = Brain(cfg)
    b.bootstrap()
    return b


def run_conversation(sets, verbose: bool, paraphrase: bool) -> tuple[dict, list[float]]:
    per_cat: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    latencies: list[float] = []
    with tempfile.TemporaryDirectory() as tmp:
        b = make_brain(tmp)
        for name, rows in sets:
            for q, expect_s, teach, cat in rows:
                if q == "@reset":
                    restore_dialog(b, {"history": [], "last_docs": [], "last_mode": "", "last_question": "", "last_topics": [], "said": []})
                    continue
                if teach:
                    b.learn_text(teach, source="https://eval/" + name)
                    b.background_step(budget_docs=100)
                expect = parse_expect(expect_s)
                variants = [q]
                if paraphrase and expect["any"] and not expect["modes"] and cat in ("facts", "robust", "basic_ja"):
                    m = re.match(r"^(.+?)(とは|って何|を教えて|について教えて)[?？]?$", q)
                    if m and is_phrase(m.group(1).lower()):
                        variants += [p.format(x=m.group(1)) for p in PARAPHRASES if p.format(x=m.group(1)) != q]
                for i, qq in enumerate(variants):
                    st = dialog_state(b) if i == 0 and len(variants) > 1 else None
                    t0 = time.perf_counter()
                    r = b.reply(qq)
                    latencies.append((time.perf_counter() - t0) * 1000)
                    ok, why = judge(r, expect)
                    key = cat if i == 0 else "paraphrase"
                    per_cat[key][0] += ok
                    per_cat[key][1] += 1
                    if verbose or not ok:
                        print(f"{'OK ' if ok else 'NG '} [{key}] {qq!r} -> {r.text[:80]!r} [{r.mode} {r.confidence}] {why}")
                    if i == 0 and st is not None:
                        base_state = dialog_state(b)  # 本問の後の状態 (言い換えの後に戻す)
                if len(variants) > 1:
                    restore_dialog(b, base_state)
    return {k: {"ok": v[0], "n": v[1], "acc": round(v[0] / v[1], 3)} for k, v in per_cat.items()}, latencies


# ---------------------------------------------------------------- 検索 / LM / 生成
def run_corpus_metrics(corpus: Path | None) -> dict:
    if corpus is None or not corpus.exists():
        return {}
    text = corpus.read_text(encoding="utf-8", errors="replace")
    sents = split_sentences(text)
    rng = random.Random(0)
    rng.shuffle(sents)
    hold = sents[: max(20, len(sents) // 20)]
    train = sents[len(hold):]
    with tempfile.TemporaryDirectory() as tmp:
        b = make_brain(tmp)
        r0 = rss_bytes()
        t0 = time.perf_counter()
        n = b.learn_text("\n".join(train), source="https://eval/corpus")
        learn_s = time.perf_counter() - t0
        b.background_step(budget_docs=100000)
        rss_mb = (rss_bytes() - r0) / 1048576
        # 検索: 文の句から疑似クエリ (奇数番目の句 + とは) → 元の文が何位か
        docs = list(b.kb.docs.values())
        rr = []
        rec3 = 0
        qn = 0
        for d in rng.sample(docs, min(200, len(docs))):
            ph = phrases(d.text)
            if len(ph) < 2:
                continue
            q = "".join(ph[::2][:3])
            hits = b.kb.search(q, k=10)
            rank = next((i for i, (_, x) in enumerate(hits) if x.id == d.id), None)
            qn += 1
            if rank is not None:
                rr.append(1.0 / (rank + 1))
                rec3 += rank < 3
        ppl = b.lm.perplexity([tokenize(s) for s in hold])
        # 生成: 多様性 (distinct-2)、繰り返し率、接地率 (生成文の 4-gram が知識にある割合)
        gens = []
        for s in hold[:20]:
            ks = [k for k in keywords(s, limit=1)]
            if ks:
                gens.append(b.generate(tokenize(ks[0]), query=ks[0], n_candidates=3))
        gens = [g for g in gens if g]
        bigrams = Counter()
        grounded = tot4 = 0
        rep_rate = []
        for g in gens:
            toks = tokenize(g)
            for i in range(len(toks) - 1):
                bigrams[(toks[i], toks[i + 1])] += 1
            for i in range(len(toks) - 3):
                tot4 += 1
                if "".join(toks[i : i + 4]) in text.lower():
                    grounded += 1
            rep_rate.append(1 - len(set(toks)) / max(len(toks), 1))
        return {
            "learn_sents_per_s": round(n / learn_s),
            "rss_mb": round(rss_mb, 1),
            "retrieval_mrr": round(sum(rr) / max(qn, 1), 3),
            "retrieval_recall3": round(rec3 / max(qn, 1), 3),
            "perplexity": round(ppl, 2),
            "gen_distinct2": round(len(bigrams) / max(sum(bigrams.values()), 1), 3),
            "gen_grounded4": round(grounded / max(tot4, 1), 3),
            "gen_repetition": round(statistics.mean(rep_rate), 3) if rep_rate else None,
            "gen_samples": gens[:3],
        }


# ---------------------------------------------------------------- 記録・比較
def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "?"


def load_last() -> dict | None:
    if not RESULTS.exists():
        return None
    lines = RESULTS.read_text(encoding="utf-8").splitlines()
    return json.loads(lines[-1]) if lines else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", action="append", help="評価セット TSV (省略時 data/eval/*.tsv)")
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--paraphrase", action="store_true")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()
    paths = [Path(p) for p in args.set] if args.set else sorted(EVAL_DIR.glob("*.tsv"))
    sets = load_sets(paths)
    per_cat, lat = run_conversation(sets, args.verbose, args.paraphrase)
    total_ok = sum(v["ok"] for v in per_cat.values())
    total_n = sum(v["n"] for v in per_cat.values())
    lat_sorted = sorted(lat)
    result = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "rev": git_rev(),
        "conversation": {"ok": total_ok, "n": total_n, "acc": round(total_ok / max(total_n, 1), 3)},
        "categories": per_cat,
        "latency_ms": {"p50": round(lat_sorted[len(lat_sorted) // 2], 2), "p95": round(lat_sorted[int(len(lat_sorted) * 0.95)], 2), "max": round(lat_sorted[-1], 2)} if lat_sorted else {},
    }
    result.update(run_corpus_metrics(Path(args.corpus) if args.corpus else None))
    print("\n== 会話評価: %d/%d = %.3f" % (total_ok, total_n, result["conversation"]["acc"]))
    for k, v in sorted(per_cat.items()):
        print(f"   {k:<12} {v['ok']}/{v['n']} = {v['acc']}")
    print("== 遅延 ms:", result["latency_ms"])
    for k in ("learn_sents_per_s", "rss_mb", "retrieval_mrr", "retrieval_recall3", "perplexity", "gen_distinct2", "gen_grounded4", "gen_repetition"):
        if k in result:
            print(f"== {k}: {result[k]}")
    if "gen_samples" in result:
        for g in result["gen_samples"]:
            print("   生成例:", g)
    last = load_last()
    if args.compare and last:
        print("\n== 前回 (%s, %s) との比較" % (last.get("rev"), last.get("time")))
        for k in ("conversation", "retrieval_mrr", "retrieval_recall3", "perplexity", "gen_grounded4", "learn_sents_per_s"):
            a, b_ = last.get(k), result.get(k)
            if isinstance(a, dict):
                a, b_ = a.get("acc"), b_.get("acc")
            if a is not None and b_ is not None:
                print(f"   {k:<20} {a} -> {b_}")
    if not args.no_save:
        RESULTS.parent.mkdir(parents=True, exist_ok=True)
        with RESULTS.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
