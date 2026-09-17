"""総合評価: 会話評価セット (data/eval/*.tsv) + 検索 (MRR / Recall@3) + LM (ppl) + 生成 (多様性・接地率・繰り返し)
+ 遅延 (p50 / p95) + メモリ + ニューラル LM (学習効率・取り置き ppl・RAG 忠実性・リアルタイム学習・思考)
+ 収集システム (ソースごとの収穫・新規性・情報量・価値)。結果は eval/results.jsonl に追記し、前回と比較して退行を警告する。

    python tools/eval.py                      # 会話評価 + 遅延
    python tools/eval.py --verbose            # 失敗と各問の出力を表示
    python tools/eval.py --paraphrase         # 言い換えを自動生成して頑健性も測る
    python tools/eval.py --corpus corpus.txt  # 検索/LM/生成の指標をこのコーパスで測る
    python tools/eval.py --corpus corpus.txt --neural   # 同じコーパスで小型 Transformer を固定ステップ学習し、学習効率と生成品質を測る
    python tools/eval.py --data ~/.tinyai     # 学習済みの Brain を評価 (ニューラル LM の実力 + 収集ソースの評価表)
    python tools/eval.py --compare            # 前回の結果との差分と退行 (5% 以上悪化) を表示
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


# ---------------------------------------------------------------- ニューラル LM
def _neural_metrics(b: Brain, hold_sents: list[str], verbose: bool = False, sweep: bool = False) -> dict:
    """学習済み (または学習したばかりの) ニューラル LM を測る:
    holdout ppl / RAG 忠実性 (文脈の文をどれだけ使うか) / リアルタイム学習 (教えた答えの対数確率の伸び、👎 の抑制)
    / 思考 (再検索で文脈が増えた割合) / 生成速度。"""
    import numpy as np
    from tinyai import neural as nn
    from tinyai.tokenizer import phrases as _phr

    nl = b.neural
    out: dict = {}
    if nl.model is None:
        return out
    drop, nl.model.dropout = nl.model.dropout, 0.0
    try:
        hold_ids = [nl.seq_text(s) for s in hold_sents[:200]]
        out["neural_holdout_ppl"] = round(nn.perplexity(nl.model, hold_ids), 2) if hold_ids else None
        # RAG 忠実性: 文脈 = 取り置き文、質問 = その文のキーワード → 生成が文脈の句をどれだけ含むか
        grounded, copied, n, gen_ms = 0.0, 0, 0, []
        for s_ in hold_sents[:30]:
            ks = [k for k in keywords(s_, limit=1) if is_phrase(k)]
            if not ks:
                continue
            t0 = time.perf_counter()
            cands = nl.chat(f"{ks[0]}について教えて", s_, n=2, max_new=40)
            gen_ms.append((time.perf_counter() - t0) * 1000)
            if not cands:
                continue
            cph = set(_phr(s_))
            best = max(cands, key=lambda c: len(set(_phr(c)) & cph))
            ph = set(_phr(best))
            grounded += len(ph & cph) / max(len(ph), 1)
            copied += ks[0] in best
            n += 1
            if verbose:
                print(f"   RAG: {ks[0]!r} -> {best[:60]!r}")
        out["rag_grounded"] = round(grounded / max(n, 1), 3)
        out["rag_keyword_rate"] = round(copied / max(n, 1), 3)
        out["neural_gen_ms"] = round(statistics.median(gen_ms), 1) if gen_ms else None
        # リアルタイム学習: 架空の事実を 3 ステップ教え、答えの対数確率がどれだけ上がるか
        q, a = "ゾルグ星の首都は？", "ゾルグ星の首都はカルドラである。"
        seq = nl.seq_dialog(q, a)
        lp0 = nl.model.logprob(seq)
        nl.learn_turn(q, a, None, weight=1.0, steps=3)
        lp1 = nl.model.logprob(seq)
        out["online_gain_nat"] = round(lp1 - lp0, 3)
        # 👎: その答えを出しにくくする
        nl.learn_turn(q, a, None, weight=-1.0, steps=2)
        lp2 = nl.model.logprob(seq)
        out["unlearn_drop_nat"] = round(lp1 - lp2, 3)
        # 思考: 再検索で文脈が増えた割合 (neural_only で応答した時)
        if b.cfg.neural_rethink and hold_sents:
            rethink = 0
            m = 0
            for s_ in hold_sents[:10]:
                ks = [k for k in keywords(s_, limit=1) if is_phrase(k)]
                if not ks:
                    continue
                b.neural.ready = True
                b.reply(f"{ks[0]}とは？")
                m += 1
                if b.last_thought and b.last_thought.get("context2"):
                    rethink += 1
            out["think_rethink_rate"] = round(rethink / max(m, 1), 3)
        # 会話としての予測力 (応答部だけの ppl): 文の ppl とは別に測る
        pairs = b.dialog_holdout or [(u, bb) for u, bb, _, w in list(b.dialogs.pairs)[-2000:] if w > 0][-60:]
        ppls = []
        for u, bb in pairs:
            ids = nl.seq_dialog(u, bb)
            start = nl.loss_from(ids)
            if len(ids) - start < 2:
                continue
            ppls.append(math.exp(-nl.model.logprob(ids[max(0, start - 1):])))
        if ppls:
            ppls.sort()
            out["dialog_ppl"] = round(ppls[len(ppls) // 2], 2)        # 中央値 (外れ値に強い)
            out["dialog_ppl_mean"] = round(sum(ppls) / len(ppls), 2)
        # 復号バイアスの掃引 (文脈をどれだけ使わせると接地率と自然さがどうなるか)
        if sweep:
            out["copy_bonus_sweep"] = {}
            keep = nl.decode.get("copy_bonus", 0)
            for bonus in (0.0, 1.5, 3.0, 5.0):
                nl.decode["copy_bonus"] = bonus
                g2 = n2 = 0
                fl = []
                for s_ in hold_sents[:20]:
                    ks = [k for k in keywords(s_, limit=1) if is_phrase(k)]
                    if not ks:
                        continue
                    cands = nl.chat(f"{ks[0]}について教えて", s_, n=2, max_new=40)
                    if not cands:
                        continue
                    cph = set(_phr(s_))
                    best = max(cands, key=lambda c: len(set(_phr(c)) & cph))
                    ph = set(_phr(best))
                    g2 += len(ph & cph) / max(len(ph), 1)
                    fl.append(nl.score(best) or -20)
                    n2 += 1
                out["copy_bonus_sweep"][str(bonus)] = {"grounded": round(g2 / max(n2, 1), 3), "fluency": round(sum(fl) / max(len(fl), 1), 3)}
            nl.decode["copy_bonus"] = keep
    finally:
        nl.model.dropout = drop
    return out


def run_neural_budget(corpus: Path, steps: int = 300, verbose: bool = False) -> dict:
    """同じコーパス・同じ乱数・同じステップ数で小型 Transformer を学習し、到達した損失と速度を測る = 学習効率の回帰テスト。"""
    text = corpus.read_text(encoding="utf-8", errors="replace")
    sents = split_sentences(text)
    rng = random.Random(0)
    rng.shuffle(sents)
    hold = sents[: max(20, len(sents) // 20)]
    train = sents[len(hold):]
    with tempfile.TemporaryDirectory() as tmp:
        b = make_brain(tmp)
        if not b.neural.available:
            return {}
        nl = b.neural
        nl.size, nl.min_sentences, nl.min_chars = "small", 10, 100
        nl.batch, nl.lr = 16, 1e-3
        if not nl.ensure_model(train):
            return {}
        b.learn_text("\n".join(train), source="https://eval/corpus")
        for s_ in train:
            nl.add_text(s_)
        nl._holdout = [nl.seq_text(s_) for s_ in hold[:200]]
        t0 = time.perf_counter()
        losses = []
        done = 0
        while done < steps:
            r = nl.train_some(steps=25)
            if r is None:
                break
            done += r["steps"]
            losses.append(r["loss"])
        dt = time.perf_counter() - t0
        out = {
            "neural_budget_steps": done,
            "neural_budget_loss": round(losses[-1], 3) if losses else None,
            "neural_budget_first_loss": round(losses[0], 3) if losses else None,
            "neural_tokens_per_s": round(done * nl.batch * nl.model.T / max(dt, 1e-9)),
            "neural_params": nl.model.n_params(),
        }
        out.update(_neural_metrics(b, hold, verbose))
        if verbose:
            print("   budget 学習:", out)
        return out


def run_brain_eval(data_dir: Path, verbose: bool = False, sweep: bool = False) -> dict:
    """学習済みの Brain を評価: ニューラル LM の実力と収集ソースの評価表。"""
    cfg = Config(data_dir=data_dir, memory_mb=1024, hard_limit=False, web_enabled=False, tools=True)
    b = Brain(cfg)
    b.load()
    b.neural.ensure_model()
    out: dict = {"kb_docs": len(b.kb), "dialogs": len(b.dialogs), "dialogs_by_source": b.dialogs.by_source(), "facts": b.facts.count}
    st = b.stats
    for k in ("docs_learned", "dups_dropped", "junk_dropped", "dialogs_collected", "new_terms", "neural_steps", "online_turns", "unlearned_turns"):
        if k in st:
            out["stat_" + k] = st[k]
    if b.neural.model is not None:
        out["neural"] = b.neural.stats()
        hold = [d.text for d in b.kb.random_docs(min(200, len(b.kb)), random.Random(1))]
        out.update(_neural_metrics(b, hold, verbose, sweep=sweep))
    # 収集ソースの評価表 (Collector の健全性は Brain の状態に保存されている場合のみ)
    sources = (b.agent_state.get("sources") if hasattr(b, "agent_state") else None) or {}
    try:
        from tinyai.collector import Collector
        col = Collector(None, cfg.data_dir, cfg.languages)
        sources = {k: v.to_dict() for k, v in col.health.items()} or sources
    except Exception:
        pass
    if sources:
        out["sources"] = sources
        print("== 収集ソースの評価 (成功率 / 収穫 / 新規性 / 情報量 nat / 価値 / 点数)")
        for name, h in sorted(sources.items(), key=lambda kv: -kv[1].get("score", 0))[:15]:
            print(f"   {name:<22} {h.get('ok', 0)}/{h.get('tries', 0)}  gain {h.get('avg_gain', 0)}  nov {h.get('avg_novelty', 0)}  surprise {h.get('surprise', '-')}  value {h.get('value', '-')}  score {h.get('score', 0)}")
    return out


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
    ap.add_argument("--neural", action="store_true", help="--corpus で小型 Transformer を固定ステップ学習して学習効率を測る")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--data", default=None, help="学習済み Brain のデータディレクトリを評価する")
    ap.add_argument("--sweep", action="store_true", help="復号バイアス (文脈への寄せ方) を掃引して接地率と自然さの関係を測る")
    ap.add_argument("--fail-on-regress", action="store_true", help="主要指標が前回より 5%% 以上悪化したら終了コード 1 (CI 用)")
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
    if args.neural and args.corpus:
        result.update(run_neural_budget(Path(args.corpus), steps=args.steps, verbose=args.verbose))
    if args.data:
        result["brain"] = run_brain_eval(Path(args.data), verbose=args.verbose, sweep=args.sweep)
    print("\n== 会話評価: %d/%d = %.3f" % (total_ok, total_n, result["conversation"]["acc"]))
    for k, v in sorted(per_cat.items()):
        print(f"   {k:<12} {v['ok']}/{v['n']} = {v['acc']}")
    print("== 遅延 ms:", result["latency_ms"])
    for k in ("learn_sents_per_s", "rss_mb", "retrieval_mrr", "retrieval_recall3", "perplexity", "gen_distinct2", "gen_grounded4", "gen_repetition",
              "neural_budget_steps", "neural_budget_loss", "neural_tokens_per_s", "neural_holdout_ppl", "rag_grounded", "rag_keyword_rate", "neural_gen_ms",
              "online_gain_nat", "unlearn_drop_nat", "think_rethink_rate"):
        if k in result:
            print(f"== {k}: {result[k]}")
    if "brain" in result:
        br = result["brain"]
        print("== 学習済み Brain:", {k: v for k, v in br.items() if k not in ("sources", "neural", "dialogs_by_source")})
        if "neural" in br:
            print("   ニューラル LM:", br["neural"])
    if "gen_samples" in result:
        for g in result["gen_samples"]:
            print("   生成例:", g)
    last = load_last()
    if args.compare and last:
        print("\n== 前回 (%s, %s) との比較" % (last.get("rev"), last.get("time")))
        higher_better = {"conversation": True, "retrieval_mrr": True, "retrieval_recall3": True, "perplexity": False, "gen_grounded4": True, "learn_sents_per_s": True,
                         "neural_budget_loss": False, "neural_tokens_per_s": True, "neural_holdout_ppl": False, "rag_grounded": True, "online_gain_nat": True, "unlearn_drop_nat": True}
        for k, hb in higher_better.items():
            a, b_ = last.get(k), result.get(k)
            if isinstance(a, dict):
                a, b_ = a.get("acc"), b_.get("acc")
            if a is not None and b_ is not None:
                worse = (b_ < a * 0.95) if hb else (b_ > a * 1.05)
                print(f"   {k:<20} {a} -> {b_}{'   ⚠ 退行' if worse else ''}")
    regressions = []
    if last:
        for k, hb in {"conversation": True, "retrieval_mrr": True, "retrieval_recall3": True, "perplexity": False, "gen_grounded4": True,
                      "learn_sents_per_s": True, "neural_budget_loss": False, "neural_tokens_per_s": True, "neural_holdout_ppl": False,
                      "rag_grounded": True, "online_gain_nat": True, "dialog_ppl": False}.items():
            a, b_ = last.get(k), result.get(k)
            if isinstance(a, dict):
                a, b_ = a.get("acc"), (b_ or {}).get("acc")
            if a is None or b_ is None or not a:
                continue
            if (b_ < a * 0.95) if hb else (b_ > a * 1.05):
                regressions.append(f"{k}: {a} -> {b_}")
    if regressions:
        print("\n⚠ 退行:", "; ".join(regressions))
    if "brain" in result and "copy_bonus_sweep" in result["brain"]:
        print("== 復号バイアスの掃引 (接地率 / 自然さ)")
        for k, v in result["brain"]["copy_bonus_sweep"].items():
            print(f"   copy_bonus={k}: {v['grounded']} / {v['fluency']}")
    if not args.no_save:
        RESULTS.parent.mkdir(parents=True, exist_ok=True)
        with RESULTS.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    if args.fail_on_regress and regressions:
        sys.exit(1)


if __name__ == "__main__":
    main()
