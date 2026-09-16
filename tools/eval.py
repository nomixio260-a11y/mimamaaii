"""会話品質の評価: data/eval_ja.tsv の質問に答えさせ、期待する語が含まれる割合を出す。

    python tools/eval.py [--set data/eval_ja.tsv] [--verbose]

各行は「質問 <TAB> 期待語 (| 区切り) <TAB> 事前に学習させる文 (省略可)」。
行の順番は会話の順番でもある (「もっと詳しく」「それはいつ？」は直前の文脈で答える)。
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tinyai import Brain, Config  # noqa: E402


def run(set_path: Path, verbose: bool = False) -> tuple[int, int]:
    rows = []
    for ln in set_path.read_text(encoding="utf-8").splitlines():
        if not ln.strip() or ln.startswith("#"):
            continue
        parts = ln.split("\t")
        q, expect = parts[0], parts[1].split("|")
        teach = parts[2] if len(parts) > 2 else ""
        rows.append((q, expect, teach))
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Config(data_dir=Path(tmp), memory_mb=256, hard_limit=False, web_enabled=False, seed=1)
        b = Brain(cfg)
        b.bootstrap()
        ok = 0
        for q, expect, teach in rows:
            if teach:
                b.learn_text(teach, source="https://eval/x")
            r = b.reply(q)
            hit = any(e.lower() in r.text.lower() for e in expect)
            ok += hit
            if verbose or not hit:
                print(f"{'OK ' if hit else 'NG '} {q!r} -> {r.text[:80]!r} [{r.mode} {r.confidence}] 期待={expect}")
        return ok, len(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default=str(Path(__file__).resolve().parent.parent / "data" / "eval_ja.tsv"))
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    ok, n = run(Path(args.set), args.verbose)
    print(f"正答 {ok}/{n} = {ok / max(n, 1):.3f}")


if __name__ == "__main__":
    main()
