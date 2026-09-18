"""ブラウザ用の書き出し: モデルを int8 (行ごとのスケール) に量子化して model.bin に、語彙・知識文・
検証ベクトルを JSON に書く。ブラウザ側 (web/engine.js) はこれを読み込み、float32 に戻して推論と
リアルタイム学習 (1 ターンごとの勾配更新) を行う。

model.bin の構成: meta.json の tensors に各テンソルの [名前, 形, 型, オフセット] がある。
  q8  : int8 の値 rows×cols、続いて float32 のスケール rows 個 (行ごとに |max|/127)
  f32 : float32 そのまま (RMSNorm の利得など小さいもの)
量子化で 4 分の 1 のサイズになり (base 2.9M params → 約 3 MB)、誤差は logits で 1% 程度。"""
from __future__ import annotations

import ast
import base64
import json
import re
import struct
from pathlib import Path

from . import neural
from .bpe import BOS, BOT, USR
from .textquality import strip_broken


def quantize_rows(w):
    np = neural.np
    w = np.asarray(w, dtype=np.float32)
    if w.ndim == 1:
        w = w[None, :]
    scale = np.abs(w).max(axis=1) / 127.0
    scale = np.where(scale == 0, 1.0, scale).astype(np.float32)
    q = np.clip(np.rint(w / scale[:, None]), -127, 127).astype(np.int8)
    return q, scale


def dequantize_rows(q, scale):
    return (q.astype(neural.np.float32) * scale[:, None]).astype(neural.np.float32)


def export_model(model: "neural.TinyTransformer", tok, out_dir: Path, meta_extra: dict | None = None, use_ema: bool = True) -> dict:
    """model.bin / meta.json / vocab.json / test.json を書く。返り値は meta。"""
    np = neural.np
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    src = model.ema if use_ema and model.ema is not None and all(k in model.ema and model.ema[k].shape == v.shape for k, v in model.p.items()) else model.p
    names = ["wte"]
    for i in range(model.L):
        names += [f"l{i}.rms1", f"l{i}.wqkv", f"l{i}.wo", f"l{i}.rms2", f"l{i}.w1", f"l{i}.wg", f"l{i}.w2"]
    names.append("rmsf")
    blobs = []
    tensors = []
    offset = 0
    deq = {}
    for name in names:
        w = src[name]
        if w.ndim == 1:
            b = w.astype(np.float32).tobytes()
            tensors.append({"name": name, "shape": list(w.shape), "dtype": "f32", "offset": offset})
            deq[name] = w.astype(np.float32)
        else:
            q, scale = quantize_rows(w)
            b = q.tobytes() + scale.tobytes()
            tensors.append({"name": name, "shape": list(w.shape), "dtype": "q8", "offset": offset})
            deq[name] = dequantize_rows(q, scale)
        pad = (-len(b)) % 4
        b += b"\0" * pad
        blobs.append(b)
        offset += len(b)
    raw = b"".join(blobs)
    (out_dir / "model.bin").write_bytes(raw)
    # 静的ホスティング先によっては .bin を配信できないので base64 テキストも置く (ページ側が自動で選ぶ)
    (out_dir / "model.b64.txt").write_text(base64.b64encode(raw).decode("ascii"), encoding="utf-8")
    meta = {
        "V": model.V, "d": model.d, "heads": model.h, "layers": model.L, "ctx": model.T, "ff": model.ff,
        "params": model.n_params(), "step": model.step, "bytes": offset, "tensors": tensors, "ema": src is not model.p,
    }
    meta.update(meta_extra or {})
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    (out_dir / "vocab.json").write_text(json.dumps(tok.tokens, ensure_ascii=False), encoding="utf-8")
    # 検証ベクトル: 量子化後の重みで numpy が計算した logits と損失・勾配 (JS 移植の正しさを確かめる)
    qm = neural.TinyTransformer(model.V, model.d, model.h, model.L, model.T, ff=model.ff)
    for k in qm.p:
        qm.p[k] = deq[k]
    prompt = [BOS, USR] + tok.encode("こんにちは、元気？", max_tokens=20) + [BOT]
    logits, _ = qm.forward(np.array([prompt], dtype=np.int64))
    last = logits[0, -1].astype(np.float64)
    order = np.argsort(-last)[:10]
    seq = prompt + tok.encode("元気だよ。ありがとう。", max_tokens=20) + [3]
    x = np.array([seq[:-1]], dtype=np.int64)
    y = np.array([seq[1:]], dtype=np.int64)
    w = neural.SequencePool.token_weights(len(seq) - 1, seq.index(BOT) + 1)[None, :]
    loss, g = qm.loss_and_grads(x, y, w)
    test = {
        "prompt": prompt, "top": [[int(i), float(last[i])] for i in order],
        "train": {"seq": seq, "loss": float(loss), "grad_rmsf": [float(v) for v in g["rmsf"]],
                  "grad_norm_wo0": float(np.sqrt((g["l0.wo"] ** 2).sum())), "grad_norm_wte": float(np.sqrt((g["wte"] ** 2).sum()))},
    }
    (out_dir / "test.json").write_text(json.dumps(test), encoding="utf-8")
    return meta


_STEP_RE = re.compile(r"^step (\d+) loss ([\d.]+) (\d+) tok/s\s+(\{.*?\})\s+経過 ([\d.]+) 分")
_EVAL_RE = re.compile(r"^eval (\{.*\})\s*$")
_COLLECT_RE = re.compile(r"^\s+収集 \[(\w+)\] (.+?) : (\d+) 文, 会話 (\d+)(?: \(会話計 (\d+)\))?")
# 進化の記録: 層の追加・中間次元の拡張・成長の取り消し (UI で学習曲線の上に印を出す)
_GROW_RE = re.compile(r"ニューラル LM: 層を追加 -> (\d+) 層 \((\d+) params\)")
_WIDEN_RE = re.compile(r"ニューラル LM: 中間次元を拡張 -> ff=(\d+) \((\d+) params\)")
_ROLLBACK_RE = re.compile(r"成長が裏目に出たため取り消し \(ppl ([\d.]+) -> ([\d.]+)、(\d+) 層へ戻す\)")


def history_from_logs(paths, max_points: int = 400) -> dict:
    """train コマンドのログから学習の経過 (ステップ・損失・取り置き ppl・速度) と収集の記録を取り出す。
    ブラウザ版の「サーバー側の学習の様子」に渡す。"""
    steps: list[dict] = []
    collects: list[dict] = []
    evals: list[dict] = []
    growth: list[dict] = []
    last_step = 0
    for path in paths:
        path = Path(path)
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            m = _STEP_RE.match(line)
            if m:
                try:
                    ev = ast.literal_eval(m.group(4))
                except Exception:
                    ev = {}
                last_step = int(m.group(1))
                steps.append({"step": int(m.group(1)), "loss": round(float(m.group(2)), 3),
                              "tok_s": int(m.group(3)), "ppl": ev.get("neural_ppl"), "ngram_ppl": round(ev.get("ngram_ppl"), 1) if ev.get("ngram_ppl") else None})
                continue
            m = _EVAL_RE.match(line)
            if m:
                try:
                    evals.append(json.loads(m.group(1)))
                except Exception:
                    pass
                continue
            m = _COLLECT_RE.match(line)
            if m:
                collects.append({"kind": m.group(1), "topic": m.group(2)[:60], "sentences": int(m.group(3)), "dialogs": int(m.group(4)),
                                 "total_dialogs": int(m.group(5)) if m.group(5) else None})
                continue
            m = _GROW_RE.search(line)
            if m:
                growth.append({"step": last_step, "kind": "layer", "value": int(m.group(1)), "params": int(m.group(2))})
                continue
            m = _WIDEN_RE.search(line)
            if m:
                growth.append({"step": last_step, "kind": "width", "value": int(m.group(1)), "params": int(m.group(2))})
                continue
            m = _ROLLBACK_RE.search(line)
            if m:
                growth.append({"step": last_step, "kind": "rollback", "value": int(m.group(3)), "params": None})
    steps.sort(key=lambda r: r["step"])
    dedup: list[dict] = []
    for r in steps:
        if dedup and dedup[-1]["step"] == r["step"]:
            dedup[-1] = r
        else:
            dedup.append(r)
    if len(dedup) > max_points:                       # 間引く (先頭と末尾は残す)
        k = len(dedup) / max_points
        dedup = [dedup[min(int(i * k), len(dedup) - 1)] for i in range(max_points)]
    evals.sort(key=lambda r: r.get("step", 0))
    growth.sort(key=lambda r: r["step"])
    return {"train_log": dedup, "collect_log": collects[-60:], "collect_total": len(collects),
            "eval_log": evals[-200:], "growth_log": growth[-60:]}


def select_docs(kb, max_docs: int, max_chars: int) -> list[str]:
    """ブラウザ版に配る知識文を選ぶ。出典ごとに点数順に並べ、出典を順番に回して取る。

    点数順にそのまま取ると、よく引かれる 1 つの出典 (青空文庫の 1 作品など) が上位を埋めてしまい、
    配る知識が偏る。話題の広さは文の数より出典の数で決まる。"""
    by_src: dict[str, list[str]] = {}
    for d in sorted(kb.docs.values(), key=lambda d: (-(d.hits + d.score), d.id)):
        t = strip_broken(d.text).strip()      # 文字化けの記号は落とす (JSON に読めない文字を出さない)
        if len(t) < 8 or len(t) > 300:
            continue
        by_src.setdefault(d.source, []).append(t)
    docs: list[str] = []
    total = 0
    queues = [iter(v) for v in by_src.values()]
    while queues and len(docs) < max_docs and total < max_chars:
        alive = []
        for q in queues:
            t = next(q, None)
            if t is None:
                continue
            docs.append(t)
            total += len(t)
            alive.append(q)
            if len(docs) >= max_docs or total >= max_chars:
                break
        queues = alive
    return docs


def export_brain(brain, out_dir: Path, max_docs: int = 12000, max_chars: int = 1_500_000, replay: int = 400, logs=None) -> dict:
    """Brain 全体からブラウザ用の一式を書き出す (モデル + 知識文 + 再生用の会話例 + 進化の統計)。"""
    nl = brain.neural
    if nl.model is None or nl.tok is None:
        raise RuntimeError("ニューラル LM がまだありません (train で学習してください)")
    # どちらの重みを配るか決めるために、書き出す前に評価する (EMA が劣化していたら生の重みを配る)
    if getattr(nl, "_holdout", None):
        nl.evaluate()
    out_dir = Path(out_dir)
    docs = select_docs(brain.kb, max_docs, max_chars)
    pairs = [[strip_broken(u), strip_broken(b)] for u, b, _, w, *_ in list(brain.dialogs.pairs)[-replay * 3 :]
             if w > 0 and len(b) <= 200][-replay:]
    kb = {"docs": docs, "replay": pairs}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "kb.json").write_text(json.dumps(kb, ensure_ascii=False), encoding="utf-8")
    stats = nl.stats()
    extra = {
        "size": nl.size, "trained_tokens": nl.trained_tokens, "holdout_ppl": nl.holdout_ppl, "ngram_ppl": nl.ngram_ppl,
        "growth_records": getattr(nl, "growth_records", [])[-10:],   # 成長の前後で品質がどう動いたか
        "grown_layers": nl.grown, "widened": getattr(nl, "widened", 0), "vocab_added": nl.vocab_added, "online_steps": nl.online_steps, "decode": getattr(nl, "decode_stable", None) or nl.decode,
        "last_loss": stats.get("last_loss"), "loss_hist": [round(x, 3) for x in nl.loss_hist[-100:]],
        "kb_docs": len(brain.kb), "facts": brain.facts.count,
        # ディスクに貯めた学習トークン (UI で「学習素材」と「トークン/パラメータ」を出すため)
        "corpus_tokens": nl.corpus.tokens if getattr(nl, "corpus", None) else 0,
        "corpus_seqs": len(nl.corpus) if getattr(nl, "corpus", None) else 0,
        "pool_tokens": nl.pool.total_tokens if getattr(nl, "pool", None) else 0,
        "dialogs": len(brain.dialogs), "dialogs_with_history": brain.dialogs.with_history(), "exported_docs": len(docs), "dialogs_by_source": brain.dialogs.by_source(),
        "stats": {k: v for k, v in brain.stats.items() if isinstance(v, (int, float))},
    }
    if logs:
        extra.update(history_from_logs(logs))
        if extra.get("train_log") and not extra.get("loss_hist"):
            extra["loss_hist"] = [r["loss"] for r in extra["train_log"]]
    return export_model(nl.model, nl.tok, out_dir, extra, use_ema=getattr(nl, "use_ema", True))
