"""ブラウザ用の書き出し: モデルを int8 (行ごとのスケール) に量子化して model.bin に、語彙・知識文・
検証ベクトルを JSON に書く。ブラウザ側 (web/engine.js) はこれを読み込み、float32 に戻して推論と
リアルタイム学習 (1 ターンごとの勾配更新) を行う。

model.bin の構成: meta.json の tensors に各テンソルの [名前, 形, 型, オフセット] がある。
  q8  : int8 の値 rows×cols、続いて float32 のスケール rows 個 (行ごとに |max|/127)
  f32 : float32 そのまま (RMSNorm の利得など小さいもの)
量子化で 4 分の 1 のサイズになり (base 2.9M params → 約 3 MB)、誤差は logits で 1% 程度。"""
from __future__ import annotations

import json
import struct
from pathlib import Path

from . import neural
from .bpe import BOS, BOT, USR


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


def export_model(model: "neural.TinyTransformer", tok, out_dir: Path, meta_extra: dict | None = None) -> dict:
    """model.bin / meta.json / vocab.json / test.json を書く。返り値は meta。"""
    np = neural.np
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    src = model.ema if model.ema is not None and all(k in model.ema and model.ema[k].shape == v.shape for k, v in model.p.items()) else model.p
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
    (out_dir / "model.bin").write_bytes(b"".join(blobs))
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


def export_brain(brain, out_dir: Path, max_docs: int = 12000, max_chars: int = 1_500_000, replay: int = 400) -> dict:
    """Brain 全体からブラウザ用の一式を書き出す (モデル + 知識文 + 再生用の会話例 + 進化の統計)。"""
    nl = brain.neural
    if nl.model is None or nl.tok is None:
        raise RuntimeError("ニューラル LM がまだありません (train で学習してください)")
    out_dir = Path(out_dir)
    docs = []
    total = 0
    for d in sorted(brain.kb.docs.values(), key=lambda d: (-(d.hits + d.score), d.id)):
        t = d.text.strip()
        if len(t) < 8 or len(t) > 300:
            continue
        docs.append(t)
        total += len(t)
        if len(docs) >= max_docs or total >= max_chars:
            break
    pairs = [[u, b] for u, b, _, w in list(brain.dialogs.pairs)[-replay * 3 :] if w > 0 and len(b) <= 200][-replay:]
    kb = {"docs": docs, "replay": pairs}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "kb.json").write_text(json.dumps(kb, ensure_ascii=False), encoding="utf-8")
    stats = nl.stats()
    extra = {
        "size": nl.size, "trained_tokens": nl.trained_tokens, "holdout_ppl": nl.holdout_ppl, "ngram_ppl": nl.ngram_ppl,
        "grown_layers": nl.grown, "vocab_added": nl.vocab_added, "online_steps": nl.online_steps, "decode": nl.decode,
        "last_loss": stats.get("last_loss"), "loss_hist": [round(x, 3) for x in nl.loss_hist[-100:]],
        "kb_docs": len(brain.kb), "facts": brain.facts.count,
        "dialogs": len(brain.dialogs), "exported_docs": len(docs),
    }
    return export_model(nl.model, nl.tok, out_dir, extra)
