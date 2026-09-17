"""ニューラル言語モデル本体: LLaMA 系のデコーダ専用 Transformer を numpy だけで実装。

構成 (現代的な小型 LLM の標準構成):
  * RMSNorm (pre-norm)、RoPE (回転位置埋め込み)、SwiGLU の MLP、バイアス無し、入出力埋め込みの共有
  * 学習: 交差エントロピー、AdamW、勾配クリップ、ウォームアップ + コサイン減衰、系列パッキング (pad の無駄なし)
  * 生成: KV キャッシュによる逐次デコード、温度、top-k、top-p (nucleus)、繰り返しペナルティ
順伝播・逆伝播ともに手書き (自動微分なし)。tests/ の有限差分検査で勾配の正しさを確認している。

サイズプリセット (語彙 6,000 の場合のパラメータ数):
  small: d=128, 2 層, 4 ヘッド, 文脈 64   ≈ 1.2M
  base : d=192, 4 層, 6 ヘッド, 文脈 128  ≈ 3.0M
  large: d=256, 6 層, 8 ヘッド, 文脈 128  ≈ 6.5M
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

# BLAS のスレッド数は既定 1: 要素ごとの演算が支配的なので、プロセス並列 (neural_parallel) の方が速い。
# 環境変数で上書きできる (単一プロセスで大きな行列積を回す時は 4 など)
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, os.environ.get("TINYAI_BLAS_THREADS", "1"))

try:
    import numpy as np
except ImportError:  # numpy が無ければこのモジュールは使えない (Brain 側で判定)
    np = None

from .bpe import PAD, UNK, BOS, EOS, USR, BOT, CTX, SEP, SPECIALS, SubwordTokenizer  # noqa: F401

PRESETS = {
    "small": dict(d=128, layers=2, heads=4, ctx=64, ff=384, batch=32, lr=1e-3),
    "base": dict(d=192, layers=4, heads=6, ctx=128, ff=512, batch=12, lr=6e-4),
    "large": dict(d=256, layers=6, heads=8, ctx=128, ff=704, batch=8, lr=4e-4),
    "xl": dict(d=384, layers=8, heads=8, ctx=160, ff=1024, batch=6, lr=3e-4),
}
# 成長の上限 (プリセット名 -> 最大層数)。層は損失が停滞した時に 1 層ずつ、関数を保ったまま追加される
MAX_LAYERS = {"small": 4, "base": 6, "large": 8, "xl": 12}


def available() -> bool:
    return np is not None


# ---------------------------------------------------------------- 基本演算
def _rms_forward(x, g, eps=1e-5):
    ms = (x * x).mean(-1, keepdims=True)
    inv = 1.0 / np.sqrt(ms + eps)
    xn = x * inv
    return xn * g, (xn, inv, g)


def _rms_backward(dy, cache):
    xn, inv, g = cache
    dg = (dy * xn).sum(axis=tuple(range(dy.ndim - 1)))
    dxn = dy * g
    # d/dx [x * inv]: inv * (dxn - xn * mean(dxn * xn))
    dx = inv * (dxn - xn * (dxn * xn).mean(-1, keepdims=True))
    return dx, dg


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _silu(x):
    return x * _sigmoid(x)


def _silu_grad_from_sig(x, s):
    """順伝播で計算した sigmoid を使い回す (exp を 3 回節約)。"""
    return s * (1.0 + x * (1.0 - s))


def _softmax(x):
    m = x.max(-1, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(-1, keepdims=True)


def _rope_tables(ctx: int, dh: int, dtype, base: float = 10000.0):
    half = dh // 2
    freqs = 1.0 / (base ** (np.arange(0, half, dtype=np.float64) / half))
    pos = np.arange(ctx, dtype=np.float64)[:, None] * freqs[None, :]
    return np.cos(pos).astype(dtype), np.sin(pos).astype(dtype)  # (ctx, half)


def _rope(x, cos, sin):
    """x: (B, h, T, dh)。前半と後半を対で回転。cos/sin: (T, half)。"""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)


def _rope_backward(d, cos, sin):
    half = d.shape[-1] // 2
    d1, d2 = d[..., :half], d[..., half:]
    return np.concatenate([d1 * cos + d2 * sin, -d1 * sin + d2 * cos], axis=-1)


# ---------------------------------------------------------------- モデル
class TinyTransformer:
    def __init__(self, vocab_size: int, d: int = 192, heads: int = 6, layers: int = 4, ctx: int = 128, ff: int | None = None, seed: int = 0, dtype=None, dropout: float = 0.0):
        assert np is not None, "numpy が必要です"
        self.dropout = float(dropout)          # 残差ブロック出力のドロップアウト率 (学習時のみ、過学習の抑制)
        self._drop_rng = np.random.default_rng(seed + 99)
        assert d % heads == 0 and (d // heads) % 2 == 0
        self.V, self.d, self.h, self.L, self.T = vocab_size, d, heads, layers, ctx
        self.ff = ff or (8 * d // 3 // 64 * 64 or 64)
        self.dtype = dtype or np.float32
        rng = np.random.default_rng(seed)
        s = 0.02
        p = {"wte": (rng.standard_normal((vocab_size, d)) * s).astype(self.dtype)}
        for i in range(layers):
            p[f"l{i}.rms1"] = np.ones(d, self.dtype)
            p[f"l{i}.wqkv"] = (rng.standard_normal((d, 3 * d)) * s).astype(self.dtype)
            p[f"l{i}.wo"] = (rng.standard_normal((d, d)) * s / math.sqrt(2 * layers)).astype(self.dtype)
            p[f"l{i}.rms2"] = np.ones(d, self.dtype)
            p[f"l{i}.w1"] = (rng.standard_normal((d, self.ff)) * s).astype(self.dtype)
            p[f"l{i}.wg"] = (rng.standard_normal((d, self.ff)) * s).astype(self.dtype)
            p[f"l{i}.w2"] = (rng.standard_normal((self.ff, d)) * s / math.sqrt(2 * layers)).astype(self.dtype)
        p["rmsf"] = np.ones(d, self.dtype)
        self.p = p
        self.m = {k: np.zeros_like(v) for k, v in p.items()}
        self.v = {k: np.zeros_like(v) for k, v in p.items()}
        self.step = 0
        self.mask = np.triu(np.full((ctx, ctx), -1e9, self.dtype), 1)
        self.cos, self.sin = _rope_tables(ctx, d // heads, self.dtype)

    def grow_layer(self) -> int:
        """関数を保ったまま層を 1 つ追加する (Net2Net 型): 新しい層の出力射影 wo, w2 をゼロにすると
        追加直後は恒等写像で、既に学んだ振る舞いを壊さずに容量だけ増える。"""
        rng = np.random.default_rng(self.step + 7)
        i = self.L
        d, ff = self.d, self.ff
        s = 0.02
        new = {
            f"l{i}.rms1": np.ones(d, self.dtype),
            f"l{i}.wqkv": (rng.standard_normal((d, 3 * d)) * s).astype(self.dtype),
            f"l{i}.wo": np.zeros((d, d), self.dtype),
            f"l{i}.rms2": np.ones(d, self.dtype),
            f"l{i}.w1": (rng.standard_normal((d, ff)) * s).astype(self.dtype),
            f"l{i}.wg": (rng.standard_normal((d, ff)) * s).astype(self.dtype),
            f"l{i}.w2": np.zeros((ff, d), self.dtype),
        }
        for k, v in new.items():
            self.p[k] = v
            self.m[k] = np.zeros_like(v)
            self.v[k] = np.zeros_like(v)
        self.L += 1
        return self.L

    def add_tokens(self, n: int) -> int:
        """語彙を n 語増やす (新しい埋め込みは既存の平均 + 小さな乱数)。トークナイザ側と同期して呼ぶ。"""
        if n <= 0:
            return self.V
        rng = np.random.default_rng(self.step + 11)
        mean = self.p["wte"].mean(axis=0, keepdims=True)
        extra = (mean + rng.standard_normal((n, self.d)) * 0.01).astype(self.dtype)
        self.p["wte"] = np.concatenate([self.p["wte"], extra], axis=0)
        self.m["wte"] = np.concatenate([self.m["wte"], np.zeros_like(extra)], axis=0)
        self.v["wte"] = np.concatenate([self.v["wte"], np.zeros_like(extra)], axis=0)
        self.V += n
        return self.V

    @classmethod
    def from_preset(cls, vocab_size: int, name: str = "base", seed: int = 0) -> "TinyTransformer":
        cfg = PRESETS[name]
        return cls(vocab_size, d=cfg["d"], heads=cfg["heads"], layers=cfg["layers"], ctx=cfg["ctx"], ff=cfg["ff"], seed=seed)

    def n_params(self) -> int:
        return int(sum(v.size for v in self.p.values()))

    # ------------------------------------------------------------ 順伝播
    def forward(self, ids, train: bool = False):
        p = self.p
        B, T = ids.shape
        dh = self.d // self.h
        cos, sin = self.cos[:T], self.sin[:T]
        mask = self.mask[:T, :T]
        x = p["wte"][ids]
        caches = []
        for i in range(self.L):
            h, c1 = _rms_forward(x, p[f"l{i}.rms1"])
            qkv = h @ p[f"l{i}.wqkv"]
            q, k, v = np.split(qkv, 3, axis=-1)
            q = _rope(q.reshape(B, T, self.h, dh).transpose(0, 2, 1, 3), cos, sin)
            k = _rope(k.reshape(B, T, self.h, dh).transpose(0, 2, 1, 3), cos, sin)
            v = v.reshape(B, T, self.h, dh).transpose(0, 2, 1, 3)
            att = _softmax(q @ k.transpose(0, 1, 3, 2) / math.sqrt(dh) + mask)
            a = (att @ v).transpose(0, 2, 1, 3).reshape(B, T, self.d)
            ao = a @ p[f"l{i}.wo"]
            m1 = m2 = None
            if train and self.dropout > 0:
                m1 = (self._drop_rng.random(ao.shape) >= self.dropout).astype(self.dtype) * (1.0 / (1.0 - self.dropout))
                ao *= m1
            x2 = x + ao
            h2, c2 = _rms_forward(x2, p[f"l{i}.rms2"])
            u = h2 @ p[f"l{i}.w1"]
            gt = h2 @ p[f"l{i}.wg"]
            sig = _sigmoid(gt)
            silu = gt * sig
            act = silu * u
            mo = act @ p[f"l{i}.w2"]
            if train and self.dropout > 0:
                m2 = (self._drop_rng.random(mo.shape) >= self.dropout).astype(self.dtype) * (1.0 / (1.0 - self.dropout))
                mo *= m2
            x3 = x2 + mo
            if train:
                caches.append((h, c1, q, k, v, att, a, h2, c2, u, gt, sig, silu, act, m1, m2))
            x = x3
        xf, cf = _rms_forward(x, p["rmsf"])
        logits = xf @ p["wte"].T
        return logits, (ids, caches, xf, cf)

    def loss_and_grads(self, ids, targets, weights=None):
        """weights: (B,) の系列ごと、または (B, T) のトークンごとの重み。正なら通常の交差エントロピー、
        負なら unlikelihood (その系列を出しにくくする: L = -log(1 - p_target))。None なら全て +1。
        重み 0 のトークンは損失に入らない (損失マスク)。"""
        p = self.p
        logits, (ids, caches, xf, cf) = self.forward(ids, train=True)
        B, T, V = logits.shape
        dh = self.d // self.h
        cos, sin = self.cos[:T], self.sin[:T]
        logits -= logits.max(-1, keepdims=True)
        np.exp(logits, out=logits)
        logits /= logits.sum(-1, keepdims=True)
        probs = logits
        valid = (targets != PAD)
        n = max(int(valid.sum()), 1)
        valid = valid.astype(self.dtype)
        bi = np.arange(B)[:, None]
        ti = np.arange(T)[None, :]
        pt = probs[bi, ti, targets]
        if weights is None:
            w = np.ones((B, 1), self.dtype)
        else:
            w = np.asarray(weights, self.dtype)
            w = w.reshape(B, 1) if w.ndim == 1 else w.reshape(B, T)  # 系列ごと or トークンごとの重み
        pos = (w > 0)
        # 損失: 正例は -log p、負例は -log(1-p)
        loss_pos = -np.log(np.maximum(pt, 1e-9))
        loss_neg = -np.log(np.maximum(1.0 - pt, 1e-9))
        loss = float(((np.where(pos, loss_pos, loss_neg) * np.abs(w)) * valid).sum() / n)
        g = {}
        dlogits = probs
        dlogits[bi, ti, targets] -= 1.0          # = p - onehot  (交差エントロピーの勾配)
        # 負例: d(-log(1-p_t))/dz = -(p_t/(1-p_t)) (p - onehot)。係数は暴走しないよう 5 で頭打ち
        factor = np.where(pos, 1.0, -np.minimum(pt / np.maximum(1.0 - pt, 1e-6), 5.0)) * np.abs(w)
        dlogits *= (factor * valid / n)[:, :, None].astype(self.dtype)
        g["wte"] = dlogits.reshape(-1, V).T @ xf.reshape(-1, self.d)
        dxf = dlogits @ p["wte"]
        dx, g["rmsf"] = _rms_backward(dxf, cf)
        for i in reversed(range(self.L)):
            h, c1, q, k, v, att, a, h2, c2, u, gt, sig, silu, act, m1, m2 = caches[i]
            # MLP (SwiGLU)
            dmo = dx * m2 if m2 is not None else dx
            g[f"l{i}.w2"] = act.reshape(-1, self.ff).T @ dmo.reshape(-1, self.d)
            dact = dmo @ p[f"l{i}.w2"].T
            du = dact * silu
            dgt = dact * u * _silu_grad_from_sig(gt, sig)
            g[f"l{i}.w1"] = h2.reshape(-1, self.d).T @ du.reshape(-1, self.ff)
            g[f"l{i}.wg"] = h2.reshape(-1, self.d).T @ dgt.reshape(-1, self.ff)
            dh2 = du @ p[f"l{i}.w1"].T + dgt @ p[f"l{i}.wg"].T
            dx2, g[f"l{i}.rms2"] = _rms_backward(dh2, c2)
            dx2 = dx2 + dx
            # Attention
            dao = dx2 * m1 if m1 is not None else dx2
            g[f"l{i}.wo"] = a.reshape(-1, self.d).T @ dao.reshape(-1, self.d)
            da = (dao @ p[f"l{i}.wo"].T).reshape(B, T, self.h, dh).transpose(0, 2, 1, 3)
            datt = da @ v.transpose(0, 1, 3, 2)
            dv = att.transpose(0, 1, 3, 2) @ da
            dscore = att * (datt - (datt * att).sum(-1, keepdims=True)) / math.sqrt(dh)
            dq = _rope_backward(dscore @ k, cos, sin)
            dk = _rope_backward(dscore.transpose(0, 1, 3, 2) @ q, cos, sin)
            dqkv = np.concatenate([
                dq.transpose(0, 2, 1, 3).reshape(B, T, self.d),
                dk.transpose(0, 2, 1, 3).reshape(B, T, self.d),
                dv.transpose(0, 2, 1, 3).reshape(B, T, self.d),
            ], axis=-1)
            g[f"l{i}.wqkv"] = h.reshape(-1, self.d).T @ dqkv.reshape(-1, 3 * self.d)
            dh1 = dqkv @ p[f"l{i}.wqkv"].T
            dx1, g[f"l{i}.rms1"] = _rms_backward(dh1, c1)
            dx = dx1 + dx2
        flat = ids.reshape(-1)
        order = np.argsort(flat, kind="stable")
        uniq, start = np.unique(flat[order], return_index=True)
        g["wte"][uniq] += np.add.reduceat(dx.reshape(-1, self.d)[order], start, axis=0)
        # 系列 (行) ごとの平均損失: 再生バッファの優先度 (難しい系列を多く出す) に使う
        row_valid = valid.sum(1)
        self.last_row_loss = (np.where(pos, loss_pos, loss_neg) * valid).sum(1) / np.maximum(row_valid, 1)
        return loss, g

    def adamw(self, grads, lr: float = 3e-4, beta1=0.9, beta2=0.99, wd=0.05, clip=1.0) -> float:
        self.step += 1
        norm = math.sqrt(sum(float(np.dot(v.ravel(), v.ravel())) for v in grads.values()))
        scale = min(1.0, clip / (norm + 1e-6))
        b1t = 1 - beta1 ** self.step
        b2t = 1 - beta2 ** self.step
        lr_eff = lr / b1t
        for k, gk in grads.items():
            m, v = self.m[k], self.v[k]
            # m = b1 m + (1-b1) g ; v = b2 v + (1-b2) g²   (全部インプレース、一時配列は 1 つ)
            m *= beta1
            m += ((1 - beta1) * scale) * gk
            tmp = gk * gk
            tmp *= (1 - beta2) * scale * scale
            v *= beta2
            v += tmp
            np.multiply(v, 1.0 / b2t, out=tmp)
            np.sqrt(tmp, out=tmp)
            tmp += 1e-8
            np.divide(m, tmp, out=tmp)
            if gk.ndim >= 2:
                self.p[k] *= (1 - lr * wd)
            self.p[k] -= (lr_eff * tmp).astype(self.dtype, copy=False)
        return norm

    # ------------------------------------------------------------ 推論
    def _step(self, tok: int, pos: int, cache: list) -> np.ndarray:
        """KV キャッシュを使って 1 トークン進め、次トークンの logits を返す。"""
        p = self.p
        dh = self.d // self.h
        x = p["wte"][tok][None, :]  # (1, d)
        cos, sin = self.cos[pos : pos + 1], self.sin[pos : pos + 1]
        for i in range(self.L):
            h, _ = _rms_forward(x, p[f"l{i}.rms1"])
            qkv = h @ p[f"l{i}.wqkv"]
            q, k, v = np.split(qkv, 3, axis=-1)
            q = _rope(q.reshape(1, self.h, 1, dh), cos, sin)
            k = _rope(k.reshape(1, self.h, 1, dh), cos, sin)
            v = v.reshape(1, self.h, 1, dh)
            K, Vc = cache[i]
            K = np.concatenate([K, k], axis=2) if K is not None else k
            Vc = np.concatenate([Vc, v], axis=2) if Vc is not None else v
            cache[i] = (K, Vc)
            att = _softmax(q @ K.transpose(0, 1, 3, 2) / math.sqrt(dh))
            a = (att @ Vc).reshape(1, self.d)
            x = x + a @ p[f"l{i}.wo"]
            h2, _ = _rms_forward(x, p[f"l{i}.rms2"])
            x = x + (_silu(h2 @ p[f"l{i}.wg"]) * (h2 @ p[f"l{i}.w1"])) @ p[f"l{i}.w2"]
        xf, _ = _rms_forward(x, p["rmsf"])
        return (xf @ p["wte"].T)[0]

    def _step_batch(self, toks: np.ndarray, pos: int, cache: list) -> np.ndarray:
        """B 本の系列を同時に 1 トークン進める (候補を並列に生成するため)。toks: (B,)"""
        p = self.p
        dh = self.d // self.h
        B = toks.shape[0]
        x = p["wte"][toks]  # (B, d)
        cos, sin = self.cos[pos : pos + 1], self.sin[pos : pos + 1]
        for i in range(self.L):
            h, _ = _rms_forward(x, p[f"l{i}.rms1"])
            qkv = h @ p[f"l{i}.wqkv"]
            q, k, v = np.split(qkv, 3, axis=-1)
            q = _rope(q.reshape(B, self.h, 1, dh), cos, sin)
            k = _rope(k.reshape(B, self.h, 1, dh), cos, sin)
            v = v.reshape(B, self.h, 1, dh)
            K, Vc = cache[i]
            K = np.concatenate([K, k], axis=2) if K is not None else k
            Vc = np.concatenate([Vc, v], axis=2) if Vc is not None else v
            cache[i] = (K, Vc)
            att = _softmax(q @ K.transpose(0, 1, 3, 2) / math.sqrt(dh))
            a = (att @ Vc).reshape(B, self.d)
            x = x + a @ p[f"l{i}.wo"]
            h2, _ = _rms_forward(x, p[f"l{i}.rms2"])
            x = x + (_silu(h2 @ p[f"l{i}.wg"]) * (h2 @ p[f"l{i}.w1"])) @ p[f"l{i}.w2"]
        xf, _ = _rms_forward(x, p["rmsf"])
        return xf @ p["wte"].T  # (B, V)

    def generate_batch(self, prompt: list[int], n: int = 3, max_new: int = 40, temperature: float = 0.8, top_k: int = 40, top_p: float = 0.9, repetition_penalty: float = 1.3, rng=None, stop=(EOS,)) -> list[list[int]]:
        """同じプロンプトから n 本の候補を同時に生成 (1 本ずつより約 n 倍速い)。"""
        rng = rng or np.random.default_rng()
        prompt = prompt[-(self.T - 1):]
        cache = [(None, None) for _ in range(self.L)]
        logits = None
        for pos, tok in enumerate(prompt):
            logits = self._step_batch(np.full(n, tok, dtype=np.int64), pos, cache)
        outs: list[list[int]] = [[] for _ in range(n)]
        alive = np.ones(n, dtype=bool)
        pos = len(prompt)
        for _ in range(max_new):
            if pos >= self.T or not alive.any():
                break
            z = logits.astype(np.float64)
            z[:, PAD] = -1e9
            z[:, UNK] = -1e9
            next_toks = np.zeros(n, dtype=np.int64)
            for b in range(n):
                if not alive[b]:
                    continue
                zb = z[b]
                if repetition_penalty > 1.0 and outs[b]:
                    for t in set(outs[b][-20:]):
                        zb[t] = zb[t] / repetition_penalty if zb[t] > 0 else zb[t] * repetition_penalty
                zb = zb / max(temperature, 1e-3)
                if top_k and top_k < len(zb):
                    thr = np.partition(zb, -top_k)[-top_k]
                    zb[zb < thr] = -1e9
                pr = _softmax(zb)
                if 0 < top_p < 1.0:
                    order = np.argsort(-pr)
                    cum = np.cumsum(pr[order])
                    cut = order[cum > top_p]
                    if len(cut) > 1:
                        pr[cut[1:]] = 0.0
                        pr /= pr.sum()
                tok = int(rng.choice(len(pr), p=pr))
                if tok in stop:
                    alive[b] = False
                    continue
                outs[b].append(tok)
                next_toks[b] = tok
            logits = self._step_batch(next_toks, pos, cache)
            pos += 1
        return outs

    def logprob(self, ids: list[int]) -> float:
        if len(ids) < 2:
            return -20.0
        ids = ids[-(self.T + 1):]
        logits, _ = self.forward(np.array([ids[:-1]], dtype=np.int64))
        lp = np.log(_softmax(logits[0].astype(np.float64)) + 1e-12)
        tgt = np.array(ids[1:])
        return float(lp[np.arange(len(tgt)), tgt].mean())

    def generate(self, prompt: list[int], max_new: int = 40, temperature: float = 0.8, top_k: int = 40, top_p: float = 0.9, repetition_penalty: float = 1.3, rng=None, stop=(EOS,)) -> list[int]:
        rng = rng or np.random.default_rng()
        prompt = prompt[-(self.T - 1):]
        cache = [(None, None) for _ in range(self.L)]
        logits = None
        for pos, tok in enumerate(prompt):
            logits = self._step(tok, pos, cache)
        out: list[int] = []
        pos = len(prompt)
        for _ in range(max_new):
            if pos >= self.T:
                break
            z = logits.astype(np.float64)
            z[PAD] = -1e9
            z[UNK] = -1e9
            if repetition_penalty > 1.0 and out:
                recent = set(out[-20:])
                for t in recent:
                    z[t] = z[t] / repetition_penalty if z[t] > 0 else z[t] * repetition_penalty
            z = z / max(temperature, 1e-3)
            if top_k and top_k < len(z):
                thr = np.partition(z, -top_k)[-top_k]
                z[z < thr] = -1e9
            pr = _softmax(z)
            if 0 < top_p < 1.0:
                order = np.argsort(-pr)
                cum = np.cumsum(pr[order])
                cut = order[cum > top_p]
                if len(cut) > 1:
                    pr[cut[1:]] = 0.0
                    pr /= pr.sum()
            tok = int(rng.choice(len(pr), p=pr))
            if tok in stop:
                break
            out.append(tok)
            logits = self._step(tok, pos, cache)
            pos += 1
        return out

    # ------------------------------------------------------------ 保存
    def save(self, path: Path, tokenizer: SubwordTokenizer, meta: dict | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **{f"p.{k}": v for k, v in self.p.items()}, **{f"m.{k}": v for k, v in self.m.items()}, **{f"v.{k}": v for k, v in self.v.items()},
                            step=np.array(self.step), shape=np.array([self.V, self.d, self.h, self.L, self.T, self.ff]), dropout=np.array(self.dropout))
        tokenizer.save(Path(str(path) + ".vocab.json"))
        Path(str(path) + ".meta.json").write_text(json.dumps(meta or {}, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> tuple["TinyTransformer", SubwordTokenizer, dict]:
        path = Path(path)
        z = np.load(path)
        V, d, h, L, T, ff = [int(x) for x in z["shape"]]
        model = cls(V, d, h, L, T, ff=ff, dropout=float(z["dropout"]) if "dropout" in z else 0.0)
        for k in list(model.p):
            model.p[k] = z[f"p.{k}"]
            model.m[k] = z[f"m.{k}"]
            model.v[k] = z[f"v.{k}"]
        model.step = int(z["step"])
        tok = SubwordTokenizer.load(Path(str(path) + ".vocab.json"))
        meta_path = Path(str(path) + ".meta.json")
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        return model, tok, meta


# ---------------------------------------------------------------- データと学習
class SequencePool:
    """学習系列の再生バッファ。バッチは複数の系列を <eos> 区切りで詰めて作る (系列パッキング)。

    各系列は (ids, weight, loss_from) を持つ。loss_from より前のトークン (会話の文脈・発話 = プロンプト部) は
    PROMPT_WEIGHT の小さな重みでしか学習しない (損失マスク)。応答部に勾配を集中させると同じ計算量で
    会話の質が上がり、プロンプト部にも弱い言語モデル信号を残すので文脈の読みは壊れない。"""

    PROMPT_WEIGHT = 0.2
    PRIORITY_ALPHA = 0.6      # 優先度の鋭さ (0 で一様)
    PRIORITY_MIX = 0.5        # 一様サンプリングと混ぜる割合 (過学習と忘却の両方を避ける)

    def __init__(self, capacity: int = 30000, seed: int = 0):
        self.capacity = capacity
        self.items: list[tuple[list[int], float, int]] = []
        self.pos = 0
        self.rng = np.random.default_rng(seed) if np is not None else None
        self.total_tokens = 0
        # 優先再生: 系列ごとの直近の損失 (未学習は大きめの初期値)。損失の大きい = まだ覚えていない系列を多く出す
        self.priority = np.zeros(capacity, dtype=np.float32) if np is not None else None
        self.init_priority = 3.0
        self.last_rows: list[list[int]] = []   # 直近のバッチで各行に入った系列の添字 (優先度更新用)
        self._cum = None
        self._cum_n = -1
        self.journal: list | None = None       # ワーカー同期用: None なら記録しない

    def add(self, ids: list[int], weight: float = 1.0, loss_from: int = 0) -> None:
        """weight < 0 は負例 (unlikelihood)。負例は詰め込まず単独の系列として学習する。
        loss_from: この添字以降のトークンを本来の重みで学習する (それより前はプロンプト部)。"""
        if len(ids) < 3:
            return
        item = (ids, float(weight), int(loss_from))
        if len(self.items) < self.capacity:
            self.priority[len(self.items)] = self.init_priority
            self.items.append(item)
        else:
            self.total_tokens -= len(self.items[self.pos][0])
            self.items[self.pos] = item
            self.priority[self.pos] = self.init_priority
            self.pos = (self.pos + 1) % self.capacity
        self.total_tokens += len(ids)
        self._cum_n = -1
        if self.journal is not None:
            self.journal.append(item)

    def _pick(self) -> int:
        """優先度 ∝ (損失)^α と一様の混合でサンプリング。"""
        n = len(self.items)
        if self.PRIORITY_ALPHA <= 0 or self.rng.random() < self.PRIORITY_MIX or n < 8:
            return int(self.rng.integers(n))
        if self._cum_n != n:
            pr = np.power(np.maximum(self.priority[:n], 0.05), self.PRIORITY_ALPHA)
            self._cum = np.cumsum(pr)
            self._cum_n = n
        r = self.rng.random() * self._cum[-1]
        return min(int(np.searchsorted(self._cum, r)), n - 1)

    def update(self, row_loss) -> None:
        """直近のバッチの行ごとの損失で優先度を更新 (指数移動平均)。"""
        if not self.last_rows or row_loss is None:
            return
        for idxs, l in zip(self.last_rows, row_loss):
            for i in idxs:
                if i < len(self.items):
                    self.priority[i] = 0.7 * self.priority[i] + 0.3 * float(l)
        self._cum_n = -1

    @classmethod
    def token_weights(cls, length: int, loss_from: int, weight: float = 1.0):
        """系列 (長さ length) の各「予測対象」の重み: 対象 j は ids[j+1] なので j+1 < loss_from をプロンプト扱い。"""
        tw = np.full(length, abs(weight), dtype=np.float32)
        k = min(max(loss_from - 1, 0), length)
        if k:
            tw[:k] *= cls.PROMPT_WEIGHT
        return tw

    def batch(self, B: int, T: int):
        """(x, y, weights)。weights は (B, T) のトークンごとの重み。正例は複数系列を詰めて作り、負例は単独の系列 (pad) にする。"""
        x = np.full((B, T), PAD, dtype=np.int64)
        y = np.full((B, T), PAD, dtype=np.int64)
        w = np.ones((B, T), dtype=np.float32)
        n = len(self.items)
        self.last_rows = []
        for b in range(B):
            j = self._pick()
            seq, wt, lf = self.items[j]
            rows = [j]
            self.last_rows.append(rows)
            if wt < 0:
                seq = seq[: T + 1]
                L = len(seq) - 1
                x[b, :L] = seq[:L]
                y[b, :L] = seq[1 : L + 1]
                w[b, :L] = -self.token_weights(L, lf, wt)
                continue
            buf: list[int] = []
            tw: list = []
            while len(buf) < T + 1:
                if len(seq) > T + 1:
                    s_ = int(self.rng.integers(len(seq) - T))
                    seq = seq[s_ : s_ + T + 1]
                    lf = max(0, lf - s_)
                buf.extend(seq)
                tw.append(self.token_weights(len(seq) - 1, lf, max(wt, 0.1)))
                tw.append(np.array([max(wt, 0.1)], dtype=np.float32))  # 系列末 <eos> から次系列先頭への予測
                if len(buf) < T + 1:
                    j = self._pick()
                    seq, wt, lf = self.items[j]
                    while wt < 0:
                        j = self._pick()
                        seq, wt, lf = self.items[j]
                    rows.append(j)
            buf = buf[: T + 1]
            x[b] = buf[:-1]
            y[b] = buf[1:]
            w[b] = np.concatenate(tw)[:T]
        return x, y, w

    def __len__(self) -> int:
        return len(self.items)


def lr_at(step: int, base_lr: float, warmup: int, total: int, min_ratio: float = 0.1) -> float:
    if step < warmup:
        return base_lr * (step + 1) / warmup
    if total <= warmup:
        return base_lr
    t = min(1.0, (step - warmup) / max(1, total - warmup))
    return base_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * t)))


def train_steps(model: TinyTransformer, pool: SequencePool, steps: int, batch: int = 16, lr: float = 5e-4, warmup: int = 200, total: int = 20000, log=None) -> dict:
    if len(pool) == 0:
        return {"steps": 0}
    t0 = time.perf_counter()
    losses = []
    for _ in range(steps):
        x, y, w = pool.batch(batch, model.T)
        loss, g = model.loss_and_grads(x, y, w)
        pool.update(model.last_row_loss)
        model.adamw(g, lr=lr_at(model.step, lr, warmup, total))
        losses.append(loss)
        if log and model.step % 50 == 0:
            log(model.step, loss)
    dt = time.perf_counter() - t0
    return {"steps": steps, "loss": sum(losses[-10:]) / max(len(losses[-10:]), 1), "first_loss": losses[0], "tokens_per_s": round(steps * batch * model.T / max(dt, 1e-9)), "seconds": round(dt, 1)}


def perplexity(model: TinyTransformer, seqs: list[list[int]]) -> float:
    lp = 0.0
    n = 0
    for s in seqs:
        s = s[: model.T + 1]
        if len(s) < 2:
            continue
        logits, _ = model.forward(np.array([s[:-1]], dtype=np.int64))
        l = np.log(_softmax(logits[0].astype(np.float64)) + 1e-12)
        tgt = np.array(s[1:])
        lp += float(l[np.arange(len(tgt)), tgt].sum())
        n += len(tgt)
    return math.exp(-lp / max(n, 1))
