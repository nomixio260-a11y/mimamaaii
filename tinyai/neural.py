"""本物の (小型) ニューラル言語モデル: デコーダ専用 Transformer を numpy だけで実装 (順伝播・逆伝播・AdamW)。

* 依存は numpy のみ (任意依存)。無ければ tinyai は従来の n-gram + 検索で動く
* 語彙は n-gram LM の頻度上位 (CJK 1 文字 / 英単語) + 特殊トークン (<pad> <unk> <bos> <eos> <usr> <bot>)
* 学習データは知識文と会話ペア (<usr> 質問 <bot> 応答 <eos>) の再生バッファ
* 既定は d=128, 4 ヘッド, 2 層, 文脈 64 トークン, 語彙 4096 → 約 0.93M パラメータ (float32 で 4MB、Adam 込み 12MB)
* 生成 (温度・top-k サンプリング) と対数尤度による採点を提供する

自前実装なので、テスト (tests/) で有限差分による勾配検査を行っている。
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

try:
    import numpy as np
except ImportError:  # numpy が無ければこのモジュールは使えない (Brain 側で判定)
    np = None

PAD, UNK, BOS, EOS, USR, BOT = 0, 1, 2, 3, 4, 5
SPECIALS = ["<pad>", "<unk>", "<bos>", "<eos>", "<usr>", "<bot>"]


def available() -> bool:
    return np is not None


# ---------------------------------------------------------------- 語彙
class NeuralVocab:
    def __init__(self, words: list[str]):
        self.words = list(SPECIALS) + [w for w in words if w not in SPECIALS]
        self.index = {w: i for i, w in enumerate(self.words)}

    @classmethod
    def from_counts(cls, counts: dict[str, int], size: int = 4096) -> "NeuralVocab":
        top = sorted(counts.items(), key=lambda x: -x[1])[: max(0, size - len(SPECIALS))]
        return cls([w for w, _ in top])

    def encode(self, tokens) -> list[int]:
        idx = self.index
        return [idx.get(t, UNK) for t in tokens]

    def decode(self, ids) -> list[str]:
        return [self.words[i] for i in ids if i >= len(SPECIALS)]

    def __len__(self) -> int:
        return len(self.words)


# ---------------------------------------------------------------- 演算 (順伝播 + 逆伝播)
def _act(x):
    """活性化は ReLU (GELU は tanh と x**3 で学習時間の 6 割を占めた。数百万パラメータ規模では差が出ない)。"""
    return np.maximum(x, 0.0)


def _act_grad(x):
    return (x > 0.0).astype(x.dtype)


def _ln_forward(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    inv = 1.0 / np.sqrt(var + eps)
    xhat = (x - mu) * inv
    return xhat * g + b, (xhat, inv, g)


def _ln_backward(dy, cache):
    xhat, inv, g = cache
    dg = (dy * xhat).sum(axis=tuple(range(dy.ndim - 1)))
    db = dy.sum(axis=tuple(range(dy.ndim - 1)))
    dxhat = dy * g
    n = xhat.shape[-1]
    dx = inv * (dxhat - dxhat.mean(-1, keepdims=True) - xhat * (dxhat * xhat).mean(-1, keepdims=True))
    return dx, dg, db


def _softmax(x):
    m = x.max(-1, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(-1, keepdims=True)


# ---------------------------------------------------------------- モデル
class TinyTransformer:
    def __init__(self, vocab_size: int, d: int = 128, heads: int = 4, layers: int = 2, ctx: int = 64, seed: int = 0, dtype=None):
        assert np is not None, "numpy が必要です"
        self.V, self.d, self.h, self.L, self.T = vocab_size, d, heads, layers, ctx
        self.dtype = dtype or np.float32
        rng = np.random.default_rng(seed)
        s = 0.02
        p = {}
        p["wte"] = (rng.standard_normal((vocab_size, d)) * s).astype(self.dtype)
        p["wpe"] = (rng.standard_normal((ctx, d)) * s).astype(self.dtype)
        for i in range(layers):
            p[f"l{i}.ln1g"] = np.ones(d, self.dtype)
            p[f"l{i}.ln1b"] = np.zeros(d, self.dtype)
            p[f"l{i}.wqkv"] = (rng.standard_normal((d, 3 * d)) * s).astype(self.dtype)
            p[f"l{i}.bqkv"] = np.zeros(3 * d, self.dtype)
            p[f"l{i}.wo"] = (rng.standard_normal((d, d)) * s / math.sqrt(2 * layers)).astype(self.dtype)
            p[f"l{i}.bo"] = np.zeros(d, self.dtype)
            p[f"l{i}.ln2g"] = np.ones(d, self.dtype)
            p[f"l{i}.ln2b"] = np.zeros(d, self.dtype)
            p[f"l{i}.w1"] = (rng.standard_normal((d, 4 * d)) * s).astype(self.dtype)
            p[f"l{i}.b1"] = np.zeros(4 * d, self.dtype)
            p[f"l{i}.w2"] = (rng.standard_normal((4 * d, d)) * s / math.sqrt(2 * layers)).astype(self.dtype)
            p[f"l{i}.b2"] = np.zeros(d, self.dtype)
        p["lnfg"] = np.ones(d, self.dtype)
        p["lnfb"] = np.zeros(d, self.dtype)
        self.p = p
        self.m = {k: np.zeros_like(v) for k, v in p.items()}
        self.v = {k: np.zeros_like(v) for k, v in p.items()}
        self.step = 0
        self.mask = np.triu(np.full((ctx, ctx), -1e9, self.dtype), 1)

    def n_params(self) -> int:
        return int(sum(v.size for v in self.p.values()))

    # ------------------------------------------------------------ 順伝播
    def forward(self, ids, train: bool = False):
        """ids: (B, T) int。戻り値 logits (B, T, V) と逆伝播用キャッシュ。"""
        p = self.p
        B, T = ids.shape
        x = p["wte"][ids] + p["wpe"][:T][None, :, :]
        caches = []
        dh = self.d // self.h
        mask = self.mask[:T, :T]
        for i in range(self.L):
            h, ln1c = _ln_forward(x, p[f"l{i}.ln1g"], p[f"l{i}.ln1b"])
            qkv = h @ p[f"l{i}.wqkv"] + p[f"l{i}.bqkv"]
            q, k, v = np.split(qkv, 3, axis=-1)
            q = q.reshape(B, T, self.h, dh).transpose(0, 2, 1, 3)  # B,h,T,dh
            k = k.reshape(B, T, self.h, dh).transpose(0, 2, 1, 3)
            v = v.reshape(B, T, self.h, dh).transpose(0, 2, 1, 3)
            att = q @ k.transpose(0, 1, 3, 2) / math.sqrt(dh) + mask
            att = _softmax(att)
            a = (att @ v).transpose(0, 2, 1, 3).reshape(B, T, self.d)
            ao = a @ p[f"l{i}.wo"] + p[f"l{i}.bo"]
            x2 = x + ao
            h2, ln2c = _ln_forward(x2, p[f"l{i}.ln2g"], p[f"l{i}.ln2b"])
            pre = h2 @ p[f"l{i}.w1"] + p[f"l{i}.b1"]
            act = _act(pre)
            mo = act @ p[f"l{i}.w2"] + p[f"l{i}.b2"]
            x3 = x2 + mo
            if train:
                caches.append((h, ln1c, q, k, v, att, a, x2, h2, ln2c, pre, act))
            x = x3
        xf, lnfc = _ln_forward(x, p["lnfg"], p["lnfb"])
        logits = xf @ p["wte"].T
        return logits, (ids, caches, xf, lnfc)

    def loss_and_grads(self, ids, targets):
        """交差エントロピー (pad は無視) と全パラメータの勾配。"""
        p = self.p
        logits, (ids, caches, xf, lnfc) = self.forward(ids, train=True)
        B, T, V = logits.shape
        # softmax はその場で (float32 のまま) 計算してメモリと時間を節約
        logits -= logits.max(-1, keepdims=True)
        np.exp(logits, out=logits)
        logits /= logits.sum(-1, keepdims=True)
        probs = logits
        valid = (targets != PAD)
        n = max(int(valid.sum()), 1)
        bi = np.arange(B)[:, None]
        ti = np.arange(T)[None, :]
        loss = -np.log(np.maximum(probs[bi, ti, targets], 1e-9))
        loss = float((loss * valid).sum() / n)
        g = {}
        dlogits = probs
        dlogits[bi, ti, targets] -= 1.0
        dlogits *= (valid[:, :, None] / n).astype(self.dtype)
        g["wte"] = dlogits.reshape(-1, V).T @ xf.reshape(-1, self.d)
        dxf = dlogits @ p["wte"]
        dx, dg, db = _ln_backward(dxf, lnfc)
        g["lnfg"] = dg
        g["lnfb"] = db
        dh_ = self.d // self.h
        for i in reversed(range(self.L)):
            h, ln1c, q, k, v, att, a, x2, h2, ln2c, pre, act = caches[i]
            # MLP
            dmo = dx
            g[f"l{i}.w2"] = act.reshape(-1, 4 * self.d).T @ dmo.reshape(-1, self.d)
            g[f"l{i}.b2"] = dmo.sum(axis=(0, 1))
            dact = dmo @ p[f"l{i}.w2"].T
            dpre = dact * _act_grad(pre)
            g[f"l{i}.w1"] = h2.reshape(-1, self.d).T @ dpre.reshape(-1, 4 * self.d)
            g[f"l{i}.b1"] = dpre.sum(axis=(0, 1))
            dh2 = dpre @ p[f"l{i}.w1"].T
            dx2, dg2, db2 = _ln_backward(dh2, ln2c)
            g[f"l{i}.ln2g"] = dg2
            g[f"l{i}.ln2b"] = db2
            dx2 = dx2 + dx  # 残差
            # Attention
            dao = dx2
            g[f"l{i}.wo"] = a.reshape(-1, self.d).T @ dao.reshape(-1, self.d)
            g[f"l{i}.bo"] = dao.sum(axis=(0, 1))
            da = (dao @ p[f"l{i}.wo"].T).reshape(B, T, self.h, dh_).transpose(0, 2, 1, 3)
            datt = da @ v.transpose(0, 1, 3, 2)
            dv = att.transpose(0, 1, 3, 2) @ da
            dscore = att * (datt - (datt * att).sum(-1, keepdims=True))
            dscore = dscore / math.sqrt(dh_)
            dq = dscore @ k
            dk = dscore.transpose(0, 1, 3, 2) @ q
            dqkv = np.concatenate([
                dq.transpose(0, 2, 1, 3).reshape(B, T, self.d),
                dk.transpose(0, 2, 1, 3).reshape(B, T, self.d),
                dv.transpose(0, 2, 1, 3).reshape(B, T, self.d),
            ], axis=-1)
            g[f"l{i}.wqkv"] = h.reshape(-1, self.d).T @ dqkv.reshape(-1, 3 * self.d)
            g[f"l{i}.bqkv"] = dqkv.sum(axis=(0, 1))
            dh1 = dqkv @ p[f"l{i}.wqkv"].T
            dx1, dg1, db1 = _ln_backward(dh1, ln1c)
            g[f"l{i}.ln1g"] = dg1
            g[f"l{i}.ln1b"] = db1
            dx = dx1 + dx2  # 残差
        # 埋め込み (add.at は遅いので one-hot 行列積で)
        flat_ids = ids.reshape(-1)
        onehot = np.zeros((self.V, flat_ids.size), self.dtype)
        onehot[flat_ids, np.arange(flat_ids.size)] = 1.0
        g["wte"] += onehot @ dx.reshape(-1, self.d)
        g["wpe"] = np.zeros_like(p["wpe"])
        g["wpe"][:T] = dx.sum(axis=0)
        return loss, g

    def adamw(self, grads, lr: float = 3e-4, beta1=0.9, beta2=0.99, wd=0.01, clip=1.0):
        self.step += 1
        norm = math.sqrt(sum(float((v.astype(np.float64) ** 2).sum()) for v in grads.values()))
        scale = min(1.0, clip / (norm + 1e-6))
        b1t = 1 - beta1 ** self.step
        b2t = 1 - beta2 ** self.step
        for k, gk in grads.items():
            gk = gk * scale
            m = self.m[k]
            v = self.v[k]
            m *= beta1
            m += (1 - beta1) * gk
            v *= beta2
            v += (1 - beta2) * gk * gk
            upd = lr * (m / b1t) / (np.sqrt(v / b2t) + 1e-8)
            if gk.ndim >= 2:
                self.p[k] *= (1 - lr * wd)
            self.p[k] -= upd.astype(self.dtype)
        return norm

    # ------------------------------------------------------------ 利用
    def logprob(self, ids: list[int]) -> float:
        """系列の平均対数尤度 (文脈は末尾 T トークン)。"""
        if len(ids) < 2:
            return -20.0
        ids = ids[-(self.T + 1):]
        x = np.array([ids[:-1]], dtype=np.int64)
        logits, _ = self.forward(x)
        lp = np.log(_softmax(logits[0].astype(np.float64)) + 1e-12)
        tgt = np.array(ids[1:])
        return float(lp[np.arange(len(tgt)), tgt].mean())

    def generate(self, prompt: list[int], max_new: int = 40, temperature: float = 0.8, top_k: int = 40, rng=None, stop=(EOS,)) -> list[int]:
        rng = rng or np.random.default_rng()
        out = list(prompt)
        for _ in range(max_new):
            ctx = out[-self.T:]
            logits, _ = self.forward(np.array([ctx], dtype=np.int64))
            z = logits[0, -1].astype(np.float64) / max(temperature, 1e-3)
            z[PAD] = -1e9
            z[UNK] = -1e9
            if top_k and top_k < len(z):
                thr = np.partition(z, -top_k)[-top_k]
                z[z < thr] = -1e9
            pr = _softmax(z)
            tok = int(rng.choice(len(pr), p=pr))
            if tok in stop:
                break
            out.append(tok)
        return out[len(prompt):]

    # ------------------------------------------------------------ 保存
    def save(self, path: Path, vocab: NeuralVocab, meta: dict | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **{f"p.{k}": v for k, v in self.p.items()}, **{f"m.{k}": v for k, v in self.m.items()}, **{f"v.{k}": v for k, v in self.v.items()},
                            step=np.array(self.step), shape=np.array([self.V, self.d, self.h, self.L, self.T]))
        Path(str(path) + ".vocab.json").write_text(json.dumps({"words": vocab.words, "meta": meta or {}}, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> tuple["TinyTransformer", NeuralVocab, dict]:
        path = Path(path)
        z = np.load(path)
        V, d, h, L, T = [int(x) for x in z["shape"]]
        model = cls(V, d, h, L, T)
        for k in list(model.p):
            model.p[k] = z[f"p.{k}"]
            model.m[k] = z[f"m.{k}"]
            model.v[k] = z[f"v.{k}"]
        model.step = int(z["step"])
        vj = json.loads(Path(str(path) + ".vocab.json").read_text(encoding="utf-8"))
        vocab = NeuralVocab([w for w in vj["words"] if w not in SPECIALS])
        return model, vocab, vj.get("meta", {})


# ---------------------------------------------------------------- 学習データと学習ループ
class SequencePool:
    """学習用の系列 (トークン ID) の再生バッファ。知識文と会話ペアを混ぜる。"""

    def __init__(self, capacity: int = 20000, seed: int = 0):
        self.capacity = capacity
        self.items: list[list[int]] = []
        self.pos = 0
        self.rng = np.random.default_rng(seed) if np is not None else None

    def add(self, ids: list[int]) -> None:
        if len(ids) < 3:
            return
        if len(self.items) < self.capacity:
            self.items.append(ids)
        else:
            self.items[self.pos] = ids
            self.pos = (self.pos + 1) % self.capacity

    def batch(self, B: int, T: int):
        """(B, T) の入力と目標 (次トークン)。短い系列は pad。長い系列はランダムな窓。"""
        x = np.full((B, T), PAD, dtype=np.int64)
        y = np.full((B, T), PAD, dtype=np.int64)
        n = len(self.items)
        for b in range(B):
            seq = self.items[int(self.rng.integers(n))]
            if len(seq) > T + 1:
                s = int(self.rng.integers(len(seq) - T))
                seq = seq[s : s + T + 1]
            L = min(len(seq) - 1, T)
            x[b, :L] = seq[:L]
            y[b, :L] = seq[1 : L + 1]
        return x, y

    def __len__(self) -> int:
        return len(self.items)


def train_steps(model: TinyTransformer, pool: SequencePool, steps: int, batch: int = 32, lr: float = 3e-4, warmup: int = 100, log=None) -> dict:
    """steps 回の更新。損失の推移とトークン/秒を返す。"""
    if len(pool) == 0:
        return {"steps": 0}
    t0 = time.perf_counter()
    losses = []
    for _ in range(steps):
        x, y = pool.batch(batch, model.T)
        loss, g = model.loss_and_grads(x, y)
        cur_lr = lr * min(1.0, (model.step + 1) / warmup)
        model.adamw(g, lr=cur_lr)
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
        x = np.array([s[:-1]], dtype=np.int64)
        logits, _ = model.forward(x)
        l = np.log(_softmax(logits[0].astype(np.float64)) + 1e-12)
        tgt = np.array(s[1:])
        lp += float(l[np.arange(len(tgt)), tgt].sum())
        n += len(tgt)
    return math.exp(-lp / max(n, 1))
