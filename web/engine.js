/* tinyai ブラウザ推論・学習エンジン (依存なし, Worker でもメインスレッドでも動く)
 *
 * Python 側 (tinyai/neural.py) の LLaMA 系 Transformer をそのまま移植:
 *   RMSNorm (pre-norm) / RoPE / SwiGLU / 埋め込み共有 / KV キャッシュ
 *   順伝播 + 手書きの逆伝播 + AdamW  → ブラウザの中で 1 ターンごとに重みを更新する (リアルタイム学習)
 * 重みは model.bin (int8, 行ごとのスケール) から float32 に戻して持つ。
 * 「何を考えているか」用に、各トークンの上位候補と確率、最終層の注意 (プロンプトのどこを見たか) を記録する。
 */
(function (root) {
  'use strict';
  const SPECIALS = ['<pad>', '<unk>', '<bos>', '<eos>', '<usr>', '<bot>', '<ctx>', '<sep>'];
  const PAD = 0, UNK = 1, BOS = 2, EOS = 3, USR = 4, BOT = 5, CTX = 6;
  const SPACE = '▁';

  // ------------------------------------------------------------ トークナイザ (tinyai/bpe.py の移植)
  class Tokenizer {
    constructor(tokens) {
      this.tokens = SPECIALS.concat(tokens.filter((t) => SPECIALS.indexOf(t) < 0));
      this.index = new Map();
      this.tokens.forEach((t, i) => this.index.set(t, i));
      this.byFirst = new Map();
      this._rebuild();
    }
    _rebuild() {
      this.byFirst = new Map();
      for (let i = SPECIALS.length; i < this.tokens.length; i++) {
        const t = this.tokens[i];
        const k = t[0];
        if (!this.byFirst.has(k)) this.byFirst.set(k, []);
        this.byFirst.get(k).push(t);
      }
      for (const lst of this.byFirst.values()) lst.sort((a, b) => b.length - a.length);
    }
    static normalize(text) { return text.normalize('NFKC').toLowerCase(); }
    encode(text, maxTokens) {
      text = Tokenizer.normalize(text);
      const out = [];
      const spaceId = this.index.get(SPACE);
      const n = text.length;
      let i = 0;
      while (i < n) {
        const ch = text[i];
        if (/\s/.test(ch)) {
          i += 1;
          if (spaceId !== undefined && i < n && /[a-z0-9]/.test(text[i]) && out.length && out[out.length - 1] >= SPECIALS.length) out.push(spaceId);
          continue;
        }
        const cands = this.byFirst.get(ch);
        let hit = null;
        if (cands) for (const t of cands) { if (text.startsWith(t, i)) { hit = t; break; } }
        if (hit === null) { out.push(UNK); i += 1; } else { out.push(this.index.get(hit)); i += hit.length; }
        if (maxTokens !== undefined && out.length >= maxTokens) break;
      }
      return out;
    }
    decode(ids) {
      let s = '';
      for (const i of ids) { if (i < SPECIALS.length) continue; const t = this.tokens[i]; s += t === SPACE ? ' ' : t; }
      return s;
    }
    piece(i) { return i < SPECIALS.length ? SPECIALS[i] : (this.tokens[i] === SPACE ? '␣' : this.tokens[i]); }
    get length() { return this.tokens.length; }
    addTokens(list) {
      let added = 0;
      for (const t of list) { if (t && !this.index.has(t) && t.length <= 12) { this.index.set(t, this.tokens.length); this.tokens.push(t); added++; } }
      if (added) this._rebuild();
      return added;
    }
    // 新しいテキストに頻出するが語彙に無い 2〜4 文字の連続 (語彙の進化用)
    frequentNewUnits(texts, top, minCount) {
      const grams = new Map();
      const re = /[a-z0-9]+|[぀-ゟ]+|[゠-ヿ]+|[一-鿿㐀-䶿]+|\S/g;
      for (let text of texts) {
        text = Tokenizer.normalize(text);
        let m;
        while ((m = re.exec(text)) !== null) {
          const run = m[0];
          for (const L of [2, 3, 4]) for (let i = 0; i + L <= run.length; i++) { const g = run.slice(i, i + L); if (!this.index.has(g)) grams.set(g, (grams.get(g) || 0) + 1); }
        }
      }
      const ranked = [];
      for (const [g, n] of grams) if (n >= minCount) ranked.push([n * (g.length - 1), g]);
      ranked.sort((a, b) => b[0] - a[0]);
      return ranked.slice(0, top).map((x) => x[1]);
    }
  }

  // ------------------------------------------------------------ 行列演算 (Float32Array, 行優先)
  // 行列積: 内側のループを 4 つ展開し、2 行ずつまとめて読む (JIT が SIMD 化しやすい形)
  function matmul(A, M, K, W, N, out) { // out[M,N] = A[M,K] @ W[K,N]
    out = out || new Float32Array(M * N);
    const N4 = N - (N % 4);
    for (let i = 0; i < M; i++) {
      const ao = i * K, oo = i * N;
      let k = 0;
      for (; k + 1 < K; k += 2) {
        const a0 = A[ao + k], a1 = A[ao + k + 1];
        if (a0 === 0 && a1 === 0) continue;
        const w0 = k * N, w1 = w0 + N;
        let j = 0;
        for (; j < N4; j += 4) {
          out[oo + j] += a0 * W[w0 + j] + a1 * W[w1 + j];
          out[oo + j + 1] += a0 * W[w0 + j + 1] + a1 * W[w1 + j + 1];
          out[oo + j + 2] += a0 * W[w0 + j + 2] + a1 * W[w1 + j + 2];
          out[oo + j + 3] += a0 * W[w0 + j + 3] + a1 * W[w1 + j + 3];
        }
        for (; j < N; j++) out[oo + j] += a0 * W[w0 + j] + a1 * W[w1 + j];
      }
      for (; k < K; k++) {
        const a = A[ao + k];
        if (a === 0) continue;
        const wo = k * N;
        for (let j = 0; j < N; j++) out[oo + j] += a * W[wo + j];
      }
    }
    return out;
  }
  function matmulBT(A, M, K, W, N, out) { // out[M,N] = A[M,K] @ W[N,K]^T
    out = out || new Float32Array(M * N);
    const K4 = K - (K % 4);
    for (let i = 0; i < M; i++) {
      const ao = i * K, oo = i * N;
      for (let j = 0; j < N; j++) {
        const wo = j * K;
        let s0 = 0, s1 = 0, s2 = 0, s3 = 0, k = 0;
        for (; k < K4; k += 4) {
          s0 += A[ao + k] * W[wo + k]; s1 += A[ao + k + 1] * W[wo + k + 1];
          s2 += A[ao + k + 2] * W[wo + k + 2]; s3 += A[ao + k + 3] * W[wo + k + 3];
        }
        for (; k < K; k++) s0 += A[ao + k] * W[wo + k];
        out[oo + j] = s0 + s1 + s2 + s3;
      }
    }
    return out;
  }
  function matmulAT(A, M, K, D, N, out) { // out[K,N] = A[M,K]^T @ D[M,N]
    out = out || new Float32Array(K * N);
    const N4 = N - (N % 4);
    for (let i = 0; i < M; i++) {
      const ao = i * K, dofs = i * N;
      for (let k = 0; k < K; k++) {
        const a = A[ao + k];
        if (a === 0) continue;
        const oo = k * N;
        let j = 0;
        for (; j < N4; j += 4) {
          out[oo + j] += a * D[dofs + j]; out[oo + j + 1] += a * D[dofs + j + 1];
          out[oo + j + 2] += a * D[dofs + j + 2]; out[oo + j + 3] += a * D[dofs + j + 3];
        }
        for (; j < N; j++) out[oo + j] += a * D[dofs + j];
      }
    }
    return out;
  }
  function rmsForward(x, T, d, g, eps) {
    const y = new Float32Array(T * d), inv = new Float32Array(T), xn = new Float32Array(T * d);
    for (let t = 0; t < T; t++) {
      let ms = 0;
      for (let j = 0; j < d; j++) { const v = x[t * d + j]; ms += v * v; }
      const iv = 1 / Math.sqrt(ms / d + (eps || 1e-5));
      inv[t] = iv;
      for (let j = 0; j < d; j++) { const n = x[t * d + j] * iv; xn[t * d + j] = n; y[t * d + j] = n * g[j]; }
    }
    return { y, xn, inv };
  }
  function rmsBackward(dy, T, d, cache, g, dg) { // returns dx; accumulates dg
    const { xn, inv } = cache;
    const dx = new Float32Array(T * d);
    for (let t = 0; t < T; t++) {
      let mean = 0;
      for (let j = 0; j < d; j++) { const dyj = dy[t * d + j]; dg[j] += dyj * xn[t * d + j]; mean += dyj * g[j] * xn[t * d + j]; }
      mean /= d;
      for (let j = 0; j < d; j++) dx[t * d + j] = inv[t] * (dy[t * d + j] * g[j] - xn[t * d + j] * mean);
    }
    return dx;
  }
  function softmaxInPlace(z, n) {
    let m = -Infinity;
    for (let i = 0; i < n; i++) if (z[i] > m) m = z[i];
    let s = 0;
    for (let i = 0; i < n; i++) { const e = Math.exp(z[i] - m); z[i] = e; s += e; }
    for (let i = 0; i < n; i++) z[i] /= s;
  }

  // ------------------------------------------------------------ モデル
  class Model {
    constructor(meta, bin) {
      this.V = meta.V; this.d = meta.d; this.h = meta.heads; this.L = meta.layers; this.T = meta.ctx; this.ff = meta.ff;
      this.dh = this.d / this.h;
      this.step = meta.step || 0;
      this.p = {};
      const dv = new DataView(bin);
      for (const t of meta.tensors) {
        const n = t.shape.reduce((a, b) => a * b, 1);
        let arr;
        if (t.dtype === 'f32') {
          arr = new Float32Array(bin.slice(t.offset, t.offset + n * 4));
        } else { // q8: int8 の値 rows×cols, 続いて float32 スケール rows 個
          const rows = t.shape[0], cols = t.shape[1];
          const q = new Int8Array(bin, t.offset, n);
          const so = t.offset + n;
          arr = new Float32Array(n);
          for (let r = 0; r < rows; r++) {
            const s = dv.getFloat32(so + r * 4, true);
            for (let c = 0; c < cols; c++) arr[r * cols + c] = q[r * cols + c] * s;
          }
        }
        this.p[t.name] = arr;
      }
      this.shapes = {};
      for (const t of meta.tensors) this.shapes[t.name] = t.shape.slice();
      this.m = {}; this.v = {};
      this._ropeTables();
      this.lastAttention = null; // 生成時: [生成トークンごとの (プロンプト長) 注意配列]
    }
    _ropeTables() {
      const half = this.dh / 2, T = this.T;
      this.cos = new Float32Array(T * half); this.sin = new Float32Array(T * half);
      for (let t = 0; t < T; t++) for (let j = 0; j < half; j++) {
        const f = 1 / Math.pow(10000, j / half);
        this.cos[t * half + j] = Math.cos(t * f); this.sin[t * half + j] = Math.sin(t * f);
      }
    }
    nParams() { let n = 0; for (const k in this.p) n += this.p[k].length; return n; }
    // 語彙の拡張 (既存埋め込みの平均 + 小さな乱数)
    addTokens(n) {
      if (n <= 0) return this.V;
      const d = this.d, old = this.p.wte, V = this.V;
      const mean = new Float32Array(d);
      for (let v = 0; v < V; v++) for (let j = 0; j < d; j++) mean[j] += old[v * d + j] / V;
      const nw = new Float32Array((V + n) * d);
      nw.set(old);
      for (let v = V; v < V + n; v++) for (let j = 0; j < d; j++) nw[v * d + j] = mean[j] + (Math.random() * 2 - 1) * 0.017;
      this.p.wte = nw;
      if (this.m.wte) { const m2 = new Float32Array((V + n) * d); m2.set(this.m.wte); this.m.wte = m2; const v2 = new Float32Array((V + n) * d); v2.set(this.v.wte); this.v.wte = v2; }
      this.V += n; this.shapes.wte = [this.V, d];
      return this.V;
    }
    _rope(x, T, offset) { // x: (T, d) を h ヘッドに分けて回転 (インプレース)。offset: 位置のオフセット
      const d = this.d, dh = this.dh, half = dh / 2, h = this.h;
      for (let t = 0; t < T; t++) {
        const co = (t + offset) * half;
        for (let hh = 0; hh < h; hh++) {
          const base = t * d + hh * dh;
          for (let j = 0; j < half; j++) {
            const x1 = x[base + j], x2 = x[base + j + half], c = this.cos[co + j], s = this.sin[co + j];
            x[base + j] = x1 * c - x2 * s; x[base + j + half] = x1 * s + x2 * c;
          }
        }
      }
    }
    _ropeBackward(dq, T) {
      const d = this.d, dh = this.dh, half = dh / 2, h = this.h;
      for (let t = 0; t < T; t++) {
        const co = t * half;
        for (let hh = 0; hh < h; hh++) {
          const base = t * d + hh * dh;
          for (let j = 0; j < half; j++) {
            const d1 = dq[base + j], d2 = dq[base + j + half], c = this.cos[co + j], s = this.sin[co + j];
            dq[base + j] = d1 * c + d2 * s; dq[base + j + half] = -d1 * s + d2 * c;
          }
        }
      }
    }
    // 系列全体の順伝播 (学習用: キャッシュを返す)。ids: 配列 (長さ T ≤ ctx)
    forward(ids, train) {
      const T = ids.length, d = this.d, h = this.h, dh = this.dh, ff = this.ff, p = this.p, V = this.V;
      const scale = 1 / Math.sqrt(dh);
      let x = new Float32Array(T * d);
      for (let t = 0; t < T; t++) x.set(p.wte.subarray(ids[t] * d, ids[t] * d + d), t * d);
      const caches = [];
      for (let i = 0; i < this.L; i++) {
        const r1 = rmsForward(x, T, d, p[`l${i}.rms1`]);
        const qkv = matmul(r1.y, T, d, p[`l${i}.wqkv`], 3 * d);
        const q = new Float32Array(T * d), k = new Float32Array(T * d), v = new Float32Array(T * d);
        for (let t = 0; t < T; t++) { q.set(qkv.subarray(t * 3 * d, t * 3 * d + d), t * d); k.set(qkv.subarray(t * 3 * d + d, t * 3 * d + 2 * d), t * d); v.set(qkv.subarray(t * 3 * d + 2 * d, t * 3 * d + 3 * d), t * d); }
        this._rope(q, T, 0); this._rope(k, T, 0);
        const att = new Float32Array(h * T * T); // 因果マスク付き softmax
        const a = new Float32Array(T * d);
        for (let hh = 0; hh < h; hh++) {
          for (let t = 0; t < T; t++) {
            const row = att.subarray(hh * T * T + t * T, hh * T * T + t * T + t + 1);
            for (let s = 0; s <= t; s++) { let dot = 0; for (let j = 0; j < dh; j++) dot += q[t * d + hh * dh + j] * k[s * d + hh * dh + j]; row[s] = dot * scale; }
            softmaxInPlace(row, t + 1);
            for (let s = 0; s <= t; s++) { const w = row[s]; if (w === 0) continue; for (let j = 0; j < dh; j++) a[t * d + hh * dh + j] += w * v[s * d + hh * dh + j]; }
          }
        }
        const x2 = matmul(a, T, d, p[`l${i}.wo`], d);
        for (let n = 0; n < T * d; n++) x2[n] += x[n];
        const r2 = rmsForward(x2, T, d, p[`l${i}.rms2`]);
        const u = matmul(r2.y, T, d, p[`l${i}.w1`], ff);
        const gt = matmul(r2.y, T, d, p[`l${i}.wg`], ff);
        const sig = new Float32Array(T * ff), silu = new Float32Array(T * ff), act = new Float32Array(T * ff);
        for (let n = 0; n < T * ff; n++) { const s = 1 / (1 + Math.exp(-gt[n])); sig[n] = s; silu[n] = gt[n] * s; act[n] = silu[n] * u[n]; }
        const x3 = matmul(act, T, ff, p[`l${i}.w2`], d);
        for (let n = 0; n < T * d; n++) x3[n] += x2[n];
        if (train) caches.push({ r1, q, k, v, att, a, r2, u, gt, sig, silu, act });
        x = x3;
      }
      const rf = rmsForward(x, T, d, p.rmsf);
      const logits = matmulBT(rf.y, T, d, p.wte, V);
      return { logits, ids, caches, rf, T };
    }
    // 損失と勾配 (バッチ 1)。weights[t] > 0 は交差エントロピー、< 0 は unlikelihood、0 は無視
    lossAndGrads(ids, targets, weights) {
      const fw = this.forward(ids, true);
      const T = fw.T, d = this.d, V = this.V, h = this.h, dh = this.dh, ff = this.ff, p = this.p;
      const probs = fw.logits;
      for (let t = 0; t < T; t++) softmaxInPlace(probs.subarray(t * V, t * V + V), V);
      let n = 0; for (let t = 0; t < T; t++) if (targets[t] !== PAD && weights[t] !== 0) n++;
      n = Math.max(n, 1);
      let loss = 0;
      const dl = probs; // dlogits (インプレース)
      for (let t = 0; t < T; t++) {
        const y = targets[t], w = weights[t];
        if (y === PAD || w === 0) { dl.fill(0, t * V, t * V + V); continue; }
        const pt = probs[t * V + y];
        const aw = Math.abs(w);
        let factor;
        if (w > 0) { loss += -Math.log(Math.max(pt, 1e-9)) * aw; factor = aw; } else { loss += -Math.log(Math.max(1 - pt, 1e-9)) * aw; factor = -Math.min(pt / Math.max(1 - pt, 1e-6), 5) * aw; }
        dl[t * V + y] -= 1;
        const f = factor / n;
        for (let v = 0; v < V; v++) dl[t * V + v] *= f;
      }
      loss /= n;
      const g = {};
      g.wte = matmulAT(dl, T, V, fw.rf.y, d);            // (V, d)
      const dxf = matmul(dl, T, V, p.wte, d);              // (T, d)
      g.rmsf = new Float32Array(d);
      let dx = rmsBackward(dxf, T, d, fw.rf, p.rmsf, g.rmsf);
      for (let i = this.L - 1; i >= 0; i--) {
        const c = fw.caches[i];
        g[`l${i}.w2`] = matmulAT(c.act, T, ff, dx, d);
        const dact = matmulBT(dx, T, d, p[`l${i}.w2`], ff);  // w2: (ff, d) → dact = dx @ w2^T
        const du = new Float32Array(T * ff), dgt = new Float32Array(T * ff);
        for (let m = 0; m < T * ff; m++) { du[m] = dact[m] * c.silu[m]; const s = c.sig[m]; dgt[m] = dact[m] * c.u[m] * s * (1 + c.gt[m] * (1 - s)); }
        g[`l${i}.w1`] = matmulAT(c.r2.y, T, d, du, ff);
        g[`l${i}.wg`] = matmulAT(c.r2.y, T, d, dgt, ff);
        const dh2 = matmulBT(du, T, ff, p[`l${i}.w1`], d);
        { const tmp = matmulBT(dgt, T, ff, p[`l${i}.wg`], d); for (let m = 0; m < T * d; m++) dh2[m] += tmp[m]; }
        g[`l${i}.rms2`] = new Float32Array(d);
        const dx2 = rmsBackward(dh2, T, d, c.r2, p[`l${i}.rms2`], g[`l${i}.rms2`]);
        for (let m = 0; m < T * d; m++) dx2[m] += dx[m];
        g[`l${i}.wo`] = matmulAT(c.a, T, d, dx2, d);
        const da = matmulBT(dx2, T, d, p[`l${i}.wo`], d);
        const dq = new Float32Array(T * d), dk = new Float32Array(T * d), dv = new Float32Array(T * d);
        const scale = 1 / Math.sqrt(dh);
        for (let hh = 0; hh < h; hh++) {
          for (let t = 0; t < T; t++) {
            const arow = c.att.subarray(hh * T * T + t * T, hh * T * T + t * T + t + 1);
            const datt = new Float32Array(t + 1);
            let dot = 0;
            for (let s = 0; s <= t; s++) {
              let x = 0;
              for (let j = 0; j < dh; j++) { x += da[t * d + hh * dh + j] * c.v[s * d + hh * dh + j]; dv[s * d + hh * dh + j] += arow[s] * da[t * d + hh * dh + j]; }
              datt[s] = x; dot += x * arow[s];
            }
            for (let s = 0; s <= t; s++) {
              const ds = arow[s] * (datt[s] - dot) * scale;
              if (ds === 0) continue;
              for (let j = 0; j < dh; j++) { dq[t * d + hh * dh + j] += ds * c.k[s * d + hh * dh + j]; dk[s * d + hh * dh + j] += ds * c.q[t * d + hh * dh + j]; }
            }
          }
        }
        this._ropeBackward(dq, T); this._ropeBackward(dk, T);
        const dqkv = new Float32Array(T * 3 * d);
        for (let t = 0; t < T; t++) { dqkv.set(dq.subarray(t * d, t * d + d), t * 3 * d); dqkv.set(dk.subarray(t * d, t * d + d), t * 3 * d + d); dqkv.set(dv.subarray(t * d, t * d + d), t * 3 * d + 2 * d); }
        g[`l${i}.wqkv`] = matmulAT(c.r1.y, T, d, dqkv, 3 * d);
        const dh1 = matmulBT(dqkv, T, 3 * d, p[`l${i}.wqkv`], d);
        g[`l${i}.rms1`] = new Float32Array(d);
        const dx1 = rmsBackward(dh1, T, d, c.r1, p[`l${i}.rms1`], g[`l${i}.rms1`]);
        for (let m = 0; m < T * d; m++) dx1[m] += dx2[m];
        dx = dx1;
      }
      for (let t = 0; t < T; t++) { const o = ids[t] * d; for (let j = 0; j < d; j++) g.wte[o + j] += dx[t * d + j]; }
      return { loss, g };
    }
    adamw(g, lr, beta1, beta2, wd, clip) {
      beta1 = beta1 === undefined ? 0.9 : beta1; beta2 = beta2 === undefined ? 0.99 : beta2; wd = wd === undefined ? 0.05 : wd; clip = clip === undefined ? 1.0 : clip;
      this.step += 1;
      let norm = 0;
      for (const k in g) { const a = g[k]; for (let i = 0; i < a.length; i++) norm += a[i] * a[i]; }
      norm = Math.sqrt(norm);
      const scale = Math.min(1, clip / (norm + 1e-6));
      const b1t = 1 - Math.pow(beta1, this.step), b2t = 1 - Math.pow(beta2, this.step);
      const lrEff = lr / b1t;
      for (const k in g) {
        if (!this.m[k]) { this.m[k] = new Float32Array(this.p[k].length); this.v[k] = new Float32Array(this.p[k].length); }
        const m = this.m[k], v = this.v[k], w = this.p[k], gk = g[k];
        const is2d = this.shapes[k].length >= 2;
        for (let i = 0; i < w.length; i++) {
          const gi = gk[i] * scale;
          m[i] = beta1 * m[i] + (1 - beta1) * gi;
          v[i] = beta2 * v[i] + (1 - beta2) * gi * gi;
          if (is2d) w[i] *= 1 - lr * wd;
          w[i] -= lrEff * m[i] / (Math.sqrt(v[i] / b2t) + 1e-8);
        }
      }
      return norm;
    }
    // KV キャッシュで 1 トークン進める。cache = {K:[layer][pos*d], V:...}
    stepToken(tok, pos, cache, wantAttention) {
      const d = this.d, h = this.h, dh = this.dh, ff = this.ff, p = this.p, V = this.V;
      const scale = 1 / Math.sqrt(dh);
      let x = new Float32Array(d);
      x.set(p.wte.subarray(tok * d, tok * d + d));
      let attnOut = null;
      for (let i = 0; i < this.L; i++) {
        const r1 = rmsForward(x, 1, d, p[`l${i}.rms1`]);
        const qkv = matmul(r1.y, 1, d, p[`l${i}.wqkv`], 3 * d);
        const q = qkv.slice(0, d), k = qkv.slice(d, 2 * d), v = qkv.slice(2 * d, 3 * d);
        this._rope(q, 1, pos); this._rope(k, 1, pos);
        cache.K[i].set(k, pos * d); cache.V[i].set(v, pos * d);
        const a = new Float32Array(d);
        const n = pos + 1;
        const Kc = cache.K[i], Vc = cache.V[i];
        const attSum = (wantAttention && i === this.L - 1) ? new Float32Array(n) : null;
        for (let hh = 0; hh < h; hh++) {
          const row = new Float32Array(n);
          for (let s = 0; s < n; s++) { let dot = 0; for (let j = 0; j < dh; j++) dot += q[hh * dh + j] * Kc[s * d + hh * dh + j]; row[s] = dot * scale; }
          softmaxInPlace(row, n);
          for (let s = 0; s < n; s++) { const w = row[s]; for (let j = 0; j < dh; j++) a[hh * dh + j] += w * Vc[s * d + hh * dh + j]; if (attSum) attSum[s] += w / h; }
        }
        const x2 = matmul(a, 1, d, p[`l${i}.wo`], d);
        for (let j = 0; j < d; j++) x2[j] += x[j];
        const r2 = rmsForward(x2, 1, d, p[`l${i}.rms2`]);
        const u = matmul(r2.y, 1, d, p[`l${i}.w1`], ff), gt = matmul(r2.y, 1, d, p[`l${i}.wg`], ff);
        for (let m = 0; m < ff; m++) u[m] *= gt[m] / (1 + Math.exp(-gt[m]));
        const x3 = matmul(u, 1, ff, p[`l${i}.w2`], d);
        for (let j = 0; j < d; j++) x3[j] += x2[j];
        x = x3;
        if (attSum) attnOut = attSum;
      }
      const rf = rmsForward(x, 1, d, p.rmsf);
      const logits = matmulBT(rf.y, 1, d, p.wte, V);
      return { logits, attention: attnOut };
    }
    newCache() { const K = [], Vv = []; for (let i = 0; i < this.L; i++) { K.push(new Float32Array(this.T * this.d)); Vv.push(new Float32Array(this.T * this.d)); } return { K, V: Vv }; }
    // 生成: 各トークンの候補 (上位 5 の確率) と注意を記録して返す
    generate(prompt, opts) {
      opts = opts || {};
      const maxNew = opts.maxNew || 40, temperature = opts.temperature || 0.7, topK = opts.topK || 40, topP = opts.topP || 0.9, rep = opts.repetitionPenalty || 1.3;
      const stop = opts.stop || [EOS];
      const rand = opts.rand || Math.random;
      prompt = prompt.slice(-(this.T - 1));
      const cache = this.newCache();
      let out = null;
      for (let pos = 0; pos < prompt.length; pos++) out = this.stepToken(prompt[pos], pos, cache, false);
      const tokens = [], trace = [], attention = [];
      let pos = prompt.length, logpSum = 0;
      const V = this.V;
      for (let step = 0; step < maxNew; step++) {
        if (pos >= this.T) break;
        const z = new Float64Array(out.logits);
        z[PAD] = -1e9; z[UNK] = -1e9;
        if (rep > 1 && tokens.length) { const recent = new Set(tokens.slice(-20)); for (const t of recent) z[t] = z[t] > 0 ? z[t] / rep : z[t] * rep; }
        for (let i = 0; i < V; i++) z[i] /= Math.max(temperature, 1e-3);
        // top-k (部分選択: 全体をソートしない)
        const K = Math.min(topK, V);
        const kept = new Array(K).fill(-1); const kv = new Float64Array(K).fill(-Infinity);
        for (let i = 0; i < V; i++) {
          const zi = z[i]; if (zi <= kv[K - 1]) continue;
          let j = K - 1; while (j > 0 && kv[j - 1] < zi) { kv[j] = kv[j - 1]; kept[j] = kept[j - 1]; j--; }
          kv[j] = zi; kept[j] = i;
        }
        let m = z[kept[0]], s = 0;
        const pr = kept.map((i) => { const e = Math.exp(z[i] - m); s += e; return e; });
        for (let i = 0; i < pr.length; i++) pr[i] /= s;
        // top-p
        let cum = 0, cut = pr.length;
        for (let i = 0; i < pr.length; i++) { cum += pr[i]; if (cum > topP) { cut = i + 1; break; } }
        let tot = 0; for (let i = 0; i < cut; i++) tot += pr[i];
        let r = rand() * tot, chosen = kept[cut - 1], chosenP = pr[cut - 1] / tot;
        for (let i = 0; i < cut; i++) { r -= pr[i]; if (r <= 0) { chosen = kept[i]; chosenP = pr[i] / tot; break; } }
        trace.push({ token: chosen, p: chosenP, top: kept.slice(0, 5).map((i, j) => [i, pr[j] / tot]) });
        if (stop.indexOf(chosen) >= 0) break;
        tokens.push(chosen);
        logpSum += Math.log(Math.max(chosenP, 1e-9));
        out = this.stepToken(chosen, pos, cache, true);
        if (out.attention) attention.push(Array.from(out.attention.subarray(0, prompt.length)));
        pos += 1;
      }
      return { tokens, trace, attention, meanLogp: tokens.length ? logpSum / tokens.length : -20, promptLength: prompt.length };
    }
    // 保存用 (重みだけ。Adam の状態は捨てても継続学習に大きな影響はない)
    exportWeights() { const o = {}; for (const k in this.p) o[k] = this.p[k]; return { step: this.step, V: this.V, L: this.L, p: o, shapes: this.shapes }; }
    importWeights(w) {
      if (!w || w.L !== this.L) return false;
      for (const k in w.p) { if (!this.p[k]) return false; }
      this.p = {}; for (const k in w.p) this.p[k] = new Float32Array(w.p[k]);
      this.shapes = w.shapes || this.shapes; this.V = w.V; this.step = w.step; this.m = {}; this.v = {};
      return true;
    }
  }

  // ------------------------------------------------------------ 検索 (文字 2-gram の BM25)
  class Retriever {
    constructor(docs) {
      this.docs = []; this.df = new Map(); this.postings = new Map(); this.len = []; this.avg = 1;
      this.addAll(docs || []);
    }
    static grams(text) {
      const t = Tokenizer.normalize(text).replace(/\s+/g, '');
      const out = [];
      for (let i = 0; i + 1 < t.length; i++) { const g = t.slice(i, i + 2); if (/^[。、・「」『』（）！？〜ー…,.!?]/.test(g) || /[。、・「」『』（）！？〜ー…,.!?]$/.test(g)) continue; out.push(g); }
      if (out.length === 0 && t.length) out.push(t);
      return out;
    }
    add(text) {
      const id = this.docs.length;
      this.docs.push(text);
      const gs = Retriever.grams(text);
      const tf = new Map();
      for (const g of gs) tf.set(g, (tf.get(g) || 0) + 1);
      for (const [g, n] of tf) { if (!this.postings.has(g)) this.postings.set(g, []); this.postings.get(g).push([id, n]); this.df.set(g, (this.df.get(g) || 0) + 1); }
      this.len.push(gs.length);
      return id;
    }
    addAll(list) { for (const t of list) this.add(t); let s = 0; for (const l of this.len) s += l; this.avg = this.len.length ? s / this.len.length : 1; }
    search(query, k) {
      const N = this.docs.length; if (!N) return [];
      const qs = Array.from(new Set(Retriever.grams(query)));
      const scores = new Map();
      const k1 = 1.4, b = 0.6;
      for (const g of qs) {
        const post = this.postings.get(g); if (!post) continue;
        const df = this.df.get(g);
        if (df > N * 0.2 && qs.length > 2) continue;
        const idf = Math.log(1 + (N - df + 0.5) / (df + 0.5));
        for (const [id, tf] of post) { const s = idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * this.len[id] / this.avg)); scores.set(id, (scores.get(id) || 0) + s); }
      }
      const arr = Array.from(scores.entries()).sort((a, b2) => b2[1] - a[1]).slice(0, k || 3);
      const max = arr.length ? arr[0][1] : 1;
      return arr.map(([id, s]) => ({ id, text: this.docs[id], score: s, rel: s / max }));
    }
  }

  // ------------------------------------------------------------ エンジン (会話 + リアルタイム学習)
  class Engine {
    constructor(meta, bin, vocab, kb) {
      this.meta = meta;
      this.tok = new Tokenizer(vocab);
      this.model = new Model(meta, bin);
      this.kb = new Retriever(kb && kb.docs);
      this.replay = (kb && kb.replay) ? kb.replay.slice() : [];
      this.decode = Object.assign({ temperature: 0.7, top_p: 0.9, repetition_penalty: 1.3 }, meta.decode || {});
      this.lr = 3e-4;
      this.stats = { onlineSteps: 0, onlineTokens: 0, idleSteps: 0, turns: 0, good: 0, bad: 0, taught: 0, vocabAdded: 0, lossHist: [], learnedDocs: 0 };
      this.history = []; // [user, bot]
      this._fb = [0, 0]; this._fbBest = 0.5; this._trial = null;
    }
    seqDialog(user, bot, context) {
      const T = this.model.T;
      const c = context ? this.tok.encode(context, T >> 1) : [];
      const u = this.tok.encode(user, T >> 2);
      const room = T - c.length - u.length - 5;
      const b = this.tok.encode(bot, Math.max(8, room));
      let seq = [BOS]; if (c.length) seq = seq.concat([CTX], c);
      return seq.concat([USR], u, [BOT], b, [EOS]);
    }
    promptDialog(user, context) {
      const T = this.model.T;
      const c = context ? this.tok.encode(context, T >> 1) : [];
      const u = this.tok.encode(user, T >> 2);
      let seq = [BOS]; if (c.length) seq = seq.concat([CTX], c);
      return seq.concat([USR], u, [BOT]);
    }
    retrieve(user, k) {
      const hits = this.kb.search(user, k || 3);
      const ctx = hits.map((h) => h.text).join(' ').slice(0, 200);
      return { hits, context: ctx || null };
    }
    _generateCands(user, context, n, maxNew, pass) {
      const prompt = this.promptDialog(user, context);
      const cands = [];
      for (let i = 0; i < n; i++) {
        const g = this.model.generate(prompt, { maxNew, temperature: this.decode.temperature, topP: this.decode.top_p, repetitionPenalty: this.decode.repetition_penalty });
        const text = this.tok.decode(g.tokens).trim();
        // 候補の点数: 平均対数確率 + 長さと文脈との重なりの補正 (接地)
        let overlap = 0;
        if (context) { const cg = new Set(Retriever.grams(context)); for (const g2 of Retriever.grams(text)) if (cg.has(g2)) overlap++; }
        const score = g.meanLogp + Math.min(text.length, 30) * 0.02 + Math.min(overlap, 10) * 0.05;
        cands.push({ text, tokens: g.tokens, trace: g.trace, attention: g.attention, meanLogp: g.meanLogp, score, promptLength: g.promptLength, pass, prompt });
      }
      return cands;
    }
    // 応答 = 考える: 1) 検索して下書きを生成 2) 下書きの語で再検索し文脈を広げてもう一度生成 3) 全候補を採点して選ぶ
    reply(user, opts) {
      opts = opts || {};
      const n = opts.candidates || 3, t0 = Date.now(), maxNew = opts.maxNew || 48;
      const { hits, context } = this.retrieve(user, 3);
      let cands = this._generateCands(user, context, n, maxNew, 1);
      let rethink = null;
      if (opts.rethink !== false) {
        const draft = cands.slice().sort((a, b) => b.score - a.score)[0];
        if (draft && draft.text.length >= 4) {
          const hits2 = this.kb.search(user + ' ' + draft.text, 4).filter((h) => !hits.some((x) => x.id === h.id)).slice(0, 2);
          if (hits2.length) {
            const context2 = (hits.slice(0, 2).map((h) => h.text).join(' ') + ' ' + hits2.map((h) => h.text).join(' ')).slice(0, 220);
            const more = this._generateCands(user, context2, Math.max(2, n - 1), maxNew, 2);
            rethink = { hits: hits2, context: context2, candidates: more.length };
            cands = cands.concat(more);
          }
        }
      }
      cands.sort((a, b) => b.score - a.score);
      const best = cands.find((c) => c.text.length >= 2) || cands[0];
      const ctxUsed = best.pass === 2 && rethink ? rethink.context : context;
      this.history.push([user, best.text]);
      this.stats.turns += 1;
      return { text: best.text, hits, context: ctxUsed, rethink, prompt: best.prompt, promptPieces: best.prompt.map((i) => this.tok.piece(i)), candidates: cands, best, ms: Date.now() - t0 };
    }
    // 常時学習: 会話の合間に再生バッファの会話か知識文を 1 系列だけ学ぶ (数百 ms)。忘却を防ぎつつ少しずつ賢くなる
    idleStep() {
      const useDialog = this.replay.length && Math.random() < 0.5;
      let seq, lf = 0, kind;
      if (useDialog) { const [u, b] = this.replay[Math.floor(Math.random() * this.replay.length)]; seq = this.seqDialog(u, b, null); lf = seq.indexOf(BOT) + 1; kind = 'dialog'; }
      else if (this.kb.docs.length) { const t = this.kb.docs[Math.floor(Math.random() * this.kb.docs.length)]; seq = [BOS].concat(this.tok.encode(t, this.model.T - 2), [EOS]); kind = 'text'; }
      else return null;
      if (seq.length < 4) return null;
      const r = this._trainSeq(seq, lf, 0.5);
      this.stats.idleSteps = (this.stats.idleSteps || 0) + 1;
      return Object.assign(r, { kind });
    }
    // 1 ターンを即座に学習 (weight > 0: 正例、< 0: unlikelihood)。再生バッファから 1 本混ぜて忘却を防ぐ
    learnTurn(user, bot, context, weight, steps) {
      weight = weight === undefined ? 1 : weight; steps = steps || 1;
      const seq = this.seqDialog(user, bot, context);
      const lf = seq.indexOf(BOT) + 1;
      let last = null;
      for (let s = 0; s < steps; s++) {
        last = this._trainSeq(seq, lf, weight);
        if (this.replay.length && weight > 0) {
          const [ru, rb] = this.replay[Math.floor(Math.random() * this.replay.length)];
          const rs = this.seqDialog(ru, rb, null);
          this._trainSeq(rs, rs.indexOf(BOT) + 1, 0.5);
        }
      }
      if (weight > 0) { this.replay.push([user, bot]); if (this.replay.length > 600) this.replay.shift(); }
      return last;
    }
    learnText(text, weight) {
      const ids = [BOS].concat(this.tok.encode(text, this.model.T - 2), [EOS]);
      if (ids.length < 4) return null;
      const r = this._trainSeq(ids, 0, weight || 1);
      this.kb.add(text); this.stats.learnedDocs += 1;
      return r;
    }
    _trainSeq(seq, lossFrom, weight) {
      const T = Math.min(seq.length - 1, this.model.T);
      const x = seq.slice(0, T), y = seq.slice(1, T + 1);
      const w = new Float32Array(T);
      for (let t = 0; t < T; t++) { let v = Math.abs(weight); if (t + 1 < lossFrom) v *= 0.2; w[t] = weight < 0 ? -v : v; }
      const { loss, g } = this.model.lossAndGrads(x, y, w);
      const norm = this.model.adamw(g, this.lr);
      this.stats.onlineSteps += 1; this.stats.onlineTokens += T;
      this.stats.lossHist.push(Math.round(loss * 1000) / 1000); if (this.stats.lossHist.length > 200) this.stats.lossHist.shift();
      return { loss, gradNorm: norm, tokens: T };
    }
    feedback(positive) {
      this._fb[positive ? 0 : 1] += 1;
      if (positive) this.stats.good += 1; else this.stats.bad += 1;
      const n = this._fb[0] + this._fb[1];
      if (n < 4) return null;
      const rate = this._fb[0] / n;
      if (this._trial) { if (rate >= this._fbBest) this._fbBest = rate; else this.decode = this._trial; this._trial = null; } else this._fbBest = rate;
      const cand = Object.assign({}, this.decode);
      const keys = ['temperature', 'top_p', 'repetition_penalty'];
      const key = keys[Math.floor(Math.random() * 3)];
      const stepv = { temperature: 0.1, top_p: 0.05, repetition_penalty: 0.1 }[key];
      const lo = { temperature: 0.3, top_p: 0.5, repetition_penalty: 1.0 }[key], hi = { temperature: 1.2, top_p: 0.99, repetition_penalty: 2.0 }[key];
      cand[key] = Math.round(Math.min(hi, Math.max(lo, cand[key] + (Math.random() < 0.5 ? -stepv : stepv))) * 1000) / 1000;
      this._trial = Object.assign({}, this.decode); this.decode = cand; this._fb = [0, 0];
      return { decode: this.decode, rate };
    }
    evolveVocab(texts) {
      const units = this.tok.frequentNewUnits(texts, 30, 3);
      const n = this.tok.addTokens(units);
      if (n) { this.model.addTokens(n); this.stats.vocabAdded += n; }
      return n;
    }
    snapshot() {
      return { weights: this.model.exportWeights(), tokens: this.tok.tokens.slice(SPECIALS.length), stats: this.stats, decode: this.decode, replay: this.replay.slice(-300), learned: this.kb.docs.slice(this.meta.exported_docs || 0), history: this.history.slice(-50) };
    }
    restore(s) {
      if (!s) return false;
      if (s.tokens && s.tokens.length > this.tok.tokens.length - SPECIALS.length) { this.tok = new Tokenizer(s.tokens); }
      if (!this.model.importWeights(s.weights)) return false;
      this.stats = Object.assign(this.stats, s.stats || {}); this.decode = s.decode || this.decode;
      if (s.replay) this.replay = this.replay.concat(s.replay);
      if (s.learned) for (const t of s.learned) this.kb.add(t);
      if (s.history) this.history = s.history;
      return true;
    }
    summary() {
      const m = this.model;
      return { params: m.nParams(), layers: m.L, vocab: this.tok.length, d: m.d, heads: m.h, ctx: m.T, ff: m.ff, step: m.step, pretrainStep: this.meta.step, docs: this.kb.docs.length, replay: this.replay.length, decode: this.decode, stats: this.stats };
    }
  }

  const api = { Tokenizer, Model, Retriever, Engine, matmul, matmulBT, matmulAT, SPECIALS, PAD, UNK, BOS, EOS, USR, BOT, CTX };
  if (typeof module !== 'undefined' && module.exports) module.exports = api; else root.TinyAI = api;
})(typeof self !== 'undefined' ? self : this);
