"""データ並列学習 (複数プロセス + 共有メモリ)。

numpy の行列積は BLAS で多スレッド化されるが、softmax や活性化などの要素ごとの演算は 1 スレッドで、
学習時間の大半を占める。そこで CPU コアごとにワーカープロセスを fork し、各ワーカーが別々のミニバッチの
勾配を計算して共有メモリに書き、親プロセスが平均して AdamW を適用する (同期 SGD)。

* パラメータは共有メモリ上の配列なので、親の更新はワーカーから即座に見える (コピー無し)
* 勾配はワーカーごとの共有メモリ領域に書く。通信はパイプで「バッチ番号」と「損失」だけ
* ワーカー内の BLAS スレッドは 1 に固定 (過剰なスレッドを避ける)
4 コアで約 2.5〜3 倍。Linux/macOS (fork) 向け。使えない環境では単一プロセスに自動で戻る。
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import os
import time
from multiprocessing import shared_memory

try:
    import numpy as np
except ImportError:
    np = None

from . import neural

log = logging.getLogger("tinyai.parallel")


def _share(arr):
    """配列を共有メモリに移し、(共有ビュー, shm) を返す。"""
    shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
    view = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
    view[...] = arr
    return view, shm


def _worker_main(model, pool, grad_names, grad_shapes, conn, seed):
    """ワーカー: 指示を受けてミニバッチの勾配を共有メモリに書く。"""
    try:
        import threadpoolctl  # noqa: F401
    except ImportError:
        pass
    pool.rng = np.random.default_rng(seed)
    grads = {}
    for k in model.p:
        shm = shared_memory.SharedMemory(name=grad_names[k])
        grads[k] = (np.ndarray(grad_shapes[k], dtype=model.p[k].dtype, buffer=shm.buf), shm)
    try:
        while True:
            msg = conn.recv()
            if msg is None:
                break
            if msg[0] == "add":                       # 親で新しく加わった系列 (継続学習: 収集したデータを即座に反映)
                for ids, wt, lf in msg[1]:
                    pool.add(ids, wt, lf)
                continue
            batch, T = msg
            x, y, w = pool.batch(batch, T)
            loss, g = model.loss_and_grads(x, y, w)
            pool.update(model.last_row_loss)          # 優先再生の更新はワーカーごと
            for k, gk in g.items():
                grads[k][0][...] = gk
            conn.send(loss)
    finally:
        for _, shm in grads.values():
            shm.close()
        conn.close()


class ParallelTrainer:
    def __init__(self, model: "neural.TinyTransformer", pool: "neural.SequencePool", workers: int | None = None):
        self.model = model
        self.pool = pool
        cpu = os.cpu_count() or 1
        self.workers = max(1, min(workers or max(1, cpu - 1), 8))
        self._shms: list = []
        self._procs: list = []
        self._conns: list = []
        self._grad_views: list[dict] = []
        self.started = False

    def start(self) -> bool:
        if self.started or np is None or self.workers <= 1:
            return False
        if mp.get_start_method(allow_none=True) not in (None, "fork") and "fork" not in mp.get_all_start_methods():
            return False
        ctx = mp.get_context("fork")
        # パラメータを共有メモリへ (親のモデルもそのビューを使う)
        for k in list(self.model.p):
            view, shm = _share(self.model.p[k])
            self.model.p[k] = view
            self._shms.append(shm)
        for w in range(self.workers):
            names, shapes, views = {}, {}, {}
            for k, v in self.model.p.items():
                shm = shared_memory.SharedMemory(create=True, size=v.nbytes)
                self._shms.append(shm)
                names[k] = shm.name
                shapes[k] = v.shape
                views[k] = np.ndarray(v.shape, dtype=v.dtype, buffer=shm.buf)
            self._grad_views.append(views)
            parent, child = ctx.Pipe()
            proc = ctx.Process(target=_worker_main, args=(self.model, self.pool, names, shapes, child, 1000 + w), daemon=True)
            proc.start()
            child.close()
            self._procs.append(proc)
            self._conns.append(parent)
        self.started = True
        self.pool.journal = []      # ここから先に加わった系列はワーカーへ送る
        log.info("データ並列学習: %d ワーカー", self.workers)
        return True

    def sync(self) -> int:
        """親の再生バッファに新しく加わった系列をワーカーへ送る (パイプ、数千系列でも数十 ms)。"""
        j = self.pool.journal
        if not j:
            return 0
        chunk = j[:5000]
        del j[:len(chunk)]
        for c in self._conns:
            c.send(("add", chunk))
        return len(chunk)

    def stop(self) -> None:
        for c in self._conns:
            try:
                c.send(None)
            except Exception:
                pass
        for p in self._procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        self.pool.journal = None
        # パラメータを通常メモリに戻す
        for k in list(self.model.p):
            self.model.p[k] = np.array(self.model.p[k], copy=True)
        for shm in self._shms:
            try:
                shm.close()
                shm.unlink()
            except Exception:
                pass
        self._shms, self._procs, self._conns, self._grad_views = [], [], [], []
        self.started = False

    def train(self, steps: int, batch: int, lr: float, total: int, warmup: int = 200) -> dict:
        """各ワーカーが batch 本ずつ処理する = 実効バッチ workers × batch。"""
        if not self.started:
            return neural.train_steps(self.model, self.pool, steps=steps, batch=batch, lr=lr, warmup=warmup, total=total)
        t0 = time.perf_counter()
        losses = []
        T = self.model.T
        self.sync()
        for _ in range(steps):
            for c in self._conns:
                c.send((batch, T))
            step_losses = [c.recv() for c in self._conns]
            g = {}
            for k in self.model.p:
                acc = self._grad_views[0][k].copy()
                for views in self._grad_views[1:]:
                    acc += views[k]
                acc *= 1.0 / self.workers
                g[k] = acc
            self.model.adamw(g, lr=neural.lr_at(self.model.step, lr, warmup, total))
            losses.append(sum(step_losses) / len(step_losses))
        dt = time.perf_counter() - t0
        return {"steps": steps, "loss": sum(losses[-10:]) / max(len(losses[-10:]), 1), "first_loss": losses[0],
                "tokens_per_s": round(steps * batch * self.workers * T / max(dt, 1e-9)), "seconds": round(dt, 1), "workers": self.workers}
