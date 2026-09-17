"""コマンドラインインターフェース。"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from . import __version__
from .brain import Brain
from .config import Config
from .evolve import Evolver, NeuralTrainer
from .web import Fetcher


def _build_config(args) -> Config:
    cfg = Config()
    if getattr(args, "data", None):
        cfg.data_dir = Path(args.data).expanduser()
    if getattr(args, "memory", None):
        cfg.memory_mb = args.memory
    if getattr(args, "offline", False):
        cfg.web_enabled = False
    if getattr(args, "no_hard_limit", False):
        cfg.hard_limit = False
    if getattr(args, "interval", None) is not None:
        cfg.evolve_interval = args.interval
    if getattr(args, "tools", False):
        cfg.tools = True
    if getattr(args, "symbolic", False):
        cfg.neural_only = False
    return cfg


def _setup_logging(cfg: Config, verbose: bool) -> None:
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    handlers = [logging.FileHandler(cfg.data_dir / "tinyai.log", encoding="utf-8")]
    if verbose:
        handlers.append(logging.StreamHandler(sys.stderr))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s", handlers=handlers)


def _open_brain(cfg: Config) -> Brain:
    brain = Brain(cfg)
    if not brain.load():
        brain.bootstrap()
    return brain


# ---------------------------------------------------------------- chat
def cmd_chat(args) -> int:
    cfg = _build_config(args)
    _setup_logging(cfg, args.verbose)
    brain = _open_brain(cfg)
    evolver = None
    trainer = None
    if not args.no_auto:
        evolver = Evolver(brain)
        evolver.start()
    if cfg.background_training and brain.neural.available:
        trainer = NeuralTrainer(brain)
        trainer.start()
    ns = brain.neural.stats() if brain.neural.available else {}
    print(f"tinyai v{__version__}  gen={brain.generation}  docs={len(brain.kb)}  mem={brain.guard.describe()['rss_mb']}MB  neural={ns.get('size')}/{ns.get('params', 0):,}params step={ns.get('steps', 0)} ready={ns.get('ready')}")
    print("終了: /quit  状態: /stats  保存: /save  新着学習: /news  教える: 覚えて: <文>  調べさせる: 調べて: <話題>  評価: 👍 / 👎")
    try:
        while True:
            try:
                line = input("あなた> ").strip()
            except EOFError:
                break
            if not line:
                continue
            if line in ("/quit", "/exit", "/q"):
                break
            if line == "/stats":
                print(brain.describe_json())
                if evolver:
                    print(json.dumps(evolver.describe(), ensure_ascii=False))
                continue
            if line == "/save":
                print("保存:", brain.save())
                continue
            if line == "/news":
                for topic, sentence, src in brain.take_notices():
                    print(f"[学習] {topic}: {sentence}  ({src})")
                continue
            if line == "/evolve":
                print(json.dumps(brain.evolve_step(), ensure_ascii=False))
                continue
            r = brain.reply(line)
            print(f"AI> {r.text}")
            if evolver and brain.notices and args.verbose:
                print(f"   [裏で {len(brain.notices)} 件の新しい学習があります: /news で表示]")
            if args.verbose:
                extra = f"   [{r.mode} conf={r.confidence}"
                if r.sources:
                    extra += f" src={r.sources[0]}"
                if r.learned_topics:
                    extra += f" 調査予定={','.join(r.learned_topics)}"
                print(extra + "]")
    finally:
        if evolver:
            evolver.stop(wait=False)
        if trainer:
            trainer.stop()
        brain.save()
        print("保存しました。")
    return 0


def cmd_ask(args) -> int:
    cfg = _build_config(args)
    _setup_logging(cfg, args.verbose)
    brain = _open_brain(cfg)
    r = brain.reply(" ".join(args.text))
    print(r.text)
    if args.verbose:
        print(f"[{r.mode} conf={r.confidence} src={r.sources}]", file=sys.stderr)
    brain.save()
    return 0


# ---------------------------------------------------------------- learn
def n_fmt(n: int) -> str:
    return f"{n:,}"


def cmd_learn(args) -> int:
    cfg = _build_config(args)
    _setup_logging(cfg, args.verbose)
    brain = _open_brain(cfg)
    fetcher = Fetcher(cfg.user_agent, cfg.fetch_timeout, cfg.max_page_bytes) if cfg.web_enabled else None
    total = 0
    for src in args.sources:
        if src.startswith(("http://", "https://")):
            if fetcher is None:
                print("オフラインなので URL は読めません:", src)
                continue
            res = fetcher.get_text(src)
            n = brain.learn_text(res[0], source=src) if res else 0
        elif src.startswith("topic:"):
            n = brain.learn_from_web(src[6:], fetcher) if fetcher else 0
        elif src.startswith("dump:") or src.lower().endswith((".xml.bz2", ".xml.gz")) or (src.lower().endswith(".xml") and "wiki" in src.lower()):
            from .dumps import learn_wiki_dump

            path = src[5:] if src.startswith("dump:") else src
            pages, n = learn_wiki_dump(brain, path, max_pages=args.max_pages, progress=lambda pg, sn: print(f"  {pg} ページ / {n_fmt(sn)} 文  mem={brain.guard.describe()['rss_mb']}MB"))
            print(f"  ダンプ {pages} ページ")
        elif src.lower().endswith((".zip", ".gz", ".bz2", ".xz")):
            from .dumps import iter_archive_texts

            n = sum(brain.learn_text(text, source=f"file:{nm}") for nm, text in iter_archive_texts(Path(src)))
        else:
            p = Path(src)
            if p.is_dir():
                n = sum(brain.learn_file(f) for f in sorted(p.rglob("*")) if f.is_file() and f.suffix.lower() in (".txt", ".md", ".html", ".htm"))
            else:
                n = brain.learn_file(p)
        print(f"{src}: {n} 文")
        total += n
    brain.enforce_memory()
    brain.save()
    print(f"合計 {total} 文を学習。docs={len(brain.kb)} lm_entries={brain.lm.entries} mem={brain.guard.describe()['rss_mb']}MB")
    return 0


# ---------------------------------------------------------------- evolve
def cmd_evolve(args) -> int:
    cfg = _build_config(args)
    _setup_logging(cfg, True)
    brain = _open_brain(cfg)
    ev = Evolver(brain, interval=cfg.evolve_interval, max_cycles=args.cycles, max_seconds=args.seconds)
    ev.start()
    try:
        while ev.is_alive():
            ev.join(timeout=1.0)
    except KeyboardInterrupt:
        ev.stop()
    print(brain.describe_json())
    return 0


def cmd_train(args) -> int:
    """ニューラル LM (Transformer) を前面で学習する (numpy が必要)。
    --hours を付けると収集システムからデータを流し込みながら長時間学習する。"""
    cfg = _build_config(args)
    if args.size:
        cfg.neural_size = args.size
    _setup_logging(cfg, True)
    brain = _open_brain(cfg)
    if not brain.neural.available:
        print("numpy がありません: pip install numpy")
        return 1
    brain.neural.min_sentences = min(brain.neural.min_sentences, max(200, len(brain.kb)))
    brain.neural.min_chars = min(brain.neural.min_chars, max(20000, sum(len(d.text) for d in brain.kb.docs.values())))
    if brain.neural_step(steps=1, budget_seconds=0.01) is None and brain.neural.model is None:
        print("学習データが足りません (文が少なすぎます)。まず learn や evolve で集めてください。")
        return 1
    nl = brain.neural
    # 知識文と会話を全部プールへ (再生バッファの容量まで。pending の上限を迂回)
    for d in list(brain.kb.docs.values()):
        nl.add_text(d.text)
    for u, b, _, w in list(brain.dialogs.pairs):
        nl.add_dialog(u, b, weight=w)
    nl.set_workers(args.workers if args.workers is not None else cfg.neural_workers)
    print(f"モデル {nl.size}: パラメータ {nl.model.n_params():,}, 語彙 {len(nl.tok)}, プール {len(nl.pool)} 系列 / {nl.pool.total_tokens:,} トークン, 会話 {len(brain.dialogs)}, 並列 {nl.workers}")
    collector = None
    if args.hours and cfg.web_enabled:
        from .collector import Collector
        from .web import Fetcher

        collector = Collector(Fetcher(cfg.user_agent, cfg.fetch_timeout, cfg.max_page_bytes), cfg.data_dir, cfg.languages, interest=brain.interest_score)
        print("収集しながら学習します (Ctrl+C で停止)")
    t0 = time.time()
    limit_s = (args.hours * 3600) if args.hours else args.seconds
    done = 0
    last_log = 0
    try:
        while (limit_s is None or time.time() - t0 < limit_s) and (limit_s is not None or done < args.steps):
            if collector is not None and done % 100 == 0:
                batch = collector.collect_stream() or collector.collect_link()
                if batch is None:
                    topic = brain.next_topic()
                    batch = collector.collect(topic) if topic else None
                if batch is not None:
                    n = brain.learn_batch(batch, collector)
                    brain.background_step(budget_docs=400)
                    print(f"  収集 [{batch.kind}] {batch.topic[:30]} : {n} 文, 会話 {len(batch.dialogs)}")
            before = nl.model.step
            r = brain.neural_step(steps=10, budget_seconds=60)
            if r is None:
                print("学習データが足りません")
                break
            done += nl.model.step - before
            if done - last_log >= 100:
                last_log = done
                ev = nl.evaluate(brain.lm.perplexity(brain.holdout) if brain.holdout else None)
                print(f"step {nl.model.step} loss {r['loss']:.3f} {r['tokens_per_s']} tok/s  {ev}  経過 {(time.time() - t0) / 60:.1f} 分")
                brain.enforce_memory()
    except KeyboardInterrupt:
        pass
    nl.evaluate(brain.lm.perplexity(brain.holdout) if brain.holdout else None)
    nl.stop_parallel()
    brain.save()
    print(json.dumps(nl.stats(), ensure_ascii=False))
    for q in ("こんにちは", "機械学習とは？", "宇宙について教えて"):
        hits = brain._search(q)
        ctx = " ".join(d.text for _, d in hits[:2])[:200] or None
        print(q, "->", nl.chat(q, ctx, n=2))
    return 0


def cmd_export(args) -> int:
    """ブラウザ版 (web/) 用にモデル・語彙・知識文を書き出す。"""
    cfg = _build_config(args)
    _setup_logging(cfg, False)
    brain = _open_brain(cfg)
    if not brain.neural.available:
        print("numpy がありません: pip install numpy")
        return 1
    brain.neural.ensure_model()
    from .export import export_brain

    try:
        meta = export_brain(brain, Path(args.out), max_docs=args.max_docs)
    except RuntimeError as e:
        print(e)
        return 1
    print(json.dumps({k: v for k, v in meta.items() if k != "tensors"}, ensure_ascii=False))
    print(f"書き出し先: {args.out} (model.bin {meta['bytes'] / 1e6:.1f} MB)。web/index.html, engine.js, worker.js と同じ場所に置いて配信してください。")
    return 0


def cmd_stats(args) -> int:
    cfg = _build_config(args)
    _setup_logging(cfg, False)
    brain = _open_brain(cfg)
    print(brain.describe_json())
    return 0


# ---------------------------------------------------------------- serve
def cmd_serve(args) -> int:
    cfg = _build_config(args)
    _setup_logging(cfg, args.verbose)
    brain = _open_brain(cfg)
    evolver = None
    trainer = None
    if not args.no_auto:
        evolver = Evolver(brain)
        evolver.start()
    if cfg.background_training and brain.neural.available:
        trainer = NeuralTrainer(brain)
        trainer.start()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, payload, ctype="application/json; charset=utf-8"):
            body = payload if isinstance(payload, bytes) else json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlsplit(self.path)
            if u.path == "/":
                self._send(200, _INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
            elif u.path == "/stats":
                d = brain.describe()
                if evolver:
                    d["evolver"] = evolver.describe()
                self._send(200, d)
            elif u.path == "/ask":
                q = parse_qs(u.query).get("q", [""])[0]
                r = brain.reply(q) if q else None
                self._send(200, r.__dict__ if r else {"error": "q required"})
            elif u.path == "/notices":
                self._send(200, {"notices": brain.take_notices()})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            u = urlsplit(self.path)
            n = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(min(n, 1_000_000)).decode("utf-8", "replace")
            try:
                js = json.loads(raw) if raw else {}
            except ValueError:
                js = {"text": raw}
            if u.path == "/ask":
                r = brain.reply(js.get("text", ""))
                self._send(200, r.__dict__)
            elif u.path == "/learn":
                n = brain.learn_text(js.get("text", ""), source=js.get("source", "api"))
                self._send(200, {"learned": n})
            else:
                self._send(404, {"error": "not found"})

        def log_message(self, fmt, *a):
            logging.getLogger("tinyai.http").info(fmt, *a)

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"http://{args.host}:{args.port}/  (Ctrl+C で終了)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if evolver:
            evolver.stop(wait=False)
        if trainer:
            trainer.stop()
        brain.save()
    return 0


_INDEX_HTML = """<!doctype html><meta charset=utf-8><title>tinyai</title>
<style>body{font-family:sans-serif;max-width:640px;margin:2em auto}#log{white-space:pre-wrap;border:1px solid #ccc;padding:1em;min-height:12em}input{width:80%}</style>
<h1>tinyai</h1><div id=log></div><form id=f><input id=q autofocus placeholder="話しかけてください"><button>送信</button></form>
<script>
const log=document.getElementById('log');
document.getElementById('f').onsubmit=async e=>{e.preventDefault();const q=document.getElementById('q');const t=q.value;q.value='';log.textContent+='あなた> '+t+'\\n';
const r=await fetch('/ask',{method:'POST',body:JSON.stringify({text:t})});const j=await r.json();log.textContent+='AI> '+j.text+'  ['+j.mode+' '+j.confidence+']\\n';};
</script>"""


# ---------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="tinyai", description="超小型・自己進化型の会話 AI")
    ap.add_argument("--data", help="保存ディレクトリ (既定 ~/.tinyai または $TINYAI_DATA)")
    ap.add_argument("--memory", type=int, help="メモリ上限 MB (既定 256 または $TINYAI_MEMORY_MB)")
    ap.add_argument("--offline", action="store_true", help="Web 探索を無効化")
    ap.add_argument("--no-hard-limit", action="store_true", help="OS のメモリ強制上限 (rlimit) を掛けない")
    ap.add_argument("--interval", type=float, help="自律学習サイクルの間隔 (秒)")
    ap.add_argument("--tools", action="store_true", help="計算・日付・単位換算などの道具を有効にする (既定はニューラル生成のみ)")
    ap.add_argument("--symbolic", action="store_true", help="ニューラル専用モードを切り、事実の即答や検索応答も使う")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--version", action="version", version=f"tinyai {__version__}")
    sub = ap.add_subparsers(dest="cmd")

    p = sub.add_parser("chat", help="対話する (裏で自動学習)")
    p.add_argument("--no-auto", action="store_true", help="バックグラウンド学習をしない")
    p.set_defaults(func=cmd_chat)

    p = sub.add_parser("ask", help="1 問だけ聞く")
    p.add_argument("text", nargs="+")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("learn", help="ファイル/ディレクトリ/URL/topic:話題/dump:ウィキダンプ(.xml.bz2)/書庫(.zip .gz) から学習")
    p.add_argument("sources", nargs="+")
    p.add_argument("--max-pages", type=int, default=None, help="ダンプから読む最大ページ数")
    p.set_defaults(func=cmd_learn)

    p = sub.add_parser("evolve", help="自律学習ループを前面で実行")
    p.add_argument("--cycles", type=int, help="サイクル数で停止")
    p.add_argument("--seconds", type=float, help="秒数で停止")
    p.set_defaults(func=cmd_evolve)

    p = sub.add_parser("train", help="ニューラル LM (Transformer) を前面で学習 (numpy が必要)")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--seconds", type=float, default=None)
    p.add_argument("--hours", type=float, default=None, help="収集しながら長時間学習する")
    p.add_argument("--size", choices=["small", "base", "large", "xl"], default=None)
    p.add_argument("--workers", type=int, default=None, help="データ並列のプロセス数 (既定: CPU 数 - 1)")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("stats", help="状態を表示")
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("export", help="ブラウザ版 (web/) 用にモデルを int8 で書き出す")
    p.add_argument("--out", default="web/dist")
    p.add_argument("--max-docs", type=int, default=12000)
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("serve", help="HTTP API + 簡易 Web UI")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-auto", action="store_true")
    p.set_defaults(func=cmd_serve)

    args = ap.parse_args(argv)
    if not args.cmd:
        args.cmd = "chat"
        args.no_auto = False
        args.func = cmd_chat
    return args.func(args)
