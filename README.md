# tinyai — 超小型・自己進化型の会話 AI

会話しながら、テキスト / 学習データ / Web を**自分で探索して学習し続ける**、
標準ライブラリだけで動く超小型 AI です。メモリ使用量は上限 (既定 **256MB**) を
超えないよう常時プルーニングされます。

```
python -m tinyai            # 対話 (裏で自動学習が回る)
```

## 特徴

| 機能 | 仕組み |
|---|---|
| 会話 | BM25 検索メモリで最も関連する知識を答え、確信が持てない時は n-gram 言語モデルで文を生成 |
| 自動学習 | 文章・ファイル・URL・`inbox/` フォルダに置いたファイルを文単位で取り込み |
| Web 探索 | 会話で分からなかった話題や知識の薄い語を、Wikipedia (Wikimedia Core API / Action API)、DuckDuckGo、`sources.txt` のサイトから自動で調べる。robots.txt と 429 バックオフを尊重 |
| 進化 | 自己評価 (取り置き文のパープレキシティ + 検索の自己テスト) を適応度とし、パラメータを変異させて改善した時だけ採用。世代番号が上がる |
| フィードバック | `👍` / `👎` で知識の信頼度と回答しきい値を調整 |
| メモリ制限 | 推定サイズ + 実 RSS の 2 段監視。超えると低頻度 n-gram と古い/役に立たない知識から削除。OS の `RLIMIT_DATA` も設定 |
| 依存なし | Python 3.10+ の標準ライブラリのみ。保存ファイルは gzip pickle 1 個 |

## 使い方

```bash
# 対話 (Web 探索スレッドが裏で走る)。終了時に自動保存
python -m tinyai chat
python -m tinyai chat --offline        # Web 探索なし
python -m tinyai chat --memory 128     # 上限 128MB

# 1 問だけ
python -m tinyai ask 日本の首都は？

# 学習: ファイル / ディレクトリ / URL / 話題
python -m tinyai learn notes.txt docs/ https://example.com/page topic:量子コンピュータ

# 自律学習ループだけを前面で回す (Ctrl+C で停止、10 サイクル or 1 時間で自動停止)
python -m tinyai evolve --cycles 10
python -m tinyai evolve --seconds 3600 --interval 30

# 状態 (世代・適応度・メモリなど)
python -m tinyai stats

# HTTP API + 簡易 Web UI (http://127.0.0.1:8765/)
python -m tinyai serve
curl -X POST localhost:8765/ask   -d '{"text":"こんにちは"}'
curl -X POST localhost:8765/learn -d '{"text":"覚えさせたい文章。","source":"api"}'
curl localhost:8765/stats
```

### 会話中のコマンド

| 入力 | 動作 |
|---|---|
| `覚えて: 文章` / `remember: text` | その文をそのまま記憶 (優先度高) |
| `調べて: 話題` / `search: topic` | 次のサイクルでその話題を Web 検索 |
| `👍` / `👎` | 直前の答えを評価 |
| `/stats` `/save` `/evolve` `/quit` | 状態表示 / 保存 / 進化を 1 回試す / 終了 |

### 完全自動で学ばせる

* **フォルダ投入**: `~/.tinyai/inbox/` に `.txt` `.md` `.html` `.csv` `.json` を置くと自動で取り込み、`learned/` に移動します。
* **サイト巡回**: `~/.tinyai/sources.txt` に URL を 1 行ずつ書くと、同一ドメイン内のリンクをたどって定期的に読みます。
* **話題の種**: `data/topics.txt` の話題から出発し、読んだ文の中の語へ好奇心で広がっていきます。

## 設定

| 環境変数 / 引数 | 既定 | 意味 |
|---|---|---|
| `TINYAI_DATA` / `--data` | `~/.tinyai` | 保存先 (`brain.pkl.gz`, `tinyai.log`, `inbox/`, `sources.txt`) |
| `TINYAI_MEMORY_MB` / `--memory` | `256` | プロセスのメモリ上限 MB |
| `--interval` | `20` | 自律学習サイクルの間隔 (秒) |
| `--offline` | – | Web を使わない |
| `--no-hard-limit` | – | `RLIMIT_DATA` を掛けない (macOS など rlimit が効かない環境向け) |

HTTPS プロキシ環境では `SSL_CERT_FILE` に CA バンドルを指定すると Fetcher がそれを読み込みます。

## 仕組み

```
ユーザー発話 ─┬─ コマンド? (覚えて/調べて/👍/👎)
             └─ 語に分解 (ラテン語 + CJK 文字 bigram + 漢字/カタカナ連続語)
                  ├─ 知識ベース検索 (BM25 + 語の情報量重み + カバー率) ─ 確信度 ≥ しきい値 → そのまま回答 (recall)
                  │                                                 ─ 中程度 → 「たぶん…」と回答し話題を調査キューへ (guess)
                  └─ 言語モデル生成 (補間付き絶対ディスカウント n-gram、次数は進化で可変) → 回答 + 調査キューへ (generate)

自律ループ (別スレッド): inbox 取込 → 話題選択 (調査キュー > 好奇心 > 知識の薄い語 > 種) → Web 検索・読解 → 学習
                     → 数サイクルに 1 回 パラメータ変異 & 自己評価 → メモリ整理 → 保存
```

**進化するパラメータ**: n-gram 使用次数、ディスカウント、BM25 の `k1` / `b`、句一致ボーナス。
回答しきい値は 👍/👎 でオンラインに動きます。適応度が上がった変異だけが採用されるので、
評価指標の上では単調に賢くなります。

**メモリ制御**: 言語モデルは「エントリ数 × 推定コスト」、知識は「文字数と語数」で
サイズを推定し、予算 (上限 × 0.85 − インタプリタ基礎分) の 55% / 35% に収めます。
実 RSS が上限の 85% を超えたら予算そのものを 20% 締めて削り直します。

## テスト

```bash
python -m unittest discover -s tests -v
```

## 制限

* 形態素解析器も埋め込みも使わない極小構成なので、「賢さ」は蓄えた文の検索と
  n-gram 生成の範囲です。数百 MB の知識ベースでも、深い推論はできません。
* 極端に小さい上限 (64MB など) で大量の文章を一気に学習させると、Python が解放済みメモリを OS に返さないため、ピーク RSS が上限を 1〜2 割超えることがあります。既定の 256MB では十分な余裕があります。
* Web の検索エンジンや API はレート制限や bot 対策で失敗することがあります。
  失敗は記録して次のサイクルで別の話題に進みます (ループは止まりません)。
