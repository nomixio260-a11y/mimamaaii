# アーキテクチャ

## モジュール構成

```
tinyai/
  config.py       設定 (メモリ上限、次数、先読みワーカー数、間隔)。環境変数 TINYAI_* で上書き
  tokenizer.py    正規化・文分割・LM トークン・検索語・話題語・質問判定 (形態素解析器なし)
  memory.py       RSS 監視と RLIMIT_DATA。予算 = 上限×0.85 − インタプリタ基礎分
  lm.py           n-gram 言語モデル (整数パック文脈、KN 継続カウント、剪定) + 会話キャッシュ LM
  knowledge.py    文単位の知識ベース (BM25 転置索引、近似重複排除、品質、追い出し、関連語)
  facts.py        主語-関係-目的語の抽出・記憶・質問解析・多段推論
  semantic.py     意味ベクトル (Random Indexing、int16 × 128 次元、SimHash スケッチで近傍探索)
  suffix.py       接尾辞配列 (最長一致の続きの分布; 生成用)
  reranker.py     学習型リランカー (オンライン ロジスティック回帰)
  brain_types.py  質問タイプの判定とタイプ別リランク
  evolution.py    進化するパラメータ (Params) と自己評価・変異・採用の判定
  brain.py        中核: 学習パイプライン、会話パイプライン、関心・通知、整理、メモリ制御、保存
  web.py          取得器 (robots.txt、ホスト間隔、429 バックオフ)、HTML/フィード解析、Wikipedia API
  collector.py    収集システム (ソース健全性、フロンティア、フィード、ランダム記事、先読みワーカー)
  evolve.py       自律学習スレッド (イベント駆動、調査キュー優先、進化・整理・保存の周期)
  cli.py          chat / ask / learn / evolve / stats / serve
tools/
  bench.py        学習速度・メモリ・応答速度・自問自答
  eval.py         会話品質の評価セット (data/eval_ja.tsv)
  experiment.py   研究実験 (次数、Count-Min Sketch LM、接尾辞配列)
docs/
  ARCHITECTURE.md この文書
  RESEARCH.md     技術サーベイと実験結果
```

## データフロー

### 学習 (1 文あたり約 180µs)

```
テキスト ─ split_sentences ─┬─ (長い Web 文書の 1/40 は取り置き = 自己評価用)
                           └─ 各文:
                                is_junk / 完全重複 / 近似重複 (句の集合の鍵) → 捨てる
                                extract_facts (文末フィルタ → 先頭アンカーの正規表現、~3µs)
                                sentence_quality (定義文・数値・話題語・長さ)
                                admission (メモリ充填率 60% 超で品質しきい値が上がる)
                                kb.add (BM25 索引; 投稿 1 件は整数パック)
                                facts.add
                                lm.learn (次数ごとにストリーミング; 文脈は整数 1 個)
```

後回しの学習 (`Brain.background_step`): 知識ベースに入った文書 ID をキューに積み、自律ループの各サイクル (400 文)、
整理時 (2,000 文)、応答の合間 (8 文、スケッチ更新なし) に意味ベクトルへ取り込む。接尾辞配列は文書が 10% 増えるごとに再構築
(5 万トークンで 0.1 秒)。会話の即時応答経路はこれらを待たない。

不変条件:
* 知識ベースに入らなかった文は LM にも事実にも入らない (学習の重複コストがゼロ)。
* 文書を消すと `kb.on_remove` で事実も消える。転置索引に消えた文書は残らない (`_post_remove`)。
* LM の `cont_total == sum(cont.values())` は剪定後も保たれる。

### 会話

```
発話 ─ コマンド (覚えて/調べて/👍/👎/もっと詳しく)
     ─ 話題語抽出 → 関心プロファイル更新 → 話題語が無ければ前の話題を補う
     ─ facts.answer  (XのYは? / Xとは? / Xはいつ? / AのBのCは?) → fact
     ─ kb.search (+ 低確信なら PMI 関連語で拡張) → 質問タイプ別リランク + 意味類似 + 学習型リランカー → recall / guess
     ─ generate: 接尾辞配列の最長一致分布 × suffix_weight + n-gram/キャッシュ LM の補間、候補 4 本から最良 (generate)
     ─ 裏で調べ終えた話題 (notices) を一言添える
     ─ 確信が低い話題は調査キューへ → on_gap → 収集を即起動
```

### 収集と自律学習

```
Collector (先読みワーカー × N, 既定 2)
   ワーカー 0: 話題検索 (健全なソース順: wikimedia > wikipedia > wikidata > duckduckgo) / 5 回に 1 回フィード / 7 回に 1 回ランダム記事
   ワーカー 1: フロンティア (アンカー文字列の関心 + 新規性 − 深さ) 優先
   → ready キュー (最大 prefetch_depth)

Evolver (1 スレッド)
   inbox 取込 → 調査キューがあれば同期で collect (リアルタイム) / 無ければ ready から消費
   → brain.learn_batch (学習 + リンクをフロンティアへ + ソースへ収穫報告)
   → evolve_every サイクルごとに evolution.step、その 5 倍ごとに consolidate
   → enforce_memory → save_every ごとに保存
   → 調査キューが空なら interval 待つ (on_gap で即起床)
```

## メモリ制御

| 層 | 仕組み |
|---|---|
| 推定 | LM: エントリ数×60 + 文脈数×50 + 語彙×70。KB: 文字数×2 + 語数×60 + 200/文 |
| 予算 | (上限×0.85 − 基礎 RSS) × 0.8^tighten。LM 45% / KB 30% / 意味ベクトル 10% / 接尾辞配列と余裕 15% |
| 剪定 | LM: カウントしきい値を段階的に上げる → それでも超えれば最高次数を落とす。KB: 信頼度 + 品質 + 使用回数 − 経過日数 の低い順に 5% ずつ |
| 実測 | RSS がソフト上限 (85%) を超えたら予算を 20% 締める (最大 6 回)。Python は解放済みメモリを OS に返さないので、推定側が主、RSS は保険 |
| OS | RLIMIT_DATA = 上限 + max(96MB, 上限/2) |
| 選択的学習 | 充填率 60% 超で低品質文を取り込まない (`admission`) |

## 拡張点

* **ソースの追加**: `Collector._ordered_sources` に `(名前, fn(topic, n) -> [(url, text, anchors)])` を足すだけ。健全性は自動で記録される。
* **事実パターンの追加**: `facts._JA_PATTERNS` / `_EN_PATTERNS` に (種類, 正規表現) を足す。文末フィルタ `_JA_TAIL_RE` も合わせる。
* **適応度の変更**: `evolution.Evolution.evaluate` を差し替える。パラメータは `Params` に項目と `mutate` の分岐を足す。
* **質問タイプ**: `brain_types._QTYPE_PATTERNS` と `_rerank_bonus`。
* **リランカーの特徴量**: `reranker.FEATURES` と `Reranker.features()`。学習は `Brain._train_reranker` (👍/👎 時)。
* **生成の混合**: `Brain._sample_sentence` (接尾辞配列・n-gram・キャッシュの重みは `Params`)、候補の採点は `_score_candidate`。
* **保存形式**: `Brain.save/load` の `version` を上げ、`load` で旧版を読み替える。

## 設計上の判断

* 純 Python + 標準ライブラリ: 配布とメモリの予測可能性を優先。numpy があれば密ベクトル検索や小型ニューラル LM も候補になるが、現状の制約では n-gram + 転置索引 + 事実ストアが最も費用対効果が高い (docs/RESEARCH.md)。
* 学習コストは「取り込まない判断」を早く安くすることで下げる (完全重複 → 近似重複 → ジャンク → 品質 → 事実)。
* 「賢さ」は 3 層: 事実 (正確・即答) > 文の検索 (広い) > 生成 (最後の手段、確信度は低く表示)。
* 「LLM らしさ」は、ニューラルネットではなく (純 Python では 30 倍遅く精度も出ない、docs/RESEARCH.md 実験 [4])、
  分散表現 (意味) + 接尾辞配列 (長い文脈) + キャッシュ (会話の流れ) + フィードバック学習 (好み) の組み合わせで実現する。
