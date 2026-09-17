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
  dialog.py       会話データの保持 (発話, 応答, 出典, 重み) と「」の応酬の抽出、個人情報らしい文字列の除外
  bpe.py          サブワードトークナイザ (頻度ベース WordPiece 風の最長一致、基本文字は常に語彙に含む、▁ で英単語の空白を保持)
  neural.py       numpy だけで書いた LLaMA 系 Transformer (RMSNorm・RoPE・SwiGLU・KV キャッシュ・top-p・コサイン LR・系列パッキング)。勾配は有限差分で検査済み
  neural_lm.py    Brain との接続: 語彙学習 (十分なデータが溜まってから固定)、平文/会話/RAG/合成 QA/抽出練習の系列化、再生バッファ、継続学習、ppl による使用判定、RAG 生成 (候補を一括生成)
  neural_parallel.py データ並列学習: fork したワーカーが共有メモリ上のパラメータで勾配を計算し、親が平均して AdamW (同期 SGD)
                  neural.py には unlikelihood 損失 (重み < 0 の系列)、関数保存の層追加 grow_layer、語彙拡張 add_tokens もある
  brain_types.py  質問タイプの判定とタイプ別リランク
  evolution.py    進化するパラメータ (Params) と自己評価・変異・採用の判定
  brain.py        中核: 学習パイプライン、会話パイプライン、関心・通知、整理、メモリ制御、保存
  agent.py        エージェント層: 意図 → 道具 (計算・日付・単位・比較・列挙・要約・調査・プロファイル)、書式指示、出典
  web.py          取得器 (robots.txt、ホスト間隔、429 バックオフ)、HTML/フィード解析、Wikipedia API
  collector.py    収集システム (ソースのプラグイン登録、健全性 = 成功率 × 収穫 × 新規性、フロンティア、フィード、供給源、先読みワーカー)
  dumps.py        大量データのストリーミング取り込み (Wikipedia XML ダンプ、書庫、ディレクトリ)
  wikitext.py     ウィキテキスト → 平文
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

段階ごとの所要時間は `Brain.timers` に累積され `/stats` の `timers_ms` で見える (analyze / facts+quality / kb / lm / semantic / neural)。
ニューラル LM の学習データは 4 種類を再生バッファ (3 万系列、系列パッキングで pad 無し) に積む:
平文 (取り込んだ文の 1/3、品質 0.7 以上は全部)、会話 (`<usr>` 発話 `<bot>` 応答、👍 は 3 倍)、RAG 会話 (`<ctx>` 応答時に検索した文 + 会話)、
合成 QA (事実ストアの (主語, 関係, 目的語) からテンプレートで質問を作り、文脈付き/無しの両方)。
`Brain.neural_step` (自律ループ各サイクル 2 秒、`train` コマンドでは連続) が AdamW で更新し、200 ステップごとに取り置き ppl を測って使用可否を更新する。
909 文の内訳の例: analyze 22ms, facts+quality 9ms, kb 48ms, lm 63ms。

不変条件:
* 知識ベースに入らなかった文は LM にも事実にも入らない (学習の重複コストがゼロ)。近似重複で弾いた文の出典は、元の文の裏付け (`facts.extra_sources`) として数える。
* 文書を消すと `kb.on_remove` で事実も消える。転置索引に消えた文書は残らない (`_post_remove`)。
* LM の `cont_total == sum(cont.values())` は剪定後も保たれる。

### 会話 (ニューラル専用モード、既定)

```
発話 ─ コマンド (覚えて/👍/👎/もっと詳しく)
     ─ 検索 (BM25 + 意味ベクトル) と事実ストアから文脈を作る
     ─ Transformer が <ctx> 文脈 <usr> 発話 <bot> の続きを候補 4 本生成 → 接地率 + 自然さで 1 本選ぶ (mode neural)
     ─ 直後に (発話, 応答, 文脈) で勾配更新 (online)。👍 → 重み 3 で更新、👎 → unlikelihood で更新
常時学習スレッド (NeuralTrainer): 再生バッファ (平文・会話・RAG・合成 QA・抽出練習) で学習、200 ステップごとに評価、
                                損失停滞で層を追加、新語で語彙を拡張、5 分ごとに保存
```

### 会話 (従来経路: --symbolic、または未学習の冷間起動時)

```
発話 ─ コマンド (覚えて/👍/👎/もっと詳しく)
     ─ 書式指示 (箇条書きで / 一言で / N 文字以内) を分離
     ─ エージェント層: 道具で正確に答えられるか (計算・日付・単位・比較・列挙・要約・調べて:・プロファイル) → tool:* で即答
     ─ 話題語抽出 → 関心プロファイル更新 → 話題語が無ければ前の話題を補う
     ─ 「X について教えて」 → 要約 (事実 + 出典の異なる文、summary)
     ─ facts.answer  (XのYは? / Xとは? / Xはいつ? / AのBのCは?) → 多出典投票で最有力を即答、食い違いは併記 (fact)
     ─ 未知ガード: 主語を知らない / 属性を含む文が無い → 確信度を抑え「まだ知りません」と正直に答える
     ─ kb.search (+ 低確信なら PMI 関連語で拡張) → 質問タイプ別リランク + 意味類似 + 学習型リランカー → recall / guess
     ─ generate: 接尾辞配列の最長一致分布 × suffix_weight + n-gram/キャッシュ LM の補間、候補 4 本 (+ ニューラル LM が使える段階なら 2 本) から
                 LM スコア + 知識の語 + 質問の句 + ニューラル LM の対数尤度 で最良 (generate)
     ─ neural (主経路): 検索の確信が neural_override_conf (0.8) 未満なら、検索文を <ctx> にして Transformer が応答を生成。
                 候補 3 本を 接地率 (句の 6 割以上が文脈/質問にある)・自然さ (平均対数尤度 ≥ −2.5)・文末・崩れ の検査にかけ、通れば採用、通らなければ検索応答
     ─ 裏で調べ終えた話題 (notices) を一言添える
     ─ Web 由来なら出典 (ホスト名) を添え、書式指示を適用
     ─ 確信が低い話題は調査キューへ → on_gap → 収集を即起動
```

### 収集と自律学習

```
Collector (先読みワーカー × N, 既定 2)
   話題ソース (言語ごと): wikimedia / wikipedia / wiktionary / wikidata / wikinews / wikibooks / duckduckgo を健全性 × 重み の順に
   供給源 (話題に依らない): random (Wikipedia ランダム記事) / aozora (青空文庫) / gutenberg / hfdatasets (公開対話・指示データ) / stackexchange を健全性 × 重み で抽選
   供給源の Batch には dialogs [(発話, 応答)] が付き、Brain.learn_batch が会話ストアとニューラル LM の再生バッファに入れる
   ワーカー 0: 話題検索 / 5 回に 1 回フィード / 4 回に 1 回供給源
   ワーカー 1: フロンティア (アンカー文字列の関心 + 新規性 − 深さ) 優先
   Collector.register(Source) で独自ソースを追加できる
   → ready キュー (最大 prefetch_depth)

Evolver (1 スレッド)
   inbox 取込 → 調査キューがあれば同期で collect (リアルタイム) / 無ければ ready から消費
   → brain.learn_batch (学習 + リンクをフロンティアへ + ソースへ収穫報告)
   → evolve_every サイクルごとに evolution.step、その 5 倍ごとに consolidate
   → enforce_memory → save_every ごとに保存
   → 調査キューが空なら適応的な間隔だけ待つ (直近の収穫が多ければ 1/4、ゼロが続けば 4 倍; on_gap で即起床)
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

## ブラウザ版 (web/)

```
tinyai export ──> model.bin (int8, 行スケール) + meta.json + vocab.json + kb.json + test.json
                       │
web/index.html ── fetch ──> worker.js ── importScripts ──> engine.js
   (UI: 会話 / 思考 / 進化)        (別スレッド)            Tokenizer / Model / Retriever / Engine
```

* `engine.js` は `tinyai/neural.py` の順伝播・逆伝播・AdamW・KV キャッシュ生成をそのまま移植したもの (Float32Array、依存なし)。
  `web/validate.js` が `test.json` の numpy 計算値 (logits 上位・損失・勾配ノルム) と突き合わせる
* 応答: 文字 2-gram の BM25 で知識文を引き、`<ctx>` に入れて候補を複数生成、平均対数確率 + 長さ + 文脈との重なりで選ぶ。
  生成中に各トークンの上位 5 候補と確率、最終層の注意 (プロンプトのどこを見たか) を記録し UI に渡す
* リアルタイム学習: ターンごとに `<ctx>…<usr>…<bot>…<eos>` 系列で 1 ステップ更新 (応答部の重み 1.0、プロンプト部 0.2)、
  再生バッファから 1 本混ぜて忘却を防ぐ。👍 は重み 3、👎 は unlikelihood。復号パラメータは 👍 率で山登り、語彙は会話に頻出する新しい単位で拡張
* 常時学習: 会話していない間も 3 秒ごとに再生バッファの会話か知識文を 1 系列学ぶ (Worker で実行するので画面は止まらない)
* 学習後の重み (float32 約 12MB) と統計は IndexedDB に保存し、次回起動時に復元する。サーバーには何も送らない

## 評価システム (tools/eval.py)

| 対象 | 指標 |
|---|---|
| 会話 | 評価セット (カテゴリ別正答率、言い換え頑健性、モード・確信度の条件) |
| 検索 / n-gram / 生成 | MRR, Recall@3, 取り置き ppl, distinct-2, 接地率, 繰り返し率 |
| ニューラル LM (`--neural`) | 同じコーパス・同じ乱数・固定ステップで小型モデルを学習して到達損失と tok/s (= 学習効率の回帰テスト)、取り置き ppl、RAG 忠実性 (文脈の句をどれだけ使うか、キーワード再現率)、リアルタイム学習 (教えた答えの対数確率の伸び、👎 による低下)、思考 (再検索で文脈が増えた割合)、生成遅延 |
| 学習済み Brain (`--data`) | ニューラル LM の実力 + 収集ソースの評価表 (成功率・収穫・新規性・驚き・価値・点数) + 会話データの出所 |
| 退行検出 (`--compare`) | 前回より 5% 以上悪化した指標に ⚠ を付ける |

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
