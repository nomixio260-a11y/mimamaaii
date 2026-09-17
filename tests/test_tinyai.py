"""python -m unittest discover -s tests  (pytest でも動く)"""
import re
import random
import tempfile
import threading
import unittest
from pathlib import Path

from tinyai import Brain, Config
from tinyai.evolve import Evolver
from tinyai.knowledge import KnowledgeBase
from tinyai.lm import NGramLM
from tinyai.memory import MemoryGuard, rss_bytes
from tinyai.brain import question_type, sentence_quality
from tinyai.collector import Batch, Collector
from tinyai.facts import FactStore, extract_facts, parse_question
from tinyai.dumps import iter_archive_texts, iter_wiki_pages, learn_wiki_dump
from tinyai.knowledge import is_junk
from tinyai.wikitext import wikitext_to_text
from tinyai.lm import CacheLM
from tinyai.reranker import Reranker
from tinyai.semantic import SemanticSpace
from tinyai.suffix import SuffixIndex
from tinyai.tokenizer import phrases
from tinyai.tokenizer import detokenize, is_phrase, is_question, keywords, split_sentences, terms, tokenize
from tinyai.web import html_to_text


def make_brain(tmp: str, **kw) -> Brain:
    cfg = Config(data_dir=Path(tmp), memory_mb=kw.pop("memory_mb", 128), hard_limit=False, web_enabled=False, seed=7, tools=True, **kw)
    b = Brain(cfg)
    b.bootstrap()
    return b


class TokenizerTest(unittest.TestCase):
    def test_tokenize_mixed(self):
        self.assertEqual(tokenize("人工知能 is AI!"), ["人", "工", "知", "能", "is", "ai", "!"])

    def test_terms_cjk_bigrams_and_runs(self):
        t = terms("人工知能とは")
        self.assertIn("人工", t)
        self.assertIn("人工知能", t)

    def test_terms_never_empty_for_text(self):
        self.assertTrue(terms("who are you"))

    def test_split_sentences(self):
        self.assertEqual(split_sentences("今日は晴れ。明日は雨です！行きますか？"), ["今日は晴れ。", "明日は雨です!", "行きますか?"])
        # 閉じ括弧は前の文に付く。英語のピリオドは大文字が続く時だけ区切る
        self.assertEqual(split_sentences("彼は「行くよ。」と言った。Mr. Smith left. He came back."), ["彼は「行くよ。」と言った。", "Mr. Smith left.", "He came back."])

    def test_detokenize(self):
        self.assertEqual(detokenize(["hello", "world", "!"]), "hello world!")
        self.assertEqual(detokenize(["こ", "ん", "に", "ち", "は"]), "こんにちは")

    def test_question(self):
        self.assertTrue(is_question("日本の首都はどこ？"))
        self.assertTrue(is_question("What is AI"))
        self.assertFalse(is_question("今日は晴れです。"))

    def test_keywords(self):
        self.assertEqual(keywords("東京タワーの高さは？")[0], "東京タワー")

    def test_is_phrase(self):
        self.assertTrue(is_phrase("機械学習"))
        self.assertTrue(is_phrase("python"))
        self.assertFalse(is_phrase("習に"))
        self.assertFalse(is_phrase("the"))


class LMTest(unittest.TestCase):
    def test_learn_prob_generate(self):
        lm = NGramLM(max_order=3)
        for _ in range(5):
            lm.learn(tokenize("the cat sat on the mat"))
        self.assertGreater(lm.prob(["the"], "cat"), lm.prob(["the"], "dog"))
        gen = lm.generate(["the"], rng=random.Random(1))
        self.assertTrue(gen)
        self.assertTrue(all(lm.knows(t) for t in gen))

    def test_perplexity_lower_on_seen_text(self):
        lm = NGramLM(max_order=3)
        seen = tokenize("今日は良い天気ですね")
        for _ in range(3):
            lm.learn(seen)
        self.assertLess(lm.perplexity([seen]), lm.perplexity([tokenize("量子力学は難しい")]))

    def test_prune_reduces_entries(self):
        lm = NGramLM(max_order=3)
        for i in range(200):
            lm.learn([f"w{i}", f"w{i+1}", "x"])
        before = lm.entries
        removed = lm.prune(min_count=2)
        self.assertGreater(removed, 0)
        self.assertEqual(lm.entries, before - removed)
        self.assertLess(lm.shrink_to(1000) + lm.estimated_bytes(), before * 200)
        # 継続カウントの整合性
        self.assertEqual(lm.cont_total, sum(lm.cont.values()))

    def test_state_roundtrip_and_packed_contexts(self):
        lm = NGramLM(max_order=4)
        for _ in range(3):
            lm.learn(tokenize("今日は良い天気ですね"))
        lm.learn(tokenize("今日は雨です"))
        lm2 = NGramLM.from_state(lm.state())
        self.assertEqual(lm2.entries, lm.entries)
        self.assertAlmostEqual(lm2.prob(["今", "日"], "は"), lm.prob(["今", "日"], "は"))
        # 後続が 1 種類の文脈は int に圧縮されている
        self.assertTrue(any(type(v) is int for k, v in lm.ctx.items() if k))
        self.assertTrue(any(type(v) is dict for k, v in lm.ctx.items() if k))
        self.assertGreater(lm.prob(["今", "日"], "は"), lm.prob(["今", "日"], "雨"))

    def test_focus_biases_generation(self):
        lm = NGramLM(max_order=3)
        for _ in range(20):
            lm.learn(["a", "b", "c"])
            lm.learn(["a", "x", "y"])
        rng = random.Random(0)
        with_focus = sum(1 for _ in range(50) if lm.generate(["a"], rng=rng, focus={"x"}, focus_bonus=50.0)[1:2] == ["x"])
        self.assertGreater(with_focus, 40)


class KnowledgeTest(unittest.TestCase):
    def test_junk_filter(self):
        self.assertTrue(is_junk("| a | b | c | d |"))
        self.assertTrue(is_junk("1990 2000 2010 2020 2030"))
        self.assertFalse(is_junk("人工知能とは何かを説明します。"))

    def test_associate_and_related(self):
        kb = KnowledgeBase()
        for i in range(30):
            kb.add(f"機械学習の手法 {i} はデータから規則を学ぶ。ニューラルネットワークも機械学習の一種である。", "t")
            kb.add(f"宇宙 {i} は広い。銀河や恒星が宇宙にはたくさんある。", "t")
        d = kb.search("ニューラルネットワーク")[0][1]
        self.assertGreater(kb.associate(d.id, "ディープラーニングって何？"), 0)
        self.assertEqual(kb.search("ディープラーニングって何？")[0][1].id, d.id)
        rel = [t for t, _ in kb.related_terms("機械学習", k=5)]
        self.assertIn("ニューラルネットワーク", rel)
        self.assertNotIn("銀河", rel)
        kb.remove(d.id)
        self.assertTrue(all(d.id not in kb.posting_ids(t) for t in list(kb.index)))
        self.assertNotIn(d.id, kb.assoc)
        # 単一投稿は int、複数は dict
        kb.add("これはオメガシグマという語を含む文です。", "t")
        self.assertIs(type(kb.index["オメガシグマ"]), int)
        self.assertEqual(kb.posting_ids("オメガシグマ"), [kb.next_id - 1])
        self.assertTrue(any(type(p) is dict for p in kb.index.values()))

    def test_near_duplicate_rejected(self):
        kb = KnowledgeBase()
        self.assertIsNotNone(kb.add("東京タワーの高さは 333 メートルである。", "t"))
        self.assertIsNone(kb.add("東京タワーの高さは 333 メートルである！", "t"))      # 記号だけ違う
        self.assertIsNone(kb.add("333 メートルである、東京タワーの高さは。", "t"))     # 語順だけ違う
        self.assertIsNotNone(kb.add("東京タワーの高さは 333 メートル (公式) である。", "t"))  # 句が増えた
        self.assertEqual(len(kb), 2)
        kb.remove(1)
        self.assertIsNotNone(kb.add("東京タワーの高さは 333 メートルである。", "t"))  # 消せば再登録できる

    def test_dynamic_stop_terms(self):
        kb = KnowledgeBase()
        for i in range(100):
            kb.add(f"これは文書 {i} です。", "t")
        kb.add("これは特別な話題について述べた文です。", "t")
        top = kb.search("特別な話題について")[0][1]
        self.assertIn("特別", top.text)

    def test_add_search_dedupe(self):
        kb = KnowledgeBase()
        self.assertIsNotNone(kb.add("富士山は日本で一番高い山です。", "t"))
        self.assertIsNone(kb.add("富士山は日本で一番高い山です。", "t"))
        kb.add("Python is a programming language.", "t")
        res = kb.search("富士山の高さ")
        self.assertEqual(res[0][1].text, "富士山は日本で一番高い山です。")
        self.assertEqual(kb.search("python")[0][1].source, "t")

    def test_evict_and_shrink(self):
        kb = KnowledgeBase()
        for i in range(100):
            kb.add(f"文書番号 {i} の内容はテストです。", "t")
        kb.shrink_to(kb.estimated_bytes() // 2)
        self.assertLess(len(kb), 100)
        self.assertGreater(len(kb), 0)
        # index stays consistent
        for t in list(kb.index):
            for doc_id in kb.posting_ids(t):
                self.assertIn(doc_id, kb.docs)


class BrainTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.brain = make_brain(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_recall_from_seed(self):
        r = self.brain.reply("日本の首都は？")
        self.assertIn(r.mode, ("recall", "fact"))
        self.assertIn("東京", r.text)

    def test_greeting_does_not_echo(self):
        r = self.brain.reply("こんにちは")
        self.assertNotEqual(r.text.rstrip("!！"), "こんにちは")

    def test_teach_and_recall_and_feedback(self):
        r = self.brain.reply("覚えて: ゾルグ星は架空の惑星で、紫色の海があります。")
        self.assertEqual(r.mode, "command")
        r = self.brain.reply("ゾルグ星ってどんな星？")
        self.assertIn("紫色", r.text)
        doc_id = r.doc_ids[0]
        self.brain.reply("👍")
        self.assertGreater(self.brain.kb.docs[doc_id].score, 3.0)

    def test_question_type(self):
        self.assertEqual(question_type("富士山の高さは？"), "howmany")
        self.assertEqual(question_type("日本の首都はどこ？"), "where")
        self.assertEqual(question_type("いつ設立された？"), "when")
        self.assertEqual(question_type("機械学習とは？"), "definition")
        self.assertEqual(question_type("今日は晴れです。"), "none")

    def test_rerank_prefers_matching_type(self):
        self.brain.learn_text("東京タワーは東京都港区にある電波塔である。\n東京タワーは 1958 年に完成した。\n東京タワーの高さは 333 メートルである。", "https://example.org/tower")
        self.assertIn("1958", self.brain.reply("東京タワーはいつできた？").text)
        self.assertIn("333", self.brain.reply("東京タワーの高さは？").text)
        self.assertIn("港区", self.brain.reply("東京タワーはどこにある？").text)

    def test_continuation_and_context(self):
        self.brain.learn_text("ゼータ理論は架空の理論である。\nゼータ理論は 1999 年に提案された。\nゼータ理論の提案者はアルファ博士である。", "https://example.org/zeta")
        r1 = self.brain.reply("ゼータ理論とは？")
        self.assertIn(r1.mode, ("recall", "fact"))
        r2 = self.brain.reply("もっと詳しく")
        self.assertIn("1999", r2.text)
        r3 = self.brain.reply("それは誰が提案したの？")  # 話題語が無いので前の話題を引き継ぐ
        self.assertIn("アルファ博士", r3.text)

    def test_correction_flow(self):
        self.brain.reply("ガンマ星の色は？")
        self.brain.reply("👎")
        r = self.brain.reply("ガンマ星は緑色に輝く架空の恒星です。")
        self.assertEqual(r.mode, "command")
        self.assertIn("緑色", self.brain.reply("ガンマ星の色は？").text)
        self.assertGreaterEqual(len(self.brain.qa_log), 1)
        self.assertGreater(self.brain.evaluate(seed=1)["qa"], 0)

    def test_strategy_bandit(self):
        class F:
            def search_and_read(self, q, langs=(), max_pages=3):
                return [("https://f/" + q, f"{q} に関する文です。{q} は話題です。")] if q != "人工知能" else []

        f = F()
        for _ in range(6):
            t = self.brain.next_topic()
            self.assertIsNotNone(t)
            self.brain.learn_from_web(t, f)
        tried = sum(v[0] for v in self.brain.strategy_stats.values())
        self.assertGreaterEqual(tried, 6)

    def test_unknown_topic_becomes_gap(self):
        r = self.brain.reply("超弦理論について教えて")
        self.assertLess(r.confidence, 0.7)
        self.assertIn("超弦理論", self.brain.gaps)
        self.assertEqual(self.brain.next_topic(), "超弦理論")

    def test_short_web_text_is_not_held_out(self):
        # 短い文書は取り置きせず全文学ぶ (39 文までは holdout 無し)
        text = "\n".join(f"短い項目 {i} は記録されています。" for i in range(10, 49))
        self.brain._holdout_counter = 39  # 次の文が 40 番目
        n = self.brain.learn_text(text, source="https://example.org/short")
        self.assertEqual(n, 39)
        self.assertEqual(len(self.brain.holdout), 0)

    def test_learn_text_and_holdout(self):
        text = "\n".join(f"項目 {i} は第 {i} 番目の事実として記録されています。" for i in range(10, 110))
        n = self.brain.learn_text(text, source="https://example.org/x")
        self.assertGreater(n, 90)
        self.assertGreaterEqual(len(self.brain.holdout), 1)
        r = self.brain.reply("項目 42 は何番目？")
        self.assertIn("42", r.text)

    def test_evolve_step_keeps_or_improves_fitness(self):
        self.brain.learn_text("\n".join(f"サンプル文 {i} はテスト用の文章です。" for i in range(60)), "https://example.org/y")
        base = self.brain.evaluate(seed=3)["fitness"]
        for _ in range(5):
            rec = self.brain.evolve_step()
            self.assertIn("accepted", rec)
        self.assertGreaterEqual(self.brain.evaluate(seed=3)["fitness"], base - 1e-9)

    def test_save_load_roundtrip(self):
        self.brain.reply("覚えて: テスト用の固有名詞アルファゼータは合言葉です。")
        path = self.brain.save()
        self.brain.reply("アルファゼータとは？")
        self.brain.reply("👍")
        self.brain.evolve_step()
        path = self.brain.save()
        b2 = Brain(self.brain.cfg)
        self.assertTrue(b2.load(path))
        self.assertEqual(len(b2.kb), len(self.brain.kb))
        self.assertEqual(b2.lm.entries, self.brain.lm.entries)
        self.assertEqual(b2.params, self.brain.params)
        self.assertEqual(len(b2.kb.assoc), len(self.brain.kb.assoc))
        self.assertEqual(len(b2.qa_log), len(self.brain.qa_log))
        self.assertIn("合言葉", b2.reply("アルファゼータとは？").text)

    def test_memory_budget_enforced(self):
        b = make_brain(self.tmp.name, memory_mb=40)
        budget = b.guard.budget
        rng = random.Random(0)
        words = [chr(0x4E00 + i // 60) + chr(0x4E00 + i % 60) + "語" for i in range(3000)]  # 数字を含むとジャンク扱いになる
        for i in range(400):
            para = "\n".join(" ".join(rng.choice(words) for _ in range(12)) + "。" for _ in range(25))
            b.learn_text(para, source=f"https://example.org/{i}")
        b.enforce_memory()
        self.assertLessEqual(b.lm.estimated_bytes(), budget * 0.55 + 1)
        self.assertLessEqual(b.kb.estimated_bytes(), budget * 0.35 + 1)
        self.assertGreater(b.stats["pruned_lm"] + b.stats["pruned_kb"], 0)
        self.assertLess(rss_bytes(), b.guard.limit * 3)  # 極端な超過はしない


class FakeCollector(Collector):
    """ネットワークを使わない収集システム。"""

    def __init__(self, data_dir):
        super().__init__(fetcher=object(), data_dir=data_dir)  # fetcher は「有効」であればよい
        self.collected: list[str] = []

    def collect(self, topic, max_pages=3):
        self.collected.append(topic)
        text = f"{topic}とは、テスト用の話題である。{topic}の説明文です。{topic}は 2001 年に作られた。"
        return Batch(topic, "topic", [(f"https://fake/{topic}", text, [("https://fake/link", f"{topic}の関連")])], "fake", 0.0)

    def collect_link(self):
        return None

    def collect_feeds(self, min_interval=1800.0, max_items=5):
        return None

    def collect_site(self, index):
        return None

    def start_prefetch(self, topic_fn):
        pass

    def next_ready(self, timeout=0.0):
        return None


class EvolverTest(unittest.TestCase):
    def test_cycle_offline_with_fake_collector(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = make_brain(tmp)
            col = FakeCollector(Path(tmp))
            inbox = Path(tmp) / "inbox"
            inbox.mkdir()
            (inbox / "note.txt").write_text("インボックスの文章は自動で取り込まれます。", encoding="utf-8")
            ev = Evolver(b, interval=0, max_cycles=3, collector=col)
            b.add_gap("架空話題")  # on_gap 経由で起こされる
            ev.run()
            self.assertEqual(ev.cycles, 3)
            self.assertIn("架空話題", b.explored)
            self.assertFalse((inbox / "note.txt").exists())
            self.assertTrue((Path(tmp) / "learned" / "note.txt").exists())
            self.assertIn("架空話題", b.reply("架空話題とは？").text)
            self.assertTrue((Path(tmp) / "brain.pkl.gz").exists())
            self.assertEqual(col.collected[0], "架空話題")
            self.assertGreater(len(col.frontier), 0)  # リンクがフロンティアへ入った
            self.assertEqual(b.facts.lookup("架空話題", "definition")[0][1], "テスト用の話題")

    def test_realtime_gap_notice(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = make_brain(tmp)
            col = FakeCollector(Path(tmp))
            ev = Evolver(b, interval=0.05, collector=col)
            ev.start()
            r = b.reply("ホゲ理論について教えて")  # 分からない -> gap -> 即収集
            self.assertLess(r.confidence, 0.7)
            for _ in range(100):
                if b.notices or "ホゲ理論" in b.explored:
                    break
                threading.Event().wait(0.05)
            ev.stop()
            self.assertIn("ホゲ理論", b.explored)
            r2 = b.reply("ホゲ理論とは？")
            self.assertIn("ホゲ理論", r2.text)
            self.assertIn(r2.mode, ("fact", "recall"))

    def test_thread_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = make_brain(tmp)
            ev = Evolver(b, collector=Collector(None, Path(tmp)), interval=0.05)
            ev.start()
            threading.Event().wait(0.3)
            ev.stop()
            self.assertFalse(ev.is_alive())
            self.assertGreater(ev.cycles, 0)


class FactsTest(unittest.TestCase):
    def test_extract_and_answer(self):
        fs = FactStore()
        for i, s in enumerate(["東京タワーの高さは 333 メートルである。", "東京タワーは東京都港区に位置する。", "機械学習とは、データから規則性を学ぶ手法のことである。", "日本の首都は東京です。", "Tokyo Tower was built in 1958."]):
            fs.add_from_sentence(s, i)
        self.assertEqual(fs.stats()["facts"], 5)
        self.assertIn("333", fs.answer("東京タワーの高さは？")[0])
        self.assertIn("港区", fs.answer("東京タワーはどこ？")[0])
        self.assertIn("規則性", fs.answer("機械学習って何？")[0])
        self.assertIn("東京", fs.answer("日本の首都はどこ？")[0])
        self.assertIn("1958", fs.answer("When was Tokyo Tower built?")[0] or "") if fs.answer("When was Tokyo Tower built?") else None
        self.assertIsNone(fs.answer("日本の面積は？"))
        self.assertEqual(parse_question("富士山の標高は？"), ("富士山", "標高"))
        self.assertEqual(extract_facts("彼の名前は太郎です。"), [])
        fs.remove_doc(0)
        self.assertIsNone(fs.answer("東京タワーの高さは？"))
        # 定義が無ければ知っている事実を並べる
        self.assertIn("港区", fs.answer("東京タワーとは？")[0])

    def test_quality(self):
        self.assertGreater(sentence_quality("東京タワーとは、東京都港区にある高さ 333 メートルの電波塔である。", True), sentence_quality("うん。"))

    def test_multihop_and_synonyms(self):
        fs = FactStore()
        for i, t in enumerate(["アルファ社の本社は大阪市にある。", "大阪市の人口は約 270 万人である。", "Python の作者は Guido van Rossum である。", "大阪市の市長は横山英幸である。"]):
            fs.add_from_sentence(t, i)
        self.assertEqual(fs.lookup("アルファ社", "本社")[0][1], "大阪市")  # 「にある」は目的語から外れる
        self.assertIn("270", fs.answer("アルファ社の本社の人口は？")[0])
        self.assertIn("横山", fs.answer("アルファ社の本社の市長は誰？")[0])
        ans = fs.answer("Who is the creator of Python?")[0]
        self.assertIn("Guido", ans)
        self.assertIn("creator", ans)  # 質問側の言語で関係名を表現する
        self.assertEqual(parse_question("アルファ社の本社の人口は？"), ("アルファ社の本社", "人口"))


class CollectorTest(unittest.TestCase):
    def test_frontier_priority_and_health(self):
        with tempfile.TemporaryDirectory() as tmp:
            interest = {"人工知能": 1.0}
            col = Collector(None, Path(tmp), interest=lambda t: interest.get(t, 0.0))
            n = col.push_links([("./人工知能", "人工知能"), ("./雑談", "雑談"), ("#x", ""), ("./深い", "深い")], depth=1, base_url="https://ex/")
            self.assertEqual(n, 3)
            target, anchor, depth = col.pop_link()
            self.assertEqual(anchor, "人工知能")
            self.assertEqual(target, "https://ex/人工知能")
            # 同じリンクは二度取らない
            col.push_links([("./人工知能", "人工知能")], 1, "https://ex/")
            self.assertNotEqual(col.pop_link()[1], "人工知能")
            h = col._h("wikimedia:ja")
            h.record(True, 1.0, 0.5)
            h.record(False, 0.0, 2.0)
            self.assertGreater(h.score, col._h("duckduckgo").score * 0.5)
            self.assertIsNone(col.collect("x").pages or None)


class BrainMemoryTest(unittest.TestCase):
    def test_consolidate_merges_duplicates_and_admission(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = make_brain(tmp)
            b.learn_text("東京タワーの高さは 333 メートルである。", "https://a/1")
            # 語順・記号だけ違う文は取り込み時点で近似重複として弾かれる
            self.assertEqual(b.learn_text("東京タワーの高さは 333 メートルである！", "https://a/2"), 0)
            # 少し言い回しが違う同じ事実は取り込まれ、整理で統合される
            self.assertEqual(b.learn_text("東京タワーの高さは 333 メートル (公式) である。", "https://a/3"), 1)
            self.assertEqual(len(b.facts.lookup("東京タワー", "高さ")), 2)
            rec = b.consolidate()
            self.assertEqual(rec["merged"], 1)
            self.assertEqual(len(b.facts.lookup("東京タワー", "高さ")), 1)
            b.admission = 0.9
            n = b.learn_text("うん、そうだね。", "https://a/3")
            self.assertEqual(n, 0)
            self.assertGreater(b.stats["skipped_low_quality"], 0)

    def test_notice_attached_to_reply(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = make_brain(tmp)
            b.learn_text("ピヨ理論とは、架空の理論である。", "https://a/p")
            b.notices.append(("ピヨ理論", "ピヨ理論とは、架空の理論である。", "https://a/p"))
            r = b.reply("ピヨ理論について何か知ってる？")
            self.assertIn("ピヨ理論", r.text)
            self.assertEqual(len(b.notices), 0)


class SemanticTest(unittest.TestCase):
    def test_similar_terms_cluster(self):
        sp = SemanticSpace()
        rng = random.Random(0)
        animals = ["犬", "猫", "馬", "牛"]
        foods = ["寿司", "天ぷら", "蕎麦", "饂飩"]
        for _ in range(300):
            a = rng.sample(animals, 2)
            sp.learn(["動物", a[0], "飼育", a[1], "牧場"])
            f = rng.sample(foods, 2)
            sp.learn(["料理", f[0], "食事", f[1], "店"])
        sp.refresh_sketches(limit=1000)
        near = [t for t, _ in sp.similar("犬", k=3)]
        self.assertTrue(set(near) & set(animals))
        self.assertFalse(set(near) & set(foods))
        v = sp.text_vector(["犬", "猫"])
        self.assertGreater(sp.cosine(v, sp.text_vector(["馬"])), sp.cosine(v, sp.text_vector(["寿司"])))
        sp2 = SemanticSpace.from_state(sp.state())
        self.assertEqual(len(sp2.vec), len(sp.vec))
        self.assertGreater(sp.shrink_to(10 * (2 * sp.dim + 160)), 0)

    def test_phrases(self):
        self.assertEqual(phrases("機械学習とは、データから規則性を学ぶ手法である。Python is great"), ["機械学習", "データ", "規則性", "手法", "python", "great"])


class SuffixTest(unittest.TestCase):
    def test_continuations(self):
        seqs = [[3, 4, 5, 6, 7], [3, 4, 5, 6, 8], [9, 4, 5, 10]]
        si = SuffixIndex.build(seqs)
        L, c = si.continuations([1, 3, 4, 5, 6])
        self.assertEqual(L, 4)
        self.assertEqual(set(c), {7, 8})
        L, c = si.continuations([4, 5])
        self.assertEqual(L, 2)
        self.assertEqual(c[6], 2)
        self.assertEqual(c[10], 1)
        self.assertEqual(si.continuations([99, 98])[0], 0)


class CacheAndRerankerTest(unittest.TestCase):
    def test_cache_lm(self):
        c = CacheLM()
        c.push([5, 6, 7, 5, 6])
        self.assertGreater(c.prob(5, 6), c.prob(5, 7))
        self.assertEqual(c.prob(None, 42), 0.0)
        c.clear()
        self.assertEqual(c.prob(5, 6), 0.0)

    def test_reranker_learns(self):
        r = Reranker()
        good = Reranker.features(8, 0.9, 1.0, 0.5, 0.8, 2.0, "user", 40, False, True)
        bad = Reranker.features(2, 0.2, 0.0, 0.0, 0.3, -1.0, "https://x", 200, True, False)
        for _ in range(30):
            r.update(good, 1)
            r.update(bad, 0)
        self.assertGreater(r.predict(good), 0.8)
        self.assertLess(r.predict(bad), 0.3)
        r2 = Reranker.from_state(r.state())
        self.assertAlmostEqual(r2.predict(good), r.predict(good))


class GenerationTest(unittest.TestCase):
    def test_suffix_backed_generation_and_feedback_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = make_brain(tmp)
            text = "\n".join(["ゼータ星は紫色の海を持つ架空の惑星である。", "ゼータ星の住民は歌で会話する。", "ゼータ星には二つの月がある。"] * 3)
            b.learn_text(text.replace("ゼータ星", "ゼータ星") , "https://x/zeta")
            b.background_step(budget_docs=1000)
            self.assertGreater(len(b.suffix), 0)
            self.assertGreater(len(b.semantic.vec), 0)
            out = b.generate(tokenize("ゼータ星"), query="ゼータ星", n_candidates=3)
            self.assertTrue(out.startswith("ゼータ星"))
            self.assertGreater(len(out), 5)
            b.reply("ゼータ星の海の色は？")
            b.reply("👍")
            self.assertGreaterEqual(b.reranker.samples, 1)
            b.reply("ゼータ星の月は？")
            b.reply("👎")
            path = b.save()
            b2 = Brain(b.cfg)
            self.assertTrue(b2.load(path))
            self.assertEqual(b2.reranker.samples, b.reranker.samples)
            self.assertEqual(len(b2.semantic.vec), len(b.semantic.vec))
            b2.background_step()
            self.assertEqual(len(b2.suffix), len(b.suffix))


class WebTest(unittest.TestCase):
    def test_html_to_text(self):
        html = "<html><head><title>T</title><script>x()</script></head><body><nav>menu</nav><p>これは本文の段落です。十分に長い文章。</p><a href='/x'>link</a></body></html>"
        text, links, title = html_to_text(html)
        self.assertIn("本文", text)
        self.assertNotIn("x()", text)
        self.assertNotIn("menu", text)
        self.assertEqual(links, ["/x"])
        self.assertEqual(title, "T")


class MemoryGuardTest(unittest.TestCase):
    def test_budget_positive(self):
        g = MemoryGuard(64, hard=False)
        self.assertGreater(g.budget, 0)
        self.assertLess(g.budget, 64 * 1024 * 1024)
        self.assertGreaterEqual(g.pressure(), 0.0)


class DumpTest(unittest.TestCase):
    def _dump(self, tmp: Path) -> Path:
        import bz2
        bold = "'" * 3
        pages = "".join(
            f"<page><title>架空記事{i}</title><ns>0</ns><revision><text xml:space=\"preserve\">{bold}架空記事{i}{bold}は[[架空の国]]にある都市である。\n架空記事{i}の人口は約 {1000 + i} 人である。{{{{Infobox|x=y}}}}\n[[Category:架空]]</text></revision></page>\n"
            for i in range(10, 40)
        ) + "<page><title>転送</title><ns>0</ns><revision><text>#転送 [[架空記事10]]</text></revision></page><page><title>Wikipedia:方針</title><ns>4</ns><revision><text>方針です。記事ではありません。</text></revision></page>"
        path = tmp / "testwiki.xml.bz2"
        with bz2.open(path, "wt", encoding="utf-8") as f:
            f.write("<mediawiki>\n" + pages + "</mediawiki>\n")
        return path

    def test_wikitext_cleaner(self):
        bold = "'" * 3
        out = wikitext_to_text("{{Infobox|a=b}}" + bold + "東京タワー" + bold + "は[[東京都|東京]]の[[電波塔]]である<ref>x</ref>。\n== 概要 ==\n{| class=\"wikitable\"\n| a || b\n|}\n* 高さは333メートル。")
        self.assertEqual(out, "東京タワーは東京の電波塔である。\n高さは333メートル。")

    def test_stream_dump_and_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._dump(Path(tmp))
            pages = list(iter_wiki_pages(path))
            self.assertEqual(len(pages), 31)  # 名前空間 4 は除外
            b = make_brain(tmp)
            n_pages, n_sents = learn_wiki_dump(b, path)
            self.assertEqual(n_pages, 30)  # 転送は除外
            self.assertGreaterEqual(n_sents, 55)
            self.assertIn("1015", b.reply("架空記事15の人口は？").text)
            import zipfile
            zp = Path(tmp) / "texts.zip"
            with zipfile.ZipFile(zp, "w") as z:
                z.writestr("a.txt", "書庫の中の文章です。二つ目の文もあります。")
                z.writestr("b.bin", b"\x00\x01")
            items = list(iter_archive_texts(zp))
            self.assertEqual(len(items), 1)
            self.assertIn("書庫", items[0][1])


class CollectorRegistryTest(unittest.TestCase):
    def test_registry_and_health(self):
        with tempfile.TemporaryDirectory() as tmp:
            col = Collector(object(), Path(tmp), languages=("ja",))
            names = col.describe()["registered"]
            self.assertIn("wiktionary:ja", names)
            self.assertIn("aozora:ja", names)
            col.report("wikimedia:ja", 120, 300)
            h = col._h("wikimedia:ja")
            self.assertGreater(h.novelty, 0)
            self.assertGreater(h.score, col._h("duckduckgo:ja").score)

    def test_ascii_junk_rule(self):
        self.assertFalse(is_junk("Artificial intelligence is the field of building computer systems that perform tasks."))
        self.assertTrue(is_junk("aaaaaaaaaaaaaaaaaaaaaaaaaa"))


class GuardAndVotingTest(unittest.TestCase):
    def test_unknown_subject_and_attribute(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = make_brain(tmp)
            b.learn_text("富士山の標高は 3776 メートルです。", "https://a/f")
            r = b.reply("富士山の面積は？")
            self.assertIn("知りません", r.text)  # 別の属性 (標高) で答えない
            self.assertLess(r.confidence, 0.5)
            r = b.reply("ゾンビ星の直径は？")
            self.assertLess(r.confidence, 0.3)
            r = b.reply("富士山の高さは？")  # 高さ ↔ 標高 は同義
            self.assertIn("3776", r.text)

    def test_voting_conflict_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = make_brain(tmp)
            b.learn_text("東京タワーの高さは 333 メートルである。東京タワーは東京都港区にある電波塔である。", "https://ja.wikipedia.org/wiki/t")
            b.learn_text("東京タワーの高さは 333 メートルだ。東京タワーは 1958 年に完成した。", "https://example.com/t")
            b.learn_text("東京タワーの高さは 300 メートルである。", "https://bad.example/x")
            groups = b.facts.support(b.facts.lookup("東京タワー", "高さ"))
            self.assertEqual(groups[0][0][1], "333 メートル")
            self.assertEqual(groups[0][1], 2)  # 近似重複の別出典が裏付けとして数えられる
            r = b.reply("東京タワーの高さは？")
            self.assertIn("333", r.text)
            self.assertIn("300", r.text)  # 矛盾を可視化
            r = b.reply("東京タワーについて教えて")
            self.assertEqual(r.mode, "summary")
            self.assertIn("1958", r.text)
            self.assertGreaterEqual(len(r.sources), 2)


class EvalToolTest(unittest.TestCase):
    def test_parse_and_judge(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("evaltool", Path(__file__).resolve().parent.parent / "tools" / "eval.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        from tinyai.brain import Reply
        ex = mod.parse_expect("東京|!大阪|mode:fact,recall|conf>=0.5")
        self.assertEqual(ex["any"], ["東京"])
        self.assertEqual(ex["not"], ["大阪"])
        self.assertTrue(mod.judge(Reply("首都は東京です", 0.9, "fact", [], [], []), ex)[0])
        self.assertFalse(mod.judge(Reply("首都は東京と大阪です", 0.9, "fact", [], [], []), ex)[0])
        self.assertFalse(mod.judge(Reply("首都は東京です", 0.2, "fact", [], [], []), ex)[0])
        unknown = mod.parse_expect("")
        self.assertTrue(mod.judge(Reply("知りません", 0.1, "generate", [], [], []), unknown)[0])
        self.assertFalse(mod.judge(Reply("何か", 0.8, "recall", [], [], []), unknown)[0])


class DialogTest(unittest.TestCase):
    def test_store_and_quotes(self):
        from tinyai.dialog import DialogStore, extract_quote_pairs
        ds = DialogStore(capacity=5)
        self.assertTrue(ds.add("こんにちは", "こんにちは！今日はどうしましたか", "chat"))
        self.assertFalse(ds.add("こんにちは", "こんにちは！今日はどうしましたか", "chat"))  # 重複
        self.assertFalse(ds.add("連絡先は？", "test@example.com です", "chat"))  # 個人情報らしきもの
        for i in range(10):
            ds.add(f"質問{i}", f"答え{i}", "hf")
        self.assertEqual(len(ds), 5)  # 上限
        ds2 = DialogStore.from_state(ds.state())
        self.assertEqual(len(ds2), 5)
        self.assertEqual(extract_quote_pairs("「行くのか」と聞いた。「行くよ」と答えた。"), [("行くのか", "行くよ")])
        chain = extract_quote_pairs("「行くのか」「行くよ」「いつだ」「明日だ」", with_history=True)
        self.assertEqual(chain[0][:2], ("行くのか", "行くよ"))     # 最初の応酬には履歴が無い
        self.assertEqual(chain[1][4], [("行くのか", "行くよ")])     # 次からは手前の応酬が履歴になる
        self.assertEqual(len(chain[-1][4]), 2)                      # 履歴は直近 2 組まで

    def test_hf_row_parsing(self):
        from tinyai.collector import HuggingFaceDatasets
        conv = {"conversations": [{"from": "human", "value": "こんにちは"}, {"from": "gpt", "value": "こんにちは！"}, {"from": "human", "value": "元気？"}, {"from": "gpt", "value": "元気です"}]}
        # 多ターン: 2 ターン目以降は「これまでのやり取り」を history として運ぶ
        pairs = HuggingFaceDatasets._pairs_from_row(conv, "conversations")
        self.assertEqual(pairs[0], ("こんにちは", "こんにちは！"))
        self.assertEqual(pairs[1][:2], ("元気？", "元気です"))
        self.assertEqual(pairs[1][4], [("こんにちは", "こんにちは！")])
        squad = {"question": "富士山の高さは？", "context": "富士山は日本一高い山である。標高は3776メートルで、静岡県と山梨県にまたがる。", "answers": {"text": ["3776メートル"], "answer_start": [17]}}
        (q, a, ctx), = HuggingFaceDatasets._pairs_from_row(squad, "squad")
        self.assertEqual((q, a), ("富士山の高さは？", "3776メートル。"))
        self.assertIn("3776メートル", ctx)
        inst = {"instruction": "首都は？", "input": "日本", "output": "東京"}
        self.assertEqual(HuggingFaceDatasets._pairs_from_row(inst, "instruction"), [("首都は？\n日本", "東京")])


class NeuralTest(unittest.TestCase):
    def setUp(self):
        from tinyai import neural
        if not neural.available():
            self.skipTest("numpy なし")
        self.neural = neural

    def test_gradient_check(self):
        import numpy as np
        nn = self.neural
        m = nn.TinyTransformer(vocab_size=13, d=8, heads=2, layers=2, ctx=6, ff=16, seed=1, dtype=np.float64)
        for k in m.p:
            if m.p[k].ndim >= 2:
                m.p[k] *= 25.0  # 勾配を大きくして丸め誤差の影響を減らす
        rng = np.random.default_rng(0)
        x = rng.integers(8, 13, size=(2, 6))
        y = rng.integers(8, 13, size=(2, 6))
        y[1, 5] = nn.PAD
        _, g = m.loss_and_grads(x, y)
        worst = 0.0
        for k in list(m.p):
            flat = m.p[k].reshape(-1)
            gk = g[k].reshape(-1)
            for idx in rng.choice(flat.size, size=min(3, flat.size), replace=False):
                old = flat[idx]
                eps = 1e-5
                flat[idx] = old + eps
                lp, _ = m.loss_and_grads(x, y)
                flat[idx] = old - eps
                lm, _ = m.loss_and_grads(x, y)
                flat[idx] = old
                num = (lp - lm) / (2 * eps)
                worst = max(worst, abs(num - gk[idx]) / max(abs(num) + abs(gk[idx]), 1e-8))
        self.assertLess(worst, 1e-5)

    def test_kv_cache_matches_full_forward(self):
        import numpy as np
        nn = self.neural
        m = nn.TinyTransformer(vocab_size=50, d=16, heads=2, layers=2, ctx=10, ff=32, seed=2)
        rng = np.random.default_rng(1)
        seq = [int(t) for t in rng.integers(8, 50, size=7)]
        full, _ = m.forward(np.array([seq]))
        cache = [(None, None)] * 2
        for pos, t in enumerate(seq):
            inc = m._step(t, pos, cache)
        self.assertLess(float(np.abs(full[0, -1] - inc).max()), 1e-4)

    def test_training_reduces_loss_and_roundtrip(self):
        import numpy as np
        nn = self.neural
        from tinyai.bpe import SubwordTokenizer
        tok = SubwordTokenizer([f"w{i}" for i in range(30)])
        m = nn.TinyTransformer(len(tok), d=16, heads=2, layers=1, ctx=12, ff=32, seed=0)
        pool = nn.SequencePool(seed=0)
        seq = [nn.BOS] + [tok.index[f"w{i % 30}"] for i in range(11)] + [nn.EOS]
        for _ in range(64):
            pool.add(seq)
        first = nn.train_steps(m, pool, steps=1, batch=8, lr=1e-2, warmup=1, total=100)["first_loss"]
        r = nn.train_steps(m, pool, steps=60, batch=8, lr=1e-2, warmup=1, total=100)
        self.assertLess(r["loss"], first * 0.7)
        lp = m.logprob(seq)
        out = m.generate(seq[:3], max_new=5, temperature=0.5, rng=np.random.default_rng(0))
        self.assertLessEqual(len(out), 5)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "n.npz"
            m.save(path, tok, meta={"x": 1})
            m2, t2, meta = nn.TinyTransformer.load(path)
            self.assertEqual(meta["x"], 1)
            self.assertEqual(len(t2), len(tok))
            self.assertAlmostEqual(m2.logprob(seq), lp, places=4)

    def test_generate_batch_and_parallel_trainer(self):
        import numpy as np
        nn = self.neural
        from tinyai.neural_parallel import ParallelTrainer
        m = nn.TinyTransformer(vocab_size=60, d=16, heads=2, layers=1, ctx=12, ff=32, seed=3)
        rng = np.random.default_rng(0)
        outs = m.generate_batch([8, 9, 10], n=3, max_new=6, rng=rng)
        self.assertEqual(len(outs), 3)
        self.assertTrue(all(len(o) <= 6 for o in outs))
        pool = nn.SequencePool(seed=0)
        seq = [nn.BOS] + list(range(8, 19)) + [nn.EOS]
        for _ in range(64):
            pool.add(seq)
        pt = ParallelTrainer(m, pool, workers=2)
        if not pt.start():
            self.skipTest("fork が使えない環境")
        try:
            first = pt.train(steps=1, batch=4, lr=1e-2, total=100, warmup=1)["first_loss"]
            r = pt.train(steps=40, batch=4, lr=1e-2, total=100, warmup=1)
            self.assertEqual(r["workers"], 2)
            self.assertLess(r["loss"], first * 0.8)  # 並列でも学習が進む
        finally:
            pt.stop()
        self.assertGreater(m.logprob(seq), -3.0)  # 停止後も (通常メモリに戻した) パラメータが有効

    def test_subword_tokenizer(self):
        from tinyai.bpe import SubwordTokenizer
        tok = SubwordTokenizer.train(["機械学習とは、データから規則性を学ぶ手法である。"] * 5 + ["Tokyo Tower was built in 1958."] * 5, size=200)
        ids = tok.encode("機械学習とは? Tokyo Tower 1958!")
        self.assertNotIn(1, ids)  # 基本文字 (ASCII・句読点) は必ず語彙にあるので <unk> は出ない
        self.assertEqual(tok.decode(tok.encode("tokyo tower was built")), "tokyo tower was built")
        self.assertIn("機械学習", tok.tokens)

    def test_brain_neural_step_and_dialog_feed(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = make_brain(tmp)
            b.learn_text("\n".join(f"サンプル文 {i} は学習用の文章であり、内容は番号 {i} に関する説明です。" for i in range(10, 90)), "https://x/nn")
            b.neural.min_sentences, b.neural.min_chars, b.neural.size = 10, 100, "small"
            r = b.neural_step(steps=2, budget_seconds=0.5)
            self.assertIsNotNone(b.neural.model)
            self.assertGreater(len(b.neural.pool), 0)
            self.assertIsNotNone(b.neural.score("サンプル文 12 は学習用の文章です。"))
            b.reply("覚えて: ゼータ星は紫色の海を持つ架空の惑星である。")
            b.reply("ゼータ星の海は？")
            b.reply("👍")
            self.assertGreaterEqual(len(b.dialogs), 1)
            path = b.save()
            self.assertTrue((Path(tmp) / "neural.npz").exists())
            b2 = Brain(b.cfg)
            b2.load(path)
            self.assertEqual(len(b2.dialogs), len(b.dialogs))
            b2.neural_step(steps=1, budget_seconds=0.2)
            self.assertGreaterEqual(b2.neural.model.step, b.neural.model.step + 1)


class AgentTest(unittest.TestCase):
    def test_tools(self):
        from tinyai.agent import calculate, answer_datetime, convert_units, apply_format, extract_items
        self.assertEqual(calculate("12×34は？")[1], 408)
        self.assertEqual(calculate("1000円の10%は？")[1], 100)
        self.assertEqual(calculate("2の10乗")[1], 1024)
        self.assertIsNone(calculate("2024年は何年"))
        self.assertIn("曜", answer_datetime("今日は何曜日？"))
        self.assertIn("3.1", convert_units("5kmはマイルで何？"))
        self.assertIn("212", convert_units("100℃は°Fで？"))
        self.assertEqual(apply_format("一つ目。二つ目。", "箇条書きで"), "・一つ目。\n・二つ目。")
        self.assertEqual(extract_items(["手法には決定木、SVM、ニューラルネットワークなどがある。"], "手法", 3), ["決定木", "SVM", "ニューラルネットワーク"])

    def test_agent_in_brain(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = make_brain(tmp)
            self.assertEqual(b.reply("12×34は？").mode, "tool:calc")
            b.reply("私の名前は太郎です")
            self.assertIn("太郎", b.reply("私の名前は？").text)
            b.learn_text("東京タワーの高さは 333 メートルである。スカイツリーの高さは 634 メートルである。", "https://ja.wikipedia.org/wiki/t")
            r = b.reply("東京タワーとスカイツリーはどちらが高い？")
            self.assertIn("スカイツリーの方が高い", r.text)
            r = b.reply("東京タワーの高さを一言で")
            self.assertIn("333", r.text)
            self.assertIn("出典", b.reply("東京タワーの高さは？").text)  # Web 由来には出典
            path = b.save()
            b2 = Brain(b.cfg)
            b2.load(path)
            self.assertEqual(b2.agent.profile.get("名前"), "太郎")


class RealtimeLearningTest(unittest.TestCase):
    def setUp(self):
        from tinyai import neural
        if not neural.available():
            self.skipTest("numpy なし")
        self.neural = neural

    def test_unlikelihood_and_growth_and_vocab(self):
        import numpy as np
        nn = self.neural
        m = nn.TinyTransformer(vocab_size=30, d=16, heads=2, layers=1, ctx=12, ff=32, seed=0)
        rng = np.random.default_rng(0)
        seq = [nn.BOS] + list(range(8, 19)) + [nn.EOS]
        before = m.logprob(seq)
        m.grow_layer()
        self.assertAlmostEqual(m.logprob(seq), before, places=4)  # 成長は関数を保つ
        self.assertEqual(m.L, 2)
        m.add_tokens(5)
        self.assertEqual(m.p["wte"].shape[0], 35)
        pool = nn.SequencePool(seed=0)
        good = [nn.BOS] + list(range(8, 14)) + [nn.EOS]
        bad = [nn.BOS] + list(range(14, 20)) + [nn.EOS]
        for _ in range(30):
            pool.add(good, 1.0)
            pool.add(bad, 1.0)
        nn.train_steps(m, pool, steps=60, batch=8, lr=1e-2, warmup=1, total=200)
        lb0 = m.logprob(bad)
        pool2 = nn.SequencePool(seed=1)
        for _ in range(30):
            pool2.add(good, 1.0)
            pool2.add(bad, -1.0)
        nn.train_steps(m, pool2, steps=40, batch=8, lr=5e-3, warmup=1, total=200)
        self.assertLess(m.logprob(bad), lb0 - 1.0)  # 負例の確率が下がる
        self.assertGreater(m.logprob(good), -1.0)   # 正例は保たれる

    def test_brain_online_learning_and_neural_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = make_brain(tmp)
            b.learn_text("\n".join(f"サンプル文 {i} は学習用の文章であり、内容は番号 {i} に関する説明です。" for i in range(10, 90)), "https://x/nn")
            b.neural.min_sentences, b.neural.min_chars, b.neural.size = 10, 100, "small"
            b.neural_step(steps=2, budget_seconds=0.5)
            self.assertIsNotNone(b.neural.model)
            b.reply("覚えて: ゼータ星は紫色の海を持つ架空の惑星である。")
            r = b.reply("ゼータ星の海は？")
            self.assertGreaterEqual(b.stats["online_turns"], 1)  # ターン直後に学習した
            steps_before = b.neural.model.step
            b.reply("👍")
            self.assertGreater(b.neural.model.step, steps_before)  # 👍 で追加学習
            b.reply("ゼータ星の海は？")
            steps_before = b.neural.model.step
            b.reply("👎")
            self.assertGreater(b.neural.model.step, steps_before)  # 👎 で unlikelihood 学習
            self.assertGreaterEqual(b.stats["unlearned_turns"], 1)
            # 準備が整ったことにして、応答がニューラル生成になるか
            b.neural.ready = True
            r = b.reply("サンプル文 12 について教えて")
            self.assertIn(r.mode, ("neural", "summary", "recall", "fact"))
            # 語彙の進化
            added = b.neural.evolve_vocab(["新語ホゲホゲ理論が何度も出てくる。新語ホゲホゲ理論とは新語ホゲホゲ理論である。"] * 6, top=5)
            self.assertGreater(added, 0)
            for _ in range(6):
                b.neural.feedback(True)
            self.assertIsNotNone(b.neural._decode_trial)  # 復号パラメータの試行が始まる

    def test_loss_mask_and_parallel_restart_on_vocab_growth(self):
        try:
            from tinyai import neural as nn
        except Exception:
            self.skipTest("numpy なし")
        if not nn.available():
            self.skipTest("numpy なし")
        from tinyai.bpe import BOT
        pool = nn.SequencePool(seed=1)
        seq = [2, 6, 10, 11, 4, 20, 21, 5, 30, 31, 32, 3]
        pool.add(seq, loss_from=seq.index(BOT) + 1)
        x, y, w = pool.batch(1, len(seq) - 1)
        self.assertEqual(w.shape, (1, len(seq) - 1))
        # <bot> より前の予測対象は PROMPT_WEIGHT、応答部は 1.0
        lf = seq.index(BOT) + 1
        self.assertTrue(all(abs(v - nn.SequencePool.PROMPT_WEIGHT) < 1e-6 for v in w[0, : lf - 1]))
        self.assertTrue(all(abs(v - 1.0) < 1e-6 for v in w[0, lf - 1 :]))
        m = nn.TinyTransformer(60, d=32, heads=2, layers=1, ctx=16)
        loss_masked, _ = m.loss_and_grads(x, y, w)
        loss_full, _ = m.loss_and_grads(x, y, None)
        self.assertLess(loss_masked, loss_full)  # プロンプト部の損失が小さく数えられる
        # 語彙進化・層追加は並列ワーカーを止めてから形を変える
        with tempfile.TemporaryDirectory() as tmp:
            from tinyai.neural_lm import NeuralLM
            nl = NeuralLM(Path(tmp), size="small")
            texts = [f"文 {i} は語彙学習のためのサンプルであり、番号 {i} に関する説明文である。" for i in range(300)]
            nl.min_sentences, nl.min_chars = 10, 100
            self.assertTrue(nl.ensure_model(texts))
            for t in texts:
                nl.add_text(t)
            nl.set_workers(2)
            r = nl.train_some(steps=2, batch=4)
            if nl._parallel is not None:
                V0 = nl.model.V
                added = nl.evolve_vocab(["新語ホゲホゲ理論が何度も出てくる。新語ホゲホゲ理論とは新語ホゲホゲ理論である。"] * 6, top=5)
                self.assertGreater(added, 0)
                self.assertIsNone(nl._parallel)              # 形が変わる前に止めた
                r = nl.train_some(steps=2, batch=4)          # 新しい形で作り直して続行できる
                self.assertIsNotNone(r)
                self.assertEqual(nl.model.V, V0 + added)
            nl.stop_parallel()

    def test_export_for_web(self):
        try:
            from tinyai import neural as nn
        except Exception:
            self.skipTest("numpy なし")
        if not nn.available():
            self.skipTest("numpy なし")
        import json
        from tinyai.export import export_model, quantize_rows, dequantize_rows
        from tinyai.bpe import SubwordTokenizer
        np = nn.np
        w = np.random.default_rng(0).standard_normal((5, 8)).astype(np.float32)
        q, sc = quantize_rows(w)
        self.assertLess(float(np.abs(dequantize_rows(q, sc) - w).max()), float(np.abs(w).max()) / 100)
        texts = [f"文 {i} は書き出しテストのための文章である。" for i in range(50)]
        tok = SubwordTokenizer.train(texts, size=300)
        m = nn.TinyTransformer(len(tok), d=32, heads=2, layers=2, ctx=32, ff=64)
        with tempfile.TemporaryDirectory() as tmp:
            meta = export_model(m, tok, Path(tmp))
            self.assertEqual(meta["V"], len(tok))
            size = (Path(tmp) / "model.bin").stat().st_size
            self.assertEqual(size, meta["bytes"])
            self.assertLess(size, m.n_params() * 1.3)   # int8 なので 4 バイト/パラメータより大幅に小さい
            test = json.loads((Path(tmp) / "test.json").read_text(encoding="utf-8"))
            self.assertEqual(len(test["top"]), 10)
            self.assertGreater(test["train"]["loss"], 0)
            vocab = json.loads((Path(tmp) / "vocab.json").read_text(encoding="utf-8"))
            self.assertEqual(vocab[:8], ["<pad>", "<unk>", "<bos>", "<eos>", "<usr>", "<bot>", "<ctx>", "<sep>"])

    def test_prioritized_replay_and_worker_sync(self):
        try:
            from tinyai import neural as nn
        except Exception:
            self.skipTest("numpy なし")
        if not nn.available():
            self.skipTest("numpy なし")
        pool = nn.SequencePool(seed=0)
        for i in range(40):
            pool.add([2] + [10 + (i % 5)] * 6 + [3])
        m = nn.TinyTransformer(60, d=32, heads=2, layers=1, ctx=8)
        x, y, w = pool.batch(4, 7)
        self.assertEqual(len(pool.last_rows), 4)
        m.loss_and_grads(x, y, w)
        self.assertEqual(m.last_row_loss.shape, (4,))
        pool.update(m.last_row_loss)
        touched = {i for rows in pool.last_rows for i in rows}
        self.assertTrue(any(abs(pool.priority[i] - pool.init_priority) > 1e-6 for i in touched))  # 優先度が更新された
        # 優先度の高い系列ほど多く出る
        pool.priority[:40] = 0.05
        pool.priority[7] = 50.0
        nn.SequencePool.PRIORITY_MIX, mix = 0.0, nn.SequencePool.PRIORITY_MIX
        try:
            picks = [pool._pick() for _ in range(300)]
        finally:
            nn.SequencePool.PRIORITY_MIX = mix
        self.assertGreater(picks.count(7), 100)
        # ワーカー同期: journal に記録され、sync でワーカーに届く
        from tinyai.neural_parallel import ParallelTrainer
        pt = ParallelTrainer(m, pool, workers=2)
        if pt.start():
            try:
                pool.add([2, 20, 21, 22, 23, 3])
                self.assertEqual(len(pool.journal), 1)
                self.assertEqual(pt.sync(), 1)
                self.assertEqual(len(pool.journal), 0)
                r = pt.train(steps=2, batch=4, lr=1e-3, total=100)
                self.assertEqual(r["workers"], 2)
            finally:
                pt.stop()
            self.assertIsNone(pool.journal)

    def test_dropout_and_surprise_and_thinking(self):
        try:
            from tinyai import neural as nn
        except Exception:
            self.skipTest("numpy なし")
        if not nn.available():
            self.skipTest("numpy なし")
        import numpy as np
        m = nn.TinyTransformer(60, d=32, heads=2, layers=1, ctx=8, dropout=0.5)
        x = np.array([[2, 10, 11, 12, 13, 14, 15, 3]])
        a, _ = m.forward(x, train=True)
        b, _ = m.forward(x, train=True)
        c, _ = m.forward(x, train=False)
        d, _ = m.forward(x, train=False)
        self.assertGreater(float(np.abs(a - b).max()), 0.0)   # 学習時はマスクが違う
        self.assertEqual(float(np.abs(c - d).max()), 0.0)     # 推論時は決定的
        with tempfile.TemporaryDirectory() as tmp:
            b_ = make_brain(tmp)
            b_.learn_text("\n".join(f"サンプル文 {i} は学習用の文章であり、内容は番号 {i} に関する説明です。" for i in range(10, 90)), "https://x/nn")
            b_.neural.min_sentences, b_.neural.min_chars, b_.neural.size = 10, 100, "small"
            b_.neural_step(steps=2, budget_seconds=0.5)
            self.assertIsNotNone(b_.neural.model)
            self.assertEqual(b_.neural.model.dropout, b_.cfg.neural_dropout)
            s1 = b_.neural.surprise(["サンプル文 12 は学習用の文章であり、内容は番号 12 に関する説明です。"])
            s2 = b_.neural.surprise(["zzqx wvv kkjjl pqrst mnbvc lkjhg"])
            self.assertIsNotNone(s1)
            self.assertGreater(s2, 0)
            # 収集ソースの価値: 驚きが評価表に入る
            from tinyai.collector import Batch, Collector
            col = Collector(None, Path(tmp), ["ja"])
            b_.learn_batch(Batch("t", "topic", [("https://src/a", "新しい文 1 は説明である。新しい文 2 も説明である。新しい文 3 は違う説明である。", [])], "srcA", 0.1), col)
            h = col.health["srcA"].to_dict()
            self.assertIn("surprise", h)
            self.assertIn("value", h)
            # 思考: 2 段階生成の記録
            b_.neural.ready = True
            r = b_.reply("サンプル文 12 について教えて")
            self.assertIsNotNone(b_.last_thought)
            self.assertIn("draft", b_.last_thought)
            self.assertIn("scores", b_.last_thought)

    def test_wsd_schedule_and_context_extension(self):
        try:
            from tinyai import neural as nn
        except Exception:
            self.skipTest("numpy なし")
        if not nn.available():
            self.skipTest("numpy なし")
        import numpy as np
        # WSD: ウォームアップの後は一定 (終わりの無い継続学習で学習率が枯れない)
        self.assertAlmostEqual(nn.lr_at(0, 1e-3, 10, 1000), 1e-4)
        self.assertAlmostEqual(nn.lr_at(500, 1e-3, 10, 1000), 1e-3)
        self.assertAlmostEqual(nn.lr_at(5000, 1e-3, 10, 1000), 1e-3)   # total を超えても止まらない
        self.assertLess(nn.lr_at(900, 1e-3, 10, 1000, schedule="cosine"), 1e-3)
        # 文脈長の拡張: 重みは変わらず、伸ばした先でも生成できる
        m = nn.TinyTransformer(60, d=32, heads=2, layers=1, ctx=16)
        before = {k: v.copy() for k, v in m.p.items()}
        x = np.array([[2, 10, 11, 12, 3]])
        logits_before, _ = m.forward(x)
        self.assertEqual(m.extend_context(48), 48)
        self.assertEqual(m.T, 48)
        for k, v in before.items():
            self.assertEqual(float(np.abs(m.p[k] - v).max()), 0.0)     # パラメータは無傷
        logits_after, _ = m.forward(x)
        self.assertLess(float(np.abs(logits_after - logits_before).max()), 1e-4)  # 短い系列の出力も同じ
        long_ids = list(range(8, 8 + 40))
        out = m.generate([2] + long_ids, max_new=3)                    # 伸ばした文脈で動く
        self.assertLessEqual(len(out), 3)

    def test_preference_data_becomes_negative_example(self):
        from tinyai.collector import HuggingFaceDatasets
        row = {"conversations": [{"from": "human", "value": "質問"}, {"from": "gpt", "value": "途中の答え"}, {"from": "human", "value": "本題は？"}],
               "chosen": "良い答えです。", "rejected": "悪い答えです。"}
        pairs = HuggingFaceDatasets._pairs_from_row(row, "preference")
        self.assertEqual(pairs[0][:2], ("本題は？", "良い答えです。"))
        self.assertEqual(pairs[0][4], [("質問", "途中の答え")])   # 手前のやり取りを履歴に持つ
        self.assertEqual(pairs[1][3], -1.0)      # 不採用の応答は負例
        self.assertEqual(pairs[1][4], pairs[0][4])                # 負例も同じ履歴の続き
        try:
            from tinyai import neural as nn
        except Exception:
            return
        if not nn.available():
            return
        with tempfile.TemporaryDirectory() as tmp:
            b = make_brain(tmp)
            b.learn_text("\n".join(f"サンプル文 {i} は学習用の文章であり、内容は番号 {i} に関する説明です。" for i in range(10, 90)), "https://x/nn")
            b.neural.min_sentences, b.neural.min_chars, b.neural.size = 10, 100, "small"
            b.neural_step(steps=2, budget_seconds=0.5)
            from tinyai.collector import Batch
            b.learn_batch(Batch("t", "stream", [], "pref", 0.1, dialogs=[("本題は？", "良い答えです。"), ("本題は？", "悪い答えです。", None, -1.0)]))
            neg = [item[3] for item in b._neural_pending_dialog if item[3] < 0]
            self.assertTrue(neg, "負例が学習待ち行列に入る")
            before = len(b.neural.pool)
            b._feed_neural()
            self.assertGreater(len(b.neural.pool), before)
            self.assertTrue(any(w < 0 for _, w, _ in b.neural.pool.items), "再生バッファに負例が入る")

    def test_dialog_shortening_and_language_preference(self):
        from tinyai.dialog import DialogStore
        d = DialogStore()
        long_bot = "最初の文です。" + "続きの説明がここに入ります。" * 30
        self.assertTrue(d.add("質問", long_bot))
        _, bot, _, _, *_rest = d.pairs[-1]
        self.assertLessEqual(len(bot), DialogStore.MAX_BOT + 40)
        self.assertTrue(bot.endswith("。"))          # 途中で切れた文は残さない
        self.assertTrue(d.add("問い", "短い答え。"))
        self.assertFalse(d.add("あ", "い"))           # 短すぎるものは従来通り弾く
        # 第 1 言語の供給源が優先される
        with tempfile.TemporaryDirectory() as tmp:
            col = Collector(object(), Path(tmp), languages=("ja", "en"))
            ja = [s_ for s_ in col.streams if s_.lang == "ja"]
            en = [s_ for s_ in col.streams if s_.lang == "en"]
            if ja and en:
                for s_ in col.streams:
                    col._h(f"{s_.name}:{s_.lang}").record(True, 1.0, 0.5)
                primary = col.languages[0]
                w = {f"{s_.name}:{s_.lang}": col._h(f"{s_.name}:{s_.lang}").score * s_.weight * (1.0 if s_.lang == primary else 0.4) for s_ in col.streams}
                self.assertGreater(sum(v for k, v in w.items() if k.endswith(":ja")), sum(v for k, v in w.items() if k.endswith(":en")) * 1.2)

    def test_multi_turn_memory(self):
        try:
            from tinyai import neural as nn
        except Exception:
            self.skipTest("numpy なし")
        if not nn.available():
            self.skipTest("numpy なし")
        from tinyai.bpe import BOT, USR
        with tempfile.TemporaryDirectory() as tmp:
            b = make_brain(tmp)
            b.learn_text("\n".join(f"文 {i} は多ターン会話のテストであり、番号 {i} を説明する。" for i in range(10, 160)), "https://x/mt")
            b.neural.min_sentences, b.neural.min_chars, b.neural.size = 10, 100, "small"
            b.neural_step(steps=2, budget_seconds=0.5)
            nl = b.neural
            hist = [("こんにちは", "こんにちは、何でも聞いてください。"), ("天気は？", "今日は晴れです。")]
            p_hist = nl.prompt_dialog("じゃあ明日は？", None, history=hist)
            p_plain = nl.prompt_dialog("じゃあ明日は？", None)
            self.assertGreater(len(p_hist), len(p_plain))     # 履歴が入っている
            self.assertGreaterEqual(p_hist.count(USR), 2)
            self.assertEqual(p_hist[-1], BOT)                 # 生成は応答から始まる
            seq = nl.seq_dialog("じゃあ明日は？", "明日は雨です。", None, history=hist)
            lf = nl.loss_from(seq)
            self.assertEqual(seq[lf - 1], BOT)                # 最後の <bot> の次から学ぶ
            self.assertNotIn(BOT, seq[lf:])                   # 応答部に過去のやり取りは含まれない
            # Brain: 会話が続くと履歴が渡される
            b.reply("こんにちは")
            b.reply("元気ですか")
            turns = b.recent_turns(2)
            self.assertTrue(turns and turns[-1][0] == "元気ですか")

    def test_ema_guard_falls_back_to_raw_weights(self):
        try:
            from tinyai import neural as nn
        except Exception:
            self.skipTest("numpy なし")
        if not nn.available():
            self.skipTest("numpy なし")
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            from tinyai.neural_lm import NeuralLM
            nl = NeuralLM(Path(tmp), size="small")
            texts = [f"文 {i} は EMA 検査のための文章であり、番号 {i} を説明する。" for i in range(200)]
            nl.min_sentences, nl.min_chars = 10, 100
            self.assertTrue(nl.ensure_model(texts))
            for t in texts:
                nl.add_text(t)
            nl.train_some(steps=5, batch=4)
            nl._holdout = [nl.seq_text(t) for t in texts[:20]]
            # EMA をわざと壊す (学習が速いと EMA が遅れて悪くなる状況の再現)
            nl.model.update_ema()
            for k in nl.model.ema:
                nl.model.ema[k] = nl.model.ema[k] + np.float32(0.5)
            ev = nl.evaluate()
            self.assertFalse(nl.use_ema)                      # 悪い EMA は使わない
            self.assertIsNotNone(nl.ema_ppl)
            raw = nn.perplexity(nl.model, nl._holdout)
            self.assertLess(abs(ev["neural_ppl"] - raw), raw * 0.2)   # 報告値は生の重みの側
            # 作り直された EMA は重みと一致する
            self.assertLess(float(max(np.abs(nl.model.ema[k] - nl.model.p[k]).max() for k in nl.model.p)), 1e-6)

    def test_free_chat_scoring_and_repetition(self):
        from tinyai.brain import Brain
        self.assertGreater(Brain._repeat_ratio("これはこれはこれは良い例です"), 0.3)
        self.assertGreater(Brain._repeat_ratio("宇宙は宇宙において宇宙の膨張が宇宙で起きる"), 0.4)
        self.assertLess(Brain._repeat_ratio("人工知能とは、人間の知的な振る舞いを実現する技術です。"), 0.2)
        self.assertEqual(Brain._repeat_ratio("はい。"), 0.0)
        try:
            from tinyai import neural as nn
        except Exception:
            return
        if not nn.available():
            return
        with tempfile.TemporaryDirectory() as tmp:
            b = make_brain(tmp)
            b.learn_text("\n".join(f"文 {i} は自由な会話のテストであり、番号 {i} を説明する。" for i in range(10, 160)), "https://x/fc")
            b.neural.min_sentences, b.neural.min_chars, b.neural.size = 10, 100, "small"
            b.neural_step(steps=2, budget_seconds=0.5)
            b.neural.ready = True
            b.reply("文 12 について教えて")          # 知識の質問: 文脈に寄せる
            strong = (b.last_thought or {}).get("copy_bonus")
            b.reply("それはどういう意味ですか")        # 指示語: 寄せを緩める
            weak = (b.last_thought or {}).get("copy_bonus")
            if strong is not None and weak is not None:
                self.assertLessEqual(weak, strong)



class CandidateDiversityTest(unittest.TestCase):
    """候補ごとに温度を変え、同じ文が並ぶのを防ぐ。"""

    def test_temp_spread_changes_rows(self):
        import numpy as np
        from tinyai.neural import TinyTransformer
        m = TinyTransformer(vocab_size=60, d=32, heads=4, layers=1, ctx=40, ff=64, seed=3)
        m.p["wte"][12] *= 30.0                             # 分布を尖らせる (同じ文が並びやすい状態)
        same = m.generate_batch([3, 4], n=4, max_new=10, temperature=0.7, rng=np.random.default_rng(1))
        spread = m.generate_batch([3, 4], n=4, max_new=10, temperature=0.7, rng=np.random.default_rng(1),
                                  temp_spread=0.5)
        self.assertEqual(len(spread), 4)
        self.assertNotEqual([tuple(x) for x in same], [tuple(x) for x in spread])   # 引き方が変わる

    def test_chat_drops_duplicate_candidates(self):
        from pathlib import Path
        from tinyai.neural_lm import NeuralLM
        with tempfile.TemporaryDirectory() as tmp:
            nl = NeuralLM(Path(tmp), size="small")
            nl.min_sentences, nl.min_chars = 4, 40
            nl.ensure_model(["こんにちは。今日はいい天気です。散歩に行きましょう。%d" % i for i in range(60)])
            outs = nl.chat("こんにちは", None, n=4, max_new=12)
            self.assertEqual(len(outs), len(set(outs)))    # 同じ文は 1 本にまとめる


class DecodeQualityTest(unittest.TestCase):
    """自由生成のための復号: 言い回しのループ禁止、最短長、既定値の世代管理。"""

    def test_banned_ngram_tokens(self):
        from tinyai.neural import _banned_ngram_tokens
        out = [5, 6, 7, 5, 6]
        self.assertEqual(_banned_ngram_tokens(out, 3), {7})       # 5,6 の続きに出た 7 は禁止
        self.assertEqual(_banned_ngram_tokens(out, 2), {7})       # 直前 6 の続きも 7
        self.assertEqual(_banned_ngram_tokens([1, 2], 3), set())  # 長さが足りなければ禁止なし

    def test_min_new_and_no_repeat_in_generate(self):
        import numpy as np
        from tinyai.neural import TinyTransformer, EOS
        m = TinyTransformer(vocab_size=40, d=32, heads=4, layers=1, ctx=40, ff=64, seed=1)
        m.p["wte"][EOS] *= 50.0                                   # 終端が出やすい状態にする
        short = m.generate_batch([3, 4], n=3, max_new=12, temperature=0.7, rng=np.random.default_rng(0))
        long = m.generate_batch([3, 4], n=3, max_new=12, temperature=0.7, rng=np.random.default_rng(0), min_new=5)
        self.assertGreaterEqual(min(len(x) for x in long), 5)
        self.assertGreaterEqual(min(len(x) for x in long), min(len(x) for x in short))
        outs = m.generate_batch([3, 4], n=3, max_new=24, temperature=0.9, rng=np.random.default_rng(2),
                                no_repeat_ngram=3, min_new=8)
        for o in outs:
            grams = [tuple(o[i:i + 3]) for i in range(len(o) - 2)]
            self.assertEqual(len(grams), len(set(grams)))         # 同じ 3-gram は二度出ない

    def test_decode_version_resets_stale_value(self):
        import json
        from pathlib import Path
        from tinyai.neural_lm import NeuralLM
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            nl = NeuralLM(d, size="small")
            nl.min_sentences, nl.min_chars = 4, 40
            nl.ensure_model(["こんにちは。今日はいい天気です。散歩に行きましょう。%d" % i for i in range(60)])
            nl.save()
            meta_path = d / "neural.npz.meta.json"
            meta = json.loads(meta_path.read_text())
            self.assertGreaterEqual(meta["decode_version"], 1)
            meta.pop("decode_version")                            # 古い書き出しを模す
            meta["decode"]["copy_bonus"] = 3.0
            meta["decode"]["temperature"] = 0.55
            meta_path.write_text(json.dumps(meta))
            nl2 = NeuralLM(d, size="small")
            nl2.ensure_model()
            self.assertEqual(nl2.decode["copy_bonus"], 1.0)       # 見直した項目は既定値から
            self.assertEqual(nl2.decode["temperature"], 0.55)     # それ以外は保存値のまま


class DialogHistoryTest(unittest.TestCase):
    """会話の履歴を保存し、再生バッファに履歴つきで戻せること (会話のキャッチボールの学習)。"""

    def test_reply_markdown_is_cleaned(self):
        """応答の書式記号を落とし、箇条書きの手前までを学習対象にする。"""
        from tinyai.dialog import DialogStore
        long_md = ("ノートPCのバッテリー寿命を保つ方法はいくつかあります。以下の点に注意してください。"
                   "\n\n1. **温度管理**\n高温だと劣化が早まります。\n2. **充電**\n満充電を避けます。")
        cleaned = DialogStore.clean_reply(long_md)
        self.assertNotIn("**", cleaned)
        self.assertNotIn("\n1.", cleaned)
        self.assertIn("高温だと劣化", cleaned)                # 箇条書きの中身は地の文にして残す
        self.assertNotIn("1. ", cleaned)                      # 記号は落とす
        self.assertEqual(DialogStore.clean_reply("**太字**と`コード`"), "太字とコード")
        self.assertEqual(DialogStore.clean_reply("- 箇条書きだけ"), "- 箇条書きだけ")   # 1 行だけなら残す
        hollow = "日本の四季について説明します。\n\n- 春は桜が咲きます。\n- 夏は高温多湿です。"
        got = DialogStore.clean_reply(hollow)
        self.assertIn("春は桜", got)                         # 前置きだけの応答にしない
        self.assertNotIn("- 春", got)
        self.assertEqual(DialogStore.clean_reply("1.***太字***です"), "1.太字です")     # ** も *** も落とす
        d = DialogStore(5)
        d.add("質問", long_md)
        self.assertNotIn("**", d.pairs[-1][1])

    def test_history_is_stored_and_trimmed(self):
        from tinyai.dialog import DialogStore
        d = DialogStore(10)
        hist = [("最初の質問", "最初の答え"), ("次の質問", "次の答え"), ("直前の質問", "直前の答え")]
        self.assertTrue(d.add("それで?", "続きはこうです。", source="chat", history=hist))
        item = d.pairs[-1]
        self.assertEqual(len(item), 5)
        self.assertEqual(len(item[4]), DialogStore.MAX_HISTORY)      # 直近 2 組だけ持つ
        self.assertEqual(item[4][-1][0], "直前の質問")               # 新しい方を残す
        self.assertEqual(d.with_history(), 1)
        self.assertTrue(d.add("履歴なし", "普通の返事"))
        self.assertEqual(d.with_history(), 1)
        long_hist = [("あ" * 400, "い" * 400)]
        d.add("長い履歴", "返事", history=long_hist)
        self.assertLessEqual(len(d.pairs[-1][4][0][0]), DialogStore.MAX_HISTORY_CHARS)

    def test_multiturn_dialogues_are_weighted_higher(self):
        """履歴つきの会話は再生バッファに厚めに入れる (前の発話を踏まえる練習を増やす)。"""
        from pathlib import Path
        from tinyai.neural_lm import NeuralLM
        with tempfile.TemporaryDirectory() as tmp:
            nl = NeuralLM(Path(tmp), size="small")
            nl.min_sentences, nl.min_chars = 4, 40
            nl.ensure_model(["こんにちは。今日はいい天気です。散歩に行きましょう。%d" % i for i in range(60)])
            n0 = len(nl.pool)
            nl.add_dialog("質問", "答えです。")
            single = len(nl.pool) - n0
            n1 = len(nl.pool)
            nl.add_dialog("質問2", "答えです2。", history=[("前の質問", "前の答え")])
            multi = len(nl.pool) - n1
            self.assertEqual(single, 1)
            self.assertEqual(multi, 2)

    def test_history_survives_save_and_restore(self):
        from tinyai.dialog import DialogStore
        d = DialogStore(10)
        d.add("それで?", "続きはこうです。", history=[("宇宙の話", "宇宙は広いです。")])
        d.add("履歴なし", "普通の返事")
        restored = DialogStore.from_state(d.state(), 10)
        self.assertEqual(restored.with_history(), 1)
        self.assertEqual(restored.pairs[0][4][0][0], "宇宙の話")
        old_style = [("質問", "答え", "web", 1.0)]                   # 旧形式 (4 要素) も読める
        self.assertEqual(len(DialogStore.from_state(old_style, 10)), 1)

    def test_replay_keeps_history(self):
        """保存済みの会話から再生バッファを積み直す時、履歴つきの系列になる。"""
        from tinyai.dialog import DialogStore
        with tempfile.TemporaryDirectory() as tmp:
            from pathlib import Path
            from tinyai.neural_lm import NeuralLM
            nl = NeuralLM(Path(tmp), size="small")
            nl.min_sentences, nl.min_chars = 4, 40
            nl.ensure_model(["こんにちは。今日はいい天気です。散歩に行きましょう。%d" % i for i in range(60)])
            d = DialogStore(10)
            d.add("それで?", "続きはこうです。", history=[("宇宙の話", "宇宙は広いです。")])
            plain = len(nl.seq_dialog("それで?", "続きはこうです。"))
            item = d.pairs[-1]
            with_hist = len(nl.seq_dialog(item[0], item[1], history=item[4]))
            self.assertGreater(with_hist, plain)                     # 履歴の分だけ長い系列になる


class ReplayReservoirTest(unittest.TestCase):
    """再生バッファの入れ替え: 一部を貯水池抽出にして古い分布を残す。"""

    def test_reservoir_keeps_old_sequences(self):
        import numpy as np
        from tinyai.neural import SequencePool
        cap, total = 400, 4000
        share = SequencePool.RESERVOIR_SHARE
        try:
            SequencePool.RESERVOIR_SHARE = 0.0
            fifo = SequencePool(cap, seed=1)
            SequencePool.RESERVOIR_SHARE = 0.25
            res = SequencePool(cap, seed=1)
            for i in range(total):
                ids = np.array([8, 9, 10, 11], dtype=np.int32)
                fifo.add(ids, weight=float(i))
                res.add(ids, weight=float(i))
            self.assertEqual(len(fifo.items), cap)
            self.assertEqual(len(res.items), cap)
            old_fifo = sum(1 for it in fifo.items if it[1] < total * 0.25)
            old_res = sum(1 for it in res.items if it[1] < total * 0.25)
            self.assertEqual(old_fifo, 0)              # 先入れ先出しは古い系列を全部押し出す
            self.assertGreater(old_res, cap * 0.02)    # 貯水池には古い時期の系列が残る
            newest = max(it[1] for it in res.items)
            self.assertGreaterEqual(newest, total - 10)  # 新しい系列にもきちんと追従する
        finally:
            SequencePool.RESERVOIR_SHARE = share

    def test_recent_holdout_is_separate(self):
        """入れ替わる取り置きは学習に使わず、固定の取り置きとは別に持つ。"""
        from pathlib import Path
        from tinyai.neural_lm import NeuralLM
        with tempfile.TemporaryDirectory() as tmp:
            nl = NeuralLM(Path(tmp), size="small")
            nl.min_sentences, nl.min_chars = 4, 40
            nl.ensure_model(["こんにちは。今日はいい天気です。散歩に行きましょう。%d" % i for i in range(60)])
            nl._holdout = [[8, 9, 10, 11]] * 300        # 固定分は満杯にしておく
            before = len(nl.pool)
            for i in range(3000):
                nl.add_text("これは取り置きの確認のための文です。番号は %d 番。" % i)
            self.assertGreater(len(nl._holdout_recent), 0)
            self.assertLessEqual(len(nl._holdout_recent), 150)
            self.assertEqual(len(nl._holdout), 300)     # 固定分は増えない
            self.assertGreater(len(nl.pool), before)


class PoolPersistenceTest(unittest.TestCase):
    """再生バッファを保存して復元する (再起動のたびに作り直さない)。"""

    def test_pool_round_trip(self):
        import numpy as np
        from pathlib import Path
        from tinyai.neural import SequencePool
        with tempfile.TemporaryDirectory() as tmp:
            p = SequencePool(300, seed=2)
            for i in range(700):
                p.add(np.array([8, 9, 10, 11, 12], dtype=np.int32), weight=float(i), loss_from=2,
                      kind="dialog" if i % 3 else "text")
            f = Path(tmp) / "pool.npz"
            p.save(f)
            q = SequencePool(300, seed=2)
            self.assertEqual(q.load(f), len(p))
            self.assertEqual(q.kind_counts, p.kind_counts)
            self.assertEqual(q.total_tokens, p.total_tokens)
            self.assertEqual(q.seen, p.seen)                 # 貯水池抽出の確率は見た総数で決まる
            x, y, w = q.batch(2, 16)
            self.assertEqual(x.shape, (2, 16))
            self.assertEqual(w.shape, (2, 16))
            self.assertEqual(SequencePool(300, seed=2).load(Path(tmp) / "無い.npz"), 0)

    def test_neural_lm_keeps_pool_across_restart(self):
        from pathlib import Path
        from tinyai.neural_lm import NeuralLM
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            nl = NeuralLM(d, size="small")
            nl.min_sentences, nl.min_chars = 4, 40
            nl.ensure_model(["こんにちは。今日はいい天気です。散歩に行きましょう。%d" % i for i in range(60)])
            for i in range(200):
                nl.add_text("学習用の文です。番号は %d 番になります。" % i)
            before = len(nl.pool)
            self.assertGreater(before, 50)
            nl.save()
            nl2 = NeuralLM(d, size="small")
            nl2.ensure_model()
            self.assertEqual(len(nl2.pool), before)          # 再開しても同じ系列が残っている


class GrowthTest(unittest.TestCase):
    """成長の判断: 損失が停滞したら層を増やす。悪化中は増やさない。"""

    def _lm(self, tmp):
        from pathlib import Path
        from tinyai.neural_lm import NeuralLM
        nl = NeuralLM(Path(tmp), size="small")
        nl.min_sentences, nl.min_chars = 4, 40
        nl.ensure_model(["こんにちは。今日はいい天気です。散歩に行きましょう。%d" % i for i in range(60)])
        nl.loss_hist = [3.0] * 10 + [2.999] * 10        # 20 回分、ほぼ横ばい = 停滞
        return nl

    def test_grows_when_loss_plateaus(self):
        with tempfile.TemporaryDirectory() as tmp:
            nl = self._lm(tmp)
            layers = nl.model.L
            self.assertTrue(nl.maybe_grow(True))        # 停滞していれば成長する
            self.assertEqual(nl.model.L, layers + 1)

    def test_does_not_grow_while_generalisation_worsens(self):
        with tempfile.TemporaryDirectory() as tmp:
            nl = self._lm(tmp)
            nl.recent_hist = [10.0, 10.0, 20.0, 20.0]   # 今の分布でも悪化中 = 過学習
            layers = nl.model.L
            self.assertFalse(nl.maybe_grow(True))
            self.assertEqual(nl.model.L, layers)

    def test_growth_cost_is_small(self):
        """成長のメモリ費用はモデル自身の分だけ (データ側の圧力で止めない)。"""
        with tempfile.TemporaryDirectory() as tmp:
            nl = self._lm(tmp)
            cost = nl.growth_bytes()
            self.assertGreater(cost, 0)
            per_param = cost / max(nl.model.n_params(), 1)
            self.assertLess(per_param, 64)        # 1 パラメータあたり数十バイトの範囲に収まる

    def test_learning_rate_scales_with_depth(self):
        """層が増えたら学習率を下げる (プリセットの値はその層数で調整したもの)。"""
        with tempfile.TemporaryDirectory() as tmp:
            nl = self._lm(tmp)
            base = nl._depth_lr()
            self.assertEqual(base, nl.lr)                                  # プリセットの層数なら変えない
            nl.model.L += 4
            deeper = nl._depth_lr()
            self.assertLess(deeper, nl.lr)
            nl.model.L += 4
            self.assertLess(nl._depth_lr(), deeper)                        # さらに深ければさらに下げる

    def test_learning_rate_ramps_back_after_growth(self):
        """成長直後は学習率を下げ、少しずつ元に戻す。"""
        with tempfile.TemporaryDirectory() as tmp:
            nl = self._lm(tmp)
            self.assertEqual(nl._effective_lr(), nl._depth_lr())           # 成長前は深さ補正のみ
            need = nl.TOKENS_PER_PARAM * nl.model.n_params()
            self.assertTrue(nl.maybe_grow(True, data_tokens=need + 1))
            just_after = nl._effective_lr()
            self.assertLess(just_after, nl._depth_lr() * 0.5)              # 直後は大きく下げる
            nl.model.step = nl._last_grow_step + nl.GROW_WARMUP // 2
            mid = nl._effective_lr()
            self.assertGreater(mid, just_after)                            # 少しずつ戻る
            nl.model.step = nl._last_grow_step + nl.GROW_WARMUP
            self.assertEqual(nl._effective_lr(), nl._depth_lr())           # 馴染んだら深さ補正のみに戻る

    def test_quality_history_survives_restart(self):
        """品質の履歴は保存する (10 分ごとに再開する運用で規則が発火しないのを防ぐ)。"""
        from pathlib import Path
        from tinyai.neural_lm import NeuralLM
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            nl = NeuralLM(d, size="small")
            nl.min_sentences, nl.min_chars = 4, 40
            nl.ensure_model(["こんにちは。今日はいい天気です。散歩に行きましょう。%d" % i for i in range(60)])
            nl.dialog_hist = [75.0, 74.0, 76.0, 75.0, 110.0, 115.0]
            nl.recent_hist = [20.0, 21.0]
            nl.lr_scale = 0.5
            nl.save()
            nl2 = NeuralLM(d, size="small")
            nl2.ensure_model()
            self.assertEqual(len(nl2.dialog_hist), 6)
            self.assertEqual(nl2.recent_hist[-1], 21.0)
            self.assertEqual(nl2.lr_scale, 0.5)

    def test_learning_rate_is_damped_when_dialogue_keeps_falling(self):
        """会話の質が続けて落ちたら学習率を自動で下げる。"""
        with tempfile.TemporaryDirectory() as tmp:
            nl = self._lm(tmp)
            nl.model.step = 5000
            nl.dialog_hist = [75.0, 74.0, 76.0, 75.0, 110.0, 115.0]     # 直近 2 回が大きく悪化
            before = nl._effective_lr()
            self.assertTrue(nl.maybe_damp_lr())
            self.assertLess(nl._effective_lr(), before)
            self.assertFalse(nl.maybe_damp_lr())                        # 続けては下げない
            nl.model.step += 1500
            self.assertTrue(nl.maybe_damp_lr())                         # 間隔を空ければまた下げる

    def test_learning_rate_is_kept_when_dialogue_is_stable(self):
        with tempfile.TemporaryDirectory() as tmp:
            nl = self._lm(tmp)
            nl.model.step = 5000
            nl.dialog_hist = [75.0, 74.0, 76.0, 75.0, 74.0, 73.0]       # 安定している
            before = nl._effective_lr()
            self.assertFalse(nl.maybe_damp_lr())
            self.assertEqual(nl._effective_lr(), before)

    def test_growth_stops_when_dialogue_gets_worse(self):
        """平文の指標が良くても、会話の質が落ちていれば成長させない。"""
        with tempfile.TemporaryDirectory() as tmp:
            nl = self._lm(tmp)
            nl.recent_hist = [20.0, 20.0, 19.5, 19.0]                 # 平文は改善中
            nl.dialog_hist = [75.0, 80.0, 100.0, 110.0]               # 会話は悪化中
            need = nl.TOKENS_PER_PARAM * nl.model.n_params()
            layers = nl.model.L
            self.assertFalse(nl.maybe_grow(True, data_tokens=need * 10))
            self.assertEqual(nl.model.L, layers)
            nl.dialog_hist = [110.0, 100.0, 80.0, 75.0]               # 会話も改善したら成長できる
            self.assertTrue(nl.maybe_grow(True, data_tokens=need * 10))

    def test_rollback_also_looks_at_dialogue(self):
        with tempfile.TemporaryDirectory() as tmp:
            nl = self._lm(tmp)
            nl.recent_ppl = 20.0
            nl.dialog_hist = [70.0, 70.0, 70.0, 70.0]
            need = nl.TOKENS_PER_PARAM * nl.model.n_params()
            layers = nl.model.L
            self.assertTrue(nl.maybe_grow(True, data_tokens=need + 1))
            nl.model.step = nl._last_grow_step + nl.GROW_COOLDOWN
            nl.recent_ppl = 20.0                                       # 平文は変わらず
            nl.dialog_hist.append(70.0 * nl.GROW_ROLLBACK_RATIO + 1)   # 会話だけ大きく悪化
            self.assertTrue(nl.check_growth())
            self.assertEqual(nl.model.L, layers)

    def test_bad_growth_is_rolled_back(self):
        """成長後に大きく悪化していたら、成長前の重みに戻す。"""
        with tempfile.TemporaryDirectory() as tmp:
            nl = self._lm(tmp)
            nl.recent_ppl = 20.0
            need = nl.TOKENS_PER_PARAM * nl.model.n_params()
            layers = nl.model.L
            self.assertTrue(nl.maybe_grow(True, data_tokens=need + 1))
            self.assertEqual(nl.model.L, layers + 1)
            self.assertTrue(nl._pregrow_path.exists())              # 成長前の重みを取ってある
            nl.model.step = nl._last_grow_step + nl.GROW_COOLDOWN
            nl.recent_ppl = 20.0 * nl.GROW_ROLLBACK_RATIO + 1       # 明らかに悪化
            self.assertTrue(nl.check_growth())
            self.assertEqual(nl.model.L, layers)                    # 元の層数に戻る

    def test_good_growth_is_kept(self):
        with tempfile.TemporaryDirectory() as tmp:
            nl = self._lm(tmp)
            nl.recent_ppl = 20.0
            need = nl.TOKENS_PER_PARAM * nl.model.n_params()
            self.assertTrue(nl.maybe_grow(True, data_tokens=need + 1))
            layers = nl.model.L
            nl.model.step = nl._last_grow_step + nl.GROW_COOLDOWN
            nl.recent_ppl = 19.0                                    # 良くなっている
            self.assertFalse(nl.check_growth())
            self.assertEqual(nl.model.L, layers)

    def test_growth_has_a_cooldown(self):
        """一度成長したら、しばらくは次の成長を待つ (増やす→悪化→また増やす の悪循環を避ける)。"""
        with tempfile.TemporaryDirectory() as tmp:
            nl = self._lm(tmp)
            need = nl.TOKENS_PER_PARAM * nl.model.n_params()
            self.assertTrue(nl.maybe_grow(True, data_tokens=need + 1))     # 1 回目は成長する
            layers = nl.model.L
            nl.loss_hist = [3.0] * 10 + [2.999] * 10
            self.assertFalse(nl.maybe_grow(True, data_tokens=need * 10))   # 直後は成長しない
            self.assertEqual(nl.model.L, layers)
            nl.model.step = nl._last_grow_step + nl.GROW_COOLDOWN          # 十分に回した後なら成長する
            self.assertTrue(nl.maybe_grow(True, data_tokens=need * 10))
            self.assertEqual(nl.model.L, layers + 1)

    def test_grows_when_data_outgrows_capacity(self):
        """損失が下がり続けていても、データ量が容量に対して多すぎれば先回りして大きくする。"""
        with tempfile.TemporaryDirectory() as tmp:
            nl = self._lm(tmp)
            nl.loss_hist = [3.0 - i * 0.05 for i in range(20)]       # まだ順調に下がっている = 停滞ではない
            layers = nl.model.L
            self.assertFalse(nl.maybe_grow(True))                     # データ量を渡さなければ成長しない
            need = nl.TOKENS_PER_PARAM * nl.model.n_params()
            self.assertTrue(nl.maybe_grow(True, data_tokens=need + 1))
            self.assertEqual(nl.model.L, layers + 1)

    def test_grows_when_more_data_stops_helping(self):
        """データは増え続けているのに今の分布での ppl が良くならない = 容量不足とみなす。"""
        with tempfile.TemporaryDirectory() as tmp:
            nl = self._lm(tmp)
            nl.loss_hist = [3.0 - i * 0.05 for i in range(20)]        # 停滞ではない
            nl.recent_hist = [20.0, 20.0, 21.0, 21.5]                 # 少しずつ悪化 (25% 未満)
            soft = nl.TOKENS_PER_PARAM_SOFT * nl.model.n_params()
            layers = nl.model.L
            self.assertFalse(nl.maybe_grow(True, data_tokens=soft // 2))   # データがまだ少なければ成長しない
            self.assertTrue(nl.maybe_grow(True, data_tokens=soft + 1))
            self.assertEqual(nl.model.L, layers + 1)

    def test_no_growth_when_data_is_still_helping(self):
        with tempfile.TemporaryDirectory() as tmp:
            nl = self._lm(tmp)
            nl.loss_hist = [3.0 - i * 0.05 for i in range(20)]
            nl.recent_hist = [22.0, 21.0, 20.0, 19.0]                 # 良くなっている = まだ余地がある
            soft = nl.TOKENS_PER_PARAM_SOFT * nl.model.n_params()
            layers = nl.model.L
            self.assertFalse(nl.maybe_grow(True, data_tokens=soft * 2))
            self.assertEqual(nl.model.L, layers)

    def test_data_rich_growth_still_respects_the_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            nl = self._lm(tmp)
            nl.recent_hist = [10.0, 10.0, 20.0, 20.0]                 # 急激に悪化 (2 倍) = 学習が不安定
            need = nl.TOKENS_PER_PARAM * nl.model.n_params()
            layers = nl.model.L
            self.assertFalse(nl.maybe_grow(True, data_tokens=need * 10))
            self.assertEqual(nl.model.L, layers)

    def test_forgetting_alone_does_not_block_growth(self):
        """固定の取り置きだけが悪化 (= 忘却) なら、容量を増やす判断は止めない。"""
        with tempfile.TemporaryDirectory() as tmp:
            nl = self._lm(tmp)
            nl.ppl_hist = [10.0, 10.0, 30.0, 30.0]      # 昔の文は忘れている
            nl.recent_hist = [10.0, 10.0, 9.8, 9.7]     # 今の分布では悪化していない
            layers = nl.model.L
            self.assertTrue(nl.maybe_grow(True))
            self.assertEqual(nl.model.L, layers + 1)


class PoolMixTest(unittest.TestCase):
    """再生バッファの種類の混ざり具合: 会話に偏らせない。"""

    def test_dialog_share_is_capped(self):
        import numpy as np
        from tinyai.neural import SequencePool
        p = SequencePool(1000, seed=0)
        mix = ["dialog"] * 70 + ["text"] * 20 + ["copy"] * 7 + ["qa"] * 3   # 収集は会話に偏る
        for i in range(12000):
            p.add(np.array([8, 9, 10, 11, 12], dtype=np.int32), kind=mix[i % len(mix)])
        n = len(p.items)
        self.assertEqual(n, 1000)
        share = {k: v / n for k, v in p.kind_counts.items()}
        self.assertLessEqual(share["dialog"], 0.60)        # 会話は上限で頭打ちになる
        self.assertGreater(share.get("text", 0), 0.15)     # 平文の居場所が残る

    def test_rebalances_an_existing_dialog_heavy_pool(self):
        """会話に偏ったバッファでも、平文を流し込めば比率が戻る。"""
        import numpy as np
        from tinyai.neural import SequencePool
        p = SequencePool(500, seed=1)
        ids = np.array([8, 9, 10, 11, 12], dtype=np.int32)
        p.max_share = {}                                   # 上限を入れる前に貯めたバッファを模す
        for _ in range(500):
            p.add(ids, kind="dialog")
        self.assertEqual(p.kind_counts["dialog"], 500)
        p.max_share = SequencePool(1).max_share             # 以後は上限つきで運用する
        for _ in range(2000):
            p.add(ids, kind="text")
        self.assertGreater(p.kind_counts.get("text", 0) / len(p.items), 0.3)


class PoolCapacityTest(unittest.TestCase):
    """再生バッファの容量はメモリに比例し、手持ちの記憶で埋める。"""

    def test_capacity_scales_with_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            small = make_brain(tmp, memory_mb=200)
            self.assertEqual(small.neural.pool.capacity, 30000)        # 下限は 3 万
        with tempfile.TemporaryDirectory() as tmp:
            big = make_brain(tmp, memory_mb=1024)
            self.assertEqual(big.neural.pool.capacity, 102400)         # 1 MB あたり 100 系列

    def test_refill_makes_long_text_sequences(self):
        """補充した平文の系列が長いこと (1 文ずつだと段落の流れを学べない)。"""
        try:
            from tinyai import neural as nn
        except Exception:
            return
        if not nn.available():
            return
        with tempfile.TemporaryDirectory() as tmp:
            b = make_brain(tmp, memory_mb=200)
            b.neural.min_sentences, b.neural.min_chars, b.neural.size = 10, 100, "base"   # 文脈長 256
            from tinyai.neural import SequencePool
            b.neural.pool = SequencePool(600, seed=1)
            b.learn_text("\n".join("第 %d 文です。これは十分な長さを持った説明文であり、続きものとして読めます。" % i
                                    for i in range(10, 400)), "https://x/nn")
            b.neural_step(budget_seconds=0.1)
            lens = [len(it[0]) for i, it in enumerate(b.neural.pool.items) if b.neural.pool.kinds[i] == "text"]
            self.assertTrue(lens)
            self.assertGreater(sum(lens) / len(lens), 80)     # 1 文だけの短い系列ばかりではない

    def test_pool_is_filled_from_stored_memory(self):
        try:
            from tinyai import neural as nn
        except Exception:
            return
        if not nn.available():
            return
        with tempfile.TemporaryDirectory() as tmp:
            b = make_brain(tmp, memory_mb=200)
            b.neural.min_sentences, b.neural.min_chars, b.neural.size = 10, 100, "small"
            from tinyai.neural import SequencePool
            b.neural.pool = SequencePool(400, seed=1)                   # 小さくして埋まり方を見る
            b.learn_text("\n".join("文 %d は学習用の文章であり、内容は番号 %d に関する説明です。" % (i, i) for i in range(10, 300)), "https://x/nn")
            b.neural_step(budget_seconds=0.1)
            self.assertGreater(len(b.neural.pool), 50)                  # 手持ちの知識文で埋まる


class TokenCorpusTest(unittest.TestCase):
    """ディスクの追記型コーパス: RAM に入りきらない分の学習トークンを貯めて循環させる。"""

    def test_ring_keeps_newest_and_survives_restart(self):
        import numpy as np
        from pathlib import Path
        from tinyai.neural import TokenCorpus
        with tempfile.TemporaryDirectory() as tmp:
            c = TokenCorpus(Path(tmp) / "corpus.bin", max_tokens=1000)
            for i in range(300):
                c.append(np.arange(8, 18, dtype=np.int32) + i, loss_from=3, kind="dialog")
            self.assertEqual(c.tokens, 1000)                 # 容量までしか持たない
            self.assertEqual(c.written, 3000)                # 書いた総数は数え続ける
            self.assertEqual(len(c), 100)
            got = c.sample(5, np.random.default_rng(0))
            self.assertEqual(len(got), 5)
            for ids, lf, kind in got:
                self.assertEqual(len(ids), 10)
                self.assertEqual(lf, 3)
                self.assertEqual(kind, "dialog")
                self.assertGreaterEqual(int(ids[0]), 8 + 200)   # 上書きされた古い分は残っていない
            c.save()
            d = TokenCorpus(Path(tmp) / "corpus.bin", max_tokens=1000)
            self.assertEqual(len(d), len(c))
            self.assertEqual(d.sample(1, np.random.default_rng(0))[0][2], "dialog")

    def test_capacity_change_keeps_what_fits(self):
        """容量を変えても貯めたトークンを捨てない (増やす時はそのまま、減らす時は収まる分だけ)。"""
        import numpy as np
        from pathlib import Path
        from tinyai.neural import TokenCorpus
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "corpus.bin"
            c = TokenCorpus(f, max_tokens=1000)
            for i in range(50):
                c.append(np.arange(8, 18, dtype=np.int32) + i)
            c.save()
            big = TokenCorpus(f, max_tokens=4000)
            self.assertEqual(len(big), 50)                     # 増やした時は索引をそのまま使える
            for i in range(50):
                big.append(np.arange(8, 18, dtype=np.int32) + i + 100)
            big.save()
            self.assertEqual(big.tokens, 1000)
            small = TokenCorpus(f, max_tokens=600)
            self.assertEqual(small.tokens, 600)                # 減らした時は収まる分だけ残る
            self.assertEqual(len(small.sample(1)[0][0]), 10)   # 残った系列はちゃんと読める

    def test_prefer_long_sampling(self):
        """長い系列を優先して引ける (段落の流れを学ぶ系列をバッファに増やすため)。"""
        import numpy as np
        from pathlib import Path
        from tinyai.neural import TokenCorpus
        with tempfile.TemporaryDirectory() as tmp:
            c = TokenCorpus(Path(tmp) / "c.bin", max_tokens=200_000)
            for i in range(2000):
                c.append(np.full(20 if i % 2 else 200, 9, dtype=np.int32))
            plain = [len(x[0]) for x in c.sample(100, np.random.default_rng(1))]
            longer = [len(x[0]) for x in c.sample(100, np.random.default_rng(1), prefer_long=True)]
            self.assertGreater(sum(longer) / len(longer), sum(plain) / len(plain) * 1.3)

    def test_refresh_from_corpus_feeds_the_pool(self):
        from pathlib import Path
        from tinyai.neural_lm import NeuralLM
        with tempfile.TemporaryDirectory() as tmp:
            nl = NeuralLM(Path(tmp), size="small")
            nl.min_sentences, nl.min_chars = 4, 40
            nl.ensure_model(["こんにちは。今日はいい天気です。散歩に行きましょう。%d" % i for i in range(60)])
            for i in range(300):
                nl.add_text("学習用の文です。番号は %d 番になります。" % i)
            self.assertGreater(len(nl.corpus), 100)          # 読んだ文はコーパスにも入る
            n = nl.refresh_from_corpus(50)
            self.assertGreater(n, 0)


class SpanLearningTest(unittest.TestCase):
    """ニューラル LM には連続した文をまとめて渡す (段落の流れを学ぶため)。"""

    def test_sentences_are_grouped_into_spans(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = make_brain(tmp)
            b._neural_pending_text.clear()
            text = "。".join("これは文書の第 %d 文であり、続きものとして読めるように書かれています" % i
                             for i in range(1, 41)) + "。"
            b.learn_text(text, "https://example.com/doc")
            spans = list(b._neural_pending_text)
            self.assertTrue(spans)
            avg = sum(len(x) for x in spans) / len(spans)
            self.assertGreater(avg, 150)                 # 1 文ずつ (40〜90 文字) より明らかに長い
            first = spans[0]                              # 取り込まれた文が連続して同じ系列に入る
            nums = [int(m) for m in re.findall(r"第 (\d+) 文", first)]
            self.assertGreater(len(nums), 3)
            self.assertEqual(nums, sorted(nums))          # 出てきた順のまま (並べ替えや欠落ではない)


class BulkCorpusTest(unittest.TestCase):
    """ページ本文を丸ごとディスクのコーパスへ流し込む (知識ベースを太らせずに学習量を増やす)。"""

    def test_page_text_goes_into_the_corpus(self):
        from pathlib import Path
        from tinyai.neural_lm import NeuralLM
        with tempfile.TemporaryDirectory() as tmp:
            nl = NeuralLM(Path(tmp), size="base", corpus_tokens=500_000)
            nl.min_sentences, nl.min_chars = 4, 40
            nl.ensure_model(["これは学習用の文章です。番号は %d 番です。" % i for i in range(200)])
            page = "\n\n".join("段落 %d です。" % i + "この段落には十分な長さの文章が入っており、続きものとして意味が通ります。" * 3
                                 for i in range(40))
            added = nl.add_corpus_text(page)
            self.assertGreater(added, 1000)                    # ページ 1 枚で数千トークン入る
            self.assertGreater(len(nl.corpus), 10)
            ids, lf, kind = nl.corpus.sample(1)[0]
            self.assertEqual(kind, "text")
            self.assertLessEqual(len(ids), nl.model.T)         # 文脈長を超える系列は作らない
            self.assertEqual(nl.add_corpus_text(""), 0)


class TextQualityTest(unittest.TestCase):
    """コーパスに入れる地の文の品質判定 (目次・数字の羅列・欠損値を落とす)。"""

    def test_good_prose(self):
        from tinyai.textquality import good_prose
        self.assertTrue(good_prose("1960年代には、ビートルズやローリング・ストーンズなど有名なバンドが登場しました。"))
        self.assertTrue(good_prose("This is an ordinary English paragraph. It has several sentences and reads naturally."))
        self.assertFalse(good_prose("ホーム 会社概要 採用情報 お問い合わせ サイトマップ プライバシーポリシー 利用規約"))
        self.assertFalse(good_prose("2020 1,234 5,678 9,012 3,456 7,890 1,234 5,678 9,012 3,456 7,890 1,234"))
        self.assertFalse(good_prose("短い"))

    def test_clean_field_drops_missing_values(self):
        from tinyai.textquality import clean_field
        self.assertEqual(clean_field("nan"), "")          # 欠損値が文字列で流れてくる
        self.assertEqual(clean_field("NaN "), "")
        self.assertEqual(clean_field(None), "")
        self.assertEqual(clean_field(" 本文 "), "本文")

    def test_corpus_skips_low_quality_paragraphs(self):
        from pathlib import Path
        from tinyai.neural_lm import NeuralLM
        with tempfile.TemporaryDirectory() as tmp:
            nl = NeuralLM(Path(tmp), size="base", corpus_tokens=200_000)
            nl.min_sentences, nl.min_chars = 4, 40
            nl.ensure_model(["これは学習用の文章です。番号は %d 番です。" % i for i in range(200)])
            junk = "\n\n".join("ホーム 会社概要 採用情報 お問い合わせ サイトマップ 利用規約 English 日本語" for _ in range(20))
            self.assertEqual(nl.add_corpus_text(junk), 0)
            good = "\n\n".join("これは意味のある段落です。日本語の文章として読めるように、読点も文末も入っています。" * 2
                                 for _ in range(10))
            self.assertGreater(nl.add_corpus_text(good), 0)


class VocabGrowthTest(unittest.TestCase):
    """語彙の追加候補は、実際に減るトークン数で順位を付ける。"""

    def test_candidates_ranked_by_real_savings(self):
        from tinyai.bpe import SubwordTokenizer
        texts = ["化学の実験をしました。化学の授業は化学室で行います。" for _ in range(10)]
        tok = SubwordTokenizer.train(["あいうえお。かきくけこ。化学。実験。授業。"], size=200)
        cand = tok.frequent_new_units(texts, top=10, min_count=3)
        self.assertTrue(cand)
        before = sum(len(tok.encode(t)) for t in texts)
        tok.add_tokens(cand)
        after = sum(len(tok.encode(t)) for t in texts)
        self.assertLess(after, before)                    # 追加すると必ずトークン数が減る

    def test_already_efficient_units_are_not_proposed(self):
        """1 トークンで表せている単位は候補にしない (削減がゼロ)。"""
        from tinyai.bpe import SubwordTokenizer
        tok = SubwordTokenizer.train(["こんにちは。こんにちは。こんにちは。" for _ in range(5)], size=200)
        cand = tok.frequent_new_units(["こんにちは。" * 10], top=20, min_count=3)
        for c in cand:
            self.assertGreater(len(tok.encode(c)), 1)


class GrowthLogTest(unittest.TestCase):
    """学習ログから進化の記録を取り出す (UI で学習曲線の上に印を出すため)。"""

    def test_parses_growth_events(self):
        from pathlib import Path
        from tinyai.export import history_from_logs
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "train.log"
            log.write_text(
                "step 100 loss 2.000 7000 tok/s  {'neural_ppl': 30.0}  経過 1.0 分\n"
                "2026-01-01 00:00:00 tinyai.neural INFO ニューラル LM: 層を追加 -> 7 層 (4306752 params)\n"
                "step 200 loss 1.900 7000 tok/s  {'neural_ppl': 29.0}  経過 2.0 分\n"
                "2026-01-01 00:01:00 tinyai.neural INFO ニューラル LM: 中間次元を拡張 -> ff=640 (5898432 params)\n"
                "2026-01-01 00:02:00 tinyai.neural WARNING 成長が裏目に出たため取り消し (ppl 20.0 -> 40.0、6 層へ戻す)\n",
                encoding="utf-8")
            h = history_from_logs([log])
            kinds = [g["kind"] for g in h["growth_log"]]
            self.assertEqual(kinds, ["layer", "width", "rollback"])
            self.assertEqual(h["growth_log"][0]["value"], 7)
            self.assertEqual(h["growth_log"][0]["step"], 100)     # 直前の step に結び付ける
            self.assertEqual(h["growth_log"][1]["value"], 640)
            self.assertEqual(h["growth_log"][2]["value"], 6)


class DialogBpcTest(unittest.TestCase):
    """会話の質は 1 文字あたりのビット数でも測る (語彙を変えても比較できるように)。"""

    def test_self_evaluate_reports_bpc(self):
        try:
            from tinyai import neural as nn
        except Exception:
            return
        if not nn.available():
            return
        with tempfile.TemporaryDirectory() as tmp:
            b = make_brain(tmp)
            b.neural.min_sentences, b.neural.min_chars, b.neural.size = 10, 100, "small"
            b.learn_text("\n".join("サンプル文 %d は学習用の文章です。内容は番号 %d の説明です。" % (i, i)
                                    for i in range(10, 120)), "https://x/nn")
            for i in range(40):
                b.dialogs.add("質問 %d は何ですか" % i, "答え %d はこうです。理由も添えて説明します。" % i)
            b.neural_step(budget_seconds=0.2)
            r = b.self_evaluate(n_docs=4, n_dialogs=12)
            if "dialog_ppl" not in r:
                return
            self.assertIn("dialog_bpc", r)
            self.assertGreater(r["dialog_bpc"], 0)
            self.assertLess(r["dialog_bpc"], 20)                  # 1 文字 20 ビットは超えない
            self.assertTrue(b.neural.dialog_hist)                 # 規則が使う履歴に入る
            if "dialog_gain_fresh" in r:
                self.assertGreater(r["dialog_bpc_fresh"], 0)      # 最近の会話でも測る
                self.assertLessEqual(r["dialog_gain_fresh"], 1.0)
                again = b.self_evaluate(n_docs=4, n_dialogs=12)   # 続けて測っても同じ値 (比べられる)
                self.assertEqual(again["dialog_bpc_fresh"], r["dialog_bpc_fresh"])
            if "dialog_bpc_unigram" in r:
                self.assertGreater(r["dialog_bpc_unigram"], 0)
                self.assertAlmostEqual(r["dialog_bpc_gain"],
                                       1 - r["dialog_bpc"] / r["dialog_bpc_unigram"], places=2)


class TermOveruseTest(unittest.TestCase):
    """同じ語を持ち出しすぎる候補を落とす (自己矛盾した応答の抑制)。"""

    def test_detects_overuse(self):
        from tinyai.brain import Brain
        bad = "犬と猫はどちらも犬よりも大きく、犬は猫と犬を飼うことができます。犬の世話は犬に向いています。"
        self.assertGreaterEqual(Brain._term_overuse(bad), 1.0)
        for good in ("犬と猫はどちらも人気のあるペットです。性格や世話の手間が違うので、生活に合う方を選ぶとよいでしょう。",
                     "富士山は静岡県と山梨県にまたがる日本最高峰の火山で、標高は3776メートルです。",
                     "機械学習とは、データから規則を見つける技術です。統計や最適化の考え方を使います。"):
            self.assertEqual(Brain._term_overuse(good), 0.0)

    def test_short_replies_are_not_penalised(self):
        from tinyai.brain import Brain
        self.assertEqual(Brain._term_overuse("はい、そうです。"), 0.0)
        self.assertEqual(Brain._term_overuse("犬です。"), 0.0)


if __name__ == "__main__":
    unittest.main()
