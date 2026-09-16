"""python -m unittest discover -s tests  (pytest でも動く)"""
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
from tinyai.knowledge import is_junk
from tinyai.tokenizer import detokenize, is_phrase, is_question, keywords, split_sentences, terms, tokenize
from tinyai.web import html_to_text


def make_brain(tmp: str, **kw) -> Brain:
    cfg = Config(data_dir=Path(tmp), memory_mb=kw.pop("memory_mb", 128), hard_limit=False, web_enabled=False, seed=7, **kw)
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

    def test_learn_text_and_holdout(self):
        text = "\n".join(f"項目 {i} は第 {i} 番目の事実として記録されています。" for i in range(100))
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
            b.learn_text("東京タワーの高さは 333 メートルである！", "https://a/2")
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


if __name__ == "__main__":
    unittest.main()
