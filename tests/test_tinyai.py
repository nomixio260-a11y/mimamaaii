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
from tinyai.tokenizer import detokenize, is_question, keywords, split_sentences, terms, tokenize
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


class KnowledgeTest(unittest.TestCase):
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
        for t, post in kb.index.items():
            for doc_id in post:
                self.assertIn(doc_id, kb.docs)


class BrainTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.brain = make_brain(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_recall_from_seed(self):
        r = self.brain.reply("日本の首都は？")
        self.assertEqual(r.mode, "recall")
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
        b2 = Brain(self.brain.cfg)
        self.assertTrue(b2.load(path))
        self.assertEqual(len(b2.kb), len(self.brain.kb))
        self.assertEqual(b2.lm.entries, self.brain.lm.entries)
        self.assertIn("合言葉", b2.reply("アルファゼータとは？").text)

    def test_memory_budget_enforced(self):
        b = make_brain(self.tmp.name, memory_mb=40)
        budget = b.guard.budget
        rng = random.Random(0)
        words = [f"語{i}" for i in range(3000)]
        for i in range(400):
            para = "\n".join(" ".join(rng.choice(words) for _ in range(12)) + "。" for _ in range(25))
            b.learn_text(para, source=f"https://example.org/{i}")
        b.enforce_memory()
        self.assertLessEqual(b.lm.estimated_bytes(), budget * 0.55 + 1)
        self.assertLessEqual(b.kb.estimated_bytes(), budget * 0.35 + 1)
        self.assertGreater(b.stats["pruned_lm"] + b.stats["pruned_kb"], 0)
        self.assertLess(rss_bytes(), b.guard.limit * 3)  # 極端な超過はしない


class EvolverTest(unittest.TestCase):
    def test_cycle_offline_with_fake_fetcher(self):
        class FakeFetcher:
            fetched = failed = 0

            def search_and_read(self, query, langs=(), max_pages=3):
                self.fetched += 1
                return [(f"https://fake/{query}", f"{query}についての説明文です。{query}はテスト用の話題です。")]

        with tempfile.TemporaryDirectory() as tmp:
            b = make_brain(tmp)
            b.add_gap("架空話題")
            inbox = Path(tmp) / "inbox"
            inbox.mkdir()
            (inbox / "note.txt").write_text("インボックスの文章は自動で取り込まれます。", encoding="utf-8")
            ev = Evolver(b, fetcher=FakeFetcher(), interval=0, max_cycles=3)
            ev.run()
            self.assertEqual(ev.cycles, 3)
            self.assertIn("架空話題", b.explored)
            self.assertFalse((inbox / "note.txt").exists())
            self.assertTrue((Path(tmp) / "learned" / "note.txt").exists())
            self.assertIn("架空話題", b.reply("架空話題とは？").text)
            self.assertTrue((Path(tmp) / "brain.pkl.gz").exists())

    def test_thread_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = make_brain(tmp)
            ev = Evolver(b, fetcher=None, interval=0.05)
            ev.start()
            threading.Event().wait(0.3)
            ev.stop()
            self.assertFalse(ev.is_alive())
            self.assertGreater(ev.cycles, 0)


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
