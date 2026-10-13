"""Smoke tests that need no GPU and no model download: the LLM is replaced by a stub.

    python -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import create_app  # noqa: E402
from llm import FinanceLLM, HeadlineRetriever, tidy_output, truncate_text  # noqa: E402
from model_training import label_from_compound  # noqa: E402
from pipeline import build_analyzer  # noqa: E402
from preprocess import assign_financial_category, match_category_rule  # noqa: E402

ANALYZER, ARTICLES = build_analyzer()


def stub_generate(messages, max_new_tokens, temperature):
    return "- stub answer\n- stub answer\n<|assistant|>"


class RuleTests(unittest.TestCase):
    def test_word_boundaries(self):
        self.assertIsNone(match_category_rule("Corporate separate generate"))  # no 'rate' inside words
        self.assertEqual(assign_financial_category("Central bank cuts rates"), "Banking & Central Banks")
        self.assertEqual(assign_financial_category("Housing data slows"), "Real Estate")
        self.assertEqual(assign_financial_category("Shares were flat"), "General Markets")

    def test_sentiment_thresholds(self):
        self.assertEqual(label_from_compound(0.3), "Positive")
        self.assertEqual(label_from_compound(-0.3), "Negative")
        self.assertEqual(label_from_compound(0.0), "Neutral")


class DataTests(unittest.TestCase):
    def test_unique_articles(self):
        self.assertEqual(ARTICLES["Headline"].is_unique, True)
        self.assertTrue(set(ARTICLES["Category"]) <= {
            "Real Estate", "Tech Stocks & Software", "Energy & Commodities",
            "Banking & Central Banks", "Trade & Macro Economy", "General Markets"})


class LLMHelperTests(unittest.TestCase):
    def test_tidy_removes_tokens_and_repeats(self):
        self.assertEqual(tidy_output("- a\n- a\n<|assistant|>"), "- a")

    def test_truncate(self):
        long = "Sentence one. " * 300
        self.assertLessEqual(len(truncate_text(long)), 1500)

    def test_retriever_grounds_answers(self):
        r = HeadlineRetriever(ARTICLES["Headline"])
        hits = r.top_k("What is happening with interest rates?", k=3)
        self.assertTrue(any("rate" in h.lower() for h in hits))
        self.assertEqual(r.top_k("zzzz qqqq", k=3), [])

    def test_no_context_skips_llm(self):
        llm = FinanceLLM(HeadlineRetriever(ARTICLES["Headline"]), generate_fn=lambda *a: 1 / 0)
        self.assertIn("could not find", llm.answer("zzzz qqqq"))


class AppTests(unittest.TestCase):
    def setUp(self):
        llm = FinanceLLM(HeadlineRetriever(ARTICLES["Headline"]), generate_fn=stub_generate)
        self.client = create_app(ANALYZER, ARTICLES, llm).test_client()
        self.text = "The Federal Reserve kept interest rates unchanged. Inflation remains elevated."

    def test_index(self):
        self.assertEqual(self.client.get("/").status_code, 200)

    def test_analyze_runs_nlp_first(self):
        data = self.client.post("/api/analyze", json={"text": self.text}).get_json()
        self.assertIn("Federal Reserve", data["analysis"]["entities"]["institutions"])
        self.assertEqual(data["analysis"]["category"], "Banking & Central Banks")

    def test_llm_endpoints_return_analysis_and_disclaimer(self):
        for path, extra in [("/api/summarize", {}), ("/api/insights", {}), ("/api/ask", {"question": "Outlook?"})]:
            r = self.client.post(path, json={"text": self.text, **extra})
            self.assertEqual(r.status_code, 200, path)
            body = r.get_json()
            self.assertTrue(body["result"] and body["analysis"] and body["disclaimer"], path)

    def test_validation_errors(self):
        self.assertEqual(self.client.post("/api/summarize", json={"text": ""}).status_code, 400)
        self.assertEqual(self.client.post("/api/ask", json={"text": self.text}).status_code, 400)
        self.assertEqual(self.client.post("/api/analyze", json={"text": "x" * 20001}).status_code, 400)

    def test_ask_without_article_uses_headlines(self):
        r = self.client.post("/api/ask", json={"question": "What about interest rates?"})
        self.assertEqual(r.status_code, 200)
        self.assertIsNone(r.get_json()["analysis"])


if __name__ == "__main__":
    unittest.main()
