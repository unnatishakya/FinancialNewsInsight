"""Flask web app. Every request runs the NLP stage first, then the LLM stage.

    python app.py            ->  http://127.0.0.1:5000
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from llm import DISCLAIMER, FinanceLLM, HeadlineRetriever, LLMError
from pipeline import build_analyzer

ROOT = Path(__file__).resolve().parent
MAX_TEXT_CHARS = 20000


def load_curated():
    with open(ROOT / "data" / "curated_articles.json", encoding="utf-8") as f:
        return json.load(f)


def create_app(analyzer=None, articles=None, llm=None) -> Flask:
    app = Flask(__name__)

    if analyzer is None or articles is None:
        analyzer, articles = build_analyzer()
    if llm is None:
        llm = FinanceLLM(retriever=HeadlineRetriever(articles["Headline"]))
    curated = load_curated()

    def read_text():
        data = request.get_json(silent=True) or {}
        text = (data.get("text") or "").strip()
        if len(text) > MAX_TEXT_CHARS:
            raise ValueError(f"Text is too long (limit {MAX_TEXT_CHARS} characters).")
        return data, text

    def respond(run):
        """Shared flow: validate -> NLP analysis -> LLM call -> JSON."""
        try:
            data, text = read_text()
            analysis = analyzer.analyze(text) if text else None
            result = run(data, text, analysis)
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        except LLMError as exc:
            return jsonify(error=str(exc), analysis=analysis if text else None), 503
        return jsonify(analysis=analysis, result=result, disclaimer=DISCLAIMER)

    @app.get("/")
    def index():
        return render_template("index.html", curated=curated, disclaimer=DISCLAIMER)

    @app.post("/api/analyze")
    def analyze():
        def run(data, text, analysis):
            if not text:
                raise ValueError("Paste or select an article first.")
            return None
        return respond(run)

    @app.post("/api/summarize")
    def summarize():
        def run(data, text, analysis):
            if not text:
                raise ValueError("Paste or select an article to summarize.")
            return llm.summarize(text, analysis)
        return respond(run)

    @app.post("/api/ask")
    def ask():
        def run(data, text, analysis):
            question = (data.get("question") or "").strip()
            if not question:
                raise ValueError("Type a question first.")
            return llm.answer(question, text or None, analysis)
        return respond(run)

    @app.post("/api/insights")
    def insights():
        def run(data, text, analysis):
            return llm.insights(text or None, analysis)
        return respond(run)

    return app


if __name__ == "__main__":
    create_app().run(debug=os.getenv("FLASK_DEBUG") == "1", port=int(os.getenv("PORT", "5000")))
