"""Traditional-NLP stage that runs BEFORE the LLM: sentiment, category, entities."""
from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from model_training import get_category_model, predict_category, vader_sentiment
from preprocess import build_unique_articles, load_data

INSTITUTIONS = [
    "Federal Reserve", "European Central Bank", "ECB", "Bank of Japan", "Bank of England",
    "IMF", "World Bank", "OPEC", "SEC",
]


def build_entity_terms(df: pd.DataFrame) -> dict:
    """Gazetteer for lightweight named-entity recognition, built from dataset columns."""
    return {
        "companies": sorted(df["Related_Company"].dropna().unique()),
        "indices": sorted(df["Market_Index"].dropna().unique()),
        "institutions": INSTITUTIONS,
    }


def find_entities(text: str, terms: dict) -> dict:
    found = {}
    for kind, names in terms.items():
        hits = []
        for name in names:
            if re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", text, flags=re.IGNORECASE):
                hits.append(name)
        found[kind] = hits
    return found


@dataclass
class NewsAnalyzer:
    category_model: object
    entity_terms: dict

    def analyze(self, text: str) -> dict:
        text = (text or "").strip()
        if not text:
            raise ValueError("No text to analyze.")
        compound, label = vader_sentiment(text)
        return {
            "sentiment": label,
            "score": compound,
            **predict_category(text, self.category_model),
            "entities": find_entities(text, self.entity_terms),
            "word_count": len(text.split()),
        }


def build_analyzer():
    """Return (NewsAnalyzer, unique_articles_df) ready for the app and the LLM stage."""
    df = load_data()
    articles = build_unique_articles(df)
    model = get_category_model(articles)
    return NewsAnalyzer(model, build_entity_terms(df)), articles
