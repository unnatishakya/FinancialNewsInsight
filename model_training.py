"""Sentiment (VADER) and category (TF-IDF + Naive Bayes) models.

Run ``python model_training.py`` to train the category model, print honest
evaluation numbers and save the model to ``models/category_model.joblib``.
"""
from __future__ import annotations

import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from preprocess import (
    build_unique_articles,
    clean_and_lemmatize,
    load_data,
    match_category_rule,
    DEFAULT_CATEGORY,
)

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "category_model.joblib"
GOLD_PATH = ROOT / "data" / "gold_eval.csv"

_sia = None


def _get_sia():
    global _sia
    if _sia is None:
        import nltk
        from nltk.sentiment.vader import SentimentIntensityAnalyzer

        nltk.download("vader_lexicon", quiet=True)
        _sia = SentimentIntensityAnalyzer()
    return _sia


def label_from_compound(compound: float) -> str:
    if compound >= 0.05:
        return "Positive"
    if compound <= -0.05:
        return "Negative"
    return "Neutral"


def vader_sentiment(text: str):
    """Return (compound, label). Multi-sentence text is scored per sentence and
    averaged, because a single compound score saturates on long paragraphs.
    Always pass the RAW text: stop-word removal deletes negations VADER needs."""
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", (text or "").strip()) if s]
    if not sentences:
        return 0.0, "Neutral"
    sia = _get_sia()
    compound = float(np.mean([sia.polarity_scores(s)["compound"] for s in sentences]))
    return round(compound, 4), label_from_compound(compound)


def make_category_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("nb", MultinomialNB(alpha=0.5)),
    ])


def evaluate_category_model(articles: pd.DataFrame) -> dict:
    """Leave-one-headline-out cross-validation.

    The table has one row per distinct headline, so no headline can appear in
    both the training and the test fold (the original 80/20 split on ~2,900 rows
    leaked duplicates). The labels come from keyword rules, so this measures how
    well the model reproduces the rules, not real-world topic accuracy.
    """
    x, y = articles["Cleaned_Headline"], articles["Category"]
    pred = cross_val_predict(make_category_pipeline(), x, y, cv=LeaveOneOut())
    return {
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "report": classification_report(y, pred, zero_division=0),
    }


MODEL_CONFIDENCE_THRESHOLD = 0.5  # chosen up front, not tuned on the gold set


def predict_category(text: str, model) -> dict:
    """Decide a category. Keyword rules win when one fires (transparent). Otherwise
    the Naive Bayes model is used only if it is confident; if not, General Markets.
    The model is trained on just ~50 distinct headlines, so it is a weak second
    opinion and is deliberately not trusted blindly."""
    proba = model.predict_proba([clean_and_lemmatize(text)])[0]
    best = int(proba.argmax())
    model_category, confidence = str(model.classes_[best]), float(proba[best])
    rule_category = match_category_rule(text)
    if rule_category:
        category, source = rule_category, "rules"
    elif confidence >= MODEL_CONFIDENCE_THRESHOLD:
        category, source = model_category, "model"
    else:
        category, source = DEFAULT_CATEGORY, "default"
    return {"category": category, "category_source": source,
            "model_category": model_category, "model_confidence": round(confidence, 3)}


def evaluate_on_gold(model, gold: pd.DataFrame) -> dict:
    """Score VADER and the category approaches on the small hand-labelled set."""
    sent_pred = [vader_sentiment(t)[1] for t in gold["text"]]
    rule_pred = [match_category_rule(t) or DEFAULT_CATEGORY for t in gold["text"]]
    ml_pred = model.predict([clean_and_lemmatize(t) for t in gold["text"]])
    hybrid_pred = [predict_category(t, model)["category"] for t in gold["text"]]
    return {
        "n": len(gold),
        "vader_sentiment_accuracy": accuracy_score(gold["sentiment"], sent_pred),
        "category_rules_accuracy": accuracy_score(gold["category"], rule_pred),
        "category_model_accuracy": accuracy_score(gold["category"], ml_pred),
        "category_hybrid_accuracy": accuracy_score(gold["category"], hybrid_pred),
    }


def train_and_save(articles: pd.DataFrame, path: Path = MODEL_PATH):
    model = make_category_pipeline().fit(articles["Cleaned_Headline"], articles["Category"])
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return model


def get_category_model(articles: pd.DataFrame | None = None, path: Path = MODEL_PATH):
    """Load the saved model, training it first if it does not exist yet."""
    if path.exists():
        return joblib.load(path)
    if articles is None:
        articles = build_unique_articles(load_data())
    return train_and_save(articles, path)


def main() -> None:
    df = load_data()
    articles = build_unique_articles(df)

    print(f"Rows with a headline: {len(df)}")
    print(f"Distinct headlines:   {len(articles)}")
    if "Sentiment" in df:
        labelled = df.dropna(subset=["Sentiment"])
        n_multi = int((labelled.groupby("Headline")["Sentiment"].nunique() > 1).sum())
        print(f"Headlines carrying more than one Sentiment label: {n_multi} "
              f"of {labelled['Headline'].nunique()} (dataset sentiment labels are unreliable)")
    print("\nRule-based category counts:")
    print(articles["Category"].value_counts().to_string())

    cv = evaluate_category_model(articles)
    print(f"\nLeave-one-headline-out CV on rule labels: accuracy={cv['accuracy']:.2%}, "
          f"macro-F1={cv['macro_f1']:.2f}")
    print(cv["report"])

    model = train_and_save(articles)
    print(f"Saved category model to {MODEL_PATH}")

    gold = pd.read_csv(GOLD_PATH)
    res = evaluate_on_gold(model, gold)
    print(f"\nHand-labelled gold set (n={res['n']}, author-labelled, small):")
    print(f"  VADER sentiment accuracy:       {res['vader_sentiment_accuracy']:.2%}")
    print(f"  Category, rules only:           {res['category_rules_accuracy']:.2%}")
    print(f"  Category, Naive Bayes only:     {res['category_model_accuracy']:.2%}")
    print(f"  Category, as deployed (rules, then confident NB): {res['category_hybrid_accuracy']:.2%}")


if __name__ == "__main__":
    main()
