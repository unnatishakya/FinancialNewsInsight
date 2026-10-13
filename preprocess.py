"""Data loading, text cleaning and rule-based category labelling.

The dataset has ~2,900 rows but only 50 distinct headlines, so most helpers work
on the de-duplicated headline table built by ``build_unique_articles``.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "financial_news_events.csv"

# Keyword rules (most hits wins; ties go to the earlier rule). Word boundaries avoid substring
# accidents such as 'rate' matching 'corporate' or 'tech' matching 'biotech'.
CATEGORY_RULES = [
    ("Real Estate",
     r"\b(housing|real estate|propert\w*|mortgage\w*|homebuilder\w*)\b"),
    ("Tech Stocks & Software",
     r"\b(tech\w*|software|ipo|cyber\w*|data breach|semiconductor\w*|crypto\w*|chip\w*|ai|artificial intelligence)\b"),
    ("Energy & Commodities",
     r"\b(oil|gas|commodit\w*|gold|energy|crude|mining|renewable\w*)\b"),
    ("Banking & Central Banks",
     r"\b(bank\w*|fed|federal reserve|central bank\w*|rates?|interest|monetary|inflation|yields?|bonds?|financ\w*|lend\w*)\b"),
    ("Trade & Macro Economy",
     r"\b(trade|tariff\w*|subsid\w*|spending|gdp|econom\w*|tax\w*|unemployment|recession|export\w*|"
     r"manufacturing|supply chain|retail sales|consumer confidence|geopolitical)\b"),
]
DEFAULT_CATEGORY = "General Markets"
CATEGORIES = [name for name, _ in CATEGORY_RULES] + [DEFAULT_CATEGORY]

_lemmatizer = None
_stop_words = None


def ensure_nltk() -> None:
    """Download the NLTK resources used by the project (no-op when present)."""
    import nltk

    for pkg in ("punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4", "vader_lexicon"):
        nltk.download(pkg, quiet=True)


def _resources():
    global _lemmatizer, _stop_words
    if _lemmatizer is None:
        ensure_nltk()
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer

        _lemmatizer = WordNetLemmatizer()
        _stop_words = set(stopwords.words("english"))
    return _lemmatizer, _stop_words


def clean_and_lemmatize(text) -> str:
    """Classic NLP cleaning for the ML models: strip markup, lowercase, drop
    stop words, lemmatize. Do NOT use this output for VADER or the LLM."""
    if not isinstance(text, str):
        return ""
    from nltk.tokenize import word_tokenize

    lemmatizer, stop_words = _resources()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text.lower())
    tokens = word_tokenize(text)
    return " ".join(lemmatizer.lemmatize(w) for w in tokens if w not in stop_words)


def clean_for_llm(text) -> str:
    """Light cleaning that keeps punctuation and sentence structure for the LLM."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def match_category_rule(text: str):
    """Return the category with the most keyword hits (ties go to the earlier rule),
    or None when no rule fires. Counting hits lets the dominant topic win, so a Fed
    statement that also mentions mortgages is still Banking, not Real Estate."""
    lowered = (text or "").lower()
    best, best_hits = None, 0
    for name, pattern in CATEGORY_RULES:
        hits = len(re.findall(pattern, lowered))
        if hits > best_hits:
            best, best_hits = name, hits
    return best


def assign_financial_category(text: str) -> str:
    return match_category_rule(text) or DEFAULT_CATEGORY


def load_data(path=None) -> pd.DataFrame:
    """Load the news CSV, drop rows without a headline and parse dates."""
    df = pd.read_csv(path or DATA_PATH)
    df = df.dropna(subset=["Headline"]).copy()
    df["Headline"] = df["Headline"].astype(str).str.strip()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df.reset_index(drop=True)


def build_unique_articles(df: pd.DataFrame) -> pd.DataFrame:
    """One row per distinct headline with cleaned text and a rule-based category."""
    unique = df.groupby("Headline", sort=False).size().rename("Row_Count").reset_index()
    unique["Cleaned_Headline"] = unique["Headline"].map(clean_and_lemmatize)
    unique["LLM_Clean_Headline"] = unique["Headline"].map(clean_for_llm)
    unique["Category"] = unique["Headline"].map(assign_financial_category)
    return unique
