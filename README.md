# Finance News Desk: NLP + LLM prototype (ITAI 2372, M04)

A small proof-of-concept for finance news triage. A user pastes or selects an article, and the app

1. runs **traditional NLP** first: VADER sentiment, a topic/category label, and company / index / institution mentions;
2. then calls a **local LLM** to summarize the text, answer questions about it, or list key takeaways.

> **Not financial advice.** Output is for information only and can be wrong or biased. See [Limitations](#limitations).

## Features

| Requirement | Where |
|---|---|
| Finance text + metadata (CSV) | `data/financial_news_events.csv` (Kaggle: *Financial News & Market Events Dataset 2025*) |
| Cleaning, tokenization, stop words, lemmatization | `preprocess.py` |
| Rule-based sentiment (VADER) | `model_training.py` (`vader_sentiment`) |
| Topic classification (rules + Naive Bayes) | `preprocess.py`, `model_training.py` |
| Named entities (gazetteer) | `pipeline.py` |
| LLM summary, grounded Q&A, key takeaways | `llm.py` |
| Web interface (Flask + HTML/CSS/JS) | `app.py`, `templates/`, `static/` |
| Business use case, limitations | below |

## Project structure

```
.
├── app.py                 # Flask app: NLP stage first, then LLM stage
├── pipeline.py            # analyze(): sentiment + category + entities
├── preprocess.py          # loading, cleaning, lemmatization, keyword category rules
├── model_training.py      # VADER helper, Naive Bayes category model, evaluation
├── llm.py                 # summarize / answer / insights, TF-IDF headline retrieval
├── templates/index.html
├── static/{style.css,app.js}
├── data/
│   ├── financial_news_events.csv
│   ├── curated_articles.json   # sample articles for the dropdown
│   └── gold_eval.csv           # 24 hand-labelled sentences for sanity checks
├── tests/test_smoke.py
├── notebooks/M04_Shakya_Unnati_ITAI2372_Finance.ipynb
├── screenshots/
└── requirements.txt
```

## Install

Python 3.10+ is recommended.

```bash
git clone <YOUR-REPO-URL>
cd <repo-folder>
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

NLTK data (punkt, stopwords, wordnet, VADER lexicon) downloads automatically on first run.

## Run

```bash
python model_training.py   # trains the category model, prints evaluation, saves models/category_model.joblib
python app.py              # then open http://127.0.0.1:5000
```

`app.py` trains the category model by itself if `models/` is empty, so the first command is optional.

The first language-model request downloads **TinyLlama-1.1B-Chat** (about 2 GB) and may take a minute to load. No Hugging Face token is needed. It runs on CPU, and is faster on a GPU.

**Use a different model.** To use a locally running Ollama server instead:

```bash
LLM_BACKEND=ollama LLM_MODEL=llama3.2 python app.py     # Windows PowerShell: $env:LLM_BACKEND="ollama"; ...
```

## Test the app

Automated (no model download; the LLM is stubbed):

```bash
python -m unittest discover -s tests -v
```

Manual:

1. Choose **Federal Reserve statement** from *Sample articles*, then click **Analyze**. You should see a sentiment gauge, a topic, and "Federal Reserve" under Mentions.
2. Click **Summarize** and **Key takeaways** for LLM output.
3. Type `What is the overall outlook?` and click **Ask**.
4. Clear the article box and ask `What is happening with interest rates?`. The answer is drawn from the most similar dataset headlines. If nothing matches, the app says so instead of guessing.

## How it works

- **Sentiment.** VADER runs on the raw text (stop-word removal would delete negations). Multi-sentence text is scored per sentence and averaged. Labels: >= 0.05 positive, <= -0.05 negative, otherwise neutral.
- **Category.** Keyword rules with word boundaries decide first: the category with the most keyword hits wins (ties go to the earlier rule). If none fires, a TF-IDF + Naive Bayes model is used only when its confidence is at least 0.5, otherwise the label is *General Markets*.
- **Q&A.** If an article is provided, it is the context (trimmed to about 1,500 characters to fit the model's context window). If not, the five most similar dataset headlines (TF-IDF cosine similarity) are used.
- **Prompting.** A finance-specific system prompt, the NLP results passed in as a hint, and a one-example (few-shot) prompt for takeaways.

## Data notes and results

- The CSV has 2,876 rows with a headline but only **50 distinct headlines**. Each of those headlines appears with all three `Sentiment` labels, and `Sector` looks random: a Naive Bayes sector classifier scored 5.90% (about chance for 18 classes), and a grouped 6-class version scored 25.52%, below the 32.8% majority-class baseline. These columns are therefore **not used as ground truth**.
- Categories come from keyword rules, so the Naive Bayes model is evaluated with leave-one-headline-out cross-validation (no headline in both train and test) and on `data/gold_eval.csv`, a small independent set labelled by the author. Treat both as sanity checks, not benchmarks.

Record your own run of `python model_training.py` here:

| Check | Result |
|---|---|
| Leave-one-headline-out accuracy / macro-F1 (rule labels) | _fill in_ |
| VADER sentiment accuracy on gold set (n=24) | _fill in_ |
| Category accuracy on gold set: rules / NB only / as deployed | _fill in_ |

## Business use case

**Scenario.** A news aggregator or an analyst desk receives hundreds of headlines a day and needs a fast first read.

**Value.** The app tags sentiment and topic, flags the companies, indices and institutions mentioned, and gives a short summary or an answer to a question. That saves first-pass triage time for investment research and risk monitoring, and it lets an analyst decide quickly which stories deserve a full read.

**Possible path to production.** Swap the static CSV for a licensed news feed, add a larger or fine-tuned model, and expose the API inside an existing research portal or sell it as a per-seat subscription.

## Privacy and compliance notes

Public news carries little personal data, but an enterprise version would need a data-retention policy, no logging of confidential client documents, and review under GDPR and the firm's own rules. Running the model locally (as here) keeps text on the machine. Outputs used in client communication fall under the firm's compliance review, since regulators treat investment recommendations as regulated activity.

## Limitations

- Not financial advice and not a substitute for professional judgment.
- VADER is a general-purpose lexicon, not tuned for finance.
- TinyLlama is small: it can repeat itself, miss nuance or state things the text does not support. Larger models (for example via Ollama) are noticeably better.
- The dataset is small and appears synthetic; it contains headlines, not full articles.
- News and model outputs can both carry bias.

## Screenshots

Add screenshots of the analysis, summary and Q&A views to `screenshots/` and link them here.
