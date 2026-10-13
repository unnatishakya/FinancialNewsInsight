"""LLM stage: summaries, grounded Q&A and key insights.

Backends (set LLM_BACKEND):
  hf      (default) local Hugging Face pipeline, TinyLlama-1.1B-Chat by default
  ollama  a locally running Ollama server (LLM_MODEL e.g. llama3.2, OLLAMA_HOST)
"""
from __future__ import annotations

import os
import random
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from preprocess import clean_and_lemmatize

DISCLAIMER = ("For information only. This is not financial or investment advice, and the "
              "output may be inaccurate or biased. Verify with primary sources.")

MAX_ARTICLE_CHARS = 1500  # keeps prompts inside TinyLlama's 2,048-token context

SYSTEM_PROMPT = (
    "You are a careful financial news assistant. Use ONLY the context provided. "
    "If the context does not contain the answer, say so. Be concise and never give "
    "investment advice."
)

# One worked example used for the few-shot insights prompt.
INSIGHT_EXAMPLE_CONTEXT = (
    "Central bank holds rates steady. Inflation data comes in hotter than expected. "
    "Bank shares dip on weaker loan demand."
)
INSIGHT_EXAMPLE_ANSWER = (
    "- Rates are unchanged, but hot inflation raises the chance of rates staying high.\n"
    "- Banks are under pressure from weaker lending demand.\n"
    "- Watch the next inflation release for any policy shift."
)


class LLMError(RuntimeError):
    """Raised when the language model cannot be reached or loaded."""


def truncate_text(text: str, limit: int = MAX_ARTICLE_CHARS) -> str:
    """Trim to ``limit`` characters, preferring to cut at a sentence boundary."""
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    end = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    return cut[: end + 1] if end > limit * 0.5 else cut


def tidy_output(text: str) -> str:
    """Remove chat tokens and repeated lines (small models like to loop)."""
    text = re.sub(r"<\|[^|>]*\|>|</s>", "", text or "").strip()
    seen, lines = set(), []
    for line in text.splitlines():
        key = line.strip().lower()
        if key and key in seen:
            continue
        seen.add(key)
        lines.append(line.rstrip())
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


class HeadlineRetriever:
    """TF-IDF retrieval over the distinct dataset headlines (the Q&A context pool)."""

    def __init__(self, headlines):
        self.headlines = list(headlines)
        self._vec = TfidfVectorizer(preprocessor=clean_and_lemmatize)
        self._matrix = self._vec.fit_transform(self.headlines)

    def top_k(self, query: str, k: int = 5):
        scores = cosine_similarity(self._vec.transform([query]), self._matrix)[0]
        order = scores.argsort()[::-1][:k]
        return [self.headlines[i] for i in order if scores[i] > 0]

    def sample(self, n: int = 12, seed: int = 42):
        rng = random.Random(seed)
        return rng.sample(self.headlines, min(n, len(self.headlines)))


class FinanceLLM:
    def __init__(self, retriever: HeadlineRetriever | None = None, backend: str | None = None,
                 model_name: str | None = None, generate_fn=None):
        self.retriever = retriever
        self.backend = (backend or os.getenv("LLM_BACKEND", "hf")).lower()
        default_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" if self.backend == "hf" else "llama3.2"
        self.model_name = model_name or os.getenv("LLM_MODEL", default_model)
        self._generate_fn = generate_fn  # injectable for tests
        self._pipe = None

    # ---- backends -------------------------------------------------------
    def _hf_pipeline(self):
        if self._pipe is None:
            try:
                import torch
                from transformers import pipeline
            except ImportError as exc:
                raise LLMError("Install transformers, torch and accelerate (see requirements.txt).") from exc
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            self._pipe = pipeline("text-generation", model=self.model_name, device_map="auto",
                                  model_kwargs={"torch_dtype": dtype})
        return self._pipe

    def _hf_generate(self, messages, max_new_tokens, temperature):
        pipe = self._hf_pipeline()
        prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        kwargs = dict(max_new_tokens=max_new_tokens, return_full_text=False, repetition_penalty=1.15,
                      pad_token_id=pipe.tokenizer.eos_token_id)
        if temperature > 0:
            kwargs.update(do_sample=True, temperature=temperature, top_p=0.9)
        else:
            kwargs.update(do_sample=False)
        return pipe(prompt, **kwargs)[0]["generated_text"]

    def _ollama_generate(self, messages, max_new_tokens, temperature):
        import requests

        host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        try:
            resp = requests.post(f"{host}/api/chat", timeout=180, json={
                "model": self.model_name, "messages": messages, "stream": False,
                "options": {"temperature": temperature, "num_predict": max_new_tokens}})
            resp.raise_for_status()
            return resp.json()["message"]["content"]
        except requests.exceptions.RequestException as exc:
            raise LLMError(f"Could not reach Ollama at {host}: {exc}") from exc

    def chat(self, messages, max_new_tokens: int = 200, temperature: float = 0.3) -> str:
        if self._generate_fn is not None:
            raw = self._generate_fn(messages, max_new_tokens, temperature)
        elif self.backend == "ollama":
            raw = self._ollama_generate(messages, max_new_tokens, temperature)
        else:
            raw = self._hf_generate(messages, max_new_tokens, temperature)
        return tidy_output(raw)

    # ---- use cases ------------------------------------------------------
    @staticmethod
    def _nlp_hint(analysis) -> str:
        if not analysis:
            return ""
        return (f"Automated analysis: sentiment is {analysis['sentiment'].lower()} "
                f"(score {analysis['score']:+.2f}); topic is {analysis['category']}.\n")

    def summarize(self, text: str, analysis=None) -> str:
        user = (f"{self._nlp_hint(analysis)}Article:\n{truncate_text(text)}\n\n"
                "Summarize the article in 2-3 sentences.")
        return self.chat([{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": user}], max_new_tokens=120)

    def answer(self, question: str, text: str | None = None, analysis=None) -> str:
        if text and text.strip():
            context = truncate_text(text)
        else:
            if self.retriever is None:
                return "No article was provided and no headline database is loaded."
            hits = self.retriever.top_k(question, k=5)
            if not hits:
                return "I could not find headlines related to that question in the dataset."
            context = "\n".join(f"- {h}" for h in hits)
        user = f"{self._nlp_hint(analysis)}Context:\n{context}\n\nQuestion: {question}"
        return self.chat([{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": user}], max_new_tokens=160)

    def insights(self, text: str | None = None, analysis=None, few_shot: bool = True) -> str:
        if text and text.strip():
            context = truncate_text(text)
        elif self.retriever is not None:
            context = "\n".join(f"- {h}" for h in self.retriever.sample(12))
        else:
            return "No article was provided and no headline database is loaded."
        task = "List 3 short bullet-point takeaways for a market analyst."
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if few_shot:
            messages += [{"role": "user", "content": f"Context:\n{INSIGHT_EXAMPLE_CONTEXT}\n\n{task}"},
                         {"role": "assistant", "content": INSIGHT_EXAMPLE_ANSWER}]
        messages.append({"role": "user", "content": f"{self._nlp_hint(analysis)}Context:\n{context}\n\n{task}"})
        return self.chat(messages, max_new_tokens=220, temperature=0.4)
