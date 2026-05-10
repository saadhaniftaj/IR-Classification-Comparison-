"""
preprocessing.py
----------------
Unified preprocessing pipeline for all three datasets.
Handles loading, cleaning, tokenization, stopword removal,
stemming, and TF-IDF vectorization.
"""

import os
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

# Download required NLTK data once
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)

STOP_WORDS = set(stopwords.words("english"))
STEMMER = PorterStemmer()


# ---------------------------------------------------------------------------
# Text Cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Lowercase, remove URLs, mentions, special chars, digits."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)          # remove URLs
    text = re.sub(r"@\w+|#\w+", "", text)               # remove mentions/hashtags
    text = re.sub(r"[^a-z\s]", " ", text)               # keep only letters
    text = re.sub(r"\s+", " ", text).strip()            # collapse whitespace
    return text


def tokenize_and_stem(text: str) -> str:
    """Tokenize, remove stopwords, apply Porter stemming."""
    tokens = text.split()
    tokens = [STEMMER.stem(t) for t in tokens if t not in STOP_WORDS and len(t) > 2]
    return " ".join(tokens)


def full_pipeline(text: str) -> str:
    return tokenize_and_stem(clean_text(text))


# ---------------------------------------------------------------------------
# Dataset Loaders
# ---------------------------------------------------------------------------

def load_twitter_health(filepath: str) -> pd.DataFrame:
    """
    Load Health News in Twitter dataset.
    Expected CSV columns: source, datetime, tweet_text
    Label is inferred from the 'source' column (news outlet = class).
    """
    df = pd.read_csv(filepath, header=None, names=["source", "datetime", "text"],
                     encoding="latin-1")
    df = df.dropna(subset=["text", "source"])
    df["text"] = df["text"].apply(full_pipeline)
    df = df[df["text"].str.strip() != ""]
    return df[["source", "text"]].rename(columns={"source": "label"})


def load_bag_of_words(docword_path: str, vocab_path: str, labels: dict = None) -> pd.DataFrame:
    """
    Load UCI Bag of Words dataset (docword.*.txt + vocab.*.txt format).
    Reconstructs sparse document-term matrix into raw text strings.

    labels: optional dict {doc_id: label_string}. If None, assigns generic labels.
    """
    # Read vocab
    with open(vocab_path, "r") as f:
        vocab = [line.strip() for line in f.readlines()]

    # Read docword: first 3 lines are metadata
    doc_words = {}
    with open(docword_path, "r") as f:
        n_docs = int(f.readline().strip())
        _ = f.readline()   # n_words
        _ = f.readline()   # n_nnz
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                doc_id, word_id, count = int(parts[0]), int(parts[1]), int(parts[2])
                word = vocab[word_id - 1]  # 1-indexed
                doc_words.setdefault(doc_id, []).extend([word] * count)

    rows = []
    for doc_id, words in doc_words.items():
        text = " ".join(words)
        text_clean = full_pipeline(text)
        lbl = labels.get(doc_id, f"class_{doc_id % 5}") if labels else f"class_{doc_id % 5}"
        rows.append({"label": lbl, "text": text_clean})

    return pd.DataFrame(rows)


def load_csv_generic(filepath: str, text_col: str, label_col: str) -> pd.DataFrame:
    """Generic loader for any CSV with explicit text and label columns."""
    df = pd.read_csv(filepath, encoding="latin-1")
    df = df.dropna(subset=[text_col, label_col])
    df["text"] = df[text_col].apply(full_pipeline)
    df = df[df["text"].str.strip() != ""]
    return df[["text", label_col]].rename(columns={label_col: "label"})


# ---------------------------------------------------------------------------
# Vectorization & Splitting
# ---------------------------------------------------------------------------

def vectorize(train_texts, test_texts, max_features: int = 20000):
    """
    Fit TF-IDF on train, transform both train and test.
    Returns: X_train_tfidf, X_test_tfidf, vectorizer
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),        # unigrams + bigrams
        sublinear_tf=True,         # apply log normalization to TF
        min_df=2,                  # ignore very rare terms
        analyzer="word"
    )
    X_train = vectorizer.fit_transform(train_texts)
    X_test  = vectorizer.transform(test_texts)
    return X_train, X_test, vectorizer


def prepare_dataset(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Encode labels, split, vectorize.
    Returns: X_train, X_test, y_train, y_test, label_encoder, vectorizer
    """
    le = LabelEncoder()
    y  = le.fit_transform(df["label"])

    X_train_txt, X_test_txt, y_train, y_test = train_test_split(
        df["text"].values, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    X_train, X_test, vectorizer = vectorize(X_train_txt, X_test_txt)
    return X_train, X_test, y_train, y_test, le, vectorizer
