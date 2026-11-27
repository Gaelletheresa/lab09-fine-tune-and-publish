"""Two baselines: the most common label, and TF-IDF with logistic regression."""

from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline


def most_common_label(train):
    """Return the label that appears most often in the training rows."""
    return Counter(train["label"]).most_common(1)[0][0]


def predict_most_common(train, texts):
    """Predict the most common training label for every text."""
    label = most_common_label(train)
    return [label] * len(texts)


def fit_tfidf(train, seed=0):
    """Fit TF-IDF plus logistic regression on the training rows only."""
    model = make_pipeline(
        TfidfVectorizer(lowercase=True, ngram_range=(1, 2)),
        LogisticRegression(max_iter=1000, random_state=seed),
    )
    model.fit(train["text"], train["label"])
    return model
