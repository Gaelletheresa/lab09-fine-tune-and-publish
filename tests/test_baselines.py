from sklearn.metrics import accuracy_score

from baselines import fit_tfidf, predict_most_common
from prepare import load_split, sample_training


def test_tfidf_beats_most_common_on_validation():
    train, validation, _ = load_split()
    sample = sample_training(train, 150, seed=0)
    common = predict_most_common(sample, validation["text"])
    tfidf = fit_tfidf(sample).predict(validation["text"])
    common_score = accuracy_score(validation["label"], common)
    tfidf_score = accuracy_score(validation["label"], tfidf)
    assert tfidf_score > common_score
