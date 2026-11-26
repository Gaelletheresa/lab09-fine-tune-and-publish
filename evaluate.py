"""Scores and plots for the experiments."""

import json
from pathlib import Path

from matplotlib.figure import Figure
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

METHOD_NAMES = {
    "most_common": "Most common label",
    "tfidf": "TF-IDF + logistic regression",
    "finetune_head": "DistilBERT, head only",
    "finetune_full": "DistilBERT, full",
}


def plot_learning_curve(results, path):
    """Plot mean validation accuracy against training size, with the seed range."""
    fig = Figure(figsize=(7, 4.5))
    ax = fig.subplots()
    for method, name in METHOD_NAMES.items():
        rows = results[results["method"] == method]
        stats = rows.groupby("size")["val_accuracy"].agg(["mean", "min", "max"])
        ax.plot(stats.index, stats["mean"], marker="o", label=name)
        ax.fill_between(stats.index, stats["min"], stats["max"], alpha=0.2)
    ax.set_xticks(sorted(results["size"].unique()))
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Validation accuracy")
    ax.set_title("Learning curve (mean and range over 3 seeds)")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)


def choose_final_method(results):
    """Pick the method with the best mean validation accuracy at the largest size."""
    largest = results["size"].max()
    means = results[results["size"] == largest].groupby("method")["val_accuracy"].mean()
    return means.idxmax(), int(largest)


def final_scores(y_true, y_pred):
    """Accuracy and macro F1 for one set of predictions."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
    }


def plot_confusion_matrix(y_true, y_pred, labels, path):
    """Save a confusion matrix with the counts written in each cell."""
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    fig = Figure(figsize=(5.5, 4.5))
    ax = fig.subplots()
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels=labels)
    ax.set_yticks(range(len(labels)), labels=labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center")
    ax.set_xlabel("Predicted queue")
    ax.set_ylabel("True queue")
    ax.set_title("Confusion matrix")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)


def save_json(data, path):
    """Write a dictionary to a JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2))