"""Scores and plots for the experiments."""

from pathlib import Path

from matplotlib.figure import Figure

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