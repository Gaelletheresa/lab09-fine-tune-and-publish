"""Run every method on every size and seed, and write reports/."""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from baselines import fit_tfidf, predict_most_common
from finetune import (
    SETTINGS,
    build_model,
    encode,
    load_tokenizer,
    make_collator,
    predict_ids,
    train_model,
)
from prepare import SAMPLE_SIZES, load_split, sample_training

SEEDS = (0, 1, 2)


def score_row(method, size, seed, y_true, y_pred):
    """One result row: validation accuracy and macro F1 for one run."""
    return {
        "method": method,
        "size": size,
        "seed": seed,
        "val_accuracy": accuracy_score(y_true, y_pred),
        "val_macro_f1": f1_score(y_true, y_pred, average="macro"),
    }


def run_grid(sizes, seeds, epochs=None):
    """Train and score every method on the same sampled rows and validation rows."""
    train, validation, _ = load_split()
    tokenizer = load_tokenizer()
    collator = make_collator(tokenizer)
    val_items = encode(validation, tokenizer)
    true_ids = [item["labels"] for item in val_items]
    true_labels = list(validation["label"])
    rows, epoch_rows = [], []
    for size in sizes:
        for seed in seeds:
            sample = sample_training(train, size, seed)
            common = predict_most_common(sample, validation["text"])
            rows.append(score_row("most_common", size, seed, true_labels, common))
            tfidf = fit_tfidf(sample, seed).predict(validation["text"])
            rows.append(score_row("tfidf", size, seed, true_labels, tfidf))
            train_items = encode(sample, tokenizer)
            for mode in ("head", "full"):
                setting = SETTINGS[mode]
                model = build_model(mode, seed)
                history = train_model(
                    model,
                    train_items,
                    val_items,
                    collator,
                    setting["lr"],
                    epochs or setting["epochs"],
                    seed,
                )
                predicted = predict_ids(model, val_items, collator)
                name = f"finetune_{mode}"
                rows.append(score_row(name, size, seed, true_ids, predicted))
                for entry in history:
                    epoch_rows.append(
                        {"method": name, "size": size, "seed": seed, **entry}
                    )
                print(f"done: {name} size={size} seed={seed}", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(epoch_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="tiny test run")
    args = parser.parse_args()
    if args.quick:
        results, _ = run_grid(sizes=(20,), seeds=(0,), epochs=1)
        print(results.round(3).to_string())
        return
    results, epoch_log = run_grid(SAMPLE_SIZES, SEEDS)
    Path("reports").mkdir(exist_ok=True)
    results.to_csv("reports/results.csv", index=False)
    epoch_log.to_csv("reports/epoch_log.csv", index=False)
    print("saved reports/results.csv and reports/epoch_log.csv")


if __name__ == "__main__":
    main()