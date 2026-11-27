"""Run every method on every size and seed, and write reports/."""

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from baselines import fit_tfidf, predict_most_common
from evaluate import (
    choose_final_method,
    final_scores,
    plot_confusion_matrix,
    plot_learning_curve,
    save_json,
)
from finetune import (
    ID2LABEL,
    SETTINGS,
    build_model,
    encode,
    load_tokenizer,
    make_collator,
    predict_ids,
    train_model,
)
from prepare import SAMPLE_SIZES, load_split, sample_training
from tickets_data import LABELS

SEEDS = (0, 1, 2)
FINAL_SEED = 0


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


def run_final():
    """Train the final method once, score the test rows once, and save outputs."""
    results = pd.read_csv("reports/results.csv")
    method, size = choose_final_method(results)
    if not method.startswith("finetune_"):
        raise SystemExit(f"Expected a fine-tuned method, got {method}")
    mode = method.removeprefix("finetune_")
    train, validation, test = load_split()
    own = pd.read_csv("data/own_messages.csv")
    tokenizer = load_tokenizer()
    collator = make_collator(tokenizer)
    sample = sample_training(train, size, FINAL_SEED)
    model = build_model(mode, FINAL_SEED)
    setting = SETTINGS[mode]
    train_model(
        model,
        encode(sample, tokenizer),
        encode(validation, tokenizer),
        collator,
        setting["lr"],
        setting["epochs"],
        FINAL_SEED,
    )
    tfidf = fit_tfidf(sample, FINAL_SEED)

    def model_labels(frame):
        ids = predict_ids(model, encode(frame, tokenizer), collator)
        return [ID2LABEL[i] for i in ids]

    def score_all(frame):
        true = list(frame["label"])
        return {
            method: final_scores(true, model_labels(frame)),
            "tfidf": final_scores(true, tfidf.predict(frame["text"])),
            "most_common": final_scores(
                true, predict_most_common(sample, frame["text"])
            ),
        }

    scores = {
        "final_method": method,
        "train_size": size,
        "seed": FINAL_SEED,
        "test": score_all(test),
        "own_40": score_all(own),
    }
    plot_confusion_matrix(
        list(test["label"]), model_labels(test), LABELS, "reports/confusion_matrix.png"
    )
    save_json(scores, "reports/final.json")
    model.save_pretrained("model")
    tokenizer.save_pretrained("model")
    print(json.dumps(scores, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="tiny test run")
    parser.add_argument("--final", action="store_true", help="final test scoring")
    args = parser.parse_args()
    if args.quick:
        results, _ = run_grid(sizes=(20,), seeds=(0,), epochs=1)
        print(results.round(3).to_string())
        return
    if args.final:
        run_final()
        return
    results, epoch_log = run_grid(SAMPLE_SIZES, SEEDS)
    Path("reports").mkdir(exist_ok=True)
    results.to_csv("reports/results.csv", index=False)
    epoch_log.to_csv("reports/epoch_log.csv", index=False)
    plot_learning_curve(results, "reports/learning_curve.png")
    print("saved reports/results.csv, epoch_log.csv and learning_curve.png")


if __name__ == "__main__":
    main()
