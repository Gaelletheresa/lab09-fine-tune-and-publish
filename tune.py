"""Choose the learning rate and number of epochs on the validation rows only."""

from pathlib import Path

import pandas as pd

from finetune import build_model, encode, load_tokenizer, make_collator, train_model
from prepare import load_split, sample_training

TUNE_SIZE = 150
TUNE_SEED = 0
EPOCHS = 6
GRID = {"head": [1e-3, 5e-3], "full": [2e-5, 5e-5]}


def main():
    train, validation, _ = load_split()
    tokenizer = load_tokenizer()
    collator = make_collator(tokenizer)
    sample = sample_training(train, TUNE_SIZE, TUNE_SEED)
    train_items = encode(sample, tokenizer)
    val_items = encode(validation, tokenizer)
    rows = []
    for mode, rates in GRID.items():
        for lr in rates:
            model = build_model(mode, TUNE_SEED)
            history = train_model(
                model, train_items, val_items, collator, lr, EPOCHS, TUNE_SEED
            )
            for entry in history:
                rows.append({"mode": mode, "lr": lr, **entry})
            print(f"done: {mode} lr={lr}")
    Path("reports").mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv("reports/tuning.csv", index=False)


if __name__ == "__main__":
    main()
