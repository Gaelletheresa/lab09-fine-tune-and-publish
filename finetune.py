"""Fine-tune DistilBERT for four-way ticket classification."""

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

from tickets_data import LABELS

MODEL_NAME = "distilbert/distilbert-base-uncased"
MAX_LENGTH = 64
SETTINGS = {
    "head": {"lr": 5e-3, "epochs": 5},
    "full": {"lr": 5e-5, "epochs": 4},
}
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}


def load_tokenizer():
    """Load the tokenizer that belongs to the base model."""
    return AutoTokenizer.from_pretrained(MODEL_NAME)


def encode(frame, tokenizer):
    """Tokenize the texts, truncating at MAX_LENGTH, with no padding yet."""
    encoded = tokenizer(list(frame["text"]), truncation=True, max_length=MAX_LENGTH)
    return [
        {
            "input_ids": encoded["input_ids"][i],
            "attention_mask": encoded["attention_mask"][i],
            "labels": LABEL2ID[label],
        }
        for i, label in enumerate(frame["label"])
    ]


def make_collator(tokenizer):
    """Pad each batch to its longest message, not to the maximum length."""
    return DataCollatorWithPadding(tokenizer)


def build_model(mode, seed):
    """Load DistilBERT with a new four-way head. mode is 'head' or 'full'."""
    if mode not in ("head", "full"):
        raise ValueError("mode must be 'head' or 'full'")
    torch.manual_seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID
    )
    if mode == "head":
        for parameter in model.distilbert.parameters():
            parameter.requires_grad = False
    return model


def count_trainable(model):
    """Count the weights that training is allowed to change."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def predict_ids(model, items, collator, batch_size=32):
    """Return the predicted label number for every item."""
    model.eval()
    loader = DataLoader(items, batch_size=batch_size, collate_fn=collator)
    predictions = []
    with torch.no_grad():
        for batch in loader:
            batch.pop("labels")
            logits = model(**batch).logits
            predictions.extend(logits.argmax(dim=-1).tolist())
    return predictions


def accuracy(model, items, collator):
    """Share of items that the model labels correctly."""
    predictions = predict_ids(model, items, collator)
    correct = sum(
        p == item["labels"] for p, item in zip(predictions, items, strict=True)
    )
    return correct / len(items)


def train_model(
    model, train_items, val_items, collator, lr, epochs, seed, batch_size=16
):
    """Train and return one log row per epoch: train loss and validation accuracy."""
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        train_items,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collator,
    )
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=lr)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total, count = 0.0, 0
        for batch in loader:
            optimizer.zero_grad()
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch["labels"])
            count += len(batch["labels"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": total / count,
                "val_accuracy": accuracy(model, val_items, collator),
            }
        )
    return history
