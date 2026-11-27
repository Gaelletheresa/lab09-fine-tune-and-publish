import pandas as pd
import pytest
import torch

from finetune import (
    accuracy,
    build_model,
    encode,
    load_tokenizer,
    make_collator,
    train_model,
)
from prepare import load_split, sample_training

HEAD_WEIGHTS = {
    "pre_classifier.weight",
    "pre_classifier.bias",
    "classifier.weight",
    "classifier.bias",
}


@pytest.fixture(scope="module")
def tokenizer():
    return load_tokenizer()


@pytest.fixture(scope="module")
def collator(tokenizer):
    return make_collator(tokenizer)


@pytest.fixture(scope="module")
def small_items(tokenizer):
    train, _, _ = load_split()
    return encode(sample_training(train, 16, seed=0), tokenizer)


def body_snapshot(model):
    return [p.detach().clone() for p in model.distilbert.parameters()]


def body_is_unchanged(model, before):
    now = list(model.distilbert.parameters())
    return all(torch.equal(a, b) for a, b in zip(before, now, strict=True))


def test_head_mode_makes_only_the_head_trainable():
    model = build_model("head", seed=0)
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert trainable == HEAD_WEIGHTS
    assert not any(p.requires_grad for p in model.distilbert.parameters())


def test_head_training_leaves_the_body_unchanged(small_items, collator):
    model = build_model("head", seed=0)
    before = body_snapshot(model)
    head_before = model.classifier.weight.detach().clone()
    train_model(model, small_items, small_items, collator, 5e-3, 1, seed=0)
    assert body_is_unchanged(model, before)
    assert not torch.equal(head_before, model.classifier.weight)


def test_full_training_changes_the_body(small_items, collator):
    model = build_model("full", seed=0)
    before = body_snapshot(model)
    train_model(model, small_items, small_items, collator, 5e-5, 1, seed=0)
    assert not body_is_unchanged(model, before)


def test_same_seed_repeats_a_run(small_items, collator):
    histories = []
    for _ in range(2):
        model = build_model("full", seed=1)
        histories.append(
            train_model(model, small_items, small_items, collator, 5e-5, 1, seed=1)
        )
    assert histories[0][0]["train_loss"] == pytest.approx(
        histories[1][0]["train_loss"], abs=1e-6
    )


def test_model_can_memorize_a_tiny_batch(tokenizer, collator):
    frame = pd.DataFrame(
        {
            "text": [
                "I was charged twice for one ride",
                "Please refund the extra fee on my statement",
                "The brakes are weak and the chain keeps slipping",
                "My rear tire is flat",
                "The dock at my corner will not let go of the bike",
                "No bikes at the station by my office",
                "I forgot my password and cannot sign in",
                "Please change the email on my profile",
            ],
            "label": [
                "billing",
                "billing",
                "bike_fault",
                "bike_fault",
                "station",
                "station",
                "account",
                "account",
            ],
        }
    )
    items = encode(frame, tokenizer)
    model = build_model("full", seed=0)
    train_model(model, items, items, collator, 1e-4, 20, seed=0, batch_size=8)
    assert accuracy(model, items, collator) == 1.0
