import pytest

from prepare import sample_training, split_tickets
from tickets_data import LABELS, make_tickets


@pytest.fixture(scope="module")
def frame():
    return make_tickets()


@pytest.mark.parametrize("seed", [0, 1, 2026])
def test_split_sizes_and_no_overlap(frame, seed):
    train, validation, test = split_tickets(frame, seed)
    assert (len(train), len(validation), len(test)) == (720, 240, 240)
    train_ids = set(train["ticket_id"])
    validation_ids = set(validation["ticket_id"])
    test_ids = set(test["ticket_id"])
    assert not train_ids & validation_ids
    assert not train_ids & test_ids
    assert not validation_ids & test_ids
    assert len(train_ids | validation_ids | test_ids) == 1200


def test_sample_has_right_size_labels_and_source(frame):
    train, _, _ = split_tickets(frame)
    sample = sample_training(train, 40, seed=1)
    assert len(sample) == 40
    assert set(sample["label"]) == set(LABELS)
    assert set(sample["ticket_id"]) <= set(train["ticket_id"])


def test_sample_repeats_with_the_same_seed(frame):
    train, _, _ = split_tickets(frame)
    first = sample_training(train, 40, seed=3)
    second = sample_training(train, 40, seed=3)
    assert list(first["ticket_id"]) == list(second["ticket_id"])
