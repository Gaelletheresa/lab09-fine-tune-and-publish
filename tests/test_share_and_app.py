import pytest
from transformers import AutoModelForSequenceClassification

from app import EMPTY_MESSAGE, score_text
from finetune import build_model, encode, load_tokenizer, make_collator, predict_ids
from prepare import load_split
from share import build_model_card
from tickets_data import LABELS

SCORES = {"accuracy": 0.9, "macro_f1": 0.9}
FINAL = {
    "final_method": "finetune_full",
    "train_size": 400,
    "seed": 0,
    "test": {"finetune_full": SCORES, "tfidf": SCORES, "most_common": SCORES},
    "own_40": {"finetune_full": SCORES, "tfidf": SCORES, "most_common": SCORES},
}


@pytest.fixture(scope="module")
def saved_model(tmp_path_factory):
    folder = tmp_path_factory.mktemp("saved")
    model = build_model("head", seed=0)
    model.save_pretrained(folder)
    load_tokenizer().save_pretrained(folder)
    return folder, model


def test_saved_model_reloads_with_the_same_predictions(saved_model):
    folder, model = saved_model
    tokenizer = load_tokenizer()
    collator = make_collator(tokenizer)
    _, validation, _ = load_split()
    items = encode(validation.head(8), tokenizer)
    reloaded = AutoModelForSequenceClassification.from_pretrained(folder)
    before = predict_ids(model, items, collator)
    after = predict_ids(reloaded, items, collator)
    assert before == after


def test_model_card_has_required_headings_and_settings():
    card = build_model_card(FINAL)
    assert card.startswith("---")
    for key in ("license:", "base_model:", "pipeline_tag:"):
        assert key in card
    for heading in (
        "## Intended use",
        "## Training data",
        "## Results",
        "## Limitations",
    ):
        assert heading in card
    assert "400" in card
    assert "invented" in card.split("## Limitations")[1]


def test_app_scores_every_queue_and_they_sum_to_one(saved_model):
    folder, _ = saved_model
    scores, message = score_text("The dock light stays red", model_id=str(folder))
    assert set(scores) == set(LABELS)
    assert sum(scores.values()) == pytest.approx(1.0, abs=1e-5)
    assert message == ""


@pytest.mark.parametrize("text", ["", "   ", None])
def test_app_gives_a_clear_message_for_empty_text(text):
    scores, message = score_text(text)
    assert scores == {}
    assert message == EMPTY_MESSAGE
