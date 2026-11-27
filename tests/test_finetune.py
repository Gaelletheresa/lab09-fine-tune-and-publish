import pandas as pd

from finetune import LABEL2ID, MAX_LENGTH, encode, load_tokenizer


def test_encode_truncates_long_messages_and_keeps_labels():
    tokenizer = load_tokenizer()
    frame = pd.DataFrame(
        {
            "text": ["bike " * 500, "short message"],
            "label": ["station", "billing"],
        }
    )
    items = encode(frame, tokenizer)
    assert len(items[0]["input_ids"]) == MAX_LENGTH
    assert len(items[1]["input_ids"]) < MAX_LENGTH
    labels = [item["labels"] for item in items]
    assert labels == [LABEL2ID["station"], LABEL2ID["billing"]]
