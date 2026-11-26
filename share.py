"""Write the model card for the saved model."""

import json
from pathlib import Path

from finetune import SETTINGS

NAMES = {
    "finetune_full": "DistilBERT, full fine-tuning",
    "finetune_head": "DistilBERT, head only",
    "tfidf": "TF-IDF + logistic regression",
    "most_common": "Most common label",
}

FRONT_MATTER = """---
license: apache-2.0
base_model: distilbert/distilbert-base-uncased
pipeline_tag: text-classification
language: en
---
"""


def results_table(scores):
    """A markdown table with one row per method."""
    lines = ["| Method | Accuracy | Macro F1 |", "|---|---|---|"]
    for key, name in NAMES.items():
        if key in scores:
            s = scores[key]
            lines.append(f"| {name} | {s['accuracy']:.3f} | {s['macro_f1']:.3f} |")
    return "\n".join(lines)


def build_model_card(final):
    """Build the model card text from the saved final scores."""
    size = final["train_size"]
    mode = final["final_method"].removeprefix("finetune_")
    setting = SETTINGS[mode]
    return f"""{FRONT_MATTER}
# Lakeshore ticket sorter

DistilBERT fine-tuned to sort short bike share support messages into four
queues: billing, bike_fault, station and account.

## Intended use

Suggest the queue for a short support message, for a person to confirm.
It is a teaching project and not meant to route real customer messages
without a human check.

## Training data

The model was trained on {size} messages, drawn with a fixed seed from the
training part (720 rows) of 1,200 invented messages. The data was split
60/20/20 into training, validation and test rows. Fine-tuning used the whole
model, a learning rate of {setting["lr"]} and {setting["epochs"]} epochs,
chosen on the validation rows.

## Results

All methods were trained on the same {size} rows and scored on the same rows.

Test rows (240 messages):

{results_table(final["test"])}

My own 40 messages, scored once:

{results_table(final["own_40"])}

This is one seed on one test set, so small differences are not reliable.

## Limitations

The training text was invented, and it repeats a small set of sentence
patterns. Real customers will write differently. About one message in eight
in the data says nothing about its topic, so no model can get every label
right. The model knows four queues only and has not been tested on real
support messages.
"""


def main():
    final = json.loads(Path("reports/final.json").read_text())
    Path("model/README.md").write_text(build_model_card(final))
    print("wrote model/README.md")


if __name__ == "__main__":
    main()