"""Gradio app: suggest the support queue for a message."""

import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_ID = "Gaelletheresa/lakeshore-ticket-sorter"
MAX_LENGTH = 64
EMPTY_MESSAGE = "Please type a message first."

_loaded = {}


def load_model(model_id=MODEL_ID):
    """Load the tokenizer and model once, and reuse them."""
    if model_id not in _loaded:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSequenceClassification.from_pretrained(model_id)
        model.eval()
        _loaded[model_id] = (tokenizer, model)
    return _loaded[model_id]


def score_text(text, model_id=MODEL_ID):
    """Return a score for every queue, and a message (empty if all is well)."""
    if text is None or not text.strip():
        return {}, EMPTY_MESSAGE
    tokenizer, model = load_model(model_id)
    inputs = tokenizer(
        text, truncation=True, max_length=MAX_LENGTH, return_tensors="pt"
    )
    with torch.no_grad():
        probabilities = torch.softmax(model(**inputs).logits, dim=-1)[0]
    scores = {
        model.config.id2label[i]: float(p) for i, p in enumerate(probabilities)
    }
    return scores, ""


def classify(text):
    """Return the queue scores and a message, for the two outputs of the page."""
    scores, message = score_text(text)
    if message:
        gr.Warning(message)
    return (scores or None), message


demo = gr.Interface(
    fn=classify,
    inputs=gr.Textbox(label="Support message", lines=3),
    outputs=[
        gr.Label(label="Queue scores", num_top_classes=4),
        gr.Textbox(label="Message"),
    ],
    title="Lakeshore ticket sorter",
    description="Type a support message to see a score for each queue.",
    flagging_mode="never",
)


if __name__ == "__main__":
    demo.launch()