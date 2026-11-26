---
title: Lakeshore Ticket Sorter
emoji: 🚲
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.32.0
app_file: app.py
pinned: false
---

# Lakeshore Ticket Sorter

Type a short bike share support message and see a score for each queue:
billing, bike_fault, station and account.

The model is a fine-tuned DistilBERT trained on invented messages, so it is a
teaching demo and not a tool for real customer messages.

Model: https://huggingface.co/Gaelletheresa/lakeshore-ticket-sorter