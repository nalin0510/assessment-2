"""Run a small CPU inference with the public TAPAS table QA checkpoint."""
import os
from pathlib import Path
import warnings

# Keep downloaded model files out of the user's global cache and this repository's git history.
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
import torch
from transformers import TapasForQuestionAnswering, TapasTokenizer

MODEL = "google/tapas-base-finetuned-wtq"

table = pd.DataFrame(
    {
        "Country": ["France", "Germany", "Italy"],
        "Capital": ["Paris", "Berlin", "Rome"],
        "Population (millions)": ["67", "83", "59"],
    }
)
question = "What is the capital of France?"

tokenizer = TapasTokenizer.from_pretrained(MODEL, clean_up_tokenization_spaces=True)
model = TapasForQuestionAnswering.from_pretrained(MODEL)
inputs = tokenizer(table=table, queries=question, padding="max_length", return_tensors="pt")
with torch.inference_mode():
    outputs = model(**inputs)

(coordinates,) = tokenizer.convert_logits_to_predictions(inputs, outputs.logits.detach())
answer = ", ".join(str(table.iat[row, col]) for row, col in coordinates[0]) or "<no cell selected>"
print("TAPAS table question answering - CPU inference")
print(f"Model: {MODEL}")
print(f"Question: {question}")
print(f"Answer: {answer}")
print("Status: SUCCESS")
