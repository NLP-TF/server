import torch
from .load_model import model, tokenizer


def predict_tf_style(text: str) -> dict:
    inputs = tokenizer(
        text, return_tensors="pt", padding="max_length", truncation=True, max_length=256
    )
    with torch.no_grad():
        logits = model(inputs["input_ids"], inputs["attention_mask"])
        probs = torch.softmax(logits, dim=1).squeeze()

    return {
        "T_prob": round(probs[0].item() * 100, 2),
        "F_prob": round(probs[1].item() * 100, 2),
    }
