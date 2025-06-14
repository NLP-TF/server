import torch
from typing import Dict, Any
from .load_model import get_model, get_tokenizer

# Get model and tokenizer instances
model = get_model()
tokenizer = get_tokenizer()


def predict_tf_style(text: str) -> Dict[str, float]:
    """
    Predict whether the input text is more T (Thinking) or F (Feeling) style.

    Args:
        text: Input text to analyze

    Returns:
        Dictionary with T_prob and F_prob percentages
    """
    try:
        # Tokenize input text
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=256,
        )

        # Move inputs to the same device as model
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get model predictions
        with torch.no_grad():
            logits = model(**inputs)
            if isinstance(logits, tuple):
                logits = logits[0]  # Handle case where model returns tuple
            probs = torch.softmax(logits, dim=1).squeeze()

        # Return probabilities as percentages
        return {
            "T_prob": round(probs[0].item() * 100, 2) if probs.dim() > 0 else 50.0,
            "F_prob": round(probs[1].item() * 100, 2) if probs.dim() > 0 else 50.0,
        }

    except Exception as e:
        print(f"Error in predict_tf_style: {e}")
        # Return neutral probabilities in case of error
        return {"T_prob": 50.0, "F_prob": 50.0}
