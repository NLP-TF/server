import os
import torch
import torch.nn as nn
from typing import Tuple, Dict, Any
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import hf_hub_download


# Define the KoBERT T/F style classification model
class KoBERT_TF_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained("monologg/kobert")
        self.classifier = nn.Linear(768, 2)  # Binary classification (T/F)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # [CLS] token
        logits = self.classifier(pooled_output)
        return logits


# Try to import the actual model, but fall back to a mock if not available
try:
    from transformers import AutoModel

    # Try to load the model and tokenizer
    try:
        print("Loading KoBERT model and tokenizer...")

        # Download and load the model
        model_path = hf_hub_download(
            repo_id="yniiiiii/kobert-tf-model-lora-nositu", filename="pytorch_model.bin"
        )

        # Initialize model and load weights
        model = KoBERT_TF_Model()
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            "monologg/kobert", trust_remote_code=True
        )

        print("Model and tokenizer loaded successfully!")
        USE_MOCK = False

    except Exception as e:
        print(f"Warning: Could not load model, using mock model instead. Error: {e}")
        USE_MOCK = True

except ImportError as e:
    print(
        f"Warning: Required packages not found, using mock model for testing. Error: {e}"
    )
    USE_MOCK = True

# Mock model for testing
if "USE_MOCK" not in locals() or USE_MOCK:
    print("Using mock model for testing")

    class MockModel:
        def __init__(self):
            self.config = type("", (), {"num_labels": 2})()

        def __call__(self, input_ids, attention_mask=None):
            # Return random logits for testing
            batch_size = input_ids.shape[0]
            return torch.randn(batch_size, 2)

        def eval(self):
            pass

        def to(self, device):
            return self

    class MockTokenizer:
        def __init__(self):
            pass

        def __call__(self, text, **kwargs):
            return {
                "input_ids": torch.zeros(1, 10, dtype=torch.long),
                "attention_mask": torch.ones(1, 10, dtype=torch.long),
            }

    model = MockModel()
    tokenizer = MockTokenizer()


def get_model():
    """Get the model instance."""
    return model


def get_tokenizer():
    """Get the tokenizer instance."""
    return tokenizer
