import os
import torch
import torch.nn as nn
import logging
from typing import Dict, Any, Optional, Tuple
from transformers import BertModel, BertForSequenceClassification, AutoTokenizer
import torch.nn.functional as F

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model and tokenizer
model = None
tokenizer = None


class KoBERT_TF_Model(nn.Module):
    def __init__(
        self, hidden_size=768, num_classes=2, situation_dim=64, num_situations=7
    ):
        super().__init__()
        # Load base BERT model
        self.bert = BertModel.from_pretrained("monologg/kobert")

        # Situation embedding with smaller dimension
        self.situ_embed = nn.Embedding(num_situations, situation_dim)

        # Classifier with late fusion architecture
        self.classifier = nn.Sequential(
            nn.Linear(
                hidden_size + situation_dim, 256
            ),  # Concatenate BERT CLS and situation embedding
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, input_ids, attention_mask, situation_id=None):
        # Get BERT outputs
        bert_out = self.bert(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )

        # Get [CLS] token representation
        cls_vec = bert_out.last_hidden_state[:, 0, :]

        # Get situation embedding and concatenate with CLS vector
        if situation_id is not None:
            situ_vec = self.situ_embed(situation_id)
            combined = torch.cat([cls_vec, situ_vec.squeeze(1)], dim=1)
            return self.classifier(combined)
        else:
            # If no situation_id is provided, just use zeros for compatibility
            batch_size = cls_vec.size(0)
            device = cls_vec.device
            situ_vec = torch.zeros(batch_size, 64, device=device)  # 64 is situation_dim
            combined = torch.cat([cls_vec, situ_vec], dim=1)
            return self.classifier(combined)


def load_model_and_tokenizer():
    """Load the model and tokenizer"""
    global model, tokenizer

    if model is not None and tokenizer is not None:
        return

    try:
        # Clear CUDA cache to free up memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("1. Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            "monologg/kobert", trust_remote_code=True
        )

        logger.info("2. Loading model...")
        # Initialize model with the same architecture as Colab
        model = KoBERT_TF_Model()

        # Load model weights
        logger.info("3. Downloading model weights...")
        from huggingface_hub import hf_hub_download

        model_path = hf_hub_download(
            repo_id="yniiiiii/kobert-tf-model-1", filename="pytorch_model.bin"
        )
        logger.info(f"✅ Model weights downloaded to: {model_path}")

        # Load state dict
        logger.info("4. Loading model weights...")
        state_dict = torch.load(model_path, map_location=torch.device("cpu"))

        # Load state dict with strict=False to handle any missing keys
        model.load_state_dict(state_dict, strict=False)

        # Set model to evaluation mode
        model.eval()
        logger.info("✅ Model set to evaluation mode.")

        # Test prediction
        logger.info("\n5. Running test prediction...")
        test_text = "테스트 문장입니다."
        logger.info(f"Test input: '{test_text}'")

        # Prepare inputs
        inputs = tokenizer(
            test_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=256,
        )

        # Make prediction
        with torch.no_grad():
            logits = model(
                inputs["input_ids"],
                inputs["attention_mask"],
                situation_id=torch.tensor([6]),  # 친구_갈등
            )
            probs = torch.softmax(logits, dim=1).squeeze()

            t_prob = float(probs[0]) * 100
            f_prob = float(probs[1]) * 100

            logger.info(
                f"Test prediction: {{'T 확률': '{t_prob:.2f}%', 'F 확률': '{f_prob:.2f}%'}}"
            )

        logger.info("\n✅ Model and tokenizer loaded successfully!")

    except Exception as e:
        logger.error(f"❌ Error loading model: {str(e)}")
        model = None
        tokenizer = None
        raise


def predict_tf_style(text: str, situation: str = "친구_갈등") -> Dict[str, str]:
    """
    Predict T/F style based on input text and situation.

    Args:
        text: Input text to analyze
        situation: The situation context (default: "친구_갈등")

    Returns:
        Dictionary with T and F probabilities
    """
    global model, tokenizer

    if model is None or tokenizer is None:
        load_model_and_tokenizer()

    situation_map = {
        "가족_갈등": 0,
        "기타": 1,
        "미래_고민": 2,
        "실수_자책": 3,
        "연인_갈등": 4,
        "직장_갈등": 5,
        "친구_갈등": 6,
    }

    try:
        # Convert situation to tensor
        situation_id = torch.tensor(
            [situation_map.get(situation, 6)]
        )  # default to 친구_갈등 if not found

        # Tokenize input
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=256,
        )

        # Get predictions
        with torch.no_grad():
            logits = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                situation_id=situation_id,
            )
            probs = torch.softmax(logits, dim=1).squeeze()

        return {
            "T 확률": f"{probs[0].item()*100:.2f}%",
            "F 확률": f"{probs[1].item()*100:.2f}%",
        }

    except Exception as e:
        logger.error(f"❌ Prediction error: {str(e)}")
        return {"error": str(e), "T 확률": "0.00%", "F 확률": "0.00%"}


def load_models_if_needed():
    """Load models if they are not already loaded."""
    global model, tokenizer
    if model is None or tokenizer is None:
        load_model_and_tokenizer()


def get_model():
    """Get the model instance."""
    load_models_if_needed()
    return model


def get_tokenizer():
    """Get the tokenizer instance."""
    load_models_if_needed()
    return tokenizer


def get_model_and_tokenizer():
    """Get both model and tokenizer instances."""
    load_models_if_needed()
    return model, tokenizer
