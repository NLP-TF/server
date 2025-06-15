import os
import torch
import torch.nn as nn
import logging
from typing import Dict, Any, Optional, Tuple
from transformers import BertModel, BertForSequenceClassification, BertTokenizer
import torch.nn.functional as F

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model and tokenizer
model = None
tokenizer = None


class KoBERT_TF_Model(nn.Module):
    def __init__(self, num_situations=7, num_labels=2):
        super().__init__()
        # Load base BERT model
        self.bert = BertModel.from_pretrained("monologg/kobert")

        # Situation embedding
        self.situation_embedding = nn.Embedding(num_situations, 768)

        # Classifier
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask, situation_id=None):
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )

        # Get [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]

        # Add situation embedding if provided
        if situation_id is not None:
            situation_emb = self.situation_embedding(situation_id)
            cls_output = cls_output + situation_emb.squeeze(1)

        # Get logits
        logits = self.classifier(cls_output)
        return logits


def load_model_and_tokenizer():
    """Load the model and tokenizer"""
    global model, tokenizer

    if model is not None and tokenizer is not None:
        return

    try:
        logger.info("1. Loading tokenizer...")
        tokenizer = BertTokenizer.from_pretrained("monologg/kobert")

        logger.info("2. Loading model...")
        model = KoBERT_TF_Model()

        # Load model weights
        logger.info("3. Loading model weights...")
        model_path = "pytorch_model.bin"
        if not os.path.exists(model_path):
            logger.info("Downloading model weights...")
            try:
                from huggingface_hub import hf_hub_download

                model_path = hf_hub_download(
                    repo_id="yniiiiii/kobert-tf-model-1",
                    filename="pytorch_model.bin",
                    cache_dir="model_cache",
                )
                logger.info(f"✅ Model weights downloaded to: {model_path}")
                logger.debug(
                    f"Model file size: {os.path.getsize(model_path) / (1024*1024):.2f} MB"
                )
            except Exception as e:
                logger.error(f"❌ Failed to download model weights: {str(e)}")
                raise

        # Load the state dict
        logger.info("4. Loading model weights...")
        try:
            state_dict = torch.load(model_path, map_location=torch.device("cpu"))

            # Handle state dict with unexpected keys
            if any(k.startswith("module.") for k in state_dict.keys()):
                logger.info("Removing 'module.' prefix from state dict keys")
                state_dict = {
                    k.replace("module.", ""): v for k, v in state_dict.items()
                }

            # Load state dict with strict=False to handle potential mismatches
            missing_keys, unexpected_keys = model.load_state_dict(
                state_dict, strict=False
            )

            if missing_keys:
                logger.warning(f"Missing keys in state dict: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys in state dict: {unexpected_keys}")

            logger.info("✅ Model weights loaded successfully.")

        except Exception as e:
            logger.error(f"❌ Failed to load model weights: {str(e)}")
            raise

        # Set to eval mode
        model.eval()
        logger.info("✅ Model set to evaluation mode.")

        # Run a test prediction
        logger.info("\n4. Running test prediction...")
        try:
            test_text = "테스트 문장입니다."
            logger.info(f"Test input: '{test_text}'")
            result = predict_tf_style(test_text)
            logger.info(f"Test prediction: {result}")
        except Exception as e:
            logger.warning(f"⚠️ Test prediction failed: {str(e)}")

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


# Load the model and tokenizer when this module is imported
if __name__ != "__main__":
    load_model_and_tokenizer()


def get_model():
    """Get the model instance."""
    if model is None:
        load_model_and_tokenizer()
    return model


def get_tokenizer():
    """Get the tokenizer instance."""
    if tokenizer is None:
        load_model_and_tokenizer()
    return tokenizer


def get_model_and_tokenizer():
    """Get both model and tokenizer instances."""
    load_model_and_tokenizer()
    return model, tokenizer
