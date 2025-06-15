import os
import torch
import torch.nn as nn
import logging
from typing import Dict, Any, Optional
from transformers import AutoModel, BertTokenizer
from huggingface_hub import hf_hub_download, HfApi
from peft import get_peft_model, LoraConfig, TaskType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# LoRA configuration
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=True,
    r=32,
    lora_alpha=128,
    lora_dropout=0.1,
    target_modules=["query", "value"],
)


def patch_bert_forward(peft_model):
    """Patch BERT forward to remove labels argument"""
    original_forward = peft_model.model.forward

    def new_forward(*args, **kwargs):
        kwargs.pop("labels", None)
        return original_forward(*args, **kwargs)

    peft_model.model.forward = new_forward
    return peft_model


class KoBERT_TF_Model(nn.Module):
    def __init__(self, hidden_size=768, num_classes=2):
        super().__init__()
        try:
            logger.info("Loading base KoBERT model...")
            base_model = AutoModel.from_pretrained(
                "monologg/kobert", trust_remote_code=True
            )
            logger.info("Applying LoRA configuration...")
            lora_model = get_peft_model(base_model, lora_config)
            self.bert = patch_bert_forward(lora_model)

            self.classifier = nn.Sequential(
                nn.Linear(hidden_size, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes),
            )
            logger.info("Model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_vec = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        return self.classifier(cls_vec)


# Initialize model and tokenizer
model: Optional[torch.nn.Module] = None
tokenizer = None

# Create model cache directory if it doesn't exist
os.makedirs("model_cache", exist_ok=True)


def load_model_and_tokenizer():
    global model, tokenizer

    # Clear any existing model and tokenizer to ensure fresh load
    model = None
    tokenizer = None

    try:
        logger.info("=" * 50)
        logger.info("Starting model and tokenizer loading process...")
        logger.info("=" * 50)

        # Load tokenizer first
        logger.info("\n1. Loading KoBERT tokenizer...")
        try:
            tokenizer = BertTokenizer.from_pretrained(
                "monologg/kobert", cache_dir="model_cache"
            )
            logger.info(
                f"✅ Tokenizer loaded successfully. Type: {type(tokenizer).__name__}"
            )
            logger.debug(f"Tokenizer config: {tokenizer}")
        except Exception as e:
            logger.error(f"❌ Failed to load tokenizer: {str(e)}", exc_info=True)
            raise

        # Download model weights
        logger.info("\n2. Downloading model weights...")
        try:
            model_path = hf_hub_download(
                repo_id="yniiiiii/kobert-tf-model-lora-nositu",
                filename="pytorch_model.bin",
                cache_dir="model_cache",
            )
            logger.info(f"✅ Model weights downloaded to: {model_path}")
            logger.debug(
                f"Model file size: {os.path.getsize(model_path) / (1024*1024):.2f} MB"
            )
        except Exception as e:
            logger.error(
                f"❌ Failed to download model weights: {str(e)}", exc_info=True
            )
            raise

        # Initialize model
        logger.info("\n3. Initializing model architecture...")
        try:
            # Initialize the custom model with LoRA
            model = KoBERT_TF_Model()
            logger.info(
                f"✅ Model architecture initialized. Type: {type(model).__name__}"
            )
            logger.debug(f"Model structure: {model}")

            # Load the state dict
            logger.info("\n4. Loading model weights...")
            try:
                state_dict = torch.load(model_path, map_location="cpu")
                logger.info(f"✅ State dict loaded. Number of keys: {len(state_dict)}")
                logger.debug(f"First 5 keys: {list(state_dict.keys())[:5]}...")

                # Handle state dict with unexpected keys
                if any(k.startswith("module.") for k in state_dict.keys()):
                    logger.info("Removing 'module.' prefix from state dict keys")
                    state_dict = {
                        k.replace("module.", ""): v for k, v in state_dict.items()
                    }

                # Load state dict with strict=False to ignore missing keys
                logger.info("Loading state dict into model...")
                missing_keys, unexpected_keys = model.load_state_dict(
                    state_dict, strict=False
                )

                if missing_keys:
                    logger.warning(f"⚠️  Missing keys in state_dict: {missing_keys}")
                if unexpected_keys:
                    logger.warning(
                        f"⚠️  Unexpected keys in state_dict: {unexpected_keys}"
                    )

                model.eval()
                logger.info("✅ Model loaded and set to eval mode")

                # Test a prediction to verify the model works
                logger.info("\n5. Running test prediction...")
                with torch.no_grad():
                    test_text = "테스트 문장입니다."
                    logger.info(f"Test input: '{test_text}'")
                    test_input = tokenizer(test_text, return_tensors="pt")

                    # Remove token_type_ids if present
                    if "token_type_ids" in test_input:
                        del test_input["token_type_ids"]

                    logger.debug(f"Tokenized test input: {test_input}")

                    # Move inputs to the same device as model
                    device = next(model.parameters()).device
                    test_input = {k: v.to(device) for k, v in test_input.items()}

                    output = model(**test_input)
                    logger.info(
                        f"✅ Test prediction successful. Output shape: {output.shape}"
                    )
                    logger.debug(f"Raw output: {output}")

                    # Apply softmax to get probabilities
                    probs = torch.softmax(output, dim=-1).squeeze()
                    logger.info(f"Test prediction probabilities: {probs.tolist()}")

            except Exception as e:
                logger.error(f"❌ Error loading model weights: {str(e)}", exc_info=True)
                raise

            logger.info("\n✅ Model and tokenizer loaded successfully!")
            logger.info("=" * 50)

        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}", exc_info=True)
            raise

        logger.info("Model and tokenizer loaded successfully!")

    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        logger.warning("Using mock model instead")

        # Initialize mock model for testing
        class MockModel:
            def __call__(self, *args, **kwargs):
                return torch.tensor([[0.69, 0.31]])

            def to(self, device):
                return self

            def eval(self):
                return self

            def parameters(self):
                return [torch.tensor([0.0])]

        model = MockModel()
        tokenizer = BertTokenizer.from_pretrained(
            "monologg/kobert"
        )


# Load the model and tokenizer when this module is imported
if __name__ != "__main__":
    load_model_and_tokenizer()


def get_model():
    """Get the model instance."""
    global model
    return model


def get_tokenizer():
    """Get the tokenizer instance."""
    global tokenizer
    return tokenizer


def get_model_and_tokenizer():
    """Get both model and tokenizer instances."""
    global model, tokenizer
    return model, tokenizer
