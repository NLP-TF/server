import os
import torch
import torch.nn as nn
import logging
import sys
import psutil
import platform
import time
from typing import Dict, Any, Optional, Tuple
from transformers import BertModel, BertForSequenceClassification, AutoTokenizer
import torch.nn.functional as F
from functools import lru_cache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("model_loading.log"),
    ],
)
logger = logging.getLogger(__name__)

# Global variables for model and tokenizer
model = None
tokenizer = None


def log_system_info():
    """Log system and environment information"""
    logger.info("=" * 50)
    logger.info("System Information")
    logger.info("=" * 50)
    logger.info(f"Python version: {sys.version}")
    logger.info(f"System: {platform.system()} {platform.release()}")
    logger.info(f"Processor: {platform.processor()}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"RAM: {psutil.virtual_memory().total / (1024**3):.2f} GB")
    logger.info("=" * 50 + "\n")


# Log system info when module is loaded
# log_system_info()  # 주석 처리 - 필요할 때만 명시적으로 호출하세요


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


def log_memory_usage(stage: str = ""):
    """Log current memory usage"""
    process = psutil.Process()
    mem_info = process.memory_info()
    logger.info(
        f"[Memory] {stage} - "
        f"RSS: {mem_info.rss / 1024 / 1024:.2f}MB | "
        f"VMS: {mem_info.vms / 1024 / 1024:.2f}MB | "
        f"CPU: {process.cpu_percent()}%"
    )


def log_gpu_info(stage: str = ""):
    """Log GPU memory usage if available"""
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.memory_allocated() / 1024**2
        gpu_mem_reserved = torch.cuda.memory_reserved() / 1024**2
        logger.info(
            f"[GPU] {stage} - "
            f"Allocated: {gpu_mem:.2f}MB | "
            f"Reserved: {gpu_mem_reserved:.2f}MB"
        )


def load_model_and_tokenizer():
    global model, tokenizer

    start_time = time.time()
    logger.info("🚀 Starting model and tokenizer loading...")
    log_memory_usage("Before loading")
    log_gpu_info("Before loading")

    if model is not None and tokenizer is not None:
        logger.info("✅ Model and tokenizer already loaded. Skipping...")
        return

    try:
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            logger.info("🧹 Clearing CUDA cache...")
            torch.cuda.empty_cache()
            log_gpu_info("After clearing cache")

        # 1. Load tokenizer
        logger.info("1. 🔄 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            "monologg/kobert",
            trust_remote_code=True,
            force_download=False,  # 기본값은 False로 설정
        )
        logger.info(f"✅ Tokenizer loaded. Type: {type(tokenizer).__name__}")
        log_memory_usage("After tokenizer load")

        # 2. Initialize model
        logger.info("\n2. 🔄 Initializing model architecture...")
        model = KoBERT_TF_Model()
        logger.info("✅ Model architecture initialized.")
        log_memory_usage("After model init")
        log_gpu_info("After model init")

        # 3. Download model weights
        logger.info("\n3. ⬇️  Downloading model weights...")
        from huggingface_hub import hf_hub_download, HfApi, HfFolder

        # Log Hugging Face cache info
        cache_dir = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        logger.info(f"Hugging Face cache directory: {cache_dir}")

        # Check if weights are already downloaded
        model_path = hf_hub_download(
            repo_id="yniiiiii/kobert-tf-model-1",
            filename="pytorch_model.bin",
            force_download=False,
        )
        logger.info(f"✅ Model weights downloaded to: {model_path}")
        logger.info(f"File size: {os.path.getsize(model_path) / 1024**2:.2f} MB")

        # 4. Load model weights
        logger.info("\n4. 🔄 Loading model weights...")
        logger.info(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

        # Load with error handling for state dict
        try:
            state_dict = torch.load(
                model_path,
                map_location=torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                ),
            )
            logger.info(
                f"State dict keys: {list(state_dict.keys())[:5]}... (truncated)"
            )

            # Load with strict=False to ignore missing keys
            missing_keys, unexpected_keys = model.load_state_dict(
                state_dict, strict=False
            )

            if missing_keys:
                logger.warning(
                    f"Missing keys in state dict: {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}"
                )
            if unexpected_keys:
                logger.warning(
                    f"Unexpected keys in state dict: {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}"
                )

            model.eval()
            logger.info("✅ Model weights loaded and set to evaluation mode.")

        except Exception as e:
            logger.error(f"❌ Error loading model weights: {str(e)}", exc_info=True)
            raise

        log_memory_usage("After model load")
        log_gpu_info("After model load")

        # 5. Run test prediction
        logger.info("\n5. 🧪 Running test prediction...")
        test_text = "테스트 문장입니다."
        logger.info(f"Test input: '{test_text}'")

        try:
            inputs = tokenizer(
                test_text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=256,
            )

            # Move inputs to the same device as model
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                logits = model(
                    inputs["input_ids"],
                    inputs["attention_mask"],
                    situation_id=torch.tensor([6], device=device),
                )
                probs = torch.softmax(logits, dim=1).squeeze()
                t_prob = float(probs[0]) * 100
                f_prob = float(probs[1]) * 100
                logger.info(f"✅ Test prediction - T: {t_prob:.2f}% | F: {f_prob:.2f}%")

        except Exception as e:
            logger.error(f"❌ Test prediction failed: {str(e)}", exc_info=True)
            raise

        logger.info("\n🎉 Model and tokenizer loaded successfully!")
        logger.info(f"⏱️  Total loading time: {time.time() - start_time:.2f} seconds")

    except Exception as e:
        logger.critical("❌ Critical error during model loading!", exc_info=True)
        model = None
        tokenizer = None
        raise
    finally:
        log_memory_usage("After loading")
        log_gpu_info("After loading")


def predict_tf_style(text: str, situation: str = "친구_갈등") -> Dict[str, str]:
    """
    Predict T/F style classification using the loaded model.
    
    Args:
        text: Input text
        situation: Situation type (default: "친구_갈등")
        
    Returns:
        Dictionary with T and F probabilities
    """
    global model, tokenizer
    
    logger.info("\n" + "="*50)
    logger.info(f"🔍 Starting prediction - Situation: {situation}")
    logger.info(f"📝 Input text: {text[:100]}{'...' if len(text) > 100 else ''}")
    start_time = time.time()
    
    try:
        # Check if model and tokenizer are loaded
        if model is None or tokenizer is None:
            error_msg = "❌ Model or tokenizer not loaded"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Map situation to ID (0-6)
        situation_map = {
            "가족_갈등": 0,
            "기타": 1,
            "미래_고민": 2,
            "실수_자책": 3,
            "연인_갈등": 4,
            "직장_갈등": 5,
            "친구_갈등": 6,
        }

        situation_id = torch.tensor(
            [situation_map.get(situation, 6)]
        )  # Default to "친구_갈등"
        logger.info(f"🎯 Using situation ID: {situation_id.item()} ({situation})")

        # Tokenize input
        tokenization_start = time.time()
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=256,
        )
        logger.info(
            f"✅ Tokenization completed in {time.time() - tokenization_start:.2f}s"
        )
        logger.debug(f"Input IDs shape: {inputs['input_ids'].shape}")
        logger.debug(f"Attention mask shape: {inputs['attention_mask'].shape}")

        # Move inputs to the same device as model
        device = next(model.parameters()).device
        logger.info(f"⚙️  Using device: {device}")
        
        move_start = time.time()
        inputs = {k: v.to(device) for k, v in inputs.items()}
        situation_id = situation_id.to(device)
        logger.info(f"✅ Inputs moved to device in {time.time() - move_start:.2f}s")

        # Inference
        inference_start = time.time()
        with torch.no_grad():
            logits = model(
                inputs["input_ids"],
                inputs["attention_mask"],
                situation_id=situation_id,
            )
            probs = torch.softmax(logits, dim=1).squeeze()
            inference_time = time.time() - inference_start
            logger.info(f"✅ Inference completed in {inference_time:.2f}s")

        # Calculate probabilities
        t_prob = float(probs[0]) * 100
        f_prob = float(probs[1]) * 100

        logger.info(f"📊 Prediction results - T: {t_prob:.2f}% | F: {f_prob:.2f}%")

        total_time = time.time() - start_time
        logger.info(f"✨ Total prediction time: {total_time:.2f} seconds")
        logger.info("=" * 50 + "\n")

        return {"T": f"{t_prob:.2f}%", "F": f"{f_prob:.2f}%"}

    except Exception as e:
        logger.error(f"❌ Error during prediction: {str(e)}", exc_info=True)
        logger.error(f"💡 Input text that caused error: {text}")
        logger.error(f"💡 Situation: {situation}")
        raise


def load_models_if_needed():
    """
    Load models if they are not already loaded.

    This function ensures that both the model and tokenizer are loaded
    before they are used for prediction.
    """
    global model, tokenizer

    logger.debug("🔍 Checking if models need to be loaded...")

    if model is None or tokenizer is None:
        logger.info("🔄 One or more models not loaded. Loading now...")
        load_model_and_tokenizer()
    else:
        logger.debug("✅ Models are already loaded")


def get_model():
    """
    Get the model instance.

    Returns:
        The loaded model instance
    """
    logger.debug("🔍 Getting model instance...")
    load_models_if_needed()

    if model is None:
        error_msg = "❌ Model is not available"
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.debug("✅ Model instance retrieved successfully")
    return model


def get_tokenizer():
    """
    Get the tokenizer instance.

    Returns:
        The loaded tokenizer instance
    """
    logger.debug("🔍 Getting tokenizer instance...")
    load_models_if_needed()

    if tokenizer is None:
        error_msg = "❌ Tokenizer is not available"
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.debug("✅ Tokenizer instance retrieved successfully")
    return tokenizer


@lru_cache(maxsize=1)
def get_model_and_tokenizer():
    """
    Get both model and tokenizer instances.

    Returns:
        Tuple of (model, tokenizer)
    """
    global model, tokenizer
    load_models_if_needed()

    if model is None or tokenizer is None:
        error_msg = f"❌ Model or tokenizer not available (model: {model is not None}, tokenizer: {tokenizer is not None})"
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.debug("✅ Both model and tokenizer instances retrieved successfully")
    return model, tokenizer
