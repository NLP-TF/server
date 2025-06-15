import torch
import logging
from typing import Dict, Any
from .load_model import get_model, get_tokenizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get model and tokenizer instances
model = get_model()
tokenizer = get_tokenizer()

def predict_tf_style(text: str, situation: str = "친구_갈등") -> Dict[str, float]:
    """
    Predicts whether the given text is more T (Thinking) or F (Feeling) style.
    
    Args:
        text: The input text to analyze
        situation: Situation context (e.g., "친구_갈등", "연인_갈등")
        
    Returns:
        Dictionary with T_prob and F_prob as percentages.
    """
    global model, tokenizer
    
    logger.info("\n" + "="*50)
    logger.info(f"📝 New prediction request for text: '{text}'")
    logger.info(f"📌 Situation context: {situation}")
    
    if model is None or tokenizer is None:
        logger.error("❌ Model or tokenizer not loaded, using default values")
        return {"T_prob": 50.0, "F_prob": 50.0}

    try:
        # Map situation to ID
        situation_map = {
            "가족_갈등": 0,
            "기타": 1,
            "미래_고민": 2,
            "실수_자책": 3,
            "연인_갈등": 4,
            "직장_갈등": 5,
            "친구_갈등": 6
        }
        
        # Default to "친구_갈등" if situation not found
        situation_id = torch.tensor([situation_map.get(situation, 6)])
        logger.info(f"🔢 Mapped situation '{situation}' to ID: {situation_id.item()}")
            
        # Tokenize input
        logger.info("🔡 Tokenizing input text...")
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=256
        )
        
        # Move tensors to the same device as model
        device = next(model.parameters()).device if hasattr(model, 'parameters') else 'cpu'
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        situation_id = situation_id.to(device)
        
        logger.debug(f"⚙️  Moved tensors to device: {device}")
        logger.debug(f"📊 Input IDs shape: {input_ids.shape}")
        logger.debug(f"🎭 Situation ID: {situation_id}")

        with torch.no_grad():
            logger.info("🤖 Running model inference...")
            logits = model(input_ids, attention_mask, situation_id)
            logger.debug(f"📦 Raw logits: {logits}")
            
            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=1).squeeze()
            logger.info(f"📊 Probabilities: {probs.tolist()}")
            
            # Get T and F probabilities
            t_prob = float(probs[0]) * 100
            f_prob = float(probs[1]) * 100
            
            logger.info(f"📊 Final probabilities - T: {t_prob:.2f}%, F: {f_prob:.2f}%")
            logger.info(f"✅ Final prediction - T: {t_prob}%, F: {f_prob}%")
            logger.info("="*50 + "\n")
            return {"T_prob": t_prob, "F_prob": f_prob}
                
    except Exception as e:
        logger.error(f"❌ Error during model inference: {str(e)}", exc_info=True)
        logger.info("="*50 + "\n")
        return {"T_prob": 50.0, "F_prob": 50.0}
