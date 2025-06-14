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

def predict_tf_style(text: str) -> Dict[str, float]:
    """
    Predicts whether the given text is more T (Thinking) or F (Feeling) style.
    Returns a dictionary with T_prob and F_prob as percentages.
    """
    global model, tokenizer
    
    logger.info("\n" + "="*50)
    logger.info(f"📝 New prediction request for text: '{text}'")
    
    if model is None or tokenizer is None:
        logger.error("❌ Model or tokenizer not loaded, using default values")
        return {"T_prob": 50.0, "F_prob": 50.0}

    try:
        # Tokenize input
        logger.info("🔡 Tokenizing input text...")
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )
        
        # Remove token_type_ids if present
        if 'token_type_ids' in inputs:
            del inputs['token_type_ids']
            
        logger.debug(f"📊 Tokenized inputs: {inputs}")

        # Move inputs to the same device as model
        device = next(model.parameters()).device if hasattr(model, 'parameters') else 'cpu'
        inputs = {k: v.to(device) for k, v in inputs.items()}
        logger.info(f"⚙️  Moved inputs to device: {device}")

        try:
            with torch.no_grad():
                logger.info("🤖 Running model inference...")
                outputs = model(**inputs)
                logger.debug(f"📦 Raw model outputs: {outputs}")
                
                # Get logits
                if isinstance(outputs, torch.Tensor):
                    logits = outputs
                    logger.debug("📊 Output is a tensor")
                elif hasattr(outputs, 'logits'):
                    logits = outputs.logits
                    logger.debug("📊 Output has 'logits' attribute")
                else:
                    logits = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
                    logger.debug(f"📊 Extracted output from {type(outputs)}")
                
                logger.info(f"📐 Logits shape: {logits.shape}")
                logger.debug(f"🔢 Logits values: {logits}")
                
                # Ensure we have a 2D tensor [batch_size, num_classes]
                if logits.dim() == 1:
                    logger.debug("🔧 Unsqueezing logits to add batch dimension")
                    logits = logits.unsqueeze(0)
                
                # Apply softmax to get probabilities
                probs = torch.softmax(logits, dim=1).squeeze()
                logger.info(f"📊 Probabilities: {probs.tolist()}")
                
                # Convert to percentages
                if probs.dim() == 0:  # Single sample, binary
                    t_prob = float(probs) * 100
                    f_prob = 100.0 - t_prob
                    logger.debug("📊 Single probability value detected")
                else:  # Batch or multi-class
                    t_prob = float(probs[0]) * 100  # First class (T)
                    f_prob = float(probs[1]) * 100  # Second class (F)
                    logger.debug(f"📊 Multiple probabilities: T={t_prob:.2f}%, F={f_prob:.2f}%")
                
                # Ensure probabilities sum to 100%
                total = t_prob + f_prob
                t_prob = round((t_prob / total) * 100, 2)
                f_prob = round(100.0 - t_prob, 2)
                
                logger.info(f"✅ Final prediction - T: {t_prob}%, F: {f_prob}%")
                logger.info("="*50 + "\n")
                return {"T_prob": t_prob, "F_prob": f_prob}
                
        except Exception as e:
            logger.error(f"❌ Error during model inference: {str(e)}", exc_info=True)
            logger.info("="*50 + "\n")
            return {"T_prob": 50.0, "F_prob": 50.0}
            
    except Exception as e:
        logger.error(f"❌ Error during tokenization or processing: {str(e)}", exc_info=True)
        logger.info("="*50 + "\n")
        return {"T_prob": 50.0, "F_prob": 50.0}
