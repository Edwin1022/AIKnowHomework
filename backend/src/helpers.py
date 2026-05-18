from transformers import AutoTokenizer
from typing import List, Dict, Any, Union

TOKENIZER_CACHE: Dict[str, Any] = {}
MAX_HISTORY_TOKENS = 125000

def get_tokenizer(model_choice: str):
    # Map API model names to their Hugging Face repos
    model_map = {
        "llama-3.3-70b-versatile": "meta-llama/Llama-3.3-70B-Instruct",
        "meta-llama/llama-4-scout-17b-16e-instruct": "meta-llama/llama-4-scout-17b-16e-instruct",
        "openai/gpt-oss-120b": "openai/gpt-oss-120b",
        "qwen/qwen3-32b": "Qwen/Qwen3-32B"
    }
    
    hf_model_id = model_map.get(model_choice, model_choice)
    
    if hf_model_id not in TOKENIZER_CACHE:
        print(f"Loading tokenizer for {hf_model_id} into cache...")
        TOKENIZER_CACHE[hf_model_id] = AutoTokenizer.from_pretrained(hf_model_id)  # type: ignore
        
    return TOKENIZER_CACHE[hf_model_id]

def count_message_tokens(content: Union[str, List[Dict[str, Any]]], model_choice: str) -> int:
    tokenizer = get_tokenizer(model_choice)
    
    if isinstance(content, str):
        return len(tokenizer.encode(content)) + 5 # +5 for ChatML formatting overhead
    else:
        # Extract text from multimodal dictionaries
        text_parts = [item["text"] for item in content if item.get("type") == "text"]
        combined_text = " ".join(text_parts)
        return len(tokenizer.encode(combined_text)) + 5
    
def get_safe_token_limit(model_choice: str) -> int:
    # Note: We reserve ~2,000 tokens from the limit so the AI has room to generate a reply
    if "llama-3.3-70b-versatile" in model_choice:
        return 10000 # Groq Free Tier Limit: typically 12,000 TPM
    elif "meta-llama/llama-4-scout-17b-16e-instruct" in model_choice:
        return 28000 # Groq Free Tier Limit: typically 30,000 TPM
    elif "openai/gpt-oss-120b" in model_choice:
        return 6000  # Groq Free Tier Limit: 8,000 TPM
    elif "qwen/qwen3-32b" in model_choice:
        return 4000 # Groq Free Tier Limit: 6,000 TPM
    else:
        return 6000  # Safe default