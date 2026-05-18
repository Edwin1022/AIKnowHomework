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

def enforce_context_window(history: List[Dict[str, Any]], model_choice: str, max_tokens: int = MAX_HISTORY_TOKENS) -> List[Dict[str, Any]]:
    tokenizer = get_tokenizer(model_choice)
    
    def count_tokens(content: Union[str, List[Dict[str, Any]]]) -> int:
        # Handle standard text content
        if isinstance(content, str):
            return len(tokenizer.encode(content))
        # Handle multimodal dict payloads by extracting the text parts
        else:
            text_parts = [item["text"] for item in content if item.get("type") == "text"]
            combined_text = " ".join(text_parts)
            return len(tokenizer.encode(combined_text))

    # Calculate initial total tokens (including a small buffer for chat template overhead)
    total_tokens = sum(count_tokens(msg.get("content", "")) + 5 for msg in history)
    
    # Prune oldest messages (protecting index 0, the System Prompt)
    while total_tokens > max_tokens and len(history) > 2:
        evicted_msg = history.pop(1) 
        evicted_tokens = count_tokens(evicted_msg.get("content", "")) + 5
        total_tokens -= evicted_tokens
        print(f"Context pruning: Evicted {evicted_msg['role']} message. Freed {evicted_tokens} tokens. Total now: {total_tokens}")

    return history