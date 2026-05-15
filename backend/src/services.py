import os
from typing import AsyncGenerator, Union
from groq import AsyncGroq
from groq.types.chat import ChatCompletionMessageParam

_client = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"))

GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_TEMPERATURE = 0.7
GROQ_MAX_TOKENS = 2048

# Pricing in USD per 1,000,000 tokens
PRICING_TIERS = {
    "llama-3.3-70b-versatile": {
        "input": 0.59,
        "output": 0.79
    },
    "meta-llama/llama-4-scout-17b-16e-instruct": {
        "input": 0.11,  # Free tier on Groq
        "output": 0.34
    },
    "openai/gpt-oss-120b": {
        "input": 0.15,
        "output": 0.60
    },
    "qwen/qwen3-32b": {
        "input": 0.29,
        "output": 0.59
    }
}

async def groq_stream(messages: list[ChatCompletionMessageParam], model: str) -> AsyncGenerator[Union[str, dict[str, object]], None]:
    stream = await _client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        temperature=GROQ_TEMPERATURE,
        max_tokens=GROQ_MAX_TOKENS,
    )
    
    async for chunk in stream:
        # 1. Yield the text tokens as normal
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
            
        # 2. Robust Usage Extraction
        usage_obj = getattr(chunk, "usage", None)
        
        if not usage_obj:
            # Bypass strict attribute checking for the undocumented field
            x_groq_obj = getattr(chunk, "x_groq", None)  # type: ignore
            
            if x_groq_obj:
                if isinstance(x_groq_obj, dict):
                    usage_obj = x_groq_obj.get("usage")  # type: ignore
                else:
                    usage_obj = getattr(x_groq_obj, "usage", None)  # type: ignore
                    
        # 3. Safely extract the integers
        if usage_obj:
            p_tokens = 0
            c_tokens = 0
            t_tokens = 0
            
            # Bypass strict dict/Any method checking
            if isinstance(usage_obj, dict):
                p_tokens = usage_obj.get("prompt_tokens", 0)  # type: ignore
                c_tokens = usage_obj.get("completion_tokens", 0)  # type: ignore
                t_tokens = usage_obj.get("total_tokens", 0)  # type: ignore
            else:
                p_tokens = getattr(usage_obj, "prompt_tokens", 0)  # type: ignore
                c_tokens = getattr(usage_obj, "completion_tokens", 0)  # type: ignore
                t_tokens = getattr(usage_obj, "total_tokens", 0)  # type: ignore
                
            yield {
                "model": chunk.model,
                "prompt_tokens": p_tokens,
                "completion_tokens": c_tokens,
                "total_tokens": t_tokens
            }

async def generate_conversation_title(question: str, answer: str) -> str:
    response = await _client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Generate a short 4-6 word title for a conversation that starts with:\n"
                    f"User: {question[:200]}\nAssistant: {answer[:200]}\n"
                    f"Reply with only the title, no quotes or punctuation."
                ),
            }
        ],
        max_tokens=20,
    )
    content = response.choices[0].message.content
    return content.strip() if content else ""