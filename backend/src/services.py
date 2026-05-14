import os
from typing import AsyncGenerator, Union
from groq import AsyncGroq
from groq.types.chat import ChatCompletionMessageParam

_client = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"))

GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_TEMPERATURE = 0.7
GROQ_MAX_TOKENS = 2048

async def groq_stream(messages: list[ChatCompletionMessageParam], model: str = GROQ_MODEL) -> AsyncGenerator[Union[str, dict[str, object]], None]:
    stream = await _client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        temperature=GROQ_TEMPERATURE,
        max_tokens=GROQ_MAX_TOKENS,
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta

    yield {"model": model, "temperature": GROQ_TEMPERATURE, "max_tokens": GROQ_MAX_TOKENS}

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