import os
from typing import AsyncGenerator, Union
from groq import AsyncGroq
from groq.types.chat import ChatCompletionMessageParam

_client = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"))

GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_TEMPERATURE = 0.7
GROQ_MAX_TOKENS = 2048


async def groq_stream(
    messages: list[ChatCompletionMessageParam],
    model: str,
) -> AsyncGenerator[Union[str, dict[str, object]], None]:
    stream = await _client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        temperature=GROQ_TEMPERATURE,
        max_tokens=GROQ_MAX_TOKENS,
        extra_body={"stream_options": {"include_usage": True}},
    )

    usage_data: dict[str, object] = {}

    async for chunk in stream:
        if chunk.usage:
            usage_data = {
                "input_tokens":  chunk.usage.prompt_tokens,
                "output_tokens": chunk.usage.completion_tokens,
            }

        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            yield delta

    yield {
        "model":         model,
        "temperature":   GROQ_TEMPERATURE,
        "max_tokens":    GROQ_MAX_TOKENS,
        "input_tokens":  usage_data.get("input_tokens", 0),
        "output_tokens": usage_data.get("output_tokens", 0),
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
