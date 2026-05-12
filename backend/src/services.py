import asyncio
from typing import AsyncGenerator

async def mock_llm_stream(prompt: str) -> AsyncGenerator[str, None]:
    words = f"This is a streamed response to: {prompt}".split()
    for word in words:
        yield word + " "
        await asyncio.sleep(0.1)

async def generate_conversation_title(question: str, answer: str) -> str:
    await asyncio.sleep(0.5)
    return f"Chat about {question[:10]}..."