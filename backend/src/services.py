import asyncio
from typing import AsyncGenerator

async def mock_llm_stream(prompt: str) -> AsyncGenerator[str, None]:
    """Mocks an asynchronous LLM streaming response."""
    words = f"This is a streamed response to: {prompt}".split()
    for word in words:
        yield word + " "
        await asyncio.sleep(0.1)

def generate_conversation_title(question: str, answer: str) -> str:
    """Mocks generating a title based on the first Q&A."""
    return f"Chat about {question[:10]}..."