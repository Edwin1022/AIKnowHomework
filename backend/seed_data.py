"""
Run from the backend/ directory:
    python seed_data.py
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from database import engine, init_db
from sqlalchemy.ext.asyncio import async_sessionmaker
from models import Conversation, Message

AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)

SEED_EMAIL = "test@example.com"

SEED_DATA = [
    {
        "title": "What is machine learning?",
        "messages": [
            {
                "role": "user",
                "content": "What is machine learning?",
            },
            {
                "role": "assistant",
                "content": "Machine learning is a subset of artificial intelligence where systems learn from data to improve performance on tasks without being explicitly programmed. It involves algorithms that identify patterns in data and make decisions with minimal human intervention.",
                "model_choice": "claude-sonnet-4-6",
                "temperature": 0.7,
                "seed": 42,
                "max_output_tokens": 1024,
            },
            {
                "role": "user",
                "content": "Can you give me a simple example?",
            },
            {
                "role": "assistant",
                "content": "Sure! Spam email filters are a classic example. The system is trained on thousands of labelled emails (spam vs. not spam). It learns patterns — certain words, sender behaviour, formatting — and uses those patterns to classify new emails automatically.",
                "model_choice": "claude-sonnet-4-6",
                "temperature": 0.7,
                "seed": 42,
                "max_output_tokens": 1024,
            },
        ],
    },
    {
        "title": "Python list comprehensions",
        "messages": [
            {
                "role": "user",
                "content": "How do list comprehensions work in Python?",
            },
            {
                "role": "assistant",
                "content": "A list comprehension gives you a concise way to build a list. The syntax is: [expression for item in iterable if condition]. For example, [x**2 for x in range(10) if x % 2 == 0] produces the squares of all even numbers from 0 to 9.",
                "model_choice": "claude-sonnet-4-6",
                "temperature": 0.7,
                "seed": 42,
                "max_output_tokens": 1024,
            },
            {
                "role": "user",
                "content": "What about nested list comprehensions?",
            },
            {
                "role": "assistant",
                "content": "You can nest them for multi-dimensional data. Example: [[row[i] for row in matrix] for i in range(len(matrix[0]))] transposes a 2D matrix. The outer comprehension iterates over column indices; the inner one collects each row's value at that index.",
                "model_choice": "claude-sonnet-4-6",
                "temperature": 0.7,
                "seed": 42,
                "max_output_tokens": 1024,
            },
            {
                "role": "user",
                "content": "Are there performance considerations?",
            },
            {
                "role": "assistant",
                "content": "List comprehensions are generally faster than equivalent for-loops because they are optimised at the bytecode level. However, if you only need to iterate once (not store results), a generator expression — using () instead of [] — is more memory-efficient since values are produced lazily.",
                "model_choice": "claude-sonnet-4-6",
                "temperature": 0.7,
                "seed": 42,
                "max_output_tokens": 1024,
            },
        ],
    },
    {
        "title": "Explain REST APIs",
        "messages": [
            {
                "role": "user",
                "content": "What is a REST API?",
            },
            {
                "role": "assistant",
                "content": "REST (Representational State Transfer) is an architectural style for building web APIs. A REST API exposes resources via URLs and uses standard HTTP methods: GET to read, POST to create, PUT/PATCH to update, and DELETE to remove. Responses are typically JSON.",
                "model_choice": "claude-sonnet-4-6",
                "temperature": 0.7,
                "seed": 42,
                "max_output_tokens": 1024,
            },
            {
                "role": "user",
                "content": "What makes an API truly RESTful?",
            },
            {
                "role": "assistant",
                "content": "There are six constraints: (1) Client-server separation, (2) Statelessness — each request carries all context the server needs, (3) Cacheability, (4) Uniform interface — consistent resource naming and HTTP verbs, (5) Layered system — clients don't need to know if they're talking to a proxy or the origin server, and (6) Code on demand (optional) — servers can send executable code like JavaScript.",
                "model_choice": "claude-sonnet-4-6",
                "temperature": 0.7,
                "seed": 42,
                "max_output_tokens": 1024,
            },
        ],
    },
]


async def seed():
    await init_db()

    async with AsyncSessionLocal() as session:
        async with session.begin():
            for conv_data in SEED_DATA:
                conv = Conversation(
                    title=conv_data["title"],
                    user_email=SEED_EMAIL,
                )
                session.add(conv)
                await session.flush()

                sequence_number = 1
                turn_number = 1
                messages = conv_data["messages"]
                i = 0
                while i < len(messages):
                    user_msg = messages[i]
                    session.add(Message(
                        conversation_id=conv.id,
                        role="user",
                        content=user_msg["content"],
                        status="completed",
                        sequence_number=sequence_number,
                        turn_number=turn_number,
                    ))
                    sequence_number += 1

                    if i + 1 < len(messages) and messages[i + 1]["role"] == "assistant":
                        asst_msg = messages[i + 1]
                        session.add(Message(
                            conversation_id=conv.id,
                            role="assistant",
                            content=asst_msg["content"],
                            status="completed",
                            sequence_number=sequence_number,
                            turn_number=turn_number,
                            model_choice=asst_msg.get("model_choice"),
                            temperature=asst_msg.get("temperature"),
                            seed=asst_msg.get("seed"),
                            max_output_tokens=asst_msg.get("max_output_tokens"),
                        ))
                        sequence_number += 1
                        i += 2
                    else:
                        i += 1

                    turn_number += 1

    print(f"Seeded {len(SEED_DATA)} conversations.")


if __name__ == "__main__":
    asyncio.run(seed())
