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

SEED_DATA = [
    {
        "title": "What is machine learning?",
        "messages": [
            ("user", "What is machine learning?"),
            ("assistant", "Machine learning is a subset of artificial intelligence where systems learn from data to improve performance on tasks without being explicitly programmed. It involves algorithms that identify patterns in data and make decisions with minimal human intervention."),
            ("user", "Can you give me a simple example?"),
            ("assistant", "Sure! Spam email filters are a classic example. The system is trained on thousands of labelled emails (spam vs. not spam). It learns patterns — certain words, sender behaviour, formatting — and uses those patterns to classify new emails automatically."),
        ],
    },
    {
        "title": "Python list comprehensions",
        "messages": [
            ("user", "How do list comprehensions work in Python?"),
            ("assistant", "A list comprehension gives you a concise way to build a list. The syntax is: [expression for item in iterable if condition]. For example, [x**2 for x in range(10) if x % 2 == 0] produces the squares of all even numbers from 0 to 9."),
            ("user", "What about nested list comprehensions?"),
            ("assistant", "You can nest them for multi-dimensional data. Example: [[row[i] for row in matrix] for i in range(len(matrix[0]))] transposes a 2D matrix. The outer comprehension iterates over column indices; the inner one collects each row's value at that index."),
            ("user", "Are there performance considerations?"),
            ("assistant", "List comprehensions are generally faster than equivalent for-loops because they are optimised at the bytecode level. However, if you only need to iterate once (not store results), a generator expression — using () instead of [] — is more memory-efficient since values are produced lazily."),
        ],
    },
    {
        "title": "Explain REST APIs",
        "messages": [
            ("user", "What is a REST API?"),
            ("assistant", "REST (Representational State Transfer) is an architectural style for building web APIs. A REST API exposes resources via URLs and uses standard HTTP methods: GET to read, POST to create, PUT/PATCH to update, and DELETE to remove. Responses are typically JSON."),
            ("user", "What makes an API truly RESTful?"),
            ("assistant", "There are six constraints: (1) Client-server separation, (2) Statelessness — each request carries all context the server needs, (3) Cacheability, (4) Uniform interface — consistent resource naming and HTTP verbs, (5) Layered system — clients don't need to know if they're talking to a proxy or the origin server, and (6) Code on demand (optional) — servers can send executable code like JavaScript."),
        ],
    },
]


async def seed():
    await init_db()

    async with AsyncSessionLocal() as session:
        async with session.begin():
            for conv_data in SEED_DATA:
                conv = Conversation(title=conv_data["title"])
                session.add(conv)
                await session.flush()  # populate conv.id before referencing it

                for seq, (role, content) in enumerate(conv_data["messages"], start=1):
                    msg = Message(
                        conversation_id=conv.id,
                        role=role,
                        content=content,
                        sequence_number=seq,
                    )
                    session.add(msg)

    print(f"Seeded {len(SEED_DATA)} conversations.")


if __name__ == "__main__":
    asyncio.run(seed())
