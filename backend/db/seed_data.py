"""
Run from the backend/ directory:
    python seed_data.py
"""

import asyncio

from backend.db.database import engine, init_db
from sqlalchemy.ext.asyncio import async_sessionmaker
from backend.db.models import Conversation, Message

AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)

USER_A = "alice@example.com"
USER_B = "bob@example.com"

SEED_DATA = [
    {
        "title": "What is machine learning?",
        "user_email": USER_A,
        "messages": [
            {
                "role": "user",
                "content": "What is machine learning?",
            },
            {
                "role": "assistant",
                "content": "Machine learning is a subset of artificial intelligence where systems learn from data to improve performance on tasks without being explicitly programmed. It involves algorithms that identify patterns in data and make decisions with minimal human intervention.",
                "model_choice": "llama-3.3-70b-versatile",
                "temperature": 0.7,
                "max_output_tokens": 1024,
            },
            {
                "role": "user",
                "content": "Can you give me a simple example?",
            },
            {
                "role": "assistant",
                "content": "Sure! Spam email filters are a classic example. The system is trained on thousands of labelled emails (spam vs. not spam). It learns patterns — certain words, sender behaviour, formatting — and uses those patterns to classify new emails automatically.",
                "model_choice": "llama-3.3-70b-versatile",
                "temperature": 0.7,
                "max_output_tokens": 1024,
            },
        ],
        # Demonstrates forking: two independent edits from the main thread
        "forks": [
            {
                # Fork of turn 1 (seq 1) — user asked about deep learning instead
                "fork_start_seq": 1,
                "messages": [
                    {
                        "role": "user",
                        "content": "What is deep learning?",
                    },
                    {
                        "role": "assistant",
                        "content": "Deep learning is a branch of machine learning that uses neural networks with many layers (hence 'deep'). These networks can automatically learn hierarchical representations of data, making them particularly powerful for tasks like image recognition, speech processing, and natural language understanding.",
                        "model_choice": "llama-3.3-70b-versatile",
                        "temperature": 0.7,
                        "max_output_tokens": 1024,
                    },
                ],
            },
            {
                # Fork of turn 2 (seq 3) — user asked for a medical example instead
                "fork_start_seq": 3,
                "messages": [
                    {
                        "role": "user",
                        "content": "Can you give me a medical example?",
                    },
                    {
                        "role": "assistant",
                        "content": "Absolutely! Medical image analysis is a great example. A model trained on thousands of labelled X-rays or MRI scans can learn to detect tumours, fractures, or diabetic retinopathy with accuracy comparable to specialist physicians — and in a fraction of the time.",
                        "model_choice": "llama-3.3-70b-versatile",
                        "temperature": 0.7,
                        "max_output_tokens": 1024,
                    },
                ],
            },
        ],
    },
    {
        "title": "Python list comprehensions",
        "user_email": USER_A,
        "messages": [
            {
                "role": "user",
                "content": "How do list comprehensions work in Python?",
            },
            {
                "role": "assistant",
                "content": "A list comprehension gives you a concise way to build a list. The syntax is: [expression for item in iterable if condition]. For example, [x**2 for x in range(10) if x % 2 == 0] produces the squares of all even numbers from 0 to 9.",
                "model_choice": "llama-3.3-70b-versatile",
                "temperature": 0.7,
                "max_output_tokens": 1024,
            },
            {
                "role": "user",
                "content": "What about nested list comprehensions?",
            },
            {
                "role": "assistant",
                "content": "You can nest them for multi-dimensional data. Example: [[row[i] for row in matrix] for i in range(len(matrix[0]))] transposes a 2D matrix. The outer comprehension iterates over column indices; the inner one collects each row's value at that index.",
                "model_choice": "llama-3.3-70b-versatile",
                "temperature": 0.7,
                "max_output_tokens": 1024,
            },
            {
                "role": "user",
                "content": "Are there performance considerations?",
            },
            {
                "role": "assistant",
                "content": "List comprehensions are generally faster than equivalent for-loops because they are optimised at the bytecode level. However, if you only need to iterate once (not store results), a generator expression — using () instead of [] — is more memory-efficient since values are produced lazily.",
                "model_choice": "llama-3.3-70b-versatile",
                "temperature": 0.7,
                "max_output_tokens": 1024,
            },
        ],
    },
    {
        "title": "Explain REST APIs",
        "user_email": USER_B,
        "messages": [
            {
                "role": "user",
                "content": "What is a REST API?",
            },
            {
                "role": "assistant",
                "content": "REST (Representational State Transfer) is an architectural style for building web APIs. A REST API exposes resources via URLs and uses standard HTTP methods: GET to read, POST to create, PUT/PATCH to update, and DELETE to remove. Responses are typically JSON.",
                "model_choice": "llama-3.3-70b-versatile",
                "temperature": 0.7,
                "max_output_tokens": 1024,
            },
            {
                "role": "user",
                "content": "What makes an API truly RESTful?",
            },
            {
                "role": "assistant",
                "content": "There are six constraints: (1) Client-server separation, (2) Statelessness — each request carries all context the server needs, (3) Cacheability, (4) Uniform interface — consistent resource naming and HTTP verbs, (5) Layered system — clients don't need to know if they're talking to a proxy or the origin server, and (6) Code on demand (optional) — servers can send executable code like JavaScript.",
                "model_choice": "llama-3.3-70b-versatile",
                "temperature": 0.7,
                "max_output_tokens": 1024,
            },
        ],
    },
]


def _add_message_pair(session, conv_id, user_raw, asst_raw, seq, turn, branch_id, fork_start_seq):
    session.add(Message(
        conversation_id=conv_id,
        role="user",
        content=user_raw["content"],
        status="completed",
        sequence_number=seq,
        turn_number=turn,
        branch_id=branch_id,
        fork_start_seq=fork_start_seq,
    ))
    session.add(Message(
        conversation_id=conv_id,
        role="assistant",
        content=asst_raw["content"],
        status="completed",
        sequence_number=seq + 1,
        turn_number=turn,
        branch_id=branch_id,
        fork_start_seq=fork_start_seq,
        model_choice=asst_raw.get("model_choice"),
        temperature=asst_raw.get("temperature"),
        max_output_tokens=asst_raw.get("max_output_tokens"),
    ))


async def seed():
    await init_db()

    async with AsyncSessionLocal() as session:
        async with session.begin():
            for conv_data in SEED_DATA:
                conv = Conversation(
                    title=conv_data["title"],
                    user_email=conv_data["user_email"],
                )
                session.add(conv)
                await session.flush()

                # --- Main branch (branch_id=0) ---
                messages = conv_data["messages"]
                seq = 1
                turn = 1
                i = 0
                while i < len(messages):
                    user_msg = messages[i]
                    asst_msg = messages[i + 1] if i + 1 < len(messages) and messages[i + 1]["role"] == "assistant" else None
                    if asst_msg:
                        _add_message_pair(session, conv.id, user_msg, asst_msg, seq, turn, branch_id=0, fork_start_seq=None)
                        seq += 2
                        i += 2
                    else:
                        session.add(Message(
                            conversation_id=conv.id,
                            role="user",
                            content=user_msg["content"],
                            status="completed",
                            sequence_number=seq,
                            turn_number=turn,
                            branch_id=0,
                            fork_start_seq=None,
                        ))
                        seq += 1
                        i += 1
                    turn += 1

                # --- Fork branches ---
                for branch_num, fork in enumerate(conv_data.get("forks", []), start=1):
                    fork_msgs = fork["messages"]
                    fork_start_seq = fork["fork_start_seq"]
                    fork_seq = 1
                    fork_turn = 1
                    j = 0
                    while j < len(fork_msgs):
                        user_msg = fork_msgs[j]
                        asst_msg = fork_msgs[j + 1] if j + 1 < len(fork_msgs) and fork_msgs[j + 1]["role"] == "assistant" else None
                        if asst_msg:
                            _add_message_pair(session, conv.id, user_msg, asst_msg, fork_seq, fork_turn, branch_id=branch_num, fork_start_seq=fork_start_seq)
                            fork_seq += 2
                            j += 2
                        else:
                            session.add(Message(
                                conversation_id=conv.id,
                                role="user",
                                content=user_msg["content"],
                                status="completed",
                                sequence_number=fork_seq,
                                turn_number=fork_turn,
                                branch_id=branch_num,
                                fork_start_seq=fork_start_seq,
                            ))
                            fork_seq += 1
                            j += 1
                        fork_turn += 1

    print(f"Seeded {len(SEED_DATA)} conversations.")


if __name__ == "__main__":
    asyncio.run(seed())