import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI
from config import OLLAMA_BASE_URL, OLLAMA_MODEL


def build_prompt(chunks, question):
    context = "\n\n".join([chunk["text"] for chunk in chunks])
    prompt = f"""Use the following context to answer the question.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question: {question}
Answer:"""
    return prompt


def get_answer(chunks, question):
    client = OpenAI(
        base_url=OLLAMA_BASE_URL,
        api_key="ollama",
    )

    prompt = build_prompt(chunks, question)

    response = client.chat.completions.create(
        model=OLLAMA_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that answers questions based only on the provided context."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content
