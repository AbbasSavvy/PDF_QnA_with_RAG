import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI
from config import OLLAMA_BASE_URL, OLLAMA_MODEL


def build_prompt(chunks, question, history=None):
    context = "\n\n".join([chunk["text"] for chunk in chunks])

    history_text = ""
    if history:
        history_text = "Previous conversation:\n"
        for past_question, past_answer in history:
            history_text += f"Q: {past_question}\nA: {past_answer}\n\n"

    prompt = f"""Use the following context to answer the question.
If the answer is not in the context, say "I don't know".

Context:
{context}

{history_text}Question: {question}
Answer:"""
    return prompt


def get_answer(chunks, question, history=None):
    client = OpenAI(
        base_url=OLLAMA_BASE_URL,
        api_key="ollama",
    )

    prompt = build_prompt(chunks, question, history)

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