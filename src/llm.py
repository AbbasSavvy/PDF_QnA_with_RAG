import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI
from dotenv import load_dotenv
from config import OLLAMA_BASE_URL, OLLAMA_MODEL, GROQ_BASE_URL, GROQ_MODEL

load_dotenv()


def _format_metadata(doc_metadata):
    """Format document metadata as a prompt prefix string."""
    if not doc_metadata:
        return ""

    lines = []
    if doc_metadata.get("title"):
        lines.append(f"Document title: {doc_metadata['title']}")
    if doc_metadata.get("authors"):
        lines.append(f"Authors: {doc_metadata['authors']}")
    if doc_metadata.get("institution"):
        lines.append(f"Institution: {doc_metadata['institution']}")

    if not lines:
        return ""

    return "Document information:\n" + "\n".join(lines)


def build_prompt(chunks, question, history=None, doc_metadata=None):
    context = "\n\n".join([chunk["text"] for chunk in chunks])

    metadata_text = _format_metadata(doc_metadata)

    history_text = ""
    if history:
        history_text = "Previous conversation:\n"
        for past_question, past_answer in history:
            history_text += f"Q: {past_question}\nA: {past_answer}\n\n"

    prompt = f"""Use the following context to answer the question.
If the answer is not in the context, say "I don't know".

{metadata_text}

Context:
{context}

{history_text}Question: {question}
Answer:"""
    return prompt


def get_answer(chunks, question, history=None, confidential=True, doc_metadata=None):

    if confidential:
        client = OpenAI(
            base_url=OLLAMA_BASE_URL,
            api_key="ollama",
        )
        model = OLLAMA_MODEL
    else:
        client = OpenAI(
            base_url=GROQ_BASE_URL,
            api_key=os.getenv("GROQ_API_KEY"),
        )
        model = GROQ_MODEL

    prompt = build_prompt(chunks, question, history, doc_metadata)

    response = client.chat.completions.create(
        model=model,
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