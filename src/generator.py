from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

def generate_answer(question: str, context_chunks: list, history: list = []) -> str:
    
    context = "\n\n---\n\n".join([chunk.page_content for chunk in context_chunks])
    
    system_prompt = """You are a biomedical research assistant specializing in cell biology and gene expression.
Answer questions based on the provided research literature.
Always cite which part of the context supports your answer.
If the context doesn't contain enough information, say so clearly.
Please respond in Chinese."""

    messages = [{"role": "system", "content": system_prompt}]
    
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    
    messages.append({
        "role": "user",
        "content": f"Context from research papers:\n{context}\n\nQuestion: {question}"
    })
    
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        temperature=0.3
    )
    
    return response.choices[0].message.content