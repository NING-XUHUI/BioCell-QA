from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

def generate_answer(question: str, context_chunks: list) -> str:
    """基于检索到的论文片段生成回答"""
    
    # 把所有检索到的片段拼成context
    context = "\n\n---\n\n".join([chunk.page_content for chunk in context_chunks])
    
    prompt = f"""You are a biomedical research assistant specializing in cell biology and gene expression.
    
Answer the following question based on the provided research literature. 
Always cite which part of the context supports your answer.
If the context doesn't contain enough information, say so clearly.

Context from research papers:
{context}

Question: {question}

Answer:"""

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    
    return response.choices[0].message.content