from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

def load_vectorstore(persist_dir: str = "./chroma_db"):
    """加载已有的向量数据库"""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )
    return vectorstore

def retrieve_chunks(question: str, vectorstore, k: int = 5):
    """检索最相关的k个论文片段"""
    chunks = vectorstore.similarity_search(question, k=k)
    return chunks