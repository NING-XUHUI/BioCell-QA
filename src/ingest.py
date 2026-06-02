from Bio import Entrez, Medline
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

Entrez.email = "1998ok4nxh@gmail.com"

def fetch_pubmed_papers(query: str, max_results: int = 50):
    """从PubMed抓取论文摘要"""
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
    record = Entrez.read(handle)
    ids = record["IdList"]

    handle = Entrez.efetch(db="pubmed", id=ids, rettype="medline", retmode="text")
    records = list(Medline.parse(handle))

    papers = []
    for r in records:
        title = r.get("TI", "")
        abstract = r.get("AB", "")
        if abstract:
            papers.append(f"Title: {title}\n\nAbstract: {abstract}")
    
    return papers


def build_vectorstore(query: str, persist_dir: str = "./chroma_db"):
    """构建向量数据库"""
    print(f"Fetching papers for: {query}")
    papers = fetch_pubmed_papers(query)
    print(f"Fetched {len(papers)} papers")
    
    # 切分文本
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.create_documents(papers)
    print(f"Created {len(chunks)} chunks")
    
    # 创建embedding并存入Chroma
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=persist_dir
    )
    print("Vectorstore built and saved!")
    return vectorstore

if __name__ == "__main__":
    build_vectorstore("cell type classification gene expression single cell")