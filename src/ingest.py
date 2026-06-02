from Bio import Entrez, Medline
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import os


Entrez.email = "your@email.com"

def fetch_pubmed_papers(query: str, max_results: int = 50):
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
    record = Entrez.read(handle)
    ids = record["IdList"]
    
    handle = Entrez.efetch(db="pubmed", id=ids, rettype="medline", retmode="text")
    records = list(Medline.parse(handle))
    
    papers = []
    for r in records:
        title = r.get("TI", "")
        abstract = r.get("AB", "")
        if abstract and len(abstract) > 100:
            papers.append({"title": title, "abstract": abstract})
    
    return papers

def build_vectorstore(persist_dir: str = "./chroma_db"):
    """用多个搜索词构建更丰富的向量数据库"""
    
    queries = [
    "cell type classification gene expression single cell",
    "supervised learning scRNA-seq cell annotation",
    "unsupervised clustering single cell RNA sequencing",
    "deep learning gene expression cell type",
    "incremental learning cell classification",
    "supervised vs unsupervised cell type annotation comparison",
    "benchmark cell type classification methods scRNA-seq",
    "machine learning methods comparison single cell transcriptomics",
    "random forest SVM deep learning cell type scRNA-seq",
    "cell annotation tools evaluation single cell",
]
    
    all_papers = []
    seen_titles = set()

    for query in queries:
        print(f"Fetching: {query}")
        papers = fetch_pubmed_papers(query, max_results=50)
        for paper in papers:
            if paper["title"] not in seen_titles:
                seen_titles.add(paper["title"])
                all_papers.append(paper)

    print(f"Total unique papers: {len(all_papers)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )

    # 把标题存为metadata
    docs = []
    for paper in all_papers:
        chunks = splitter.create_documents(
            texts=[paper["abstract"]],
            metadatas=[{"title": paper["title"]}]
    )
        docs.extend(chunks)

    chunks = docs
    
    print(f"Total chunks: {len(chunks)}")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # 删除旧的向量库，重新建
    import shutil
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)
    
    vectorstore = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=persist_dir
    )
    print("Vectorstore rebuilt!")
    return vectorstore

if __name__ == "__main__":
    build_vectorstore()