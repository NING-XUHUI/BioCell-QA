import streamlit as st
from src.ingest import build_vectorstore
from src.retriever import load_vectorstore, retrieve_chunks
from src.generator import generate_answer
import os

st.title("🔬 BioCell QA")
st.caption("Ask questions about cell type classification and gene expression research")

# 侧边栏：构建知识库
with st.sidebar:
    st.header("Knowledge Base")
    query_topic = st.text_input(
        "Research topic to fetch from PubMed:",
        value="cell type classification gene expression single cell"
    )
    if st.button("Build Knowledge Base"):
        with st.spinner("Fetching papers from PubMed..."):
            build_vectorstore(query_topic)
            st.success("Knowledge base ready!")

# 主界面：问答
if os.path.exists("./chroma_db"):
    vectorstore = load_vectorstore()
    
    question = st.text_input("Ask a question about cell biology:")
    
    if st.button("Ask") and question:
        with st.spinner("Searching papers and generating answer..."):
            chunks = retrieve_chunks(question, vectorstore)
            answer = generate_answer(question, chunks)
            
            st.markdown("### Answer")
            st.write(answer)
            
            with st.expander("📄 Source chunks"):
                for i, chunk in enumerate(chunks):
                    st.markdown(f"**Chunk {i+1}:**")
                    st.write(chunk.page_content)
else:
    st.info("👈 Please build the knowledge base first using the sidebar.")