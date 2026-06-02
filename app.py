import streamlit as st
from src.retriever import load_vectorstore, retrieve_chunks
from src.generator import generate_answer
import os

st.title("🔬 BioCell QA")
st.caption("Ask questions about cell type classification and gene expression research")

# 初始化对话历史
if "messages" not in st.session_state:
    st.session_state.messages = []

# 加载向量库
@st.cache_resource
def get_vectorstore():
    return load_vectorstore()

if os.path.exists("./chroma_db"):
    vectorstore = get_vectorstore()
    
    # 显示对话历史
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # 用户输入
    if question := st.chat_input("Ask a question about cell biology..."):
        with st.chat_message("user"):
            st.write(question)
        st.session_state.messages.append({"role": "user", "content": question})
        
        with st.chat_message("assistant"):
            with st.spinner("Searching papers..."):
                chunks = retrieve_chunks(question, vectorstore, k=10)
                history = st.session_state.messages[:-1]
                answer = generate_answer(question, chunks, history)
                st.write(answer)
                
                with st.expander("📄 Source chunks"):
                    for i, chunk in enumerate(chunks):
                        st.markdown(f"**Chunk {i+1}:**")
                        st.write(chunk.page_content)
        
        st.session_state.messages.append({"role": "assistant", "content": answer})

else:
    st.info("Knowledge base not found. Please run ingest.py first.")
    if st.button("Build Knowledge Base"):
        from src.ingest import build_vectorstore
        with st.spinner("Fetching papers from PubMed..."):
            build_vectorstore("cell type classification gene expression single cell")
            st.success("Done! Please refresh the page.")