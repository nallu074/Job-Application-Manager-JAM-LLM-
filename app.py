import os
import time
import streamlit as st
from dotenv import load_dotenv

# Load env vars (for GROQ_API_KEY)
load_dotenv()

# Local modules
from modules.m1_cover_letter_rag.loader import load_cover_letters
from modules.m1_cover_letter_rag.vector import (
    load_and_split_docs,
    create_vectorstore,
    load_vectorstore,
    build_ensemble_retriever
)
from modules.m1_cover_letter_rag.ranker import rerank_documents

# LangChain core
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

# ---- Setup ----
st.set_page_config(page_title="Job Application Manager - Cover Letter Finder", layout="wide")
st.title("📄 Job Application Copilot (Cover Letter RAG + Groq)")

# ---- Sidebar ----
with st.sidebar:
    st.header("💾 Embedding Setup")
    folder = st.text_input("📁 Cover Letter Folder", value="data/cover_letters")
    embed_btn = st.button("Build Vectorstore")
    show_sources = st.checkbox("🔍 Show Sources", value=True)

# ---- Groq LLM ----
groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(groq_api_key=groq_api_key, model_name="LLaMA3-8b-8192")

prompt = ChatPromptTemplate.from_template("""
Answer the question using ONLY the context below. Be precise and concise.

<context>
{context}
</context>

Question: {input}
""")
document_chain = create_stuff_documents_chain(llm, prompt)

# ---- Embed step ----
if embed_btn:
    with st.spinner("Loading and embedding documents..."):
        docs = load_and_split_docs(folder)
        vectorstore = create_vectorstore(docs)
        st.session_state.docs = docs
        st.session_state.vectorstore = vectorstore
        st.success("✅ Vectorstore created and stored!")

# ---- Load vectorstore if exists ----
if "vectorstore" not in st.session_state:
    vectorstore = load_vectorstore()
    if vectorstore:
        st.session_state.vectorstore = vectorstore
        st.session_state.docs = load_cover_letters(folder)

# ---- Query ----
query = st.text_input("🔎 What are you looking for?", placeholder="e.g., Cover letter for BMW Data Analyst")
if query and "vectorstore" in st.session_state:
    # Run retrieval
    with st.spinner("Retrieving documents..."):
        retriever = build_ensemble_retriever(st.session_state.docs, st.session_state.vectorstore)
        retrieved_docs = retriever.get_relevant_documents(query)

    # Rerank
    with st.spinner("Reranking top documents..."):
        reranked_docs = rerank_documents(query, retrieved_docs, top_n=3)

    # Generate answer
    with st.spinner("Generating final answer with LLaMA3..."):
        start = time.process_time()
        response = document_chain.invoke({
            "context": reranked_docs,
            "input": query
        })
        runtime = time.process_time() - start
    
    print("🧪 Raw LLM response:", response)
    final_answer = response if isinstance(response, str) else response.get('output_text', str(response))
    st.subheader("🧠 Final Answer (Groq - LLaMA3)")
    st.write(final_answer)
    st.caption(f"⚙️ Generated in {runtime:.2f} seconds")

    # Show retrieved chunks
    st.subheader("🔬 Comparison: Before vs After Reranking")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🟡 Before Reranking")
        for i, doc in enumerate(retrieved_docs[:3]):
            st.markdown(f"**Chunk {i+1}:** `{os.path.basename(doc.metadata.get('source', ''))}`")
            st.write(doc.page_content[:500] + "...")
            st.caption(f"Source: {doc.metadata.get('source')}")

    with col2:
        st.markdown("### 🟢 After Reranking")
        for i, doc in enumerate(reranked_docs):
            st.markdown(f"**Chunk {i+1}:** `{os.path.basename(doc.metadata.get('source', ''))}`")
            st.write(doc.page_content[:500] + "...")
            st.caption(f"Rerank Score: {doc.metadata.get('rerank_score', 'N/A')}, Source: {doc.metadata.get('source')}")
