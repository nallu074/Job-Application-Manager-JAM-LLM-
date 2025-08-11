import os
from typing import Optional, List
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from modules.m1_cover_letter_rag.loader import load_cover_letters

load_dotenv()  # Load keys from .env

# ---- Config ----
DATA_DIR = "data/cover_letters"
VECTORSTORE_PATH = "state/vectordb"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def load_and_split_docs(folder: str = DATA_DIR) -> List[Document]:
    print("📄 Loading documents...")
    docs = load_cover_letters(folder)
    print(f"✅ Loaded {len(docs)} documents")

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    split_docs = splitter.split_documents(docs)
    print(f"✂️ Split into {len(split_docs)} chunks")
    return split_docs


def create_vectorstore(documents: List[Document], persist: bool = True) -> FAISS:
    print("🔍 Creating FAISS vector store...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents, embeddings)

    if persist:
        os.makedirs(VECTORSTORE_PATH, exist_ok=True)
        vectorstore.save_local(VECTORSTORE_PATH)
        print(f"💾 Vectorstore saved to {VECTORSTORE_PATH}")
    return vectorstore


def load_vectorstore() -> Optional[FAISS]:
    if os.path.exists(os.path.join(VECTORSTORE_PATH, "index.faiss")):
        print("📦 Loading existing FAISS vectorstore...")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return FAISS.load_local(
            VECTORSTORE_PATH,
            embeddings,
            allow_dangerous_deserialization=True  # 👈 ADD THIS
        )
    else:
        print("⚠️ No existing vectorstore found")
        return None


def build_ensemble_retriever(docs: List[Document], vectorstore: FAISS) -> EnsembleRetriever:
    print("🔗 Building EnsembleRetriever (FAISS + BM25)...")

    # BM25 uses original full documents (not chunks)
    keyword_retriever = BM25Retriever.from_documents(docs)
    keyword_retriever.k = 5

    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    ensemble = EnsembleRetriever(
        retrievers=[vector_retriever, keyword_retriever],
        weights=[0.5, 0.5],
    )
    return ensemble
