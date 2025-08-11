import torch
from typing import List, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langchain_core.documents import Document

# ---- Load Reranker Model ----
model_name = "BAAI/bge-reranker-base"
device = "cuda" if torch.cuda.is_available() else "cpu"

reranker_model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
).to(device)

reranker_tokenizer = AutoTokenizer.from_pretrained(model_name)


# ---- Reranking Function ----
def rerank_documents(query: str, docs: List[Document], top_n: int = 3) -> List[Document]:
    if not docs:
        return []

    # Prepare (query, doc) pairs
    pairs: List[Tuple[str, str]] = [
        (query, doc.page_content if hasattr(doc, "page_content") else str(doc))
        for doc in docs
    ]

    # Tokenize
    inputs = reranker_tokenizer(
        pairs,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(device)

    # Run model
    with torch.no_grad():
        scores = reranker_model(**inputs).logits.squeeze(-1)

    # Attach scores to documents
    scored_docs = []
    for score, doc in zip(scores.tolist(), docs):
        if not hasattr(doc, "metadata"):
            doc.metadata = {}
        doc.metadata["rerank_score"] = float(score)
        scored_docs.append((score, doc))

    # Sort and return top_n
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored_docs[:top_n]]
