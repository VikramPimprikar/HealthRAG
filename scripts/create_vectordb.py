import json

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ============================================================
# LOAD JSON DATA
# ============================================================

with open("data/heart_narratives.json", "r") as f:
    data = json.load(f)

print("Narratives Loaded")

# ============================================================
# CREATE DOCUMENTS
# ============================================================

documents = []

for item in data:

    doc = Document(
        page_content=item["text"],
        metadata={
            "id": item["id"]
        }
    )

    documents.append(doc)

print("Documents Created")

# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding Model Loaded")

# ============================================================
# CREATE FAISS VECTORSTORE
# ============================================================

vectorstore = FAISS.from_documents(
    documents,
    embedding_model
)

print("FAISS Vectorstore Created")

# ============================================================
# SAVE VECTORSTORE
# ============================================================

vectorstore.save_local("vectordb")

print("Vector Database Saved Successfully")