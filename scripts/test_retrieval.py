import faiss
import pickle
import numpy as np
import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent

# ============================================================
# LOAD FAISS INDEX
# ============================================================

index_path = BASE_DIR / "vectordb" / "heart_index.faiss"
docs_path = BASE_DIR / "vectordb" / "documents.pkl"

if not index_path.exists():
    print(f"Index not found at {index_path}. Please run create_faiss_index.py first.")
    sys.exit(1)

index = faiss.read_index(str(index_path))
print(f"FAISS Index Loaded (Dimension: {index.d}, Total: {index.ntotal})")

# ============================================================
# LOAD DOCUMENTS
# ============================================================

with open(docs_path, "rb") as f:
    documents = pickle.load(f)

print(f"Documents Loaded: {len(documents)}")

# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

model_name = "pritamdeka/S-PubMedBert-MS-MARCO"
print(f"\nLoading Medical Embedding Model ({model_name})...")
model = SentenceTransformer(model_name)

# ============================================================
# RETRIEVAL FUNCTION
# ============================================================

def retrieve(query, top_k=3):
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding, dtype=np.float32)

    distances, indices = index.search(query_embedding, top_k)
    results = []
    for idx in indices[0]:
        results.append(documents[idx])
    return results

# ============================================================
# TEST QUERY
# ============================================================

query = "high cholesterol and chest pain"
results = retrieve(query)

print("\n================================================")
print("QUERY:", query)
print("\nTOP RETRIEVED DOCUMENTS:\n")

for i, result in enumerate(results):
    print(f"{i+1}. {result}")
    print()