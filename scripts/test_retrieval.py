import faiss
import pickle
import numpy as np

from sentence_transformers import SentenceTransformer

# ============================================================
# LOAD FAISS INDEX
# ============================================================

index = faiss.read_index(
    "vectordb/heart_index.faiss"
)

print("FAISS Index Loaded")

# ============================================================
# LOAD DOCUMENTS
# ============================================================

with open(
    "vectordb/documents.pkl",
    "rb"
) as f:

    documents = pickle.load(f)

print("Documents Loaded:", len(documents))

# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("\nLoading Embedding Model...")

model = SentenceTransformer(
    'all-MiniLM-L6-v2'
)

# ============================================================
# RETRIEVAL FUNCTION
# ============================================================

def retrieve(query, top_k=3):

    # Convert query to embedding
    query_embedding = model.encode(
        [query]
    )

    query_embedding = np.array(
        query_embedding,
        dtype=np.float32
    )

    # Search nearest vectors
    distances, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for idx in indices[0]:

        results.append(
            documents[idx]
        )

    return results

# ============================================================
# TEST QUERY
# ============================================================

query = "high cholesterol and chest pain"

results = retrieve(query)

print("\n================================================")
print("QUERY:")
print(query)

print("\nTOP RETRIEVED DOCUMENTS:\n")

for i, result in enumerate(results):

    print(f"{i+1}. {result}")
    print()