import json
import faiss
import pickle
import numpy as np
import os

from sentence_transformers import SentenceTransformer

# ============================================================
# PATHS
# ============================================================

INPUT_JSON = "data/heart_narratives.json"

VECTOR_DB_DIR = "vectordb"

FAISS_PATH = os.path.join(
    VECTOR_DB_DIR,
    "heart_index.faiss"
)

DOCS_PATH = os.path.join(
    VECTOR_DB_DIR,
    "documents.pkl"
)

# ============================================================
# CREATE VECTOR DB DIRECTORY
# ============================================================

os.makedirs(VECTOR_DB_DIR, exist_ok=True)

# ============================================================
# LOAD NARRATIVES
# ============================================================

with open(INPUT_JSON, "r") as f:

    docs = json.load(f)

documents = [d["text"] for d in docs]

print("Documents Loaded:", len(documents))

# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("\nLoading Embedding Model...")

model = SentenceTransformer(
    'all-MiniLM-L6-v2'
)

# ============================================================
# GENERATE EMBEDDINGS
# ============================================================

print("\nGenerating Embeddings...")

embeddings = model.encode(
    documents,
    show_progress_bar=True
)

embeddings = np.array(
    embeddings,
    dtype=np.float32
)

print("\nEmbeddings Shape:", embeddings.shape)

# ============================================================
# CREATE FAISS INDEX
# ============================================================

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

print("\nFAISS Index Created")
print("Total vectors stored:", index.ntotal)

# ============================================================
# SAVE FAISS INDEX
# ============================================================

faiss.write_index(
    index,
    FAISS_PATH
)

print(f"\nFAISS Index Saved:")
print(FAISS_PATH)

# ============================================================
# SAVE DOCUMENTS
# ============================================================

with open(DOCS_PATH, "wb") as f:

    pickle.dump(documents, f)

print(f"\nDocuments Saved:")
print(DOCS_PATH)