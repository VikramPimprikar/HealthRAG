import json
import faiss
import pickle
import numpy as np
import os
import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_JSON = BASE_DIR / "data" / "heart_narratives.json"
VECTOR_DB_DIR = BASE_DIR / "vectordb"

FAISS_PATH = VECTOR_DB_DIR / "heart_index.faiss"
DOCS_PATH = VECTOR_DB_DIR / "documents.pkl"

# ============================================================
# CREATE VECTOR DB DIRECTORY
# ============================================================

VECTOR_DB_DIR.mkdir(exist_ok=True, parents=True)

# ============================================================
# LOAD NARRATIVES
# ============================================================

with open(INPUT_JSON, "r", encoding="utf-8") as f:
    docs = json.load(f)

documents = [d["text"] for d in docs]
print(f"Documents Loaded: {len(documents)}")

# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

model_name = "pritamdeka/S-PubMedBert-MS-MARCO"
if len(sys.argv) > 1:
    model_name = sys.argv[1]

print(f"\nLoading Medical Embedding Model: {model_name}...")
model = SentenceTransformer(model_name)

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

print(f"\nEmbeddings Shape: {embeddings.shape}")

# ============================================================
# CREATE FAISS INDEX
# ============================================================

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

print("\nFAISS Index Created")
print(f"Total vectors stored: {index.ntotal} (Dimension: {dimension})")

# ============================================================
# SAVE FAISS INDEX
# ============================================================

faiss.write_index(index, str(FAISS_PATH))
print(f"\nFAISS Index Saved: {FAISS_PATH}")

# ============================================================
# SAVE DOCUMENTS
# ============================================================

with open(DOCS_PATH, "wb") as f:
    pickle.dump(documents, f)

print(f"\nDocuments Saved: {DOCS_PATH}")