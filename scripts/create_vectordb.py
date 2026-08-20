"""
Medical Vector Database Creation Script
======================================
Builds FAISS vector databases from clinical patient narratives using:
1. Primary Medical Model: pritamdeka/S-PubMedBert-MS-MARCO (768-dim)
2. Optional Baseline Model: sentence-transformers/all-MiniLM-L6-v2 (384-dim) for comparison
"""

import json
import os
import argparse
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "heart_narratives.json"
PRIMARY_VECTORDB_DIR = BASE_DIR / "vectordb"
BASELINE_VECTORDB_DIR = BASE_DIR / "vectordb_minilm"


def build_vectorstore(model_name: str, output_dir: Path):
    print(f"\n============================================================")
    print(f"Building FAISS Index with Model: {model_name}")
    print(f"Destination: {output_dir}")
    print(f"============================================================")

    # 1. Load Patient Narratives Data
    documents = []
    if DATA_PATH.exists():
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            narratives_data = json.load(f)
        print(f"[1/4] Loaded {len(narratives_data)} clinical narratives from {DATA_PATH.name}")
        for idx, item in enumerate(narratives_data):
            documents.append(
                Document(
                    page_content=item["text"],
                    metadata={
                        "id": item.get("id", f"P{idx}"),
                        "doc_type": "patient_record"
                    }
                )
            )

    # 1b. Load Medical Reference Knowledge Base
    KB_PATH = BASE_DIR / "data" / "medical_knowledge.json"
    if KB_PATH.exists():
        with open(KB_PATH, "r", encoding="utf-8") as f:
            kb_data = json.load(f)
        print(f"[1b/4] Loaded {len(kb_data)} medical knowledge articles from {KB_PATH.name}")
        for item in kb_data:
            documents.append(
                Document(
                    page_content=f"{item.get('title', '')}: {item['text']}",
                    metadata={
                        "id": item.get("id", "KB"),
                        "title": item.get("title", ""),
                        "category": item.get("category", ""),
                        "doc_type": "medical_knowledge"
                    }
                )
            )

    print(f"[2/4] Created total {len(documents)} LangChain Document objects")

    # 3. Load Embedding Model
    print(f"[3/4] Loading HuggingFace Embedding Model: {model_name}...")
    embedding_model = HuggingFaceEmbeddings(model_name=model_name)

    # 4. Create FAISS Vectorstore
    print(f"[4/4] Generating embeddings and building FAISS index...")
    vectorstore = FAISS.from_documents(documents, embedding_model)

    output_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(output_dir))
    print(f"[SUCCESS] FAISS Vectorstore saved to: {output_dir}")
    print(f"Total vectors stored: {vectorstore.index.ntotal} (Dimension: {vectorstore.index.d})")
    return vectorstore


def main():
    parser = argparse.ArgumentParser(description="Create FAISS Vector Database for RAGChainMed")
    parser.add_argument(
        "--model",
        type=str,
        default="pritamdeka/S-PubMedBert-MS-MARCO",
        help="HuggingFace / Sentence-Transformers embedding model name"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PRIMARY_VECTORDB_DIR),
        help="Output directory for FAISS vectorstore"
    )
    parser.add_argument(
        "--build-baseline",
        action="store_true",
        help="Also build baseline all-MiniLM-L6-v2 vector store in vectordb_minilm for comparison"
    )

    args = parser.parse_args()

    # Build primary medical vector store
    build_vectorstore(args.model, Path(args.output))

    # Optionally build baseline index for comparison benchmarking
    if args.build_baseline:
        build_vectorstore("sentence-transformers/all-MiniLM-L6-v2", BASELINE_VECTORDB_DIR)


if __name__ == "__main__":
    main()