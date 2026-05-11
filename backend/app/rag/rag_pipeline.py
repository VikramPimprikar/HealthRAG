from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter

# ============================================================
# LOAD KNOWLEDGE BASE
# ============================================================

def load_knowledge():
    # Get correct path to this file
    BASE_DIR = Path(__file__).resolve().parent
    file_path = BASE_DIR / "knowledge.txt"

    # Read file
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Split text into chunks
    splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    docs = splitter.split_text(text)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings()

    # Store in FAISS vector DB
    db = FAISS.from_texts(docs, embeddings)

    return db


# ============================================================
# RETRIEVE RELEVANT CONTEXT
# ============================================================

def retrieve_context(db, query):
    results = db.similarity_search(query, k=2)
    return " ".join([r.page_content for r in results])