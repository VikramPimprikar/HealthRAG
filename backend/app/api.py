from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from dotenv import load_dotenv
from groq import Groq

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

import os

# =========================================================
# LOAD ENV VARIABLES
# =========================================================

load_dotenv()

# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(
    title="RAGChainMed",
    description="Healthcare Retrieval System using FAISS + Groq",
    version="1.0"
)

# =========================================================
# ENABLE CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# BASE DIRECTORY
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

# =========================================================
# LOAD GROQ CLIENT
# =========================================================

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# =========================================================
# LOAD EMBEDDING MODEL
# =========================================================

print("Loading embedding model...")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded successfully!")

# =========================================================
# LOAD FAISS VECTORSTORE
# =========================================================

print("Loading FAISS vectorstore...")

vectorstore = FAISS.load_local(
    os.path.join(BASE_DIR, "vectordb"),
    embedding_model,
    allow_dangerous_deserialization=True
)

print("FAISS vectorstore loaded successfully!")

# =========================================================
# HOME ROUTE
# =========================================================

@app.get("/")
def home():

    return {
        "message": "RAGChainMed API Running Successfully"
    }

# =========================================================
# HEALTH ROUTE
# =========================================================

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }

# =========================================================
# QUERY ROUTE
# =========================================================

@app.get("/query")
def query_rag(query: str, user_id: str = ""):

    print(f"\nReceived Query: {query}")
    print(f"User ID: {user_id}")

    # =====================================================
    # RETRIEVE SIMILAR DOCUMENTS
    # =====================================================

    results = vectorstore.similarity_search(
        query,
        k=5
    )

    retrieved_docs = []

    for doc in results:

        retrieved_docs.append(
            doc.page_content
        )

    # =====================================================
    # CREATE CONTEXT
    # =====================================================

    context = "\n\n".join(retrieved_docs)

    # =====================================================
    # PROMPT
    # =====================================================

    prompt = f"""
You are a healthcare assistant.

Answer the user's question ONLY using the medical records below.

Medical Records:
{context}

Question:
{query}
"""

    # =====================================================
    # GROQ RESPONSE
    # =====================================================

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful healthcare retrieval assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3
    )

    answer = response.choices[0].message.content

    # =====================================================
    # RETURN RESPONSE
    # =====================================================

    return {
        "query": query,
        "user_id": user_id,
        "retrieved_records": retrieved_docs,
        "answer": answer
    }

# =========================================================
# AUDIT LOGS ROUTE
# =========================================================

@app.get("/api/v1/audit/logs")
def get_audit_logs():

    logs = [

        {
            "user_id": "D101",
            "action": "Query Search",
            "data_type": "Healthcare Records",
            "status": "success",
            "timestamp": "2026-05-10 10:30:25",
            "patient_id": "P1738",
            "details": {
                "query": "female patients",
                "records_retrieved": 5
            }
        },

        {
            "user_id": "D102",
            "action": "Query Search",
            "data_type": "Healthcare Records",
            "status": "success",
            "timestamp": "2026-05-10 10:35:11",
            "patient_id": "P1192",
            "details": {
                "query": "exercise induced angina",
                "records_retrieved": 5
            }
        },

        {
            "user_id": "admin",
            "action": "Audit Access",
            "data_type": "System Logs",
            "status": "success",
            "timestamp": "2026-05-10 10:40:44",
            "patient_id": None,
            "details": {
                "dashboard_access": True
            }
        }

    ]

    return {
        "records": logs
    }