# RAGChainMed
### Retrieval-Augmented Generation with Blockchain for Clinical Decision Support

**Sardar Patel Institute of Technology — BE Computer Engineering Major Project**  
Monil Parekh · Rohit Patil · Vikram Pimprikar  
Mentor: Prof. Jyoti Ramteke

---

## Overview

RAGChainMed is an AI-powered clinical knowledge and decision support system that combines:

- **RAG (Retrieval-Augmented Generation)** — retrieves relevant patient records and medical knowledge before generating answers, reducing hallucinations
- **EHR Narrative Conversion** — converts structured patient CSV data into readable clinical text for semantic search
- **Blockchain Audit Logging** — every query and response is hashed and logged to an immutable audit trail
- **Multi-source Knowledge Base** — combines Pima diabetes EHR data with PubMedQA, MedQA, and MedMCQA

---

## Project Structure

```
RAGChainMed/
├── data/
│   ├── raw/
│   │   └── diabetes.csv          ← Kaggle Pima dataset (not pushed)
│   └── processed/
│       ├── pima_narratives.json
│       └── pubmedqa_chunks.json
├── chroma_db/                    ← auto-generated (not pushed)
├── src/
│   ├── narrative_generator.py   ← structured CSV → clinical text
│   ├── data_ingestion.py        ← HuggingFace dataset loaders
│   ├── embed_store.py           ← embedding + ChromaDB builder
│   ├── retriever.py             ← semantic search
│   └── rag_pipeline.py          ← full RAG + Groq LLM pipeline
├── notebooks/
│   └── experimentation.ipynb    ← Colab notebook
├── requirements.txt
├── .gitignore
├── main.py
└── README.md
```

---

## Datasets Used

| Dataset | Source | Size | Type |
|---|---|---|---|
| Pima Indians Diabetes | Kaggle | 768 rows | Structured EHR |
| PubMedQA | HuggingFace | 2,000 records | Medical QA |
| MedQA USMLE | HuggingFace | 1,000 records | Clinical exam QA |
| MedMCQA | HuggingFace | 1,000 records | Clinical exam QA |

---

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/RAGChainMed.git
cd RAGChainMed
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add your API key
Create a `.env` file:
```
GROQ_API_KEY=your_key_here
```
Get a free key at [console.groq.com](https://console.groq.com)

### 4. Add the Pima dataset
Download `diabetes.csv` from [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) and place it at `data/raw/diabetes.csv`

### 5. Run the pipeline
```bash
python main.py
```

---

## How It Works

```
User Query
    ↓
Embed query with all-MiniLM-L6-v2
    ↓
ChromaDB cosine similarity search → top-5 chunks
    ↓
Separate EHR records vs medical knowledge
    ↓
Groq LLM (llama-3.3-70b) generates structured response
    ↓
Blockchain logs query hash + response hash
    ↓
Return: Analysis + Reasoning + Suggestions + Evidence
```

---

## Tech Stack

- `sentence-transformers` — text embeddings
- `chromadb` — vector database
- `groq` — LLM inference (llama-3.3-70b)
- `langchain` — text chunking
- `web3.py` — blockchain integration
- `ragas` — RAG evaluation metrics

---

## Evaluation Metrics

- Precision@K, Recall@K, MRR (retrieval quality)
- ROUGE score, BERTScore (answer quality)
- Faithfulness score via RAGAS (hallucination detection)
- Ablation: RAG vs LLM-only baseline comparison