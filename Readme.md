# RAGChainMed
### Secure Clinical Decision Support with Retrieval-Augmented Generation & Blockchain Audit Integrity

**Sardar Patel Institute of Technology — BE Computer Engineering Major Project**  
Monil Parekh · Rohit Patil · Vikram Pimprikar  
Mentor: Prof. Jyoti Ramteke

---

## 🌟 Overview

**RAGChainMed** is a healthcare clinical decision support system combining:
1. **Grounded Medical RAG (Retrieval-Augmented Generation)**: Uses `sentence-transformers/all-MiniLM-L6-v2` embeddings with a FAISS vector database to retrieve top-K relevant patient records and clinical evidence. Grounds Groq's `llama-3.1-8b-instant` LLM in verified records with strict anti-hallucination guardrails.
2. **Immutable Blockchain Audit Trail**: Native cryptographic blockchain with SHA-256 evidence and query hashing. Every query, prediction, and access attempt is committed to an immutable block with full provenance without storing raw patient text on-chain.
3. **Evidence Integrity Verification**: Real-time cryptographic proof verifying that retrieved evidence matches the on-chain recorded hash, immediately flagging any data tampering or corruption.
4. **Role-Based Access Control (RBAC)**: Fine-grained permissions enforced across all endpoints (`admin`, `doctor`, `nurse`, `auditor`, `patient`).
5. **Machine Learning Cardiovascular Risk Prediction**: Multi-class XGBoost model evaluating 11 physiological biomarkers to predict heart disease severity with automated Clinical Decision Support (CDS) recommendations.
6. **Modern React Frontend**: Clean, responsive dashboard connecting live to FastAPI backend APIs.

---

## 📂 Project Architecture

```
RAGChainMed/
├── backend/
│   ├── app/
│   │   ├── api.py                    ← FastAPI REST endpoints (RBAC, RAG, Blockchain, ML)
│   │   ├── config.py                 ← Central configuration & dataset paths
│   │   ├── main.py                   ← Application entry point
│   │   ├── predict.py                ← XGBoost prediction & feature normalization
│   │   ├── run_pipeline.py           ← ML pipeline orchestration
│   │   ├── blockchain/
│   │   │   ├── access_control.py     ← RBAC Manager & permissions
│   │   │   ├── audit_log.py          ← Blockchain & SHA-256 evidence hashing
│   │   │   └── secure_retrieval.py   ← RBAC-gated retrieval helper
│   │   ├── clinical/
│   │   │   ├── clinical_data_manager.py ← Structured EHR management
│   │   │   └── clinical_decision_support.py ← Evidence-based CDS recommendations
│   │   └── rag/
│   │       ├── enhanced_rag_pipeline.py ← Grounded RAG with anti-hallucination
│   │       └── rag_pipeline.py       ← Vector retrieval utilities
│   ├── training/                     ← Data preprocessing, feature engineering & training
│   ├── requirements.txt              ← Python backend dependencies
│   └── .env                          ← Environment variables (GROQ_API_KEY)
├── frontend/
│   ├── src/
│   │   ├── App.js                    ← Navigation & global role switcher
│   │   ├── index.css                 ← Clinical design system tokens & theme
│   │   └── pages/
│   │       ├── Home.js               ← RAG Search & Evidence Viewer
│   │       ├── ClinicalPrediction.js ← ML Heart Disease Risk Assessment
│   │       ├── AuditDashboard.js     ← Blockchain Audit Trail & Chain Verification
│   │       └── EvidenceVerifier.js   ← Cryptographic Evidence Tamper Detector
│   └── package.json                  ← Frontend dependencies (React, React Router)
├── data/
│   ├── heart_disease.csv             ← Clinical dataset (920 patient records)
│   └── heart_narratives.json         ← Generated rich clinical narratives
├── vectordb/                         ← FAISS vector database (index.faiss, index.pkl)
├── model/                            ← Trained XGBoost model & scalers
├── scripts/
│   ├── generate_narratives.py        ← Converts CSV records to text narratives
│   ├── create_vectordb.py            ← Builds FAISS vector store
│   ├── test_retrieval.py             ← Tests vector similarity search
│   └── test_system_e2e.py            ← 10-step automated integration test suite
└── README.md
```

---

## 🔐 Role-Based Access Control (RBAC) Reference

| Role | Default User ID | Permissions | Typical Operations |
|---|---|---|---|
| **Admin** | `admin` | All permissions | Full access: RAG queries, ML predictions, Audit logs, User management |
| **Doctor** | `DOC001`, `D101` | Query Knowledge Base, Request Prediction, View Patient Data, Edit Patient Data | Perform RAG medical queries & run ML risk assessments |
| **Nurse** | `NURSE001` | View Patient Data, Edit Patient Data | View patient records (RAG queries & ML predictions restricted) |
| **Auditor** | `AUDITOR001` | View Audit Logs | Inspect blockchain blocks & verify cryptographic integrity |
| **Patient** | `PATIENT001` | View Patient Data | View own medical records |

---

## 🚀 Complete Windows Setup Instructions

### Prerequisites
1. **Python 3.10+** (Tested on Python 3.12 64-bit on Windows)
2. **Node.js 18+ & npm** (Node v22+ recommended)
3. **Git**

---

### Step 1: Clone the Repository & Open Project Directory
Open **PowerShell** or **Command Prompt**:
```powershell
cd c:\Users\vikra\OneDrive\Desktop\RAG_Vector-main
```

---

### Step 2: Configure Environment Variables
Create or verify `backend/.env`:
```ini
GROQ_API_KEY=your_groq_api_key_here
```
> *Get a free Groq API key from [console.groq.com](https://console.groq.com).*

---

### Step 3: Install Python Backend Dependencies
```powershell
pip install -r backend/requirements.txt
```
Key packages installed: `fastapi`, `uvicorn`, `langchain`, `langchain-community`, `sentence-transformers`, `faiss-cpu`, `groq`, `xgboost`, `scikit-learn`, `pandas`, `python-dotenv`.

---

### Step 4: Install React Frontend Dependencies
```powershell
cd frontend
npm install
cd ..
```

---

### Step 5: (Optional) Ingest Clinical Data & Rebuild Vector DB
If you modify the dataset or wish to re-index the FAISS vector database from scratch:

1. **Generate Clinical Narratives from CSV:**
   ```powershell
   python scripts/generate_narratives.py
   ```
2. **Build FAISS Vector Database:**
   ```powershell
   python scripts/create_vectordb.py
   ```
3. **Train & Evaluate XGBoost ML Model:**
   ```powershell
   python backend/app/run_pipeline.py --full
   ```

---

### Step 6: Start the Backend Server
In PowerShell Window 1:
```powershell
python -m uvicorn app.api:app --host 127.0.0.1 --port 8000 --reload
```
- **API Base URL**: `http://127.0.0.1:8000`
- **Swagger Interactive Docs**: `http://127.0.0.1:8000/docs`

---

### Step 7: Start the React Frontend Application
In PowerShell Window 2:
```powershell
cd frontend
npm start
```
- **Frontend URL**: `http://localhost:3000`

---

## 🧪 Testing & Verification

### Run Automated End-to-End Test Suite
Run the test suite verifying all 10 core capabilities (Health, RBAC, RAG search, Anti-hallucination, ML prediction, Blockchain evidence hashing, Evidence tamper detection, and Full-chain integrity):
```powershell
python scripts/test_system_e2e.py
```

Expected output:
```
======================================================================
RUNNING RAGCHAINMED END-TO-END TEST SUITE
======================================================================
[TEST 1] Testing /health...
  [OK] Status: healthy
  [OK] Total vectors in FAISS: 920
  [OK] Blockchain intact: True
[TEST 2] Testing /api/v1/auth/users...
  [OK] Loaded 6 registered users and roles.
[TEST 3] Testing RAG Query as admin...
  [OK] Retrieved 3 evidence records.
  [OK] Evidence SHA-256 Hash: 02bd00711098637f...
  [OK] Committed to Blockchain Block #1
[TEST 4] Testing Anti-Hallucination Guardrail...
  [OK] Response received: Grounding fallback active.
[TEST 5] Testing RBAC Enforcement (NURSE001)...
  [OK] RBAC correctly denied access with 403 Forbidden
[TEST 6] Testing Clinical ML Prediction as DOC001...
  [OK] Predicted Severity: Severe (Confidence: 77.73%)
  [OK] Committed to Blockchain Block #4
[TEST 7] Testing Evidence Verification (Authentic)...
  [OK] Verification result: AUTHENTIC (verified=True)
[TEST 8] Testing Evidence Tamper Detection...
  [OK] Tamper detection result: TAMPER DETECTED (verified=False, tamper_detected=True)
[TEST 9] Testing Blockchain Full Chain Cryptographic Verification...
  [OK] Full Chain Validity: True across 5 blocks.
[TEST 10] Testing Audit Logs Retrieval as AUDITOR001...
  [OK] Retrieved 5 blocks and 4 audit records.
======================================================================
ALL 10 END-TO-END TESTS COMPLETED AND PASSED SUCCESSFULLY!
======================================================================
```

---

## 💡 Using the Web Application

1. **RAG Search & Grounded Answers (`http://localhost:3000/`)**:
   - Click preset queries or type clinical searches (e.g. *"Patients with cholesterol > 240 and exercise induced angina"*).
   - View top-K evidence cards with Patient ID, Name, Biomarkers, Similarity Score, and SHA-256 Hash.
   - Click **"Verify Integrity"** on any evidence card to verify its hash against the blockchain.
2. **Clinical Risk Prediction (`http://localhost:3000/predict`)**:
   - Select a clinical preset (e.g. *Severe Critical CAD Profile*) or enter custom measurements.
   - Click **"Run ML Risk Prediction"** (requires Doctor or Admin role).
   - View predicted severity level, multi-class probability distribution, and evidence-based clinical recommendations.
3. **Blockchain Audit Trail (`http://localhost:3000/audit`)**:
   - View all committed blocks in real time with block hashes, timestamps, user IDs, and provenance.
   - Click **"Verify Blockchain Integrity"** to verify the entire cryptographic chain.
4. **Evidence Verifier (`http://localhost:3000/verify`)**:
   - Enter a Block Index and Evidence SHA-256 Hash (or raw text).
   - Click **"Verify Against Blockchain"** to confirm authenticity, or click **"Simulate Tampered Data"** to observe real-time tamper detection.
5. **Role Switcher (Top Navigation Bar)**:
   - Switch between **Admin**, **DOC001 (Doctor)**, **NURSE001 (Nurse)**, **AUDITOR001 (Auditor)**, or **Unauthorized User** to experience RBAC permissions and access denials live.

---

## 🔒 Security & Privacy Notice

- **Zero PII on Blockchain**: Raw medical records are never written to the blockchain. Only SHA-256 cryptographic hashes, query digests, and anonymized provenance metadata are committed to blocks.
- **Environment Isolation**: API keys (`GROQ_API_KEY`) are managed strictly through `.env` and never committed or exposed in client bundles.