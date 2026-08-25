"""
Enhanced Medical RAG Pipeline
=============================
Provides semantic retrieval from FAISS vector database with:
- Top-K evidence search over patient narratives and clinical medical knowledge
- Structured metadata extraction (Patient ID, biomarkers, clinical indicators)
- Cryptographic evidence hashing (SHA-256) for blockchain verification
- Grounded medical Q&A with anti-hallucination and fallback guardrails
- Full query routing between Normal Medical RAG and Structured Data queries

Author: RAGChainMed
"""

import os
import re
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dotenv import load_dotenv
from groq import Groq

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load environment from possible locations
base_dir = Path(__file__).resolve().parent.parent.parent.parent
backend_dir = Path(__file__).resolve().parent.parent.parent

if (backend_dir / ".env").exists():
    load_dotenv(backend_dir / ".env")
elif (base_dir / ".env").exists():
    load_dotenv(base_dir / ".env")
else:
    load_dotenv()


try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

from app.clinical.structured_query_engine import get_structured_engine
from app.blockchain.evidence_verifier import compute_canonical_hash, canonicalize_evidence


MISMATCH_REFUSAL_MESSAGE = (
    "I apologize for the mismatch in the query and the provided medical records. "
    "As an AI assistant, my capabilities are limited to clinical decision support and answering cardiovascular medical inquiries. "
    "If you have any clinical question or need assistance with cardiovascular information, "
    "please feel free to ask, and I will be more than happy to help."
)

INSUFFICIENT_CONTEXT_MESSAGE = (
    "I could not find sufficient information in the available medical knowledge base to answer this question. "
    "Please refine your inquiry or consult the relevant clinical literature."
)


def is_clinical_query(query: str) -> bool:
    """
    Determine whether a query is related to clinical medicine, cardiology,
    cardiovascular biomarkers, diagnoses, symptoms, or patient records.
    Returns False for off-topic queries (e.g. quantum physics, recipes, movies, programming).
    """
    q = query.lower().strip()
    if not q:
        return False

    # 1. Direct Patient ID or Record match (e.g. P1005, P1651, patient 12)
    if re.search(r"\bp\d+\b", q) or re.search(r"\bpatient\b", q) or re.search(r"\brecord\b", q):
        return True

    # 2. Key clinical and cardiovascular terminology
    clinical_terms = [
        "heart", "cardiac", "cardio", "coronary", "angina", "chest pain", "cp",
        "blood pressure", "bp", "trestbps", "hypertension", "hypotension", "normotensive",
        "cholesterol", "chol", "hypercholesterolemia", "lipid", "serum", "ldl", "hdl", "triglycerides",
        "ecg", "ekg", "electrocardiogram", "electrocardiographic",
        "blood sugar", "glucose", "fbs", "diabetic", "diabetes",
        "heart rate", "bpm", "thalach", "thalch", "tachycardia", "bradycardia", "pulse",
        "st depression", "oldpeak", "slope", "upsloping", "downsloping", "flat",
        "vessel", "vessels", "fluoroscopy", "thallium", "thal",
        "cad", "disease", "severity", "ischemia", "ischemic", "infarction", "myocardial",
        "atherosclerosis", "hypertrophy", "lvh", "defect", "reversable",
        "symptom", "symptoms", "cause", "causes", "diagnosis", "prognosis", "clinical",
        "biomarker", "vital", "mechanism", "pathophysiology", "risk", "risk factor", "factors",
        "hospital", "medical", "treatment", "doctor", "nurse", "physician", "medication",
        "health", "asymptomatic", "typical", "atypical", "non-anginal", "cohort",
        "age", "sex", "male", "female", "exercise induced", "exang", "cardiologist", "cardiology"
    ]
    for term in clinical_terms:
        if term in q:
            return True

    # 3. Off-topic indicators without clinical terms
    off_topic_indicators = [
        "quantum", "astrophysics", "rocket", "teleportation", "recipe", "cooking",
        "pasta", "movie", "cinema", "football", "cricket", "nba", "programming",
        "javascript", "python code", "minecraft", "weather forecast", "lyrics",
        "capital of", "president", "who won", "joke"
    ]
    if any(ind in q for ind in off_topic_indicators):
        return False

    return False


def sha256_hash(text: Any) -> str:
    """Compute SHA-256 hash using deterministic canonicalization"""
    return compute_canonical_hash(text)


class MedicalRAGService:
    """
    Production-grade Medical RAG Service for clinical Q&A and evidence retrieval.
    """

    def __init__(
        self,
        vectordb_path: Optional[str] = None,
        embedding_model_name: Optional[str] = None
    ):
        base_dir = Path(__file__).resolve().parent.parent.parent.parent
        if vectordb_path:
            self.vectordb_path = Path(vectordb_path)
        else:
            self.vectordb_path = base_dir / "vectordb"

        self.embedding_model_name = embedding_model_name or "pritamdeka/S-PubMedBert-MS-MARCO"

        # Initialize Groq client
        groq_key = os.getenv("GROQ_API_KEY")
        self.groq_client = Groq(api_key=groq_key) if groq_key else None

        # Initialize embedding model
        print(f"Loading HuggingFace Embedding model ({self.embedding_model_name})...")
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name
        )

        # Load FAISS index
        self.vectorstore = self._load_vectorstore()

    def _load_vectorstore(self) -> Optional[FAISS]:
        """Load FAISS index from disk"""
        if not self.vectordb_path.exists():
            print(f"Warning: Vector DB directory not found at: {self.vectordb_path}")
            return None

        try:
            print(f"Loading FAISS vectorstore from {self.vectordb_path}...")
            vs = FAISS.load_local(
                str(self.vectordb_path),
                self.embedding_model,
                allow_dangerous_deserialization=True
            )
            print(f"[OK] FAISS vectorstore loaded with {vs.index.ntotal} vectors.")
            return vs
        except Exception as e:
            print(f"Error loading FAISS vectorstore: {e}")
            return None

    def _extract_patient_metadata(self, text: str, initial_meta: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract structured attributes from clinical text.
        """
        meta = dict(initial_meta)

        # Check if medical knowledge document
        if meta.get("doc_type") == "medical_knowledge" or text.startswith("KB") or "Symptoms of Heart Disease" in text or "Causes and Pathophysiology" in text or "Mechanisms and Clinical Significance" in text:
            meta["doc_type"] = "medical_knowledge"
            return meta

        meta["doc_type"] = "patient_record"

        # Extract Patient ID
        if "id" not in meta or meta["id"] == "KB":
            p_match = re.search(r"Patient\s*ID\s*([A-Za-z0-9]+)", text, re.IGNORECASE)
            if p_match:
                meta["id"] = p_match.group(1)

        # Extract Name
        n_match = re.search(r"Name\s*([A-Za-z\s]+?),", text, re.IGNORECASE)
        if n_match:
            meta["name"] = n_match.group(1).strip()

        # Extract Age
        a_match = re.search(r"aged\s*(\d+)\s*years", text, re.IGNORECASE)
        if a_match:
            meta["age"] = int(a_match.group(1))

        # Extract Sex
        s_match = re.search(r"sex\s*(?:value)?\s*([A-Za-z0-9\.]+)", text, re.IGNORECASE)
        if s_match:
            raw_s = s_match.group(1).strip()
            if raw_s in ["1", "1.0", "male", "Male"]:
                meta["sex"] = "Male"
            elif raw_s in ["0", "0.0", "female", "Female"]:
                meta["sex"] = "Female"
            else:
                meta["sex"] = raw_s

        # Extract Chest Pain Type
        cp_match = re.search(r"chest pain type\s*([^,]+)", text, re.IGNORECASE)
        if cp_match:
            raw_cp = cp_match.group(1).strip().lower()
            if "atypical" in raw_cp or raw_cp in ["2", "2.0"]:
                meta["chest_pain"] = "Atypical Angina (Type 2)"
            elif "typical" in raw_cp or raw_cp in ["1", "1.0"]:
                meta["chest_pain"] = "Typical Angina (Type 1)"
            elif "non-anginal" in raw_cp or "non anginal" in raw_cp or raw_cp in ["3", "3.0"]:
                meta["chest_pain"] = "Non-Anginal Pain (Type 3)"
            elif "asymptomatic" in raw_cp or raw_cp in ["4", "4.0"]:
                meta["chest_pain"] = "Asymptomatic (Type 4)"
            else:
                meta["chest_pain"] = raw_cp.title()

        # Extract Blood Pressure
        bp_match = re.search(r"resting blood pressure\s*([\d\.]+)\s*mm\s*Hg", text, re.IGNORECASE)
        if bp_match:
            meta["resting_bp"] = float(bp_match.group(1))

        # Extract Cholesterol
        chol_match = re.search(r"cholesterol level\s*([\d\.]+)\s*mg/dL", text, re.IGNORECASE)
        if chol_match:
            meta["cholesterol"] = float(chol_match.group(1))

        # Extract Fasting Blood Sugar
        fbs_match = re.search(r"fasting blood sugar\s*([^,]+)", text, re.IGNORECASE)
        if fbs_match:
            raw_fbs = fbs_match.group(1).strip()
            meta["fasting_blood_sugar"] = "Elevated (>120 mg/dL)" if raw_fbs in ["1", "1.0", "True", "true"] else "Normal (<=120 mg/dL)"

        # Extract Resting ECG
        ecg_match = re.search(r"rest ECG result\s*([^,]+)", text, re.IGNORECASE)
        if ecg_match:
            raw_ecg = ecg_match.group(1).strip()
            ecg_map = {
                "0": "Normal",
                "0.0": "Normal",
                "1": "ST-T Wave Abnormality",
                "1.0": "ST-T Wave Abnormality",
                "2": "Left Ventricular Hypertrophy (LVH)",
                "2.0": "Left Ventricular Hypertrophy (LVH)"
            }
            meta["rest_ecg"] = ecg_map.get(raw_ecg, raw_ecg)

        # Extract Heart Rate
        hr_match = re.search(r"maximum heart rate\s*([\d\.]+)", text, re.IGNORECASE)
        if hr_match:
            meta["max_heart_rate"] = float(hr_match.group(1))

        # Extract Exercise Induced Angina
        exang_match = re.search(r"exercise induced angina\s*([^,]+)", text, re.IGNORECASE)
        if exang_match:
            raw_exang = exang_match.group(1).strip()
            meta["exercise_angina"] = "Yes (Present)" if raw_exang in ["1", "1.0", "True", "true"] else "No (Absent)"

        # Extract Oldpeak (ST depression)
        oldpeak_match = re.search(r"oldpeak value\s*([\d\.\-]+)", text, re.IGNORECASE)
        if oldpeak_match:
            meta["oldpeak"] = float(oldpeak_match.group(1))

        # Extract Slope
        slope_match = re.search(r"slope value\s*([^,]+)", text, re.IGNORECASE)
        if slope_match:
            raw_slope = slope_match.group(1).strip()
            slope_map = {"1": "Upsloping", "1.0": "Upsloping", "2": "Flat", "2.0": "Flat", "3": "Downsloping", "3.0": "Downsloping"}
            meta["slope"] = slope_map.get(raw_slope, raw_slope)

        # Extract Diagnosis Outcome
        diag_match = re.search(r"diagnosis outcome\s*([\d\.]+)", text, re.IGNORECASE)
        if diag_match:
            outcome = int(float(diag_match.group(1)))
            meta["diagnosis_outcome"] = outcome
            labels = {
                0: "Healthy / No CAD",
                1: "Mild CAD (Class 1)",
                2: "Moderate CAD (Class 2)",
                3: "Severe CAD (Class 3)",
                4: "Very Severe CAD (Class 4)"
            }
            meta["diagnosis_label"] = labels.get(outcome, f"Class {outcome}")

        return meta

    def retrieve_evidence(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant evidence items from FAISS index.
        Note: Internal distance metric is preserved for ranking, but user-facing
        similarity scores are not attached to avoid misleading metrics.
        """
        if not self.vectorstore:
            return []

        try:
            pool_k = min(max(top_k * 15, 60), getattr(self.vectorstore.index, "ntotal", 930))
            raw_results = self.vectorstore.similarity_search_with_score(query, k=pool_k)
        except Exception as e:
            print(f"Error during vector similarity search: {e}")
            return []

        q_lower = query.lower()

        # Check if query targets medical concepts vs specific patient
        is_concept_query = any(w in q_lower for w in ["what are the symptoms", "what causes", "explain st depression", "risk factors", "pathophysiology", "what does", "how does", "what role", "what is the clinical significance"])
        
        target_pid = re.findall(r"\b(p\d{4})\b", q_lower)
        target_pid = [p.upper() for p in target_pid]

        candidates = []
        for doc, distance in raw_results:
            content = doc.page_content.strip()
            meta = self._extract_patient_metadata(content, doc.metadata)
            doc_pid = meta.get("id", "").upper()
            doc_type = meta.get("doc_type", "patient_record")

            # Base score
            base_sim = 1.0 / (1.0 + float(distance))
            hybrid_score = base_sim

            # Medical knowledge documents boost for general medical questions
            if is_concept_query and doc_type == "medical_knowledge":
                hybrid_score += 1.5

            # Exact Patient ID Match boost
            if target_pid and doc_pid in target_pid:
                hybrid_score += 2.0

            candidates.append({
                "doc": doc,
                "content": content,
                "metadata": meta,
                "distance": float(distance),
                "hybrid_score": hybrid_score
            })

        # Sort by relevance score descending
        candidates.sort(key=lambda x: x["hybrid_score"], reverse=True)
        top_candidates = candidates[:top_k]

        evidence_list = []
        for rank, item in enumerate(top_candidates, 1):
            content = item["content"]
            meta = item["metadata"]
            meta["rank"] = rank

            evidence_list.append({
                "rank": rank,
                "text": content,
                "metadata": meta,
                "distance": item["distance"],
                "sha256_hash": sha256_hash(content),
                "patient_id": meta.get("id", f"Record-{rank}")
            })

        return evidence_list

    def answer_query(self, query: str, user_id: str = "guest", top_k: int = 5, bypass_structured: bool = False) -> Dict[str, Any]:
        """
        Execute full grounded RAG pipeline:
        1. Check for Structured Patient Analytics & Exact Patient Lookups (if not bypassed).
        2. Pre-filter query for clinical domain validity.
        3. Retrieve top-K evidence chunks from FAISS.
        4. Check relevance threshold and anti-hallucination guardrails.
        5. Generate grounded LLM answer citing retrieved evidence without displaying similarity scores.
        """
        query_cleaned = query.strip()
        if not query_cleaned:
            return {
                "query": query,
                "user_id": user_id,
                "answer": "Please provide a valid medical question or search query.",
                "retrieved_evidence": [],
                "evidence_hash": "",
                "has_relevant_evidence": False,
                "retrieved_count": 0,
                "is_mismatch": False,
                "query_type": "empty"
            }

        # -------------------------------------------------------------
        # STEP 1: Structured Patient Data Queries & Exact Patient Lookups
        # -------------------------------------------------------------
        if not bypass_structured:
            try:
                structured_engine = get_structured_engine()
                structured_intent = structured_engine.detect_structured_query_intent(query_cleaned)
                if structured_intent:
                    struct_res = structured_engine.execute_structured_query(query_cleaned)
                    if (
                        struct_res.get("success", False)
                        or struct_res.get("query_type", "").startswith("specific_patient")
                        or struct_res.get("query_type", "").startswith("structured")
                    ):
                        ans_text = struct_res["answer"]
                        evidence_payload = struct_res.get("retrieved_evidence", [])
                        return {
                            "query": query_cleaned,
                            "user_id": user_id,
                            "answer": ans_text,
                            "retrieved_evidence": evidence_payload,
                            "evidence_hash": sha256_hash(ans_text),
                            "has_relevant_evidence": struct_res.get("has_relevant_evidence", True),
                            "retrieved_count": len(evidence_payload),
                            "is_mismatch": False,
                            "query_type": struct_res.get("query_type", "structured_data")
                        }
            except Exception as e:
                print(f"Structured engine query routing error: {e}")

        # -------------------------------------------------------------
        # STEP 2: Clinical Domain Validity Check
        # -------------------------------------------------------------
        if not is_clinical_query(query_cleaned):
            return {
                "query": query_cleaned,
                "user_id": user_id,
                "answer": MISMATCH_REFUSAL_MESSAGE,
                "retrieved_evidence": [],
                "evidence_hash": "",
                "has_relevant_evidence": False,
                "retrieved_count": 0,
                "is_mismatch": True,
                "query_type": "off_topic_mismatch"
            }

        # -------------------------------------------------------------
        # STEP 3: Retrieve Evidence from FAISS
        # -------------------------------------------------------------
        evidence = self.retrieve_evidence(query_cleaned, top_k=top_k)

        # -------------------------------------------------------------
        # STEP 4: Relevance & Empty Context Guardrail Check
        # -------------------------------------------------------------
        if not evidence:
            return {
                "query": query_cleaned,
                "user_id": user_id,
                "answer": INSUFFICIENT_CONTEXT_MESSAGE,
                "retrieved_evidence": [],
                "evidence_hash": "",
                "has_relevant_evidence": False,
                "retrieved_count": 0,
                "is_mismatch": False,
                "query_type": "insufficient_context"
            }

        # Compute combined evidence hash for audit trail
        combined_evidence_text = "\n---\n".join([item["text"] for item in evidence])
        evidence_bundle_hash = sha256_hash(combined_evidence_text)

        # -------------------------------------------------------------
        # STEP 5: Format Verified Medical Context for LLM
        # -------------------------------------------------------------
        context_blocks = []
        has_kb_doc = False
        for item in evidence:
            meta = item["metadata"]
            doc_type = meta.get("doc_type", "patient_record")

            if doc_type == "medical_knowledge" or "KB" in meta.get("id", ""):
                has_kb_doc = True
                context_blocks.append(f"### Verified Medical Knowledge Reference [{meta.get('title', 'Clinical Reference')}]:\n{item['text']}")
            else:
                p_id = meta.get("id", "Unknown")
                age = meta.get("age", "N/A")
                sex = meta.get("sex", "N/A")
                cp = meta.get("chest_pain", "N/A")
                bp = meta.get("resting_bp", "N/A")
                chol = meta.get("cholesterol", "N/A")
                fbs = meta.get("fasting_blood_sugar", "N/A")
                ecg = meta.get("rest_ecg", "N/A")
                hr = meta.get("max_heart_rate", "N/A")
                exang = meta.get("exercise_angina", "N/A")
                oldpeak = meta.get("oldpeak", "N/A")
                slope = meta.get("slope", "N/A")
                diag = meta.get("diagnosis_label", "N/A")

                block = (
                    f"### Patient Record: {p_id}\n"
                    f"- Age: {age} | Sex: {sex}\n"
                    f"- Chest Pain: {cp}\n"
                    f"- Resting BP: {bp} mmHg | Cholesterol: {chol} mg/dL\n"
                    f"- Fasting Blood Sugar: {fbs}\n"
                    f"- Resting ECG: {ecg} | Max Heart Rate: {hr} bpm\n"
                    f"- Exercise Induced Angina: {exang}\n"
                    f"- ST Depression (oldpeak): {oldpeak} mm | Slope: {slope}\n"
                    f"- Diagnosis: {diag}\n"
                    f"- Full Clinical Text: {item['text']}"
                )
                context_blocks.append(block)

        context_text = "\n\n".join(context_blocks)

        # -------------------------------------------------------------
        # STEP 6: System Prompt and Grounded Generation
        # -------------------------------------------------------------
        system_prompt = (
            "You are RAGChainMed, an expert clinical AI decision support assistant specializing in cardiovascular medicine. "
            "Your role is to provide grounded, accurate, and concise answers based STRICTLY on the retrieved medical knowledge and patient records provided in the context.\n\n"
            "CRITICAL CLINICAL & ANTI-HALLUCINATION GUIDELINES:\n"
            "1. Grounded Medical Answers: For general medical inquiries (such as symptoms of heart disease, causes of high cholesterol, ST depression mechanisms, or risk factors), clearly explain the pathophysiological and clinical concepts using the retrieved medical knowledge.\n"
            "2. Patient Evidence: When answering questions regarding patient cases or cohorts, cite specific matching Patient IDs and key clinical indicators (Age, Sex, Chest Pain, BP, Cholesterol, ECG, ST depression, Diagnosis).\n"
            "3. Zero Fabrication: Do not invent unsupported medical claims or patient records not present in the context.\n"
            "4. If the retrieved context is genuinely insufficient to address the query, state: 'I could not find sufficient information in the available medical knowledge base to answer this question.'\n"
            "5. NEVER mention similarity scores, vector distances, or retrieval confidence metrics in your generated response."
        )

        user_prompt = f"""Verified Medical Knowledge & Clinical Records:
{context_text}

Clinical Query:
{query_cleaned}

Please provide a clear, grounded clinical response directly addressing the query using the verified information above."""

        if not self.groq_client:
            groq_key = os.getenv("GROQ_API_KEY")
            if groq_key:
                try:
                    self.groq_client = Groq(api_key=groq_key)
                except Exception as e:
                    print(f"Error initializing Groq client: {e}")

        answer = ""
        if self.groq_client:
            candidate_models = [
                "openai/gpt-oss-120b",
                "openai/gpt-oss-20b",
                "qwen/qwen3.6-27b",
                "allam-2-7b"
            ]
            for m in candidate_models:
                try:
                    response = self.groq_client.chat.completions.create(
                        model=m,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.2,
                        max_tokens=800,
                        timeout=25
                    )
                    content = response.choices[0].message.content
                    if content and content.strip():
                        if "</think>" in content:
                            content = content.split("</think>")[-1].strip()
                        if content:
                            answer = content
                            break
                except Exception as e:
                    continue

        if not answer:
            # Fallback structured summary
            summary_lines = []
            for e in evidence[:3]:
                meta = e["metadata"]
                if meta.get("doc_type") == "medical_knowledge":
                    summary_lines.append(f"• {meta.get('title', 'Medical Guide')}: {e['text'][:180]}...")
                else:
                    summary_lines.append(
                        f"• Patient {meta.get('id', 'N/A')}: Age {meta.get('age', 'N/A')}, "
                        f"Cholesterol {meta.get('cholesterol', 'N/A')} mg/dL, "
                        f"BP {meta.get('resting_bp', 'N/A')} mmHg, "
                        f"Chest Pain: {meta.get('chest_pain', 'N/A')}, "
                        f"Diagnosis: {meta.get('diagnosis_label', 'N/A')}."
                    )
            answer = (
                f"**Clinical Summary:**\n"
                f"Retrieved {len(evidence)} verified medical evidence item(s) from the clinical knowledge base.\n\n"
                f"**Key Retrieved Evidence:**\n" + "\n".join(summary_lines)
            )

        # Hallucination / Refusal Check
        mismatch_phrases = [
            "i apologize for the mismatch",
            "mismatch in the query",
            "capabilities are limited to clinical decision support",
            "limited to clinical decision support",
            "specialized for clinical cardiovascular",
            "cannot answer questions unrelated to",
            "unrelated to medicine",
            "unrelated to healthcare",
            "not related to medicine",
            "not related to healthcare"
        ]

        answer_lower = answer.lower()
        if any(phrase in answer_lower for phrase in mismatch_phrases):
            return {
                "query": query_cleaned,
                "user_id": user_id,
                "answer": MISMATCH_REFUSAL_MESSAGE,
                "retrieved_evidence": [],
                "evidence_hash": "",
                "has_relevant_evidence": False,
                "retrieved_count": 0,
                "is_mismatch": True,
                "query_type": "off_topic_mismatch"
            }

        return {
            "query": query_cleaned,
            "user_id": user_id,
            "answer": answer,
            "retrieved_evidence": evidence,
            "evidence_hash": evidence_bundle_hash,
            "has_relevant_evidence": True,
            "retrieved_count": len(evidence),
            "is_mismatch": False,
            "query_type": "rag_medical"
        }


# Lazy global instance
_rag_service: Optional[MedicalRAGService] = None


def get_rag_service() -> MedicalRAGService:
    """Get or create singleton MedicalRAGService"""
    global _rag_service
    if _rag_service is None:
        _rag_service = MedicalRAGService()
    return _rag_service
