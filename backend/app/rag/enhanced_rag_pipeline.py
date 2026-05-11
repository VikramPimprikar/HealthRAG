"""
Enhanced RAG Pipeline for Medical Data
======================================
Implements a comprehensive Retrieval-Augmented Generation system
specifically designed for clinical data and medical knowledge.

Handles both structured patient records and unstructured clinical notes,
supporting intelligent search and context-aware response generation.

Author: RAGChainMed
Date: May 2026
"""

import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import json
from datetime import datetime
from dotenv import load_dotenv

try:
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
    from langchain.schema import Document
    LANGCHAIN_AVAILABLE = True
except ImportError:
    print("Warning: LangChain dependencies not fully installed. Install with: pip install langchain langchain-community")
    LANGCHAIN_AVAILABLE = False
    
    # Minimal Document class for when LangChain isn't available
    class Document:
        def __init__(self, page_content: str, metadata: Dict[str, Any] = None):
            self.page_content = page_content
            self.metadata = metadata or {}


# Load environment variables
load_dotenv()


# ============================================================
# KNOWLEDGE BASE MANAGEMENT
# ============================================================

class MedicalKnowledgeBase:
    """
    Manages a medical knowledge base with support for:
    - Clinical guidelines and protocols
    - Drug information and interactions
    - Diagnostic criteria
    - Treatment recommendations
    - Anatomical information
    """
    
    def __init__(self, base_dir: str = "knowledge_base"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # Knowledge categories
        self.categories = {
            'clinical_guidelines': self.base_dir / 'clinical_guidelines.json',
            'drug_database': self.base_dir / 'drug_database.json',
            'diagnostic_criteria': self.base_dir / 'diagnostic_criteria.json',
            'treatment_protocols': self.base_dir / 'treatment_protocols.json',
            'contraindications': self.base_dir / 'contraindications.json',
        }
        
        # Initialize knowledge files
        self._initialize_knowledge_files()
    
    def _initialize_knowledge_files(self):
        """Initialize knowledge base files"""
        for category, filepath in self.categories.items():
            if not filepath.exists():
                with open(filepath, 'w') as f:
                    json.dump({'metadata': {'created': datetime.utcnow().isoformat()}, 'records': []}, f)
    
    def add_guideline(self, condition: str, guideline: str, source: str):
        """Add a clinical guideline"""
        self._add_to_category('clinical_guidelines', {
            'condition': condition,
            'guideline': guideline,
            'source': source,
            'added_date': datetime.utcnow().isoformat()
        })
    
    def add_drug_info(self, drug_name: str, indications: str, 
                     contraindications: str, interactions: List[str]):
        """Add drug information"""
        self._add_to_category('drug_database', {
            'drug_name': drug_name,
            'indications': indications,
            'contraindications': contraindications,
            'interactions': interactions,
            'added_date': datetime.utcnow().isoformat()
        })
    
    def add_diagnostic_criteria(self, disease: str, criteria: Dict[str, Any]):
        """Add diagnostic criteria for a disease"""
        self._add_to_category('diagnostic_criteria', {
            'disease': disease,
            'criteria': criteria,
            'added_date': datetime.utcnow().isoformat()
        })
    
    def _add_to_category(self, category: str, record: Dict):
        """Add a record to a knowledge category"""
        filepath = self.categories[category]
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        data['records'].append(record)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_all_knowledge(self) -> str:
        """Get all knowledge as formatted text for embedding"""
        knowledge_text = "MEDICAL KNOWLEDGE BASE\n" + "="*50 + "\n\n"
        
        for category, filepath in self.categories.items():
            if filepath.exists():
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                if data['records']:
                    knowledge_text += f"\n{category.upper().replace('_', ' ')}\n"
                    knowledge_text += "-" * len(category) + "\n"
                    
                    for record in data['records']:
                        knowledge_text += json.dumps(record, indent=2) + "\n"
        
        return knowledge_text


# ============================================================
# MEDICAL DATA LOADER
# ============================================================

class MedicalDataLoader:
    """Loads and processes medical data from various sources"""
    
    @staticmethod
    def load_patient_records(csv_path: str) -> List[Document]:
        """
        Load patient records from CSV and convert to Documents.
        
        Args:
            csv_path: Path to patient records CSV
            
        Returns:
            List of LangChain Document objects
        """
        try:
            import pandas as pd
            
            df = pd.read_csv(csv_path)
            documents = []
            
            for idx, row in df.iterrows():
                # Convert row to text format
                content = "PATIENT RECORD\n"
                for col, val in row.items():
                    content += f"{col}: {val}\n"
                
                doc = Document(
                    page_content=content,
                    metadata={
                        'source': 'patient_records',
                        'type': 'structured_data',
                        'patient_id': str(row.get('patient_id', idx)),
                        'loaded_at': datetime.utcnow().isoformat()
                    }
                )
                documents.append(doc)
            
            return documents
        except Exception as e:
            print(f"Error loading patient records: {e}")
            return []
    
    @staticmethod
    def load_clinical_notes(directory: str) -> List[Document]:
        """
        Load clinical notes from text files.
        
        Args:
            directory: Directory containing clinical note files
            
        Returns:
            List of LangChain Document objects
        """
        documents = []
        path = Path(directory)
        
        if not path.exists():
            return documents
        
        for file_path in path.glob('*.txt'):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                doc = Document(
                    page_content=content,
                    metadata={
                        'source': str(file_path),
                        'type': 'unstructured_clinical_note',
                        'filename': file_path.name,
                        'loaded_at': datetime.utcnow().isoformat()
                    }
                )
                documents.append(doc)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        return documents


# ============================================================
# ENHANCED RAG PIPELINE
# ============================================================

class MedicalRAGPipeline:
    """
    Enhanced RAG pipeline for medical question-answering.
    
    Combines:
    - Medical knowledge base
    - Patient records (structured)
    - Clinical notes (unstructured)
    - Vector similarity search
    """
    
    def __init__(self, knowledge_base_dir: str = "knowledge_base"):
        """
        Initialize the medical RAG pipeline.
        
        Args:
            knowledge_base_dir: Directory for storing medical knowledge
        """
        self.knowledge_base = MedicalKnowledgeBase(knowledge_base_dir)
        self.data_loader = MedicalDataLoader()
        self.vector_db = None
        self.embeddings = None
        self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        """Initialize embedding model"""
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        except Exception as e:
            print(f"Error initializing embeddings: {e}")
    
    def build_knowledge_base(self, 
                            patient_records_csv: Optional[str] = None,
                            clinical_notes_dir: Optional[str] = None):
        """
        Build the vector database from medical knowledge and data.
        
        Args:
            patient_records_csv: Path to patient records CSV
            clinical_notes_dir: Directory containing clinical notes
        """
        if not LANGCHAIN_AVAILABLE:
            print("Warning: LangChain not installed. Vector database not available.")
            print("Install with: pip install langchain langchain-community sentence-transformers")
            return
            
        documents = []
        
        # Load medical knowledge
        print("Loading medical knowledge base...")
        knowledge_text = self.knowledge_base.get_all_knowledge()
        if knowledge_text.strip():
            # Split medical knowledge
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                separators=["\n\n", "\n", " ", ""]
            )
            knowledge_docs = splitter.split_text(knowledge_text)
            
            for doc_text in knowledge_docs:
                doc = Document(
                    page_content=doc_text,
                    metadata={
                        'source': 'medical_knowledge',
                        'type': 'clinical_guideline',
                        'loaded_at': datetime.utcnow().isoformat()
                    }
                )
                documents.append(doc)
        
        # Load patient records if provided
        if patient_records_csv:
            print(f"Loading patient records from {patient_records_csv}...")
            patient_docs = self.data_loader.load_patient_records(patient_records_csv)
            documents.extend(patient_docs)
        
        # Load clinical notes if provided
        if clinical_notes_dir:
            print(f"Loading clinical notes from {clinical_notes_dir}...")
            clinical_docs = self.data_loader.load_clinical_notes(clinical_notes_dir)
            documents.extend(clinical_docs)
        
        if not documents:
            print("Warning: No documents loaded for knowledge base")
            return
        
        # Create vector database
        print(f"Creating vector database with {len(documents)} documents...")
        if self.embeddings:
            self.vector_db = FAISS.from_documents(documents, self.embeddings)
            print("✓ Vector database created successfully")
    
    def retrieve_context(self, query: str, k: int = 3) -> List[Tuple[str, Dict]]:
        """
        Retrieve relevant documents based on a medical query.
        
        Args:
            query: Medical question or query
            k: Number of results to return
            
        Returns:
            List of (document_text, metadata) tuples
        """
        if not self.vector_db:
            print("Warning: Vector database not initialized")
            return []
        
        try:
            # Perform similarity search
            results = self.vector_db.similarity_search_with_score(query, k=k)
            
            # Format results
            context_list = []
            for doc, score in results:
                context_list.append((doc.page_content, {
                    'source': doc.metadata.get('source', 'unknown'),
                    'type': doc.metadata.get('type', 'unknown'),
                    'similarity_score': float(score)
                }))
            
            return context_list
        except Exception as e:
            print(f"Error retrieving context: {e}")
            return []
    
    def format_context(self, context_results: List[Tuple[str, Dict]]) -> str:
        """
        Format retrieved context for use in LLM prompts.
        
        Args:
            context_results: Results from retrieve_context
            
        Returns:
            Formatted context string
        """
        if not context_results:
            return "No relevant medical knowledge found for this query."
        
        formatted = "RELEVANT MEDICAL CONTEXT:\n" + "="*50 + "\n\n"
        
        for i, (content, metadata) in enumerate(context_results, 1):
            formatted += f"Source {i}: {metadata['source']} ({metadata['type']})\n"
            formatted += f"Relevance Score: {metadata['similarity_score']:.3f}\n"
            formatted += f"Content:\n{content}\n"
            formatted += "-"*50 + "\n\n"
        
        return formatted
    
    def answer_medical_question(self, query: str, llm_response: str) -> Dict[str, Any]:
        """
        Answer a medical question using RAG.
        
        Args:
            query: Medical question
            llm_response: Response from LLM
            
        Returns:
            Dict with query, context, and response
        """
        # Retrieve context
        context_results = self.retrieve_context(query, k=3)
        formatted_context = self.format_context(context_results)
        
        return {
            'query': query,
            'context': formatted_context,
            'response': llm_response,
            'context_sources': [m['source'] for _, m in context_results],
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def save_vector_db(self, save_path: str = "medical_vector_db"):
        """Save the vector database to disk"""
        if self.vector_db:
            self.vector_db.save_local(save_path)
            print(f"✓ Vector database saved to {save_path}")
    
    def load_vector_db(self, save_path: str = "medical_vector_db"):
        """Load a previously saved vector database"""
        try:
            if self.embeddings:
                self.vector_db = FAISS.load_local(save_path, self.embeddings)
                print(f"✓ Vector database loaded from {save_path}")
        except Exception as e:
            print(f"Error loading vector database: {e}")


# ============================================================
# BACKWARD COMPATIBILITY FUNCTIONS
# ============================================================

# Maintain compatibility with existing code
_pipeline = None

def load_knowledge():
    """Load knowledge base (for backward compatibility)"""
    global _pipeline
    if not _pipeline:
        _pipeline = MedicalRAGPipeline()
        # Try to load patient records if they exist
        patient_csv = Path(__file__).resolve().parent.parent.parent / 'data' / 'processed' / 'processed_data.csv'
        if patient_csv.exists():
            _pipeline.build_knowledge_base(patient_records_csv=str(patient_csv))
    return _pipeline.vector_db

def retrieve_context(db, query):
    """Retrieve context (for backward compatibility)"""
    if not _pipeline:
        load_knowledge()
    
    context_results = _pipeline.retrieve_context(query, k=2)
    return " ".join([content for content, _ in context_results])
