"""
Clinical Data Management Module
================================
Manages clinical data including patient records, vital signs,
medical histories, and unstructured clinical notes.

Provides data validation, privacy controls, and access logging.

Author: RAGChainMed
Date: May 2026
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import json
from pathlib import Path


class DataClassification(Enum):
    """Classification levels for medical data"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class MedicalDataType(Enum):
    """Types of medical data"""
    PATIENT_DEMOGRAPHICS = "patient_demographics"
    VITAL_SIGNS = "vital_signs"
    LABORATORY_RESULTS = "laboratory_results"
    IMAGING_RESULTS = "imaging_results"
    CLINICAL_NOTES = "clinical_notes"
    MEDICATION_HISTORY = "medication_history"
    DIAGNOSIS = "diagnosis"
    TREATMENT_PLAN = "treatment_plan"


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class PatientRecord:
    """Represents a patient's medical record"""
    patient_id: str
    first_name: str
    last_name: str
    date_of_birth: str
    gender: str
    blood_type: Optional[str] = None
    contact_number: Optional[str] = None
    email: Optional[str] = None
    allergies: Optional[List[str]] = None
    current_medications: Optional[List[str]] = None
    chronic_conditions: Optional[List[str]] = None
    emergency_contact: Optional[Dict[str, str]] = None
    created_at: str = ""
    updated_at: str = ""
    created_by: str = "system"
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat()
        if not self.updated_at:
            self.updated_at = datetime.utcnow().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class VitalSigns:
    """Represents vital signs reading"""
    patient_id: str
    timestamp: str
    heart_rate: float
    systolic_bp: float
    diastolic_bp: float
    temperature: float
    respiratory_rate: float
    oxygen_saturation: float
    recorded_by: str = "system"
    notes: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class ClinicalNote:
    """Represents an unstructured clinical note"""
    note_id: str
    patient_id: str
    note_type: str  # e.g., 'SOAP', 'Consultation', 'Progress'
    content: str
    author: str
    timestamp: str
    classification: DataClassification = DataClassification.CONFIDENTIAL
    tags: Optional[List[str]] = None
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['classification'] = self.classification.value
        return data


# ============================================================
# CLINICAL DATA MANAGER
# ============================================================

class ClinicalDataManager:
    """
    Manages clinical data storage and retrieval.
    
    Handles:
    - Patient records
    - Vital signs
    - Clinical notes
    - Medical histories
    """
    
    def __init__(self, data_dir: str = "clinical_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.patients_dir = self.data_dir / "patients"
        self.vital_signs_dir = self.data_dir / "vital_signs"
        self.notes_dir = self.data_dir / "clinical_notes"
        self.observations_dir = self.data_dir / "observations"
        
        for dir_path in [self.patients_dir, self.vital_signs_dir, 
                        self.notes_dir, self.observations_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # In-memory cache
        self.patient_cache: Dict[str, PatientRecord] = {}
        self.vital_signs_cache: Dict[str, List[VitalSigns]] = {}
        self.notes_cache: Dict[str, List[ClinicalNote]] = {}
    
    # ============================================================
    # PATIENT RECORD MANAGEMENT
    # ============================================================
    
    def add_patient(self, patient: PatientRecord) -> bool:
        """Add a new patient record"""
        if patient.patient_id in self.patient_cache:
            return False
        
        self.patient_cache[patient.patient_id] = patient
        self._save_patient(patient)
        return True
    
    def update_patient(self, patient: PatientRecord) -> bool:
        """Update patient record"""
        if patient.patient_id not in self.patient_cache:
            return False
        
        patient.updated_at = datetime.utcnow().isoformat()
        self.patient_cache[patient.patient_id] = patient
        self._save_patient(patient)
        return True
    
    def get_patient(self, patient_id: str) -> Optional[PatientRecord]:
        """Get patient record by ID"""
        if patient_id in self.patient_cache:
            return self.patient_cache[patient_id]
        
        # Try to load from disk
        patient = self._load_patient(patient_id)
        if patient:
            self.patient_cache[patient_id] = patient
        return patient
    
    def _save_patient(self, patient: PatientRecord):
        """Save patient to disk"""
        file_path = self.patients_dir / f"{patient.patient_id}.json"
        with open(file_path, 'w') as f:
            json.dump(patient.to_dict(), f, indent=2)
    
    def _load_patient(self, patient_id: str) -> Optional[PatientRecord]:
        """Load patient from disk"""
        file_path = self.patients_dir / f"{patient_id}.json"
        if file_path.exists():
            with open(file_path, 'r') as f:
                data = json.load(f)
                return PatientRecord(**data)
        return None
    
    def list_all_patients(self) -> List[str]:
        """Get list of all patient IDs"""
        patients = []
        for file_path in self.patients_dir.glob("*.json"):
            patients.append(file_path.stem)
        return patients
    
    # ============================================================
    # VITAL SIGNS MANAGEMENT
    # ============================================================
    
    def record_vital_signs(self, vital_signs: VitalSigns) -> bool:
        """Record vital signs for a patient"""
        patient_id = vital_signs.patient_id
        
        if patient_id not in self.vital_signs_cache:
            self.vital_signs_cache[patient_id] = []
        
        self.vital_signs_cache[patient_id].append(vital_signs)
        self._save_vital_signs(patient_id, vital_signs)
        return True
    
    def get_vital_signs_history(self, patient_id: str) -> List[VitalSigns]:
        """Get vital signs history for a patient"""
        if patient_id in self.vital_signs_cache:
            return self.vital_signs_cache[patient_id]
        
        # Load from disk
        vital_signs_list = self._load_vital_signs_history(patient_id)
        if vital_signs_list:
            self.vital_signs_cache[patient_id] = vital_signs_list
        return vital_signs_list
    
    def get_latest_vital_signs(self, patient_id: str) -> Optional[VitalSigns]:
        """Get the most recent vital signs reading"""
        history = self.get_vital_signs_history(patient_id)
        if history:
            return history[-1]
        return None
    
    def _save_vital_signs(self, patient_id: str, vital_signs: VitalSigns):
        """Save vital signs to disk"""
        file_path = self.vital_signs_dir / f"{patient_id}.json"
        
        # Load existing data
        existing_data = []
        if file_path.exists():
            with open(file_path, 'r') as f:
                existing_data = json.load(f)
        
        # Add new record
        existing_data.append(vital_signs.to_dict())
        
        # Save
        with open(file_path, 'w') as f:
            json.dump(existing_data, f, indent=2)
    
    def _load_vital_signs_history(self, patient_id: str) -> List[VitalSigns]:
        """Load vital signs history from disk"""
        file_path = self.vital_signs_dir / f"{patient_id}.json"
        if file_path.exists():
            with open(file_path, 'r') as f:
                data = json.load(f)
                return [VitalSigns(**record) for record in data]
        return []
    
    # ============================================================
    # CLINICAL NOTES MANAGEMENT
    # ============================================================
    
    def add_clinical_note(self, note: ClinicalNote) -> bool:
        """Add a clinical note"""
        patient_id = note.patient_id
        
        if patient_id not in self.notes_cache:
            self.notes_cache[patient_id] = []
        
        self.notes_cache[patient_id].append(note)
        self._save_clinical_note(patient_id, note)
        return True
    
    def get_clinical_notes(self, patient_id: str) -> List[ClinicalNote]:
        """Get all clinical notes for a patient"""
        if patient_id in self.notes_cache:
            return self.notes_cache[patient_id]
        
        # Load from disk
        notes = self._load_clinical_notes(patient_id)
        if notes:
            self.notes_cache[patient_id] = notes
        return notes
    
    def get_note_by_id(self, note_id: str) -> Optional[ClinicalNote]:
        """Get a specific clinical note by ID"""
        # Search in cache
        for patient_notes in self.notes_cache.values():
            for note in patient_notes:
                if note.note_id == note_id:
                    return note
        
        # TODO: Search in disk if not in cache
        return None
    
    def _save_clinical_note(self, patient_id: str, note: ClinicalNote):
        """Save clinical note to disk"""
        file_path = self.notes_dir / f"{patient_id}.json"
        
        # Load existing notes
        existing_notes = []
        if file_path.exists():
            with open(file_path, 'r') as f:
                existing_notes = json.load(f)
        
        # Add new note
        existing_notes.append(note.to_dict())
        
        # Save
        with open(file_path, 'w') as f:
            json.dump(existing_notes, f, indent=2)
    
    def _load_clinical_notes(self, patient_id: str) -> List[ClinicalNote]:
        """Load clinical notes from disk"""
        file_path = self.notes_dir / f"{patient_id}.json"
        if file_path.exists():
            with open(file_path, 'r') as f:
                data = json.load(f)
                return [ClinicalNote(
                    **{k: DataClassification(v) if k == 'classification' else v 
                       for k, v in record.items()}
                ) for record in data]
        return []
    
    # ============================================================
    # DATA EXPORT & REPORTING
    # ============================================================
    
    def generate_patient_summary(self, patient_id: str) -> Dict[str, Any]:
        """
        Generate a comprehensive summary of a patient's data.
        
        Returns:
            Dictionary with patient info, vital signs, notes, etc.
        """
        patient = self.get_patient(patient_id)
        if not patient:
            return {}
        
        vital_signs = self.get_vital_signs_history(patient_id)
        clinical_notes = self.get_clinical_notes(patient_id)
        
        return {
            'patient': patient.to_dict(),
            'vital_signs_records': len(vital_signs),
            'latest_vital_signs': vital_signs[-1].to_dict() if vital_signs else None,
            'clinical_notes_count': len(clinical_notes),
            'recent_notes': [n.to_dict() for n in clinical_notes[-5:]] if clinical_notes else [],
            'generated_at': datetime.utcnow().isoformat()
        }
    
    def export_patient_data_json(self, patient_id: str, output_path: str):
        """Export patient data to JSON file"""
        summary = self.generate_patient_summary(patient_id)
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def get_data_statistics(self) -> Dict[str, Any]:
        """Get statistics about clinical data"""
        return {
            'total_patients': len(self.list_all_patients()),
            'total_vital_signs_records': sum(
                len(self._load_vital_signs_history(pid))
                for pid in self.list_all_patients()
            ),
            'total_clinical_notes': sum(
                len(self._load_clinical_notes(pid))
                for pid in self.list_all_patients()
            ),
            'timestamp': datetime.utcnow().isoformat()
        }
