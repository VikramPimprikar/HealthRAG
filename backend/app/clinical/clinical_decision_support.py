"""
Clinical Decision Support Module
================================
Provides decision support recommendations based on patient data,
medical guidelines, and model predictions.

Integrates with RAG to provide evidence-based recommendations.

Author: RAGChainMed
Date: May 2026
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class RiskLevel(Enum):
    """Risk assessment levels"""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class RecommendationType(Enum):
    """Types of clinical recommendations"""
    MONITORING = "monitoring"
    MEDICATION = "medication"
    LIFESTYLE = "lifestyle"
    IMAGING = "imaging"
    LABORATORY = "laboratory"
    CONSULTATION = "consultation"
    URGENT = "urgent"


@dataclass
class ClinicalRecommendation:
    """Represents a clinical recommendation"""
    type: RecommendationType
    description: str
    priority: int  # 1-5, where 5 is highest
    evidence_sources: List[str]
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()


@dataclass
class RiskAssessment:
    """Represents a risk assessment"""
    patient_id: str
    risk_level: RiskLevel
    risk_score: float  # 0-1
    contributing_factors: List[str]
    recommendations: List[ClinicalRecommendation]
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()


# ============================================================
# CLINICAL DECISION SUPPORT ENGINE
# ============================================================

class ClinicalDecisionSupportEngine:
    """
    Provides clinical decision support using patient data and medical knowledge.
    """
    
    def __init__(self):
        self.guidelines = self._initialize_guidelines()
    
    def _initialize_guidelines(self) -> Dict[str, Dict[str, Any]]:
        """Initialize clinical guidelines"""
        return {
            'hypertension': {
                'risk_factors': ['systolic_bp > 140', 'diastolic_bp > 90', 'age > 60'],
                'recommendations': [
                    {'type': RecommendationType.MONITORING, 'description': 'Daily blood pressure monitoring'},
                    {'type': RecommendationType.LIFESTYLE, 'description': 'Reduce sodium intake'},
                    {'type': RecommendationType.MEDICATION, 'description': 'Consider ACE inhibitors or diuretics'},
                    {'type': RecommendationType.CONSULTATION, 'description': 'Refer to cardiologist if BP remains elevated'},
                ]
            },
            'diabetes': {
                'risk_factors': ['glucose > 126', 'BMI > 30', 'family_history'],
                'recommendations': [
                    {'type': RecommendationType.MONITORING, 'description': 'Regular glucose monitoring'},
                    {'type': RecommendationType.LABORATORY, 'description': 'HbA1c test every 3 months'},
                    {'type': RecommendationType.LIFESTYLE, 'description': 'Exercise and dietary management'},
                    {'type': RecommendationType.MEDICATION, 'description': 'Consider metformin or other agents'},
                ]
            },
            'heart_disease': {
                'risk_factors': ['chest_pain', 'previous_MI', 'high_cholesterol'],
                'recommendations': [
                    {'type': RecommendationType.URGENT, 'description': 'Immediate EKG if chest pain present'},
                    {'type': RecommendationType.IMAGING, 'description': 'Stress test or angiography'},
                    {'type': RecommendationType.MEDICATION, 'description': 'Consider beta-blockers and statins'},
                    {'type': RecommendationType.CONSULTATION, 'description': 'Refer to cardiologist'},
                ]
            }
        }
    
    def assess_risk(self, 
                   patient_id: str,
                   patient_data: Dict[str, Any],
                   model_prediction: Optional[str] = None,
                   rag_context: Optional[str] = None) -> RiskAssessment:
        """
        Perform comprehensive risk assessment.
        
        Args:
            patient_id: Patient identifier
            patient_data: Patient's vital signs and medical data
            model_prediction: Machine learning model prediction if available
            rag_context: Context from RAG system
            
        Returns:
            RiskAssessment with recommendations
        """
        # Initialize assessment
        risk_level = RiskLevel.LOW
        risk_score = 0.0
        contributing_factors = []
        recommendations = []
        
        # Analyze vital signs
        if 'systolic_bp' in patient_data and patient_data['systolic_bp'] > 140:
            risk_score += 0.3
            contributing_factors.append('Elevated blood pressure')
        
        if 'heart_rate' in patient_data:
            if patient_data['heart_rate'] > 100:
                risk_score += 0.15
                contributing_factors.append('Elevated heart rate')
        
        if 'respiratory_rate' in patient_data:
            if patient_data['respiratory_rate'] > 20:
                risk_score += 0.1
                contributing_factors.append('Elevated respiratory rate')
        
        # Analyze model prediction
        if model_prediction:
            if 'severe' in model_prediction.lower() or 'critical' in model_prediction.lower():
                risk_score += 0.25
                contributing_factors.append(f'Model prediction: {model_prediction}')
        
        # Determine risk level based on score
        if risk_score >= 0.7:
            risk_level = RiskLevel.CRITICAL
        elif risk_score >= 0.5:
            risk_level = RiskLevel.HIGH
        elif risk_score >= 0.3:
            risk_level = RiskLevel.MODERATE
        
        # Generate recommendations based on risk level
        recommendations = self._generate_recommendations(
            patient_data, 
            risk_level, 
            contributing_factors,
            rag_context
        )
        
        return RiskAssessment(
            patient_id=patient_id,
            risk_level=risk_level,
            risk_score=min(risk_score, 1.0),  # Clamp to 0-1
            contributing_factors=contributing_factors,
            recommendations=recommendations
        )
    
    def _generate_recommendations(self,
                                 patient_data: Dict[str, Any],
                                 risk_level: RiskLevel,
                                 factors: List[str],
                                 rag_context: Optional[str] = None) -> List[ClinicalRecommendation]:
        """Generate clinical recommendations based on risk assessment"""
        recommendations = []
        
        # High and critical risk require urgent evaluation
        if risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            recommendations.append(ClinicalRecommendation(
                type=RecommendationType.URGENT if risk_level == RiskLevel.CRITICAL 
                     else RecommendationType.CONSULTATION,
                description="Immediate clinical evaluation required",
                priority=5,
                evidence_sources=['Risk assessment', 'Clinical guidelines']
            ))
        
        # Add monitoring recommendations
        recommendations.append(ClinicalRecommendation(
            type=RecommendationType.MONITORING,
            description="Continuous vital signs monitoring recommended",
            priority=4 if risk_level == RiskLevel.HIGH else 3,
            evidence_sources=['Vital signs analysis']
        ))
        
        # Add laboratory recommendations
        recommendations.append(ClinicalRecommendation(
            type=RecommendationType.LABORATORY,
            description="Comprehensive metabolic panel recommended",
            priority=4,
            evidence_sources=['Patient assessment', 'Clinical guidelines']
        ))
        
        # Add medication considerations
        if 'Elevated blood pressure' in factors:
            recommendations.append(ClinicalRecommendation(
                type=RecommendationType.MEDICATION,
                description="Consider antihypertensive therapy",
                priority=4,
                evidence_sources=['Blood pressure elevation', 'Clinical guidelines']
            ))
        
        return sorted(recommendations, key=lambda r: r.priority, reverse=True)
    
    def get_guideline_recommendations(self, condition: str) -> Optional[Dict[str, Any]]:
        """Get evidence-based recommendations for a condition"""
        return self.guidelines.get(condition)
    
    def format_assessment_report(self, assessment: RiskAssessment) -> str:
        """Format risk assessment as readable report"""
        report = f"""
CLINICAL RISK ASSESSMENT REPORT
================================
Patient ID: {assessment.patient_id}
Assessment Date: {assessment.timestamp}

RISK LEVEL: {assessment.risk_level.value.upper()}
Risk Score: {assessment.risk_score:.2%}

Contributing Factors:
{chr(10).join(f'  • {factor}' for factor in assessment.contributing_factors)}

Clinical Recommendations:
{chr(10).join(f'  {i+1}. [{rec.type.value.upper()}] {rec.description} (Priority: {rec.priority}/5)' 
              for i, rec in enumerate(assessment.recommendations))}

Evidence Sources:
{chr(10).join(set(f'  • {source}' 
                  for rec in assessment.recommendations 
                  for source in rec.evidence_sources))}

Note: This assessment is generated to support clinical decision-making.
Final clinical decisions should be made by qualified healthcare professionals.
        """
        return report
