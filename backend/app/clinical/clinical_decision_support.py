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
        
        # Analyze vital signs and clinical biomarkers
        bp = float(patient_data.get('trestbps', patient_data.get('systolic_bp', 120)))
        if bp >= 140:
            risk_score += 0.25
            contributing_factors.append(f'Stage 2 Hypertension (Resting BP: {bp:.0f} mm Hg)')
        elif bp >= 130:
            risk_score += 0.15
            contributing_factors.append(f'Stage 1 Hypertension (Resting BP: {bp:.0f} mm Hg)')

        chol = float(patient_data.get('chol', 200))
        if chol >= 240:
            risk_score += 0.20
            contributing_factors.append(f'Hypercholesterolemia (Cholesterol: {chol:.0f} mg/dL)')
        elif chol >= 200:
            risk_score += 0.10
            contributing_factors.append(f'Borderline High Cholesterol ({chol:.0f} mg/dL)')

        oldpeak = float(patient_data.get('oldpeak', 0.0))
        if oldpeak >= 2.0:
            risk_score += 0.30
            contributing_factors.append(f'Significant ST Depression ({oldpeak:.1f} mm - severe myocardial ischemia)')
        elif oldpeak >= 1.0:
            risk_score += 0.15
            contributing_factors.append(f'Moderate ST Depression ({oldpeak:.1f} mm)')

        exang = int(patient_data.get('exang', 0))
        if exang == 1:
            risk_score += 0.20
            contributing_factors.append('Exercise-Induced Angina (exertional ischemia)')

        cp = int(patient_data.get('cp', 1))
        if cp == 4:
            risk_score += 0.25
            contributing_factors.append('Asymptomatic presentation (High-risk silent CAD pattern)')
        elif cp == 1:
            risk_score += 0.15
            contributing_factors.append('Typical Angina presentation (classic substernal chest pain)')

        fbs = int(patient_data.get('fbs', 0))
        if fbs == 1:
            risk_score += 0.15
            contributing_factors.append('Elevated Fasting Blood Sugar (>120 mg/dL - diabetic profile)')

        thalch = float(patient_data.get('thalch', patient_data.get('thalach', patient_data.get('heart_rate', 150))))
        if thalch < 120:
            risk_score += 0.15
            contributing_factors.append(f'Chronotropic Incompetence (Peak HR: {thalch:.0f} bpm)')

        restecg = int(patient_data.get('restecg', 0))
        if restecg == 2:
            risk_score += 0.15
            contributing_factors.append('Left Ventricular Hypertrophy (LVH) on resting ECG')
        elif restecg == 1:
            risk_score += 0.10
            contributing_factors.append('ST-T Wave Abnormality on resting ECG')

        slope = int(patient_data.get('slope', 1))
        if slope == 3:
            risk_score += 0.15
            contributing_factors.append('Downsloping ST segment at peak exertion (severe ischemia marker)')
        elif slope == 2:
            risk_score += 0.10
            contributing_factors.append('Flat ST segment response during exercise')

        age = float(patient_data.get('age', 50))
        if age >= 65:
            risk_score += 0.15
            contributing_factors.append(f'Advanced age ({age:.0f} years)')

        # Analyze model prediction
        if model_prediction:
            if 'very severe' in model_prediction.lower() or 'critical' in model_prediction.lower():
                risk_score += 0.30
                contributing_factors.append(f'ML Model Assessment: {model_prediction}')
            elif 'severe' in model_prediction.lower():
                risk_score += 0.25
                contributing_factors.append(f'ML Model Assessment: {model_prediction}')
            elif 'moderate' in model_prediction.lower():
                risk_score += 0.15
                contributing_factors.append(f'ML Model Assessment: {model_prediction}')
            elif 'mild' in model_prediction.lower():
                risk_score += 0.05
                contributing_factors.append(f'ML Model Assessment: {model_prediction}')

        # Determine risk level based on score
        if risk_score >= 0.70 or (model_prediction and ('very severe' in model_prediction.lower() or 'critical' in model_prediction.lower())):
            risk_level = RiskLevel.CRITICAL
        elif risk_score >= 0.45 or (model_prediction and 'severe' in model_prediction.lower()):
            risk_level = RiskLevel.HIGH
        elif risk_score >= 0.25 or (model_prediction and 'moderate' in model_prediction.lower()):
            risk_level = RiskLevel.MODERATE
        else:
            risk_level = RiskLevel.LOW

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
                description="Urgent cardiology consultation and comprehensive coronary evaluation recommended",
                priority=5,
                evidence_sources=['Risk assessment', 'ACC/AHA Guidelines']
            ))
            recommendations.append(ClinicalRecommendation(
                type=RecommendationType.IMAGING,
                description="Coronary angiography or stress echocardiography indicated for suspected obstructive CAD",
                priority=5,
                evidence_sources=['Ischemia Workup Guidelines']
            ))
        
        # Add monitoring recommendations
        recommendations.append(ClinicalRecommendation(
            type=RecommendationType.MONITORING,
            description="Continuous vital signs and 12-lead ECG telemetry monitoring recommended",
            priority=4 if risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL] else 3,
            evidence_sources=['Vital signs analysis']
        ))
        
        # Lipid management
        if any('Cholesterol' in f or 'Hypercholesterolemia' in f for f in factors):
            recommendations.append(ClinicalRecommendation(
                type=RecommendationType.MEDICATION,
                description="Initiate or intensify high-intensity Statin therapy (e.g. Atorvastatin 40-80mg) and dietary lipid restrictions",
                priority=4,
                evidence_sources=['AHA/ACC Cholesterol Guidelines']
            ))

        # Antihypertensive therapy
        if any('Hypertension' in f for f in factors):
            recommendations.append(ClinicalRecommendation(
                type=RecommendationType.MEDICATION,
                description="Initiate or adjust antihypertensive regimen (e.g. ACE inhibitor / ARB or beta-blocker) to target BP < 130/80 mmHg",
                priority=4,
                evidence_sources=['JNC 8 Hypertension Guidelines']
            ))

        # Anti-ischemic / Anti-platelet therapy
        if any('ST Depression' in f or 'Angina' in f or 'Silent CAD' in f for f in factors):
            recommendations.append(ClinicalRecommendation(
                type=RecommendationType.MEDICATION,
                description="Prescribe Aspirin 81-100mg daily and sublingual Nitroglycerin PRN for acute chest discomfort; consider Beta-blockers for ischemic rate control",
                priority=5 if risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL] else 4,
                evidence_sources=['Chronic Coronary Disease Guidelines']
            ))

        # Glycemic control
        if any('Blood Sugar' in f for f in factors):
            recommendations.append(ClinicalRecommendation(
                type=RecommendationType.LABORATORY,
                description="Check HbA1c, fasting lipid panel, and initiate endocrinology glycemic optimization",
                priority=3,
                evidence_sources=['ADA Diabetes Guidelines']
            ))

        # Add laboratory recommendations
        recommendations.append(ClinicalRecommendation(
            type=RecommendationType.LABORATORY,
            description="Comprehensive metabolic panel (CMP), serum electrolytes, and cardiac troponin baseline",
            priority=4,
            evidence_sources=['Patient assessment', 'Clinical guidelines']
        ))

        # Lifestyle modifications
        recommendations.append(ClinicalRecommendation(
            type=RecommendationType.LIFESTYLE,
            description="Prescribe Mediterranean cardioprotective diet, structured cardiac rehabilitation exercise, and smoking cessation counseling",
            priority=2,
            evidence_sources=['Cardiovascular Prevention Guidelines']
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
