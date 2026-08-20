"""
Medical RAG Evaluator Adapter
=============================
Unified RAG evaluation interface delegating directly to the genuine
RagasEvaluationEngine (RAGAS 0.4.3).

All heuristic word-overlap and hardcoded fallback scoring formulas have been
removed in favor of proper semantic RAG evaluation:
- Context Precision: Signal-to-noise ratio and ranking quality of retrieved contexts.
- Context Recall: Completeness of retrieved context against ground-truth clinical facts.
- Faithfulness: Verification that generated clinical claims are grounded in evidence.
- Answer Relevancy: Pertinence and alignment of answers to the clinical query.
"""

from typing import List, Dict, Any, Optional
from app.rag.ragas_evaluator import RagasEvaluationEngine


class RAGEvaluator:
    """
    Adapter providing the standard RAGEvaluator interface while delegating
    all metric computations to the genuine RagasEvaluationEngine.
    """

    def __init__(self, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.engine = RagasEvaluationEngine(embedding_model_name=embedding_model_name)

    def evaluate_sample(
        self,
        question: str,
        retrieved_contexts: List[str],
        generated_answer: str,
        reference_answer: str,
        reference_contexts: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Evaluate a single query sample using genuine RAGAS metrics."""
        return self.engine.evaluate_sample(
            question=question,
            retrieved_contexts=retrieved_contexts,
            generated_answer=generated_answer,
            reference_answer=reference_answer,
            reference_contexts=reference_contexts
        )

    def evaluate_dataset(
        self,
        dataset: List[Dict[str, Any]],
        rag_service,
        top_k: int = 5,
        progress_callback = None
    ) -> Dict[str, Any]:
        """
        Execute full RAG evaluation across a benchmark dataset using genuine RAGAS 0.4.3.
        Returns the structured summary_metrics and sample_results dictionaries.
        """
        return self.engine.evaluate_dataset(
            dataset=dataset,
            rag_service=rag_service,
            top_k=top_k,
            progress_callback=progress_callback
        )
