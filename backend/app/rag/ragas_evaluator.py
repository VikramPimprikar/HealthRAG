"""
RAGAS Evaluation Module for HealthRAG / RAGChainMed
===================================================
Production RAG evaluation implementation using genuine RAGAS 0.4.3 metrics:
1. Context Precision: Signal-to-noise ratio and ranking quality of retrieved contexts.
2. Context Recall: Completeness of retrieved context with respect to ground-truth clinical facts.
3. Faithfulness: Extent to which the generated answer is grounded in retrieved evidence (anti-hallucination).
4. Answer Relevancy: How directly and pertinently the generated answer addresses the clinical query.

Uses genuine RAGAS semantic evaluation with custom GroqChatModel (resolving Groq's n <= 1 constraint)
and HuggingFaceEmbeddings. Zero heuristic word-overlap scoring, zero arbitrary offsets.
"""

import os
import re
import json
import time
import math
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
import pandas as pd
import numpy as np

# Load environment
base_dir = Path(__file__).resolve().parent.parent.parent.parent
backend_dir = Path(__file__).resolve().parent.parent.parent

if (backend_dir / ".env").exists():
    load_dotenv(backend_dir / ".env")
elif (base_dir / ".env").exists():
    load_dotenv(base_dir / ".env")
else:
    load_dotenv()

from groq import Groq
from datasets import Dataset

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_community.embeddings import HuggingFaceEmbeddings

import ragas
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper


class GroqChatModel(BaseChatModel):
    """
    Custom LangChain BaseChatModel adapter for Groq API.
    Handles multi-candidate generation (n > 1) via sequential requests to satisfy
    Groq's constraint ('n' must be at most 1), manages token limits, and provides
    automatic model fallback.
    """
    client: Any = None
    model_name: str = "openai/gpt-oss-120b"
    fallback_models: List[str] = [
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "qwen/qwen3.6-27b",
        "allam-2-7b"
    ]
    temperature: float = 0.0
    max_tokens: int = 2048

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        api_key = kwargs.get("api_key") or os.getenv("GROQ_API_KEY")
        self.client = Groq(api_key=api_key)

    @property
    def _llm_type(self) -> str:
        return "groq-chat-model"

    def _format_messages(self, messages: List[BaseMessage]) -> List[Dict[str, str]]:
        formatted = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                formatted.append({"role": "system", "content": str(msg.content)})
            elif isinstance(msg, AIMessage):
                formatted.append({"role": "assistant", "content": str(msg.content)})
            else:
                formatted.append({"role": "user", "content": str(msg.content)})
        return formatted

    def _call_groq_single(self, groq_messages: List[Dict[str, str]], stop: Optional[List[str]] = None) -> str:
        models_to_try = [self.model_name] + [m for m in self.fallback_models if m != self.model_name]
        for model in models_to_try:
            try:
                resp = self.client.chat.completions.create(
                    model=model,
                    messages=groq_messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stop=stop,
                    timeout=35
                )
                content = resp.choices[0].message.content or ""
                if "</think>" in content:
                    content = content.split("</think>")[-1].strip()
                content = content.strip()
                if content:
                    return content
            except Exception:
                continue
        return ""

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        groq_messages = self._format_messages(messages)
        n = kwargs.get("n", 1)
        generations = []

        for _ in range(n):
            content_generated = self._call_groq_single(groq_messages, stop=stop)
            generations.append(ChatGeneration(message=AIMessage(content=content_generated)))

        return ChatResult(generations=generations)

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._generate, messages, stop, run_manager, **kwargs)


class RagasEvaluationEngine:
    """
    Evaluation Engine executing genuine RAGAS 0.4.3 metrics on the Medical RAG pipeline.
    """

    def __init__(self, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.groq_chat = GroqChatModel(api_key=self.api_key)
        self.ragas_llm = LangchainLLMWrapper(self.groq_chat)

        self.hf_embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        self.ragas_embeddings = LangchainEmbeddingsWrapper(self.hf_embeddings)

        # Configure metrics with custom LLM and Embeddings
        self.metrics = [context_precision, context_recall, faithfulness, answer_relevancy]
        for m in self.metrics:
            m.llm = self.ragas_llm
            if hasattr(m, "embeddings"):
                m.embeddings = self.ragas_embeddings

    def evaluate_sample(
        self,
        question: str,
        retrieved_contexts: List[str],
        generated_answer: str,
        reference_answer: str,
        reference_contexts: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single query sample using genuine RAGAS metrics.
        """
        sample_dict = {
            "question": [question],
            "contexts": [retrieved_contexts if retrieved_contexts else ["No context retrieved."]],
            "answer": [generated_answer if generated_answer else "No answer generated."],
            "ground_truth": [reference_answer if reference_answer else "No reference answer provided."]
        }

        dataset = Dataset.from_dict(sample_dict)
        try:
            ragas_res = evaluate(
                dataset,
                metrics=self.metrics,
                llm=self.ragas_llm,
                embeddings=self.ragas_embeddings,
                show_progress=False
            )
            df = ragas_res.to_pandas()

            def clean_val(col: str) -> float:
                if col in df.columns and len(df[col]) > 0:
                    val = df[col].iloc[0]
                    if pd.notna(val) and not math.isnan(float(val)):
                        return round(max(0.0, min(1.0, float(val))), 4)
                return float("nan")

            cp = clean_val("context_precision")
            cr = clean_val("context_recall")
            faith = clean_val("faithfulness")
            ar = clean_val("answer_relevancy")
        except Exception as e:
            print(f"[RAGAS Evaluator] Single sample evaluation failed: {e}")
            cp = float("nan")
            cr = float("nan")
            faith = float("nan")
            ar = float("nan")

        valid_scores = [s for s in [cp, cr, faith, ar] if not math.isnan(s)]
        overall = round(sum(valid_scores) / len(valid_scores), 4) if valid_scores else 0.0

        return {
            "context_precision": cp,
            "context_recall": cr,
            "faithfulness": faith,
            "answer_relevancy": ar,
            "overall_score": overall
        }

    def evaluate_dataset(
        self,
        dataset: List[Dict[str, Any]],
        rag_service,
        top_k: int = 5,
        progress_callback = None
    ) -> Dict[str, Any]:
        """
        Execute end-to-end RAG evaluation across a benchmark dataset using genuine RAGAS 0.4.3.
        1. Runs each question through the existing MedicalRAGService with pure vector retrieval.
        2. Captures retrieved context chunks, generated answers, and ground truth references.
        3. Runs RAGAS evaluation across all 4 metrics.
        4. Compiles individual sample results and summary metrics adhering to the expected interface.
        """
        questions = []
        retrieved_contexts_list = []
        generated_answers_list = []
        ground_truths_list = []
        sample_metas = []
        latencies = []

        print(f"\n[RAGAS Evaluator] Running RAG retrieval & answer generation for {len(dataset)} samples...")

        # 1. Run through existing RAG pipeline
        for idx, item in enumerate(dataset, start=1):
            q = item["question"]
            t0 = time.time()

            # Execute pure vector retrieval and grounded LLM generation
            rag_output = rag_service.answer_query(q, top_k=top_k, bypass_structured=True)
            latency = round(time.time() - t0, 3)
            latencies.append(latency)

            retrieved_chunks = [e["text"] for e in rag_output.get("retrieved_evidence", [])]
            ans = rag_output.get("answer", "")
            gt_ans = item.get("ground_truth_answer") or item.get("reference_answer", "")
            gt_contexts = item.get("ground_truth_contexts") or item.get("reference_contexts", [gt_ans])

            questions.append(q)
            retrieved_contexts_list.append(retrieved_chunks if retrieved_chunks else ["No retrieved context."])
            generated_answers_list.append(ans if ans else "No generated answer.")
            ground_truths_list.append(gt_ans if gt_ans else "No reference answer provided.")

            sample_metas.append({
                "id": item.get("id", f"EVAL_{idx:03d}"),
                "category": item.get("category", "General Clinical"),
                "question": q,
                "generated_answer": ans,
                "retrieved_contexts": retrieved_chunks,
                "reference_answer": gt_ans,
                "reference_contexts": gt_contexts,
                "latency_sec": latency
            })

            if progress_callback:
                progress_callback(idx, len(dataset), q)
            else:
                print(f"  [{idx}/{len(dataset)}] Retrieved {len(retrieved_chunks)} chunks in {latency}s: '{q[:50]}...'")

        # 2. Run RAGAS evaluate on the batch
        print(f"\n[RAGAS Evaluator] Computing RAGAS metrics across {len(dataset)} samples...")
        ragas_data = {
            "question": questions,
            "contexts": retrieved_contexts_list,
            "answer": generated_answers_list,
            "ground_truth": ground_truths_list
        }
        ragas_ds = Dataset.from_dict(ragas_data)

        per_question_results = []
        try:
            ragas_result = evaluate(
                ragas_ds,
                metrics=self.metrics,
                llm=self.ragas_llm,
                embeddings=self.ragas_embeddings,
                show_progress=True
            )
            df = ragas_result.to_pandas()
            print(f"[RAGAS Evaluator] Batch evaluation completed successfully.")

            for i, meta in enumerate(sample_metas):
                row = df.iloc[i] if i < len(df) else {}

                def parse_metric(val: Any) -> float:
                    if pd.notna(val) and not math.isnan(float(val)):
                        return round(max(0.0, min(1.0, float(val))), 4)
                    return float("nan")

                cp_val = parse_metric(row.get("context_precision"))
                cr_val = parse_metric(row.get("context_recall"))
                faith_val = parse_metric(row.get("faithfulness"))
                ar_val = parse_metric(row.get("answer_relevancy"))

                valid_scores = [s for s in [cp_val, cr_val, faith_val, ar_val] if not math.isnan(s)]
                overall = round(sum(valid_scores) / len(valid_scores), 4) if valid_scores else 0.0

                per_question_results.append({
                    "id": meta["id"],
                    "category": meta["category"],
                    "question": meta["question"],
                    "generated_answer": meta["generated_answer"],
                    "retrieved_contexts": meta["retrieved_contexts"],
                    "reference_answer": meta["reference_answer"],
                    "reference_contexts": meta["reference_contexts"],
                    "context_precision": cp_val if not math.isnan(cp_val) else 0.0,
                    "context_recall": cr_val if not math.isnan(cr_val) else 0.0,
                    "faithfulness": faith_val if not math.isnan(faith_val) else 0.0,
                    "answer_relevancy": ar_val if not math.isnan(ar_val) else 0.0,
                    "overall_score": overall,
                    "latency_sec": meta["latency_sec"]
                })
        except Exception as eval_err:
            print(f"[RAGAS Evaluator] Batch evaluate failed: {eval_err}. Evaluating per-sample...")
            for meta in sample_metas:
                sample_score = self.evaluate_sample(
                    question=meta["question"],
                    retrieved_contexts=meta["retrieved_contexts"],
                    generated_answer=meta["generated_answer"],
                    reference_answer=meta["reference_answer"],
                    reference_contexts=meta["reference_contexts"]
                )
                per_question_results.append({
                    "id": meta["id"],
                    "category": meta["category"],
                    "question": meta["question"],
                    "generated_answer": meta["generated_answer"],
                    "retrieved_contexts": meta["retrieved_contexts"],
                    "reference_answer": meta["reference_answer"],
                    "reference_contexts": meta["reference_contexts"],
                    "context_precision": sample_score["context_precision"] if not math.isnan(sample_score["context_precision"]) else 0.0,
                    "context_recall": sample_score["context_recall"] if not math.isnan(sample_score["context_recall"]) else 0.0,
                    "faithfulness": sample_score["faithfulness"] if not math.isnan(sample_score["faithfulness"]) else 0.0,
                    "answer_relevancy": sample_score["answer_relevancy"] if not math.isnan(sample_score["answer_relevancy"]) else 0.0,
                    "overall_score": sample_score["overall_score"],
                    "latency_sec": meta["latency_sec"]
                })

        # Calculate summary averages
        n = max(len(per_question_results), 1)
        mean_cp = round(sum(q["context_precision"] for q in per_question_results) / n, 4)
        mean_cr = round(sum(q["context_recall"] for q in per_question_results) / n, 4)
        mean_faith = round(sum(q["faithfulness"] for q in per_question_results) / n, 4)
        mean_ar = round(sum(q["answer_relevancy"] for q in per_question_results) / n, 4)
        mean_overall = round(sum(q["overall_score"] for q in per_question_results) / n, 4)
        avg_latency = round(sum(latencies) / n, 3)

        return {
            "summary_metrics": {
                "context_precision": mean_cp,
                "context_recall": mean_cr,
                "faithfulness": mean_faith,
                "answer_relevancy": mean_ar,
                "overall_rag_score": mean_overall,
                "average_latency_sec": avg_latency,
                "total_samples": len(per_question_results)
            },
            "sample_results": per_question_results
        }
