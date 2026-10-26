"""Fast RAGAS evaluation using cached RAG outputs when available."""

import argparse
import asyncio
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = BASE_DIR / "backend"
DATA_DIR = BASE_DIR / "data"
DATASET_PATH = DATA_DIR / "ragas_evaluation_dataset.json"
CACHE_PATH = DATA_DIR / "evaluation_results.json"
JSON_OUTPUT_PATH = DATA_DIR / "ragas_evaluation_results.json"
CSV_OUTPUT_PATH = DATA_DIR / "ragas_evaluation_results.csv"
EMBEDDING_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"
METRIC_NAMES = (
    "context_precision",
    "context_recall",
    "faithfulness",
    "answer_relevancy",
)


def create_ragas_embedding_adapter(embeddings: Any) -> Any:
    from ragas.embeddings.base import BaseRagasEmbedding

    class ExistingEmbeddingsAdapter(BaseRagasEmbedding):
        def __init__(self, langchain_embeddings: Any) -> None:
            super().__init__()
            self.langchain_embeddings = langchain_embeddings

        def embed_text(self, text: str, **kwargs: Any) -> List[float]:
            return self.langchain_embeddings.embed_query(text)

        async def aembed_text(self, text: str, **kwargs: Any) -> List[float]:
            return await asyncio.to_thread(self.embed_text, text)

        def embed_texts(
            self, texts: List[str], **kwargs: Any
        ) -> List[List[float]]:
            return self.langchain_embeddings.embed_documents(texts)

        async def aembed_texts(
            self, texts: List[str], **kwargs: Any
        ) -> List[List[float]]:
            return await asyncio.to_thread(self.embed_texts, texts)

    return ExistingEmbeddingsAdapter(embeddings)


def save_results(payload: Dict[str, Any]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    JSON_OUTPUT_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    with CSV_OUTPUT_PATH.open("w", newline="", encoding="utf-8-sig") as csv_file:
        fieldnames = [
            "id",
            "category",
            "question",
            *METRIC_NAMES,
            "source",
            "retrieval_generation_time_sec",
            "evaluation_time_sec",
            "errors",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for sample in payload.get("samples", []):
            writer.writerow(
                {
                    **{key: sample.get(key) for key in fieldnames if key != "errors"},
                    "errors": json.dumps(sample.get("errors", []), ensure_ascii=False),
                }
            )


def get_cached_outputs() -> Dict[str, Dict[str, Any]]:
    if not CACHE_PATH.exists():
        return {}
    with CACHE_PATH.open(encoding="utf-8") as cache_file:
        cache_data = json.load(cache_file)
    return {
        item["question"]: item
        for item in cache_data.get("sample_results", [])
        if isinstance(item.get("question"), str)
    }


def load_retrieval_outputs(
    dataset: List[Dict[str, Any]], refresh: bool, top_k: int
) -> tuple[List[Dict[str, Any]], Optional[Any]]:
    cache = {} if refresh else get_cached_outputs()
    samples: List[Dict[str, Any]] = []
    missing = [item for item in dataset if item["question"] not in cache]
    rag_service = None

    if missing:
        sys.path.insert(0, str(BACKEND_DIR))
        from app.rag.enhanced_rag_pipeline import MedicalRAGService

        rag_service = MedicalRAGService(
            vectordb_path=str(BASE_DIR / "vectordb"),
            embedding_model_name=EMBEDDING_MODEL,
        )

    for item in dataset:
        cached = cache.get(item["question"])
        started = time.perf_counter()
        sample_errors: List[str] = []
        if cached:
            contexts = cached.get("retrieved_contexts", [])
            answer = cached.get("generated_answer", "")
            source = "cached_evaluation_results"
        else:
            try:
                output = rag_service.answer_query(
                    item["question"], top_k=top_k, bypass_structured=True
                )
                contexts = [
                    evidence["text"]
                    for evidence in output.get("retrieved_evidence", [])
                    if evidence.get("text")
                ]
                answer = output.get("answer", "")
                source = "live_rag_pipeline"
            except Exception as exc:
                contexts = []
                answer = ""
                source = "live_rag_pipeline"
                sample_errors.append(
                    f"RAG pipeline failed: {type(exc).__name__}: {exc}"
                )

        if not isinstance(contexts, list) or not all(
            isinstance(context, str) for context in contexts
        ):
            contexts = []
            sample_errors.append("Retrieved contexts were missing or invalid.")
        if not isinstance(answer, str) or not answer.strip():
            answer = ""
            sample_errors.append("Generated answer was missing or empty.")

        samples.append(
            {
                "id": item["id"],
                "category": item["category"],
                "question": item["question"],
                "reference_answer": item["reference_answer"],
                "retrieved_contexts": contexts,
                "generated_answer": answer,
                "source": source,
                "retrieval_generation_time_sec": round(
                    time.perf_counter() - started, 3
                ),
                "evaluation_time_sec": None,
                "scores": {name: None for name in METRIC_NAMES},
                "errors": sample_errors,
            }
        )
    return samples, rag_service


def valid_sample(sample: Dict[str, Any]) -> bool:
    return bool(
        sample["retrieved_contexts"]
        and sample["generated_answer"]
        and sample["reference_answer"]
        and not sample["errors"]
    )


def run_evaluation(refresh: bool = False, top_k: int = 5) -> Dict[str, Any]:
    started = time.perf_counter()
    payload: Dict[str, Any] = {
        "dataset": str(DATASET_PATH.relative_to(BASE_DIR)),
        "cache": str(CACHE_PATH.relative_to(BASE_DIR)),
        "embedding_model": EMBEDDING_MODEL,
        "llm_provider": "Groq (configured RAGAS adapter)",
        "refresh": refresh,
        "top_k": top_k,
        "samples": [],
        "summary_metrics": {name: None for name in METRIC_NAMES},
        "errors": [],
        "evaluation_time_sec": None,
        "total_time_sec": None,
    }

    try:
        from dotenv import load_dotenv

        for env_path in (BACKEND_DIR / ".env", BASE_DIR / ".env"):
            if env_path.exists():
                load_dotenv(env_path)

        if not os.getenv("GROQ_API_KEY"):
            raise RuntimeError(
                "GROQ_API_KEY is not set in the environment, backend/.env, or .env."
            )

        with DATASET_PATH.open(encoding="utf-8") as dataset_file:
            dataset = json.load(dataset_file)
        if len(dataset) != 5:
            raise ValueError(
                f"Expected exactly 5 evaluation questions; found {len(dataset)}."
            )

        samples, rag_service = load_retrieval_outputs(dataset, refresh, top_k)
        payload["samples"] = samples
        usable_samples = [sample for sample in samples if valid_sample(sample)]

        if usable_samples:
            from ragas.llms import llm_factory
            from ragas.metrics.collections import (
                AnswerRelevancy,
                ContextPrecision,
                ContextRecall,
                Faithfulness,
            )
            from openai import AsyncOpenAI
            try:
                from langchain_huggingface import HuggingFaceEmbeddings
            except ImportError:
                from langchain_community.embeddings import HuggingFaceEmbeddings

            evaluation_started = time.perf_counter()
            embeddings = (
                rag_service.embedding_model
                if rag_service is not None
                else HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
            )
            groq_openai_client = AsyncOpenAI(
                api_key=os.getenv("GROQ_API_KEY"),
                base_url="https://api.groq.com/openai/v1",
            )
            llm = llm_factory(
                "openai/gpt-oss-120b",
                provider="openai",
                client=groq_openai_client,
                max_tokens=1024,
                temperature=0.0,
            )
            ragas_embeddings = create_ragas_embedding_adapter(embeddings)
            metrics = [
                ContextPrecision(llm=llm),
                ContextRecall(llm=llm),
                Faithfulness(llm=llm),
                AnswerRelevancy(
                    llm=llm, embeddings=ragas_embeddings, strictness=1
                ),
            ]
            metric_fields = {
                "context_precision": (
                    "user_input",
                    "retrieved_contexts",
                    "reference",
                ),
                "context_recall": (
                    "user_input",
                    "retrieved_contexts",
                    "reference",
                ),
                "faithfulness": (
                    "user_input",
                    "retrieved_contexts",
                    "response",
                ),
                "answer_relevancy": ("user_input", "response"),
            }
            metric_inputs = {
                name: [
                    {
                        "user_input": sample["question"],
                        "retrieved_contexts": sample["retrieved_contexts"],
                        "response": sample["generated_answer"],
                        "reference": sample["reference_answer"],
                    }
                    for sample in usable_samples
                ]
                for name in METRIC_NAMES
            }

            async def score_metrics() -> Dict[str, List[Any]]:
                results: Dict[str, List[Any]] = {}
                for name, metric in zip(METRIC_NAMES, metrics):
                    results[name] = []
                    for row in metric_inputs[name]:
                        metric_row = {
                            field: row[field] for field in metric_fields[name]
                        }
                        try:
                            score = await metric.ascore(**metric_row)
                        except Exception as exc:
                            score = exc
                        results[name].append(score)
                        await asyncio.sleep(1)
                return results

            metric_results = asyncio.run(score_metrics())
            evaluation_duration = round(time.perf_counter() - evaluation_started, 3)
            valid_scores: Dict[str, List[float]] = {
                name: [] for name in METRIC_NAMES
            }
            for name in METRIC_NAMES:
                for row_index, score_result in enumerate(metric_results[name]):
                    if isinstance(score_result, BaseException):
                        usable_samples[row_index]["errors"].append(
                            f"{name} evaluation failed: "
                            f"{type(score_result).__name__}: {score_result}"
                        )
                        continue
                    value = getattr(score_result, "value", score_result)
                    if value is not None and not (
                        isinstance(value, float) and math.isnan(value)
                    ):
                        numeric_value = float(value)
                        usable_samples[row_index]["scores"][name] = numeric_value
                        valid_scores[name].append(numeric_value)
                    else:
                        reason = getattr(score_result, "reason", None)
                        message = f"RAGAS returned no score for {name}."
                        if reason:
                            message += f" Reason: {reason}"
                        usable_samples[row_index]["errors"].append(message)

            for sample in usable_samples:
                sample["evaluation_time_sec"] = evaluation_duration

            for name, values in valid_scores.items():
                if values:
                    payload["summary_metrics"][name] = sum(values) / len(values)

            payload["evaluation_time_sec"] = evaluation_duration

    except Exception as exc:
        payload["errors"].append(f"{type(exc).__name__}: {exc}")
    finally:
        payload["total_time_sec"] = round(time.perf_counter() - started, 3)
        save_results(payload)

    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Quickly evaluate five existing RAG outputs with RAGAS."
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Regenerate retrieval contexts and answers instead of using cached outputs.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of retrieved evidence records for uncached questions (default: 5).",
    )
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")

    payload = run_evaluation(refresh=args.refresh, top_k=args.top_k)
    print("\nRAGAS evaluation")
    for name in METRIC_NAMES:
        score = payload["summary_metrics"][name]
        print(f"  {name}: {score if score is not None else 'ERROR / unavailable'}")
    print(f"  evaluation time: {payload['evaluation_time_sec']} seconds")
    print(f"  total time: {payload['total_time_sec']} seconds")
    if payload["errors"]:
        print("Errors:")
        for error in payload["errors"]:
            print(f"  - {error}")
    for sample in payload["samples"]:
        for error in sample["errors"]:
            print(f"  - {sample['id']}: {error}")
    print(f"JSON results: {JSON_OUTPUT_PATH}")
    print(f"CSV results: {CSV_OUTPUT_PATH}")

    has_errors = bool(payload["errors"]) or any(
        sample["errors"] for sample in payload["samples"]
    )
    has_scores = any(
        score is not None for score in payload["summary_metrics"].values()
    )
    return 0 if has_scores and not has_errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
