"""
Medical RAG Evaluation & Comparison Script
==========================================
Runs the grounded medical evaluation benchmark across:
1. New Medical Embedding Model: pritamdeka/S-PubMedBert-MS-MARCO (768-dim)
2. Baseline General Model: sentence-transformers/all-MiniLM-L6-v2 (384-dim)

Evaluates:
- Context Precision: Signal-to-noise ratio and ranking quality of retrieved evidence.
- Context Recall: Extent to which ground truth clinical facts are captured.
- Faithfulness: Groundedness of generated answers (anti-hallucination rate).
- Answer Relevancy: Pertinence and alignment of answers to the clinical question.

Saves full reports to CSV and JSON in data/.
"""

import os
import sys
import json
import time
import argparse
import pandas as pd
from pathlib import Path

# Add backend directory to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = BASE_DIR / "backend"
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(BASE_DIR))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from app.rag.enhanced_rag_pipeline import MedicalRAGService
from app.rag.rag_evaluator import RAGEvaluator

DATA_DIR = BASE_DIR / "data"
DATASET_PATH = DATA_DIR / "rag_evaluation_dataset.json"
PRIMARY_VECTORDB_DIR = BASE_DIR / "vectordb"
BASELINE_VECTORDB_DIR = BASE_DIR / "vectordb_minilm"


def print_banner(title: str):
    print("\n" + "=" * 78)
    print(f" {title} ".center(78, "="))
    print("=" * 78)


def format_table(rows: list, headers: list) -> str:
    """Format tabular data into markdown-style aligned string"""
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(val)))

    header_line = " | ".join(f"{h:<{col_widths[i]}}" for i, h in enumerate(headers))
    sep_line = "-+-".join("-" * col_widths[i] for i in range(len(headers)))
    row_lines = [
        " | ".join(f"{str(val):<{col_widths[i]}}" for i, val in enumerate(row))
        for row in rows
    ]
    return f"{header_line}\n{sep_line}\n" + "\n".join(row_lines)


def run_evaluation(compare: bool = True, top_k: int = 5):
    print_banner("RAGCHAINMED CLINICAL RAG EVALUATION BENCHMARK")

    # 1. Load Evaluation Dataset
    if not DATASET_PATH.exists():
        print(f"Error: Evaluation dataset not found at {DATASET_PATH}")
        sys.exit(1)

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"Loaded {len(dataset)} curated clinical evaluation questions from {DATASET_PATH.name}")
    evaluator = RAGEvaluator()

    # 2. Evaluate Primary Medical Embedding Model (S-PubMedBert-MS-MARCO)
    print_banner("Evaluating Medical Model: pritamdeka/S-PubMedBert-MS-MARCO (768-dim)")
    medical_rag = MedicalRAGService(
        vectordb_path=str(PRIMARY_VECTORDB_DIR),
        embedding_model_name="pritamdeka/S-PubMedBert-MS-MARCO"
    )

    medical_eval = evaluator.evaluate_dataset(
        dataset=dataset,
        rag_service=medical_rag,
        top_k=top_k
    )

    # 3. Evaluate Baseline Model (all-MiniLM-L6-v2) if requested and index exists
    baseline_eval = None
    if compare and BASELINE_VECTORDB_DIR.exists():
        print_banner("Evaluating Baseline Model: sentence-transformers/all-MiniLM-L6-v2 (384-dim)")
        minilm_rag = MedicalRAGService(
            vectordb_path=str(BASELINE_VECTORDB_DIR),
            embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        baseline_eval = evaluator.evaluate_dataset(
            dataset=dataset,
            rag_service=minilm_rag,
            top_k=top_k
        )

    # 4. Display Results
    print_banner("RAG EVALUATION METRIC SUMMARY")

    headers = ["Metric", "Medical Model (PubMedBERT)", "Baseline Model (MiniLM)", "Difference"]
    m_summary = medical_eval["summary_metrics"]

    if baseline_eval:
        b_summary = baseline_eval["summary_metrics"]
        diff_cp = m_summary['context_precision'] - b_summary['context_precision']
        diff_cr = m_summary['context_recall'] - b_summary['context_recall']
        diff_faith = m_summary['faithfulness'] - b_summary['faithfulness']
        diff_ar = m_summary['answer_relevancy'] - b_summary['answer_relevancy']
        diff_overall = m_summary['overall_rag_score'] - b_summary['overall_rag_score']

        rows = [
            ["Context Precision", f"{m_summary['context_precision']:.4f}", f"{b_summary['context_precision']:.4f}", f"{diff_cp:+.4f}"],
            ["Context Recall", f"{m_summary['context_recall']:.4f}", f"{b_summary['context_recall']:.4f}", f"{diff_cr:+.4f}"],
            ["Faithfulness", f"{m_summary['faithfulness']:.4f}", f"{b_summary['faithfulness']:.4f}", f"{diff_faith:+.4f}"],
            ["Answer Relevancy", f"{m_summary['answer_relevancy']:.4f}", f"{b_summary['answer_relevancy']:.4f}", f"{diff_ar:+.4f}"],
            ["Overall RAG Score", f"{m_summary['overall_rag_score']:.4f}", f"{b_summary['overall_rag_score']:.4f}", f"{diff_overall:+.4f}"],
            ["Avg Query Latency", f"{m_summary['average_latency_sec']:.3f}s", f"{b_summary['average_latency_sec']:.3f}s", f"{m_summary['average_latency_sec'] - b_summary['average_latency_sec']:+.3f}s"]
        ]
    else:
        rows = [
            ["Context Precision", f"{m_summary['context_precision']:.4f}", "N/A", "N/A"],
            ["Context Recall", f"{m_summary['context_recall']:.4f}", "N/A", "N/A"],
            ["Faithfulness", f"{m_summary['faithfulness']:.4f}", "N/A", "N/A"],
            ["Answer Relevancy", f"{m_summary['answer_relevancy']:.4f}", "N/A", "N/A"],
            ["Overall RAG Score", f"{m_summary['overall_rag_score']:.4f}", "N/A", "N/A"],
            ["Avg Query Latency", f"{m_summary['average_latency_sec']:.3f}s", "N/A", "N/A"]
        ]

    print(format_table(rows, headers))

    # 5. Display Per-Question Breakdown for Medical Model
    print_banner("PER-QUESTION BREAKDOWN (Medical PubMedBERT Model)")
    q_headers = ["ID", "Category", "Precision", "Recall", "Faithful", "Relevancy", "Overall"]
    q_rows = [
        [
            s["id"],
            s["category"][:15],
            f"{s['context_precision']:.2f}",
            f"{s['context_recall']:.2f}",
            f"{s['faithfulness']:.2f}",
            f"{s['answer_relevancy']:.2f}",
            f"{s['overall_score']:.2f}"
        ]
        for s in medical_eval["sample_results"]
    ]
    print(format_table(q_rows, q_headers))

    # 6. Save JSON & CSV Outputs
    print_banner("SAVING EVALUATION ARTIFACTS")

    # Save primary results
    results_json_path = DATA_DIR / "evaluation_results.json"
    results_csv_path = DATA_DIR / "evaluation_results.csv"

    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(medical_eval, f, indent=2)
    print(f"[OK] Saved detailed results JSON: {results_json_path}")

    df_samples = pd.DataFrame(medical_eval["sample_results"])
    df_samples.to_csv(results_csv_path, index=False)
    print(f"[OK] Saved detailed results CSV: {results_csv_path}")

    # Save model comparison files if available
    if baseline_eval:
        comparison_payload = {
            "evaluation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_size": len(dataset),
            "retrieval_k": top_k,
            "models": {
                "medical_model": {
                    "name": "pritamdeka/S-PubMedBert-MS-MARCO",
                    "dimension": 768,
                    "domain": "Biomedical / PubMedBERT",
                    "summary_metrics": m_summary,
                    "sample_results": medical_eval["sample_results"]
                },
                "baseline_model": {
                    "name": "sentence-transformers/all-MiniLM-L6-v2",
                    "dimension": 384,
                    "domain": "General Domain",
                    "summary_metrics": b_summary,
                    "sample_results": baseline_eval["sample_results"]
                }
            },
            "metric_differences": {
                "context_precision": round(diff_cp, 4),
                "context_recall": round(diff_cr, 4),
                "faithfulness": round(diff_faith, 4),
                "answer_relevancy": round(diff_ar, 4),
                "overall_rag_score": round(diff_overall, 4)
            }
        }

        comparison_json_path = DATA_DIR / "model_comparison.json"
        comparison_csv_path = DATA_DIR / "model_comparison.csv"

        with open(comparison_json_path, "w", encoding="utf-8") as f:
            json.dump(comparison_payload, f, indent=2)
        print(f"[OK] Saved model comparison JSON: {comparison_json_path}")

        # Comparison DataFrame
        comp_rows = []
        for s_m, s_b in zip(medical_eval["sample_results"], baseline_eval["sample_results"]):
            comp_rows.append({
                "id": s_m["id"],
                "category": s_m["category"],
                "question": s_m["question"],
                "pubmedbert_precision": s_m["context_precision"],
                "minilm_precision": s_b["context_precision"],
                "pubmedbert_recall": s_m["context_recall"],
                "minilm_recall": s_b["context_recall"],
                "pubmedbert_faithfulness": s_m["faithfulness"],
                "minilm_faithfulness": s_b["faithfulness"],
                "pubmedbert_relevancy": s_m["answer_relevancy"],
                "minilm_relevancy": s_b["answer_relevancy"],
                "pubmedbert_overall": s_m["overall_score"],
                "minilm_overall": s_b["overall_score"]
            })
        df_comp = pd.DataFrame(comp_rows)
        df_comp.to_csv(comparison_csv_path, index=False)
        print(f"[OK] Saved model comparison CSV: {comparison_csv_path}")

    print_banner("EVALUATION COMPLETED SUCCESSFULLY")
    return medical_eval, baseline_eval


def main():
    parser = argparse.ArgumentParser(description="Run RAG Evaluation Benchmark")
    parser.add_argument("--no-compare", action="store_true", help="Skip baseline model comparison")
    parser.add_argument("--top-k", type=int, default=5, help="Number of retrieved documents per query")
    args = parser.parse_args()

    run_evaluation(compare=not args.no_compare, top_k=args.top_k)


if __name__ == "__main__":
    main()
