"""Full evaluation pipeline comparing Step 1, Step 2, and Step 3 on ZentroMart benchmarks."""

import json
import logging
import sys
import io
from pathlib import Path

# Ensure UTF-8 output encoding across Windows PowerShell and Unix shells
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from src.data.loader import DataLoader
from src.agent.conversational_agent import ConversationalAgent
from src.eval.evaluator import BenchmarkEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_full_evaluation():
    loader = DataLoader()
    agent = ConversationalAgent(data_loader=loader)
    evaluator = BenchmarkEvaluator(agent=agent)

    questions = loader.load_practice_questions()
    conversations = loader.load_practice_conversations()

    print("\n=======================================================")
    print("VOICEOPS 2026: OFFICIAL CAPSTONE BENCHMARK EVALUATION")
    print("=======================================================\n")

    # Step 1: Prompting Baseline
    logger.info(f"Running Step 1 (Prompting Baseline) on {len(questions)} questions...")
    step1_res = evaluator.evaluate_step1_baseline(questions)

    # Step 2: Grounded RAG
    logger.info(f"Running Step 2 (Knowledge-Grounded RAG) on {len(questions)} questions...")
    step2_res = evaluator.evaluate_step2_rag(questions)

    # Step 3: Multi-turn Conversations
    logger.info(f"Running Step 3 (Conversations with Tools) on {len(conversations)} turns...")
    step3_res = evaluator.evaluate_step3_conversations(conversations)

    summary = {
        "step1_baseline": {
            "name": step1_res["name"],
            "questions_evaluated": step1_res["total_questions"],
            "accuracy": step1_res["accuracy"],
            "retrieval_recall": step1_res["retrieval_recall"],
            "retrieval_precision": step1_res["retrieval_precision"],
        },
        "step2_rag": {
            "name": step2_res["name"],
            "questions_evaluated": step2_res["total_questions"],
            "accuracy": step2_res["accuracy"],
            "retrieval_recall": step2_res["retrieval_recall"],
            "retrieval_precision": step2_res["retrieval_precision"],
            "retrieval_f1": step2_res["retrieval_f1"],
            "hit_rate": step2_res["hit_rate"],
        },
        "step3_conversations": {
            "name": step3_res["name"],
            "turns_evaluated": step3_res["total_turns"],
            "decision_accuracy": step3_res["decision_accuracy"],
            "tool_call_accuracy": step3_res["tool_call_accuracy"],
            "fact_groundedness": step3_res["fact_groundedness"],
            "average_evidence_recall": step3_res["average_evidence_recall"],
        }
    }

    out_dir = Path("sample_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "eval_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n-------------------------------------------------------")
    print("STEP 1 VS STEP 2: RAG GROUNDING COMPARISON (52 Questions)")
    print("-------------------------------------------------------")
    print(f"| Metric                    | Step 1 Baseline | Step 2 Grounded RAG | Delta (Gain) |")
    print(f"|---------------------------|-----------------|---------------------|--------------|")
    print(f"| Fact Groundedness Acc.    | {step1_res['accuracy']*100:6.1f}%          | {step2_res['accuracy']*100:6.1f}%             | +{(step2_res['accuracy'] - step1_res['accuracy'])*100:5.1f}%       |")
    print(f"| Evidence Retrieval Recall | {step1_res['retrieval_recall']*100:6.1f}%          | {step2_res['retrieval_recall']*100:6.1f}%             | +{(step2_res['retrieval_recall'])*100:5.1f}%       |")
    print(f"| Retrieval Precision       | {step1_res['retrieval_precision']*100:6.1f}%          | {step2_res['retrieval_precision']*100:6.1f}%             | +{(step2_res['retrieval_precision'])*100:5.1f}%       |")
    print(f"| Retrieval Hit Rate        |   0.0%          | {step2_res['hit_rate']*100:6.1f}%             | +{step2_res['hit_rate']*100:5.1f}%       |")

    print("\n-------------------------------------------------------")
    print("STEP 3: MULTI-TURN CONVERSATIONS & TOOLS (96 Turns / 23 Dialogues)")
    print("-------------------------------------------------------")
    print(f"- Decision Match Accuracy: {step3_res['decision_accuracy']*100:.1f}%")
    print(f"- Tool Invocation Accuracy: {step3_res['tool_call_accuracy']*100:.1f}%")
    print(f"- Fact Groundedness:        {step3_res['fact_groundedness']*100:.1f}%")
    print(f"- Evidence Recall:          {step3_res['average_evidence_recall']*100:.1f}%")
    print("\nEvaluation results saved to sample_outputs/eval_results.json\n")
    return summary


if __name__ == "__main__":
    run_full_evaluation()
