"""Comprehensive evaluation harness for Step 1, Step 2, and Step 3."""

import logging
from typing import Dict, List, Any
import pandas as pd

from src.agent.conversational_agent import ConversationalAgent
from src.eval.metrics import (
    calculate_retrieval_metrics,
    check_must_include,
    check_must_not_include,
)

logger = logging.getLogger(__name__)


class BenchmarkEvaluator:
    """Executes official capstone evaluation across practice questions and multi-turn conversations."""

    def __init__(self, agent: ConversationalAgent):
        self.agent = agent

    def evaluate_step1_baseline(self, questions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate Step 1 (Prompting-only baseline without RAG) on practice questions."""
        results = []
        correct_facts = 0
        total_questions = len(questions)

        for q in questions:
            qid = q["id"]
            question_text = q.get("asr_text") or q.get("question")
            gold_ev = q.get("gold_evidence")
            must_inc = q.get("must_include")
            must_not = q.get("must_not_include")

            # Call baseline
            resp = self.agent.respond_step1_baseline(question_text)
            reply = resp["reply"]

            inc_pass, missing = check_must_include(reply, must_inc)
            not_pass, violations = check_must_not_include(reply, must_not)
            is_accurate = inc_pass and not_pass
            if is_accurate:
                correct_facts += 1

            ret_metrics = calculate_retrieval_metrics(resp["evidence_ids"], gold_ev)

            results.append({
                "id": qid,
                "question": question_text,
                "reply": reply,
                "must_include_pass": inc_pass,
                "must_not_pass": not_pass,
                "is_accurate": is_accurate,
                "retrieval_recall": ret_metrics["recall"],
                "retrieval_precision": ret_metrics["precision"],
            })

        accuracy = correct_facts / total_questions if total_questions else 0.0
        return {
            "step": 1,
            "name": "Prompting Baseline (No RAG)",
            "total_questions": total_questions,
            "correct_facts_count": correct_facts,
            "accuracy": round(accuracy, 4),
            "retrieval_recall": 0.0,
            "retrieval_precision": 0.0,
            "details": results,
        }

    def evaluate_step2_rag(self, questions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate Step 2 (Knowledge Base Grounded RAG) on practice questions."""
        results = []
        correct_facts = 0
        total_questions = len(questions)
        total_recall = 0.0
        total_precision = 0.0
        total_f1 = 0.0
        hits = 0

        for q in questions:
            qid = q["id"]
            question_text = q.get("asr_text") or q.get("question")
            gold_ev = q.get("gold_evidence")
            must_inc = q.get("must_include")
            must_not = q.get("must_not_include")

            # Call grounded RAG
            resp = self.agent.respond_step2_rag(question_text)
            reply = resp["reply"]

            inc_pass, missing = check_must_include(reply, must_inc)
            not_pass, violations = check_must_not_include(reply, must_not)
            is_accurate = inc_pass and not_pass
            if is_accurate:
                correct_facts += 1

            ret_metrics = calculate_retrieval_metrics(resp["evidence_ids"], gold_ev)
            total_recall += ret_metrics["recall"]
            total_precision += ret_metrics["precision"]
            total_f1 += ret_metrics["f1"]
            hits += int(ret_metrics["hit"])

            results.append({
                "id": qid,
                "question": question_text,
                "reply": reply,
                "retrieved_evidence": resp["evidence_str"],
                "gold_evidence": gold_ev,
                "must_include_pass": inc_pass,
                "must_not_pass": not_pass,
                "is_accurate": is_accurate,
                "recall": ret_metrics["recall"],
                "precision": ret_metrics["precision"],
                "f1": ret_metrics["f1"],
            })

        acc = correct_facts / total_questions if total_questions else 0.0
        avg_rec = total_recall / total_questions if total_questions else 0.0
        avg_prec = total_precision / total_questions if total_questions else 0.0
        avg_f1 = total_f1 / total_questions if total_questions else 0.0

        return {
            "step": 2,
            "name": "Knowledge-Grounded RAG",
            "total_questions": total_questions,
            "correct_facts_count": correct_facts,
            "accuracy": round(acc, 4),
            "retrieval_recall": round(avg_rec, 4),
            "retrieval_precision": round(avg_prec, 4),
            "retrieval_f1": round(avg_f1, 4),
            "hit_rate": round(hits / total_questions, 4) if total_questions else 0.0,
            "details": results,
        }

    def evaluate_step3_conversations(self, conv_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate Step 3 (Multi-turn conversations with verification and tools)."""
        df = pd.DataFrame(conv_rows)
        # Group by conversation_id and sort by turn
        grouped = df.sort_values(by=["conversation_id", "turn"]).groupby("conversation_id")

        total_turns = len(df)
        decision_correct_count = 0
        tool_correct_count = 0
        fact_correct_count = 0
        total_recall = 0.0
        details = []

        for conv_id, group in grouped:
            # Reset agent session per conversation
            session_id = f"eval_{conv_id}"
            self.agent.reset_session(session_id)

            for _, row in group.iterrows():
                msg = row["customer_message"]
                expected_dec = str(row["expected_decision"]).strip().lower()
                expected_tool = str(row.get("expected_tool_calls", "")).strip()
                gold_ev = row.get("gold_evidence")
                must_inc = row.get("must_include")
                must_not = row.get("must_not_include")

                resp = self.agent.respond(msg, session_id=session_id, mode="conversational")
                reply = resp["reply"]
                pred_dec = resp["decision"].lower()
                pred_tools = resp["tool_calls"]

                # Check decision
                dec_match = (pred_dec == expected_dec)
                if dec_match:
                    decision_correct_count += 1

                # Check tool calls
                tool_match = True
                if expected_tool and expected_tool not in ("nan", "None", ""):
                    if "lookup_order" in expected_tool:
                        tool_match = any("lookup_order" in pt for pt in pred_tools)
                    elif "create_ticket" in expected_tool:
                        tool_match = any("create_ticket" in pt for pt in pred_tools)
                    else:
                        tool_match = (len(pred_tools) > 0)
                else:
                    tool_match = (len(pred_tools) == 0)

                if tool_match:
                    tool_correct_count += 1

                inc_pass, _ = check_must_include(reply, must_inc)
                not_pass, _ = check_must_not_include(reply, must_not)
                if inc_pass and not_pass:
                    fact_correct_count += 1

                ret_metrics = calculate_retrieval_metrics(resp["evidence_ids"], gold_ev)
                total_recall += ret_metrics["recall"]

                details.append({
                    "conversation_id": conv_id,
                    "turn": row["turn"],
                    "customer_message": msg,
                    "reply": reply,
                    "expected_decision": expected_dec,
                    "predicted_decision": pred_dec,
                    "decision_match": dec_match,
                    "expected_tool": expected_tool,
                    "predicted_tools": pred_tools,
                    "tool_match": tool_match,
                    "must_include_pass": inc_pass,
                })

        return {
            "step": 3,
            "name": "Multi-Turn Support Agent with Tools",
            "total_turns": total_turns,
            "decision_accuracy": round(decision_correct_count / total_turns, 4) if total_turns else 0.0,
            "tool_call_accuracy": round(tool_correct_count / total_turns, 4) if total_turns else 0.0,
            "fact_groundedness": round(fact_correct_count / total_turns, 4) if total_turns else 0.0,
            "average_evidence_recall": round(total_recall / total_turns, 4) if total_turns else 0.0,
            "details": details,
        }

