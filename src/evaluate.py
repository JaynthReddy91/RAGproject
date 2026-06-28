import time
import json
import re
from typing import List, Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage

from src.config import Settings, BASE_DIR
from src.vectorstore import VectorStoreManager
from src.retriever import HybridRetrieverManager
from src.generator import GeneratorManager

def get_content_str(content) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict) and "text" in part:
                text_parts.append(part["text"])
        return "".join(text_parts).strip()
    return str(content).strip()

# Test Q&A dataset based on our sample documents
EVAL_DATASET = [
    {
        "question": "What is the dimension of the embeddings and sub-layers in the Transformer encoder?",
        "ground_truth": "All sub-layers in the model, as well as the embedding layers, produce outputs of dimension d_model = 512."
    },
    {
        "question": "How many layers are in the encoder and decoder stacks of the Transformer?",
        "ground_truth": "The encoder and decoder are both composed of a stack of N = 6 identical layers."
    },
    {
        "question": "What is the formula for Scaled Dot-Product Attention?",
        "ground_truth": "Attention(Q, K, V) = softmax((Q * K^T) / sqrt(d_k)) * V"
    },
    {
        "question": "Why does self-attention have an advantage over recurrence for long-range dependencies?",
        "ground_truth": "Self-attention layers connect all positions with a constant number of sequentially executed operations, whereas recurrent layers require O(n) sequential operations, making it easier to learn long-range dependencies."
    },
    {
        "question": "What is Retrieval-Augmented Generation (RAG)?",
        "ground_truth": "RAG is a technique that optimizes the output of a large language model (LLM) by referencing an authoritative knowledge base outside of its training data sources before generating a response."
    },
    {
        "question": "What are the four major challenges of LLMs that RAG addresses?",
        "ground_truth": "RAG addresses LLM hallucinations, stale training data, lack of authority/source verification, and security/access control limitations."
    },
    {
        "question": "What are the common text chunking strategies in RAG?",
        "ground_truth": "Standard chunking strategies include fixed-size chunking, recursive character chunking, and semantic chunking."
    },
    {
        "question": "What are the key advantages of RAG over fine-tuning?",
        "ground_truth": "RAG is lower cost, allows dynamic real-time updates in the database, provides strict source citations, minimizes hallucinations, and supports simple access control."
    },
    {
        "question": "When is fine-tuning preferred over RAG?",
        "ground_truth": "Fine-tuning is preferred when adapting a model's style, tone, format outputs (like JSON), or specialized syntax, and for zero-shot efficiency to reduce input tokens."
    },
    {
        "question": "Who is the President of France?",
        "ground_truth": "Not present in the documents. System should say: 'I am sorry, but the provided documents do not contain the information required to answer this question.'"
    }
]

class RAGEvaluator:
    def __init__(self, retriever_manager: HybridRetrieverManager, generator_manager: GeneratorManager):
        self.retriever = retriever_manager
        self.generator = generator_manager
        self.llm = generator_manager.llm

    def evaluate_llm_as_judge(self, question: str, context: str, answer: str, ground_truth: str) -> Dict[str, float]:
        """
        Uses the LLM as a judge to evaluate Faithfulness, Relevance, and Correctness.
        Returns a dict of scores from 1.0 to 5.0.
        """
        judge_prompt = f"""You are an objective AI evaluator. Rate the RAG system output based on the provided Question, Context, and Ground Truth.

Inputs:
- Question: {question}
- Retrieved Context: {context}
- Generated Answer: {answer}
- Ground Truth Answer: {ground_truth}

Evaluate and rate the following metrics on a scale from 1 to 5 (where 1 is poor/incorrect and 5 is excellent/perfect). 

Metrics definition:
1. Faithfulness: Is the generated answer fully grounded in and supported ONLY by the retrieved context? It must not contain information outside the context.
2. Relevance: Does the generated answer directly address the user's question? Is it concise and helpful?
3. Correctness: Compared to the Ground Truth answer, is the generated answer factually accurate and complete?

Format your output EXACTLY as this JSON block (do not return any other text or markdown wrapper outside the json):
{{
  "faithfulness": <score 1-5>,
  "relevance": <score 1-5>,
  "correctness": <score 1-5>
}}
"""
        try:
            response = self.llm.invoke([
                SystemMessage(content="You are a strict, objective quality control evaluator. Always output clean JSON."),
                HumanMessage(content=judge_prompt)
            ])
            content = get_content_str(response.content)
            
            # Clean up JSON if LLM returned markdown wrappers
            if content.startswith("```"):
                content = re.sub(r"^```(?:json)?\n", "", content)
                content = re.sub(r"\n```$", "", content)
                
            scores = json.loads(content)
            return {
                "faithfulness": float(scores.get("faithfulness", 1.0)),
                "relevance": float(scores.get("relevance", 1.0)),
                "correctness": float(scores.get("correctness", 1.0))
            }
        except Exception as e:
            print(f"Error parsing judge scores: {e}. Defaulting to 1.0.")
            return {"faithfulness": 1.0, "relevance": 1.0, "correctness": 1.0}

    def run_evaluations(self) -> Dict[str, Any]:
        """Runs the test dataset through the pipeline and calculates summary statistics."""
        results = []
        
        total_retrieval_time = 0.0
        total_generation_time = 0.0
        
        print(f"Starting evaluation of {len(EVAL_DATASET)} Q&A pairs...")
        
        for idx, item in enumerate(EVAL_DATASET):
            question = item["question"]
            gt = item["ground_truth"]
            
            print(f"\n[{idx+1}/{len(EVAL_DATASET)}] Question: {question}")
            
            # 1. Benchmark Retrieval
            start_ret = time.time()
            retrieved_docs = self.retriever.retrieve(question)
            ret_time = time.time() - start_ret
            total_retrieval_time += ret_time
            
            # 2. Benchmark Generation
            start_gen = time.time()
            # Disable memory for individual testing
            self.generator.clear_memory()
            response_data = self.generator.generate_response(question, retrieved_docs)
            gen_time = time.time() - start_gen
            total_generation_time += gen_time
            
            answer = response_data["answer"]
            context = response_data["context_used"]
            
            # 3. Judge scores
            scores = self.evaluate_llm_as_judge(question, context, answer, gt)
            
            result = {
                "question": question,
                "ground_truth": gt,
                "generated_answer": answer,
                "retrieval_time_sec": ret_time,
                "generation_time_sec": gen_time,
                "scores": scores
            }
            results.append(result)
            
            print(f"  Retrieval: {ret_time:.2f}s | Generation: {gen_time:.2f}s")
            print(f"  Scores -> Faithfulness: {scores['faithfulness']}/5 | Relevance: {scores['relevance']}/5 | Correctness: {scores['correctness']}/5")
            
            # Sleep to respect free tier rate limits (15 RPM)
            time.sleep(4)

        # Compile summaries
        avg_ret_time = total_retrieval_time / len(EVAL_DATASET)
        avg_gen_time = total_generation_time / len(EVAL_DATASET)
        
        avg_faithfulness = sum(r["scores"]["faithfulness"] for r in results) / len(results)
        avg_relevance = sum(r["scores"]["relevance"] for r in results) / len(results)
        avg_correctness = sum(r["scores"]["correctness"] for r in results) / len(results)
        
        summary = {
            "total_questions": len(EVAL_DATASET),
            "averages": {
                "retrieval_latency_sec": avg_ret_time,
                "generation_latency_sec": avg_gen_time,
                "total_latency_sec": avg_ret_time + avg_gen_time,
                "faithfulness_score": avg_faithfulness,
                "relevance_score": avg_relevance,
                "correctness_score": avg_correctness
            },
            "detailed_results": results
        }
        
        # Write summary report to disk
        report_path = BASE_DIR / "evaluation_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
            
        print(f"\nEvaluation Completed! Report written to: {report_path}")
        print("=== EVALUATION REPORT SUMMARY ===")
        print(f"Avg Retrieval Latency:  {avg_ret_time:.3f}s")
        print(f"Avg Generation Latency: {avg_gen_time:.3f}s")
        print(f"Avg Faithfulness Score: {avg_faithfulness:.2f}/5.0")
        print(f"Avg Relevance Score:    {avg_relevance:.2f}/5.0")
        print(f"Avg Correctness Score:  {avg_correctness:.2f}/5.0")
        print("=================================")
        
        return summary

if __name__ == "__main__":
    # Simple standalone check
    vsm = VectorStoreManager()
    rm = HybridRetrieverManager(vsm)
    gm = GeneratorManager()
    
    evaluator = RAGEvaluator(rm, gm)
    evaluator.run_evaluations()
