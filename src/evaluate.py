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

# Test Q&A dataset based on our calculus integration guide
EVAL_DATASET = [
    {
        "question": "What is the constant multiple rule of integration?",
        "ground_truth": "The constant multiple rule is that the integral of k * f(x) dx is equal to k times the integral of f(x) dx, where k is a constant."
    },
    {
        "question": "What is the antiderivative of e^x?",
        "ground_truth": "The antiderivative of e^x is e^x + C."
    },
    {
        "question": "How does integration by substitution work?",
        "ground_truth": "Integration by substitution is used when the integrand contains a function and its derivative. You set u = g(x), find du = g'(x) dx, substitute u and du into the integral, evaluate it, and substitute back."
    },
    {
        "question": "What are the priority guidelines for choosing u in integration by parts (LIATE)?",
        "ground_truth": "LIATE stands for Logarithmic, Inverse trigonometric, Algebraic, Trigonometric, and Exponential functions (in order of priority for choosing u)."
    },
    {
        "question": "Explain how to evaluate the integral of x * e^x.",
        "ground_truth": "Using integration by parts with u = x and dv = e^x dx. This yields du = dx and v = e^x, giving the solution x*e^x - e^x + C."
    },
    {
        "question": "What is the result of evaluating the definite integral of sin(x) from 0 to pi?",
        "ground_truth": "The antiderivative of sin(x) is -cos(x). Evaluated from 0 to pi, it equals -cos(pi) - (-cos(0)) = -(-1) - (-1) = 2."
    },
    {
        "question": "How do you integrate 2x * cos(x^2)?",
        "ground_truth": "Using u-substitution. Let u = x^2, du = 2x dx. The integral becomes the integral of cos(u) du, which is sin(u) + C = sin(x^2) + C."
    },
    {
        "question": "What is the LIATE rule used for?",
        "ground_truth": "The LIATE rule is a priority guide (Logarithmic, Inverse trig, Algebraic, Trigonometric, Exponential) used to choose the function 'u' in integration by parts."
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
        self.judge_llm = self._init_judge_llm()

    def _init_judge_llm(self):
        """Initializes a separate high-parameter model as evaluation judge if keys exist."""
        if Settings.GEMINI_API_KEY:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                print("Initializing separate high-IQ Gemini API model as the evaluation judge...")
                return ChatGoogleGenerativeAI(
                    google_api_key=Settings.GEMINI_API_KEY,
                    model="gemini-1.5-flash",
                    temperature=0.0
                )
            except Exception as e:
                print(f"Error initializing Gemini judge: {e}. Falling back to generator LLM.")
        elif Settings.OPENAI_API_KEY:
            try:
                from langchain_openai import ChatOpenAI
                print("Initializing separate high-IQ OpenAI API model as the evaluation judge...")
                return ChatOpenAI(
                    openai_api_key=Settings.OPENAI_API_KEY,
                    model="gpt-4o-mini",
                    temperature=0.0
                )
            except Exception as e:
                print(f"Error initializing OpenAI judge: {e}. Falling back to generator LLM.")
        
        print("No cloud API keys detected. Falling back to local generator model as judge (note: potential self-grading bias).")
        return self.llm

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
            try:
                response = self.judge_llm.invoke([
                    SystemMessage(content="You are a strict, objective quality control evaluator. Always output clean JSON."),
                    HumanMessage(content=judge_prompt)
                ])
            except Exception as err:
                if self.judge_llm != self.llm:
                    print(f"Error calling judge LLM ({err}). Falling back to generator model...")
                    response = self.llm.invoke([
                        SystemMessage(content="You are a strict, objective quality control evaluator. Always output clean JSON."),
                        HumanMessage(content=judge_prompt)
                    ])
                else:
                    raise err

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

    def evaluate_single_item(self, idx: int, item: Dict[str, str]) -> Dict[str, Any]:
        """Runs the RAG pipeline for a single question and evaluates it."""
        question = item["question"]
        gt = item["ground_truth"]
        
        # 1. Benchmark Retrieval
        start_ret = time.time()
        retrieved_docs = self.retriever.retrieve(question)
        ret_time = time.time() - start_ret
        
        # 2. Benchmark Generation (using thread-local memory to avoid races)
        start_gen = time.time()
        self.generator.clear_memory()
        response_data = self.generator.generate_response(question, retrieved_docs)
        gen_time = time.time() - start_gen
        
        answer = response_data["answer"]
        context = response_data["context_used"]
        
        # 3. Judge scores
        scores = self.evaluate_llm_as_judge(question, context, answer, gt)
        
        print(f"\n[{idx+1}/{len(EVAL_DATASET)}] Completed: {question}")
        print(f"  Retrieval: {ret_time:.2f}s | Generation: {gen_time:.2f}s")
        print(f"  Scores -> Faithfulness: {scores['faithfulness']}/5 | Relevance: {scores['relevance']}/5 | Correctness: {scores['correctness']}/5")
        
        # Sleep to respect rate limits
        time.sleep(2)
        
        return {
            "question": question,
            "ground_truth": gt,
            "generated_answer": answer,
            "retrieval_time_sec": ret_time,
            "generation_time_sec": gen_time,
            "scores": scores
        }

    def run_evaluations(self) -> Dict[str, Any]:
        """Runs the test dataset through the pipeline in parallel and calculates summary statistics."""
        from concurrent.futures import ThreadPoolExecutor
        
        max_workers = 2
        print(f"Starting parallel evaluation of {len(EVAL_DATASET)} Q&A pairs (max_workers={max_workers})...")
        
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Map index and item to thread workers
            futures = [executor.submit(self.evaluate_single_item, idx, item) for idx, item in enumerate(EVAL_DATASET)]
            # Collect results as they complete
            for fut in futures:
                try:
                    results.append(fut.result())
                except Exception as e:
                    print(f"Error evaluating test case: {e}")
                    
        if not results:
            print("No evaluation results were recorded.")
            return {}
            
        # Compile summaries
        total_retrieval_time = sum(r["retrieval_time_sec"] for r in results)
        total_generation_time = sum(r["generation_time_sec"] for r in results)
        
        avg_ret_time = total_retrieval_time / len(results)
        avg_gen_time = total_generation_time / len(results)
        
        avg_faithfulness = sum(r["scores"]["faithfulness"] for r in results) / len(results)
        avg_relevance = sum(r["scores"]["relevance"] for r in results) / len(results)
        avg_correctness = sum(r["scores"]["correctness"] for r in results) / len(results)
        
        summary = {
            "total_questions": len(results),
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
