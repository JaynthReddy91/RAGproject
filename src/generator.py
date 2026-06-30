import re
from typing import List, Dict, Any, Tuple, Optional
from langchain_core.documents import Document as LCDocument
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

from src.config import Settings

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

class GeneratorManager:
    def __init__(self):
        self.llm = self._init_llm()
        self.memory: List[Tuple[str, str]] = [] # Simple memory stored as a list of (human, ai) pairs
        self.memory_limit = 5 # Keep last 5 exchanges

    def _init_llm(self):
        """Initializes the Chat LLM based on Settings configurations."""
        provider = Settings.LLM_PROVIDER
        model_name = Settings.LLM_MODEL
        
        print(f"Initializing LLM provider: '{provider}' with model: '{model_name}'...")
        
        if provider == "openai":
            if not Settings.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY is required for OpenAI LLM.")
            return ChatOpenAI(
                openai_api_key=Settings.OPENAI_API_KEY,
                model=model_name,
                temperature=0.0
            )
        elif provider == "gemini":
            if not Settings.GEMINI_API_KEY:
                raise ValueError("GEMINI_API_KEY is required for Google Gemini LLM.")
            return ChatGoogleGenerativeAI(
                google_api_key=Settings.GEMINI_API_KEY,
                model=model_name,
                temperature=0.0
            )
        elif provider in ["xai", "grok"]:
            if not Settings.XAI_API_KEY:
                raise ValueError("XAI_API_KEY / GROK_API_KEY is required for Grok/xAI.")
            return ChatOpenAI(
                openai_api_key=Settings.XAI_API_KEY,
                base_url="https://api.x.ai/v1",
                model=model_name,
                temperature=0.0
            )
        elif provider == "ollama":
            return ChatOllama(
                model=model_name,
                base_url="http://localhost:11434",
                temperature=0.0
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    def clear_memory(self):
        """Clears conversational memory."""
        self.memory = []

    def get_chat_history_string(self) -> str:
        """Formats the chat history into a string format for prompt inclusion."""
        history = ""
        for human, ai in self.memory:
            history += f"User: {human}\nAssistant: {ai}\n"
        return history

    def rewrite_query(self, query: str) -> str:
        """
        Uses LLM to rewrite a follow-up query into a standalone query 
        incorporating the conversational history if memory exists.
        """
        if not self.memory:
            return query
            
        history_str = self.get_chat_history_string()
        
        prompt = f"""Given the following conversation history and a follow-up question, rewrite the follow-up question to be a standalone question that can be search-indexed.
Do NOT answer the question. Only return the rewritten standalone question.

Conversation History:
{history_str}
Follow-up Question: {query}

Standalone Question:"""
        
        try:
            response = self.llm.invoke([
                SystemMessage(content="You are an expert query processor. Rewrite follow-up questions to be self-contained."),
                HumanMessage(content=prompt)
            ])
            rewritten = get_content_str(response.content)
            print(f"Rewrote query: '{query}' -> '{rewritten}'")
            return rewritten
        except Exception as e:
            print(f"Error rewriting query: {e}. Using raw query.")
            return query

    def format_context(self, docs: List[LCDocument]) -> str:
        """Formats retrieved documents into a context block with explicit references."""
        context_parts = []
        for idx, doc in enumerate(docs):
            src = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "?")
            context_parts.append(
                f"[Doc ID {idx+1}] Source: {src} (Page {page})\nContent:\n{doc.page_content}\n---"
            )
        return "\n\n".join(context_parts)

    def solve_math_symbolically(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Attempts to solve integration queries symbolically using SymPy.
        Returns a dictionary containing the solution details, or None if it fails
        or is not an integration query.
        """
        keywords = ["integrate", "integral", "antiderivative", "antidifferentiate", "\\int"]
        is_math = any(kw in query.lower() for kw in keywords) or "dx" in query.lower()
        if not is_math:
            return None
            
        try:
            import sympy
            from sympy import Symbol, integrate, latex
            from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application, convert_xor
            
            # Extract expression and bounds
            expr_clean = re.sub(
                r'^(integrate|evaluate|find|determine|antiderivative of|integral of|the integral of|what is the integral of|what is the antiderivative of)\s+',
                '', query, flags=re.IGNORECASE
            )
            # Check for definite bounds: e.g. "from 0 to pi"
            bounds_match = re.search(
                r'from\s+([a-zA-Z0-9_\*/\+\-\.\s]+)\s+to\s+([a-zA-Z0-9_\*/\+\-\.\s]+)', 
                expr_clean, flags=re.IGNORECASE
            )
            
            # Strip bounds text and 'dx' from expression
            expr_clean = re.sub(r'\s+from\s+.*$', '', expr_clean, flags=re.IGNORECASE)
            expr_clean = re.sub(r'\s+dx$', '', expr_clean, flags=re.IGNORECASE)
            expr_clean = expr_clean.strip()
            
            # Apply trig replacements (e.g. sinx -> sin(x))
            for func in ['sin', 'cos', 'tan', 'exp', 'log', 'ln', 'sec', 'csc', 'cot']:
                expr_clean = re.sub(rf'\b{func}\s*([a-zA-Z0-9]+)\b', rf'{func}(\1)', expr_clean)
                expr_clean = re.sub(rf'\b{func}\s*\(', rf'{func}(', expr_clean)
            
            # Replace e^x or e**x with exp(x)
            expr_clean = re.sub(r'\be\^([a-zA-Z0-9]+)\b', r'exp(\1)', expr_clean)
            expr_clean = re.sub(r'\be\*\*([a-zA-Z0-9]+)\b', r'exp(\1)', expr_clean)
            
            # Parse symbolic variables
            x = Symbol('x')
            transformations = (standard_transformations + (implicit_multiplication_application, convert_xor))
            parsed_expr = parse_expr(expr_clean, local_dict={'x': x, 'pi': sympy.pi, 'e': sympy.E}, transformations=transformations)
            
            # Integrate
            if bounds_match:
                lower_str = bounds_match.group(1).strip()
                upper_str = bounds_match.group(2).strip()
                lower_val = parse_expr(lower_str, local_dict={'pi': sympy.pi, 'e': sympy.E}, transformations=transformations)
                upper_val = parse_expr(upper_str, local_dict={'pi': sympy.pi, 'e': sympy.E}, transformations=transformations)
                
                result = integrate(parsed_expr, (x, lower_val, upper_val))
                latex_expr = f"\\int_{{{latex(lower_val)}}}^{{{latex(upper_val)}}} {latex(parsed_expr)} \\, dx"
                latex_result = latex(result)
                is_definite = True
            else:
                result = integrate(parsed_expr, x)
                latex_expr = f"\\int {latex(parsed_expr)} \\, dx"
                # Add constant of integration for indefinite integral
                latex_result = f"{latex(result)} + C"
                is_definite = False
                
            return {
                "expr": str(parsed_expr),
                "result": str(result),
                "latex_expr": latex_expr,
                "latex_result": latex_result,
                "is_definite": is_definite
            }
        except Exception as e:
            print(f"SymPy symbolic calculation failed or skipped: {e}")
            return None

    def generate_response(self, query: str, docs: List[LCDocument]) -> Dict[str, Any]:
        """
        Generates an answer based on the retrieved documents and updates the conversational memory.
        """
        context_str = self.format_context(docs)
        
        system_prompt = """You are a precise and helpful assistant. Your task is to answer the user's question using ONLY the provided text context. 

Strict Rules:
1. Cite information: For every claim or statement you make based on a document, append a citation showing which document index it came from, e.g., '[Doc ID 1]'.
2. Rely ONLY on the provided context: Do not assume, extrapolate, or use outside knowledge. 
3. Fallback: If the provided context is empty or does not contain the answer, state EXACTLY: "I am sorry, but the provided documents do not contain the information required to answer this question." Do not make up any response.
4. Keep the response factual, clear, and professional.
"""

        # Check if SymPy can solve it symbolically
        symbolic_solution = self.solve_math_symbolically(query)
        if symbolic_solution:
            verified_math_fact = f"""
VERIFIED MATHEMATICAL CALCULATION:
The symbolic math engine has calculated the exact, 100% accurate result for this query:
- Integral Equation: $${symbolic_solution['latex_expr']}$$
- Verified Symbolic Solution: $${symbolic_solution['latex_result']}$$

Strict Rule:
You MUST output the exact Verified Symbolic Solution in your final answer. Do NOT guess or write a different mathematical result. Use the retrieved context documents to explain the steps leading to this solution, citing the documents accordingly.
"""
            system_prompt += "\n" + verified_math_fact

        prompt = f"""Context documents:
{context_str}

Conversation History (for context of who is speaking, prioritize the documents above for answers):
{self.get_chat_history_string()}
Question: {query}

Answer:"""

        try:
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=prompt)
            ])
            answer = get_content_str(response.content)
            
            # Save to memory
            self.memory.append((query, answer))
            if len(self.memory) > self.memory_limit:
                self.memory.pop(0)
                
            return {
                "answer": answer,
                "retrieved_docs": docs,
                "context_used": context_str
            }
        except Exception as e:
            print(f"Error during response generation: {e}")
            return {
                "answer": f"Error generating answer: {e}",
                "retrieved_docs": docs,
                "context_used": context_str
            }
        
    def add_memory_turn(self, human: str, ai: str):
        """Manual addition of message exchange to history (e.g. from UI state)."""
        self.memory.append((human, ai))
        if len(self.memory) > self.memory_limit:
            self.memory.pop(0)
