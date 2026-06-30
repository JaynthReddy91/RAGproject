import os
import re
import sys
from pathlib import Path
from typing import List

# Add project root to python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import Settings, BASE_DIR, DATA_DIR
from src.loader import DocumentLoader
from src.chunker import TextChunker
from src.generator import GeneratorManager

# 1. Imports for Deep Learning training
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

def clean_text_for_pairs(text: str) -> str:
    """Removes extra white space and markdown formatting for clean sentence comparisons."""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def generate_synthetic_pairs() -> List[InputExample]:
    """
    Parses textbook guides, uses Ollama to synthetically generate relevant mathematical questions
    paired with paragraph contexts, and compiles them into InputExamples for SentenceTransformers.
    """
    print("=== Phase 1: Ingesting Textbook Guide and Chunking ===")
    loader = DocumentLoader()
    chunker = TextChunker(chunk_size=800, chunk_overlap=150)
    
    # Load calculus guides
    documents = loader.load_directory(DATA_DIR)
    if not documents:
        raise ValueError(f"No documents found in data folder: {DATA_DIR}")
        
    chunks = chunker.chunk_documents(documents, method="recursive")
    print(f"Ingested {len(chunks)} text chunks for dataset generation.")

    print("\n=== Phase 2: Generating Synthetic Mathematical Training Questions via Ollama ===")
    # Instantiate the LLM manager
    gm = GeneratorManager()
    
    training_examples = []
    
    # Hardcoded high-quality seed pairs in case Ollama is offline or slow
    seed_pairs = [
        ("constant multiple rule integration", "The constant multiple rule: \\int k * f(x) dx = k \\int f(x) dx"),
        ("power rule of integration", "The Power Rule: \\int x^n dx = (x^{n+1})/(n+1) + C for n != -1"),
        ("antiderivative of exponential function", "Exponential functions: \\int e^x dx = e^x + C"),
        ("integration by substitution method", "Substitution Rule (u-substitution): Used when integrand contains a function and its derivative."),
        ("LIATE rule priority", "LIATE Rule: Logarithmic, Inverse trig, Algebraic, Trigonometric, Exponential. Order of priority for choosing u."),
        ("integral of sinx", "Trigonometric integration: \\int \\sin(x) dx = -\\cos(x) + C"),
        ("integral of cosx", "Trigonometric integration: \\int \\cos(x) dx = \\sin(x) + C"),
        ("integration by parts formula", "Integration by Parts: \\int u dv = u v - \\int v du")
    ]
    
    # Add seed examples
    for query, context in seed_pairs:
        training_examples.append(InputExample(texts=[query, context]))

    # For each chunk, generate 2 synthetic questions using Ollama
    for idx, chunk in enumerate(chunks[:8]): # Limit to first 8 chunks for fast local training
        context = clean_text_for_pairs(chunk.text)
        prompt = f"""You are a math textbook editor. Read the context paragraph below and write exactly 2 diverse, short questions that are directly answered by this text. 
Write each question on a new line. Do NOT output anything else except the questions.

Context:
{context}

Questions:"""
        try:
            print(f"[{idx+1}/{min(8, len(chunks))}] Generating query pairs for chunk: '{chunk.text[:50]}...'")
            response = gm.llm.invoke(prompt)
            # Handle string responses
            content = str(response.content).strip() if hasattr(response, 'content') else str(response).strip()
            
            # Split lines and filter empty/numbered prefixes
            lines = [re.sub(r'^\d+[\.\)\-\s]+', '', l.strip()) for l in content.split('\n') if l.strip()]
            for line in lines[:2]:
                if len(line) > 10:
                    print(f"  -> Created query: \"{line}\"")
                    training_examples.append(InputExample(texts=[line, context]))
        except Exception as e:
            print(f"  -> Ollama query generation failed ({e}). Using fallback seed pairs.")

    print(f"\nGenerated {len(training_examples)} training pairs for embedding fine-tuning.")
    return training_examples

def train_math_embeddings():
    """Fine-tunes standard all-MiniLM-L6-v2 embeddings on calculus synthetic datasets."""
    # 1. Compile datasets
    try:
        train_examples = generate_synthetic_pairs()
    except Exception as e:
        print(f"Error compiling training data: {e}")
        return

    # 2. Configure model
    base_model_name = Settings.EMBEDDING_MODEL if "MiniLM" in Settings.EMBEDDING_MODEL else "all-MiniLM-L6-v2"
    print(f"\n=== Phase 3: Initializing sentence-transformers base model '{base_model_name}' ===")
    
    try:
        model = SentenceTransformer(base_model_name)
    except Exception as e:
        print(f"Error loading base model: {e}. Falling back to default 'all-MiniLM-L6-v2'.")
        model = SentenceTransformer('all-MiniLM-L6-v2')

    # 3. Setup DataLoader and Loss
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=4)
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    # 4. Define save directory
    save_dir = BASE_DIR / "models" / "math-embeddings"
    save_dir.mkdir(parents=True, exist_ok=True)

    # 5. Training Loop
    print(f"\n=== Phase 4: Commencing offline fine-tuning (3 epochs) ===")
    print(f"Saving weights to local path: {save_dir}")
    
    try:
        # Fine-tune the model
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=3,
            warmup_steps=10,
            show_progress_bar=True
        )
        # Export weights
        model.save(str(save_dir))
        print(f"\n[SUCCESS] Deep Learning training finished successfully! Weights exported to: {save_dir}")
        print("Now set EMBEDDING_MODEL=./models/math-embeddings/ in your .env file to enable your custom model!")
    except Exception as e:
        print(f"Error during SentenceTransformer training: {e}")

if __name__ == "__main__":
    train_math_embeddings()
