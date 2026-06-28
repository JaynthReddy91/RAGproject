import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import Settings, DATA_DIR
from src.loader import DocumentLoader
from src.chunker import TextChunker
from src.vectorstore import VectorStoreManager
from src.retriever import HybridRetrieverManager

def main():
    print("=== Bootstrapping Knowledge Base Indexing ===")
    
    # 1. Initialize managers
    loader = DocumentLoader()
    chunker = TextChunker(chunk_size=Settings.CHUNK_SIZE, chunk_overlap=Settings.CHUNK_OVERLAP)
    vsm = VectorStoreManager()
    
    # 2. Check and load directory
    data_dir = DATA_DIR
    print(f"Loading documents from: {data_dir}...")
    documents = loader.load_directory(data_dir)
    
    if not documents:
        print("No documents found in data/ folder. Please place txt, md, pdf, or docx files there first.")
        return
        
    print(f"Loaded {len(documents)} document pages/sections.")
    
    # 3. Chunk documents
    print("Chunking documents (recursive method)...")
    chunked_docs = chunker.chunk_documents(documents, method="recursive")
    print(f"Created {len(chunked_docs)} text chunks.")
    
    # 4. Clear existing index (optional, but good for clean boots)
    print("Clearing existing vector store collection...")
    vsm.delete_collection()
    
    # 5. Index chunks
    print("Generating embeddings and writing to ChromaDB (this might take a few moments)...")
    vsm.add_documents(chunked_docs)
    
    # 6. Initialize and verify hybrid retriever
    print("Refreshing hybrid search index...")
    retriever = HybridRetrieverManager(vsm)
    
    print("\nIndexing bootstrap finished successfully!")
    print(f"Total chunks in vector store: {vsm.get_document_count()}")
    print("You are ready to run evaluations or start the Streamlit UI!")

if __name__ == "__main__":
    main()
