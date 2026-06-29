import re
import numpy as np
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.loader import Document

class ChunkedDocument:
    def __init__(self, text: str, metadata: Dict[str, Any]):
        self.text = text
        self.metadata = metadata

    def __repr__(self):
        return f"Chunk(source={self.metadata.get('source')}, page={self.metadata.get('page')}, chunk={self.metadata.get('chunk_index')}, len={len(self.text)})"

class TextChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize LangChain's RecursiveCharacterTextSplitter with logical boundary priorities
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n\n", "\n\n", "---", "\n", ". ", " ", ""]
        )

    def split_fixed(self, text: str) -> List[str]:
        """Performs simple fixed-size character chunking with overlap."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            start += (self.chunk_size - self.chunk_overlap)
        return chunks

    def split_recursive(self, text: str) -> List[str]:
        """Performs recursive character chunking based on structure (paragraphs, sentences)."""
        return self.recursive_splitter.split_text(text)

    def split_semantic(self, text: str, embedding_model=None, threshold_percentile: float = 95.0) -> List[str]:
        """
        Splits text based on semantic similarity between consecutive sentences.
        If embedding_model is not provided, falls back to recursive character chunking.
        """
        if not embedding_model:
            # Fallback
            return self.split_recursive(text)
        
        # 1. Split text into sentences
        # Simple sentence splitter using regex
        sentence_ends = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s')
        sentences = [s.strip() for s in sentence_ends.split(text) if s.strip()]
        
        if len(sentences) < 3:
            return [text]

        # 2. Get embeddings for sentences
        # Assume embedding_model has an 'embed_documents' or 'encode' method
        try:
            if hasattr(embedding_model, "embed_documents"):
                embeddings = embedding_model.embed_documents(sentences)
            elif hasattr(embedding_model, "encode"):
                embeddings = embedding_model.encode(sentences)
            else:
                raise ValueError("Embedding model does not have 'embed_documents' or 'encode' method")
            
            embeddings = np.array(embeddings)
        except Exception as e:
            print(f"Error generating embeddings for semantic chunking: {e}. Falling back to recursive chunker.")
            return self.split_recursive(text)

        # 3. Calculate cosine distances between consecutive sentence embeddings
        similarities = []
        for i in range(len(embeddings) - 1):
            vec1 = embeddings[i]
            vec2 = embeddings[i+1]
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 > 0 and norm2 > 0:
                sim = np.dot(vec1, vec2) / (norm1 * norm2)
            else:
                sim = 0.0
            similarities.append(sim)

        distances = [1 - sim for sim in similarities]

        # 4. Determine distance threshold for splitting (e.g. 95th percentile)
        threshold = np.percentile(distances, threshold_percentile)
        
        # 5. Split sentences into groups
        chunks = []
        current_chunk_sentences = [sentences[0]]
        
        for idx, dist in enumerate(distances):
            if dist > threshold:
                # Split here
                chunks.append(" ".join(current_chunk_sentences))
                current_chunk_sentences = [sentences[idx + 1]]
            else:
                # Keep going
                current_chunk_sentences.append(sentences[idx + 1])
                
        if current_chunk_sentences:
            chunks.append(" ".join(current_chunk_sentences))
            
        return chunks

    def chunk_document(self, doc: Document, method: str = "recursive", embedding_model=None) -> List[ChunkedDocument]:
        """
        Chunks a Document object and attaches updated metadata (chunk_index, source, page).
        """
        text = doc.text
        if method == "fixed":
            text_chunks = self.split_fixed(text)
        elif method == "semantic":
            text_chunks = self.split_semantic(text, embedding_model=embedding_model)
        else: # Default is recursive
            text_chunks = self.split_recursive(text)

        chunked_docs = []
        for idx, chunk_text in enumerate(text_chunks):
            # Create a shallow copy of metadata and add chunk details
            meta = doc.metadata.copy()
            meta["chunk_index"] = idx
            meta["chunk_method"] = method
            
            # Create a simple unique ID for the chunk
            # e.g., filename_page_chunkindex
            clean_filename = re.sub(r'[^a-zA-Z0-9_\.-]', '_', meta.get("source", "doc"))
            meta["chunk_id"] = f"{clean_filename}_p{meta.get('page', 1)}_c{idx}"
            
            chunked_docs.append(ChunkedDocument(chunk_text, meta))
            
        return chunked_docs

    def chunk_documents(self, docs: List[Document], method: str = "recursive", embedding_model=None) -> List[ChunkedDocument]:
        """Helper to chunk list of Document objects."""
        all_chunks = []
        for doc in docs:
            all_chunks.extend(self.chunk_document(doc, method, embedding_model))
        return all_chunks
