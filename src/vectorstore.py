import os
from typing import List, Optional
from langchain_core.documents import Document as LCDocument
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

from src.config import Settings
from src.chunker import ChunkedDocument

class VectorStoreManager:
    def __init__(self):
        self.embedding_model = self._init_embeddings()
        self.db_path = Settings.CHROMA_DB_PATH
        self.collection_name = "rag_collection"
        self.vector_store: Optional[Chroma] = None
        
        # Load existing database if it exists
        self.get_or_create_vector_store()

    def _init_embeddings(self):
        """Initializes the embedding model based on Settings configuration."""
        provider = Settings.EMBEDDING_PROVIDER
        model_name = Settings.EMBEDDING_MODEL

        print(f"Initializing embedding provider: '{provider}' with model: '{model_name}'...")
        
        if provider == "openai":
            if not Settings.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY is required for OpenAI embeddings.")
            return OpenAIEmbeddings(
                openai_api_key=Settings.OPENAI_API_KEY,
                model=model_name
            )
        elif provider == "gemini":
            if not Settings.GEMINI_API_KEY:
                raise ValueError("GEMINI_API_KEY is required for Gemini embeddings.")
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            # Fallback to models/text-embedding-004 if default local model name is set
            model = model_name if model_name and "MiniLM" not in model_name else "models/text-embedding-004"
            return GoogleGenerativeAIEmbeddings(
                google_api_key=Settings.GEMINI_API_KEY,
                model=model
            )
        else:
            # Fallback to local HuggingFace embeddings
            return HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': 'cpu'}
            )

    def get_or_create_vector_store(self) -> Chroma:
        """Retrieves an existing ChromaDB instance or initializes a new one."""
        if self.vector_store is not None:
            return self.vector_store

        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding_model,
            persist_directory=self.db_path
        )
        return self.vector_store

    def add_documents(self, chunked_docs: List[ChunkedDocument]) -> List[str]:
        """Converts chunked docs into LangChain documents and adds them to ChromaDB."""
        if not chunked_docs:
            print("No documents to add to vector store.")
            return []

        lc_docs = [
            LCDocument(
                page_content=chunk.text,
                metadata=chunk.metadata
            )
            for chunk in chunked_docs
        ]
        
        # Extract individual unique IDs for target updates, if possible
        ids = [chunk.metadata.get("chunk_id") for chunk in chunked_docs]
        # Verify if all IDs are unique and present
        if len(ids) == len(chunked_docs) and all(ids):
            self.vector_store.add_documents(documents=lc_docs, ids=ids)
        else:
            ids = self.vector_store.add_documents(documents=lc_docs)
            
        print(f"Successfully added {len(lc_docs)} chunks to the vector store.")
        return ids

    def search_similarity(self, query: str, k: int = 5) -> List[LCDocument]:
        """Performs simple vector similarity search."""
        store = self.get_or_create_vector_store()
        return store.similarity_search(query, k=k)

    def search_similarity_with_scores(self, query: str, k: int = 5) -> List[tuple]:
        """Performs similarity search and returns tuple of (document, score)."""
        store = self.get_or_create_vector_store()
        return store.similarity_search_with_relevance_scores(query, k=k)

    def delete_collection(self):
        """Clears the collection database completely."""
        if self.vector_store is not None:
            self.vector_store.delete_collection()
            self.vector_store = None
            print("ChromaDB collection deleted successfully.")
            # Recreate an empty database instance
            self.get_or_create_vector_store()

    def get_document_count(self) -> int:
        """Returns the number of documents (chunks) currently stored."""
        if self.vector_store is not None:
            collection = self.vector_store._collection
            return collection.count()
        return 0
