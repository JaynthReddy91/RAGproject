import re
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document as LCDocument
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

from src.config import Settings
from src.vectorstore import VectorStoreManager

def math_tokenizer(text: str) -> List[str]:
    """
    A custom tokenizer for mathematical content.
    Extracts words, LaTeX commands, operations (+, -, *, /, ^, =),
    and mathematical variables/terms instead of stripping punctuation.
    """
    if not text:
        return []
    # Pattern extracts LaTeX macros, variables/equations, operators, and words
    pattern = r'\\[a-zA-Z]+|\b\d*[a-zA-Z](?:\^[a-zA-Z0-9]+)?\b|[\+\-\*/\^=]|\b\w+\b'
    tokens = re.findall(pattern, text)
    return [t.lower() if not t.startswith('\\') else t for t in tokens]

class HybridRetrieverManager:
    def __init__(self, vector_store_manager: VectorStoreManager):
        self.vsm = vector_store_manager
        self.bm25_retriever: Optional[BM25Retriever] = None
        self.ensemble_retriever: Optional[EnsembleRetriever] = None
        self.reranker_model = None
        
        # Initialize CrossEncoder if configured
        self._init_reranker()
        
        # Try to initialize retriever from existing documents in vector store
        self.refresh_retrievers()

    def _init_reranker(self):
        """Initializes the local cross-encoder model for re-ranking."""
        if Settings.RERANKER_PROVIDER == "local":
            try:
                from sentence_transformers import CrossEncoder
                print(f"Loading local Cross-Encoder re-ranker model: '{Settings.RERANKER_MODEL}'...")
                self.reranker_model = CrossEncoder(Settings.RERANKER_MODEL)
            except Exception as e:
                print(f"Error loading cross-encoder re-ranker: {e}. Re-ranking will be disabled.")
                self.reranker_model = None

    def refresh_retrievers(self):
        """
        Rebuilds the BM25 and Ensemble retrievers based on the documents currently 
        stored in the persistent vector store.
        """
        store = self.vsm.get_or_create_vector_store()
        doc_count = self.vsm.get_document_count()
        
        if doc_count == 0:
            print("Vector store is empty. Hybrid retrievers not initialized yet.")
            self.bm25_retriever = None
            self.ensemble_retriever = None
            return

        print(f"Initializing BM25 and Hybrid Ensemble Retriever on {doc_count} documents...")
        
        # 1. Fetch all documents from the vector store to build the BM25 index
        # ChromaDB allows getting all documents from the collection
        collection_data = store._collection.get(include=["documents", "metadatas"])
        
        lc_docs = []
        for text, meta in zip(collection_data["documents"], collection_data["metadatas"]):
            lc_docs.append(LCDocument(page_content=text, metadata=meta))
            
        # 2. Build BM25 retriever
        try:
            self.bm25_retriever = BM25Retriever.from_documents(lc_docs, preprocess_func=math_tokenizer)
            self.bm25_retriever.k = Settings.TOP_K_RETRIEVAL
        except Exception as e:
            print(f"Error creating BM25 retriever: {e}")
            self.bm25_retriever = None

        # 3. Create Vector Retriever
        vector_retriever = store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": Settings.TOP_K_RETRIEVAL}
        )

        # 4. Assemble Ensemble Retriever (weighting 50/50 keyword and vector search)
        if self.bm25_retriever:
            self.ensemble_retriever = EnsembleRetriever(
                retrievers=[self.bm25_retriever, vector_retriever],
                weights=[0.5, 0.5]
            )
        else:
            self.ensemble_retriever = vector_retriever

    def get_raw_retrieved_docs(self, query: str, filter_sources: Optional[List[str]] = None) -> List[LCDocument]:
        """Retrieves raw candidate documents using the configured retriever, with optional metadata filtering."""
        store = self.vsm.get_or_create_vector_store()
        doc_count = self.vsm.get_document_count()
        
        if doc_count == 0:
            print("Vector store is empty. Hybrid retrievers not initialized yet.")
            return []

        # If no specific source filters are applied, use the cached ensemble retriever
        if not filter_sources:
            if not self.ensemble_retriever:
                self.refresh_retrievers()
            if not self.ensemble_retriever:
                print("No retrievers initialized. Returning empty results.")
                return []
            return self.ensemble_retriever.invoke(query)

        # Build dynamic retriever with metadata filters
        print(f"Applying database-level source filters: {filter_sources}")
        
        # Chroma filter syntax: if list has elements, use $in operator
        if len(filter_sources) == 1:
            where_filter = {"source": filter_sources[0]}
        else:
            where_filter = {"source": {"$in": filter_sources}}
            
        # 1. Fetch filtered documents from Chroma to build a filtered BM25 index
        try:
            collection_data = store._collection.get(where=where_filter, include=["documents", "metadatas"])
        except Exception as e:
            print(f"Error fetching filtered documents from collection: {e}")
            return []
        
        lc_docs = []
        if collection_data and "documents" in collection_data and collection_data["documents"]:
            for text, meta in zip(collection_data["documents"], collection_data["metadatas"]):
                lc_docs.append(LCDocument(page_content=text, metadata=meta))
            
        # If no documents match the filters, return empty list
        if not lc_docs:
            print(f"No documents found matching source filters: {filter_sources}")
            return []

        # 2. Build filtered BM25 retriever
        try:
            dynamic_bm25 = BM25Retriever.from_documents(lc_docs, preprocess_func=math_tokenizer)
            dynamic_bm25.k = min(Settings.TOP_K_RETRIEVAL, len(lc_docs))
        except Exception as e:
            print(f"Error creating dynamic BM25 retriever: {e}")
            dynamic_bm25 = None

        # 3. Create filtered Vector Retriever
        vector_retriever = store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": Settings.TOP_K_RETRIEVAL,
                "filter": where_filter
            }
        )

        # 4. Assemble dynamic Ensemble Retriever
        if dynamic_bm25:
            dynamic_ensemble = EnsembleRetriever(
                retrievers=[dynamic_bm25, vector_retriever],
                weights=[0.5, 0.5]
            )
        else:
            dynamic_ensemble = vector_retriever

        return dynamic_ensemble.invoke(query)

    def rerank_documents(self, query: str, docs: List[LCDocument]) -> List[LCDocument]:
        """Re-ranks documents using the local cross-encoder model."""
        if not docs or not self.reranker_model:
            return docs[:Settings.TOP_K_RERANK]
            
        try:
            # Prepare pairs of (query, document_text)
            pairs = [(query, doc.page_content) for doc in docs]
            
            # Predict scores
            scores = self.reranker_model.predict(pairs)
            
            # Sort documents by scores in descending order
            scored_docs = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
            
            # Filter and return top re-ranked docs
            return [doc for score, doc in scored_docs[:Settings.TOP_K_RERANK]]
            
        except Exception as e:
            print(f"Error during re-ranking: {e}. Returning top hybrid results without re-ranking.")
            return docs[:Settings.TOP_K_RERANK]

    def retrieve(self, query: str, filter_sources: Optional[List[str]] = None) -> List[LCDocument]:
        """
        Executes the full retrieval pipeline:
        1. Fetch candidates via Hybrid Search (BM25 + Vector) with optional source filtering.
        2. Re-rank candidates using the Cross-Encoder.
        """
        raw_docs = self.get_raw_retrieved_docs(query, filter_sources=filter_sources)
        reranked_docs = self.rerank_documents(query, raw_docs)
        return reranked_docs
