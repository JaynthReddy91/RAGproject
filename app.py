import os
from pathlib import Path
import streamlit as st
from src.config import Settings
from src.loader import DocumentLoader
from src.chunker import TextChunker
from src.vectorstore import VectorStoreManager
from src.retriever import HybridRetrieverManager
from src.generator import GeneratorManager

# Initialize Streamlit Page Config
st.set_page_config(
    page_title="RAG Pipeline",
    page_icon="😒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Design & Aesthetics (Glassmorphism, Gradients, Outfit font)
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    /* Overall App Background & Font Override */
    html, body, [class*="css"], .stApp {
        font-family: 'Outfit', sans-serif;
        background: linear-gradient(135deg, #0d0e15 0%, #151828 50%, #0a0b0e 100%) !important;
        color: #e2e8f0;
    }
    
    /* Header styling */
    .app-title {
        background: linear-gradient(90deg, #a78bfa 0%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 2.8rem;
        margin-bottom: 0.2rem;
        text-align: left;
    }
    
    .app-subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        text-align: left;
    }
    
    /* Glassmorphism containers */
    .glass-card {
        background: rgba(30, 41, 59, 0.4);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    }
    
    /* Citation Card layout */
    .citation-header {
        font-weight: 600;
        color: #38bdf8;
        font-size: 0.95rem;
        margin-bottom: 0.4rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .citation-body {
        font-size: 0.85rem;
        color: #cbd5e1;
        background: rgba(15, 23, 42, 0.5);
        padding: 0.75rem;
        border-left: 3px solid #8b5cf6;
        border-radius: 4px;
        margin-top: 0.25rem;
        line-height: 1.4;
    }
    
    /* Custom Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.95) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Buttons customization */
    .stButton>button {
        background: linear-gradient(135deg, #6d28d9 0%, #4f46e5 100%) !important;
        color: white !important;
        border: none !important;
        padding: 0.6rem 1.5rem !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(109, 40, 217, 0.3) !important;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(109, 40, 217, 0.5) !important;
        background: linear-gradient(135deg, #7c3aed 0%, #6366f1 100%) !important;
    }
    
    /* Badges */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 6px;
        margin-right: 0.5rem;
    }
    .badge-primary {
        background-color: rgba(139, 92, 246, 0.2);
        color: #c084fc;
        border: 1px solid rgba(139, 92, 246, 0.4);
    }
    .badge-secondary {
        background-color: rgba(56, 189, 248, 0.2);
        color: #7dd3fc;
        border: 1px solid rgba(56, 189, 248, 0.4);
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Initialize Session State Managers
if "vsm" not in st.session_state:
    with st.spinner("Initializing Vector DB..."):
        st.session_state.vsm = VectorStoreManager()
if "retriever" not in st.session_state:
    with st.spinner("Initializing Retrievers..."):
        st.session_state.retriever = HybridRetrieverManager(st.session_state.vsm)
if "generator" not in st.session_state:
    with st.spinner("Initializing LLM Agent..."):
        st.session_state.generator = GeneratorManager()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "loader" not in st.session_state:
    st.session_state.loader = DocumentLoader()
if "chunker" not in st.session_state:
    st.session_state.chunker = TextChunker(
        chunk_size=Settings.CHUNK_SIZE,
        chunk_overlap=Settings.CHUNK_OVERLAP
    )

vsm: VectorStoreManager = st.session_state.vsm
retriever: HybridRetrieverManager = st.session_state.retriever
generator: GeneratorManager = st.session_state.generator
loader: DocumentLoader = st.session_state.loader
chunker: TextChunker = st.session_state.chunker

# --- SIDEBAR: Settings & Document Management ---
with st.sidebar:
    st.markdown("<h2 style='color: #a78bfa; margin-bottom: 0.5rem;'> INTIG RAG</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #64748b; font-size: 0.85rem; margin-bottom: 1.5rem;'>Interactive Document Intelligence Pipeline</p>", unsafe_allow_html=True)
    
    st.divider()
    
    # Pipeline Settings Status
    st.subheader("System Status")
    st.markdown(f"<span class='badge badge-primary'>LLM: {Settings.LLM_PROVIDER} ({Settings.LLM_MODEL})</span>", unsafe_allow_html=True)
    st.markdown(f"<span class='badge badge-secondary'>Embeddings: {Settings.EMBEDDING_PROVIDER}</span>", unsafe_allow_html=True)
    
    st.markdown(f"**Indexed chunks:** {vsm.get_document_count()}")
    
    st.divider()
    
    # Ingestion Block
    st.subheader("Ingest New Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, or TXT files",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
        key="file_uploader"
    )
    
    if uploaded_files:
        if st.button("Process & Index Files", key="btn_index"):
            new_chunks_count = 0
            for uploaded_file in uploaded_files:
                # Save file to local data directory
                save_path = Settings.DATA_DIR / uploaded_file.name
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Load and Parse document
                with st.spinner(f"Parsing {uploaded_file.name}..."):
                    loaded_docs = loader.load_file(save_path)
                
                # Chunk document
                with st.spinner(f"Chunking {uploaded_file.name}..."):
                    # We can use recursive or semantic. Let's use recursive by default, semantic as an option
                    chunk_method = st.session_state.get("sel_chunk_method", "recursive")
                    
                    # If semantic, we pass our vector store's embedding model
                    emb_model = vsm.embedding_model if chunk_method == "semantic" else None
                    chunked_docs = chunker.chunk_documents(
                        loaded_docs,
                        method=chunk_method,
                        embedding_model=emb_model
                    )
                
                # Index in vector store
                if chunked_docs:
                    with st.spinner(f"Embedding and Indexing {uploaded_file.name}..."):
                        vsm.add_documents(chunked_docs)
                        new_chunks_count += len(chunked_docs)
            
            # Refresh retrievers so they incorporate newly added files
            retriever.refresh_retrievers()
            st.success(f"Indexed successfully! Added {new_chunks_count} chunks.")
            st.rerun()

    # Settings configurations
    st.subheader("Chunking Strategy")
    st.selectbox(
        "Split Method",
        options=["recursive", "semantic", "fixed"],
        key="sel_chunk_method",
        index=0,
        help="Recursive is recommended for general docs. Semantic uses sentence embedding similarity splits."
    )
    
    st.divider()
    
    # Document Filtering
    st.subheader("Knowledge Filters")
    # Retrieve names of all indexed files from ChromaDB (source field in metadatas)
    all_files = set()
    if vsm.get_document_count() > 0:
        collection_data = vsm.get_or_create_vector_store()._collection.get(include=["metadatas"])
        for meta in collection_data["metadatas"]:
            if "source" in meta:
                all_files.add(meta["source"])
                
    all_files_list = sorted(list(all_files))
    
    selected_files = []
    if all_files_list:
        st.write("Query database on selected documents only:")
        for doc_name in all_files_list:
            if st.checkbox(doc_name, value=True, key=f"filter_{doc_name}"):
                selected_files.append(doc_name)
    else:
        st.info("No documents indexed. Upload files to get started!")
        
    st.divider()
    
    # Clear DB Button
    if st.button("Reset Vector Store", key="btn_clear_db"):
        with st.spinner("Deleting database..."):
            vsm.delete_collection()
            retriever.refresh_retrievers()
            generator.clear_memory()
            st.session_state.chat_history = []
            
            # Delete physical files in data dir
            for item in Settings.DATA_DIR.iterdir():
                if item.is_file():
                    try:
                        item.unlink()
                    except Exception as e:
                        print(f"Error deleting file {item}: {e}")
                        
        st.success("Vector database reset completed.")
        st.rerun()

# --- MAIN CHAT AREA ---
st.markdown("<h1 class='app-title'>⚡ INTIGRAGR AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='app-subtitle'>Retrieval-Augmented Intelligent Knowledge Assistant</p>", unsafe_allow_html=True)

# Main container for Chat
chat_container = st.container()


# Render Chat History
with chat_container:
    for chat in st.session_state.chat_history:
        role = chat["role"]
        content = chat["content"]
        
        with st.chat_message(role):
            st.write(content)
            
            # Removed citation drawer for a cleaner interface

# User Chat Input
if prompt := st.chat_input("Ask a question about your documents..."):
    # 1. Render User Message
    with chat_container:
        with st.chat_message("user"):
            st.write(prompt)
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    
    # 2. Process query & generate response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing query and scanning database..."):
            # A. Rewrite query to stand-alone if there is memory
            query_to_search = generator.rewrite_query(prompt)
            
            # B. Search Vector Store with database-level metadata filters
            if all_files_list and not selected_files:
                filtered_docs = []
                st.warning("No documents selected in Knowledge Filters. Please select at least one document.")
            else:
                filter_sources = selected_files if all_files_list and len(selected_files) < len(all_files_list) else None
                filtered_docs = retriever.retrieve(query_to_search, filter_sources=filter_sources)
                
            # D. Generate answer based on context
            response_data = generator.generate_response(prompt, filtered_docs)
            answer = response_data["answer"]
            
        # E. Render response
        st.write(answer)
        
        # Removed citation drawer for a cleaner interface
                    
    # Save assistant response to state
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer,
        "docs": filtered_docs
    })
