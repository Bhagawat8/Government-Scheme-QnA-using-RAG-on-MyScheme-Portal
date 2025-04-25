# app.py
import os

# Disable Streamlit file watcher before any imports
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"

import sys
import pickle
import streamlit as st
from loader import load_csv_data
from embedding_generator import initialize_embedding_model, generate_embeddings
from vector_store import create_vector_store
from retrieval import CustomRetriever
from augmentation import augment_query
from generation import FalconGenerator
from parser import CustomOutputParser
from langchain_core.runnables import RunnableSequence, RunnablePassthrough

# Constants
EMBEDDINGS_PATH = "chunks_with_embeddings.pkl"

# Monkey-patch for torch compatibility (kept as a fallback)
import torch
if hasattr(torch, '_classes'):
    torch._classes.__path__ = []

# Set __file__ for main module
sys.modules['__main__'].__file__ = 'app.py'

def main():
    # Set page config first
    st.set_page_config(
        page_title="GovScheme Assistant",
        page_icon="🇮🇳",
        layout="centered",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
        /* Add your custom styles here if needed */
    </style>
    """, unsafe_allow_html=True)

    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Sidebar elements
    with st.sidebar:
        st.header("System Status")
        status_placeholder = st.empty()
        st.markdown("---")
        st.markdown("**Model Information**")
        st.caption("Using Falcon-3B-Instruct model")

    # Main app header
    st.title("🇮🇳 Indian Government Scheme Assistant")
    st.caption("Ask about any government scheme in India!")

    # Initialize RAG pipeline
    @st.cache_resource(show_spinner=False)
    def init_pipeline(_status):
        """Initialize the RAG pipeline without UI elements"""
        try:
            # 1. Initialize embedding model
            embedding_model = initialize_embedding_model()
            
            # 2. Load or create embeddings
            if os.path.exists(EMBEDDINGS_PATH):
                with open(EMBEDDINGS_PATH, 'rb') as f:
                    chunks = pickle.load(f)
            else:
                docs = load_csv_data("cleaned_my_scheme_data_fixed.csv")
                chunks = generate_embeddings(embedding_model, docs)
                with open(EMBEDDINGS_PATH, 'wb') as f:
                    pickle.dump(chunks, f)

            # 3. Create vector store
            collection = create_vector_store(chunks)

            # 4. Initialize components
            retriever = CustomRetriever(
                collection=collection,
                embedding_model=embedding_model
            )
            generator = FalconGenerator()
            
            return RunnableSequence(
                RunnablePassthrough.assign(
                    context=lambda x: retriever.retrieve_chunks(x["query"])[0]
                ),
                lambda x: augment_query(x["query"], x["context"]),
                lambda messages: generator.generate([messages]),
                lambda result: CustomOutputParser().parse(result.generations[0][0].text)
            )
        except Exception as e:
            _status.error(f"Initialization failed: {str(e)}")
            raise

    try:
        # Show initialization status
        status_placeholder.info("⚙️ Initializing system...")
        rag_chain = init_pipeline(status_placeholder)
        status_placeholder.success("✅ System ready!")
    
    except Exception as e:
        status_placeholder.error(f"🚨 Critical error: {str(e)}")
        st.error("System initialization failed - please refresh the page")
        st.stop()

    # Chat interface functions
    def display_chat():
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # Main chat flow
    display_chat()
    
    if prompt := st.chat_input("Ask about government schemes..."):
        try:
            # Add user message
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("🔍 Analyzing schemes..."):
                    response = rag_chain.invoke({"query": prompt})
                    st.markdown(response)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response
                    })
                    
        except Exception as e:
            error_msg = f"⚠️ Error processing request: {str(e)}"
            st.error(error_msg)
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": error_msg
            })
            st.rerun()

if __name__ == "__main__":
    # Windows-specific event loop fix
    if sys.platform == "win32":
        import asyncio
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    main()