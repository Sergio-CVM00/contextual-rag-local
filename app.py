import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv

from src.document_processor import DocumentProcessor
from src.llm_interface import LocalLLMInterface
from src.vector_db import VectorDBManager
from src.retrieval import DualRetriever
import config

load_dotenv()

# Page config
st.set_page_config(
    page_title="Local Contextual RAG",
    page_icon="🔍",
    layout="wide"
)

# Check if Ollama is running
llm = LocalLLMInterface(config.OLLAMA_HOST, config.RESPONSE_MODEL)

if not llm.check_connection():
    st.error("❌ **Ollama is not running!**")
    st.info("""
    Please start Ollama in another terminal:
    ```bash
    ollama serve
    ```
    
    And make sure you've downloaded a model:
    ```bash
    ollama pull mistral
    ```
    """)
    st.stop()

# Initialize session state
if 'db' not in st.session_state:
    st.session_state.db = None
if 'retriever' not in st.session_state:
    st.session_state.retriever = None

# Sidebar
st.sidebar.title("⚙️ Settings")
chunk_size = st.sidebar.slider("Chunk Size", 200, 2000, config.CHUNK_SIZE, 100)
top_k = st.sidebar.slider("Results to Retrieve", 5, 20, config.TOP_K_FINAL)

st.sidebar.divider()
st.sidebar.info("""
**🎉 Zero Cost Edition**
- No API costs
- Runs completely local
- Works offline
- All open source
""")

# Main app
st.title("🔍 Local Contextual Retrieval RAG")
st.write("**100% Free. 100% Local. 100% Offline.** Powered by Ollama + Mistral")

# Tabs
tab1, tab2, tab3 = st.tabs(["📄 Upload", "❓ Query", "ℹ️ Info"])

### TAB 1: UPLOAD ###
with tab1:
    st.header("Upload & Process Documents")
    
    uploaded_file = st.file_uploader(
        "Upload a document (PDF or TXT)",
        type=['pdf', 'txt', 'md']
    )
    
    if uploaded_file and st.button("Process Document", type="primary"):
        with st.spinner("Processing..."):
            try:
                # Save file
                with open("temp_file", "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                # Process
                processor = DocumentProcessor(chunk_size=chunk_size)
                text, filename = processor.load_document("temp_file")
                
                st.success(f"✅ Loaded {len(text):,} characters")
                
                # Chunk
                chunks = processor.chunk_text(text)
                st.info(f"📦 Split into {len(chunks)} chunks")
                
                # Generate contexts (LOCAL - NO COST!)
                st.write("🧠 Generating contexts with local Mistral...")
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                context_gen = LocalLLMInterface(
                    config.OLLAMA_HOST,
                    config.CONTEXT_MODEL
                )
                
                contextualized = []
                for i, chunk in enumerate(chunks):
                    # Update progress
                    progress = (i + 1) / len(chunks)
                    progress_bar.progress(progress)
                    status_text.text(f"🔄 Processing chunk {i+1}/{len(chunks)}")
                    
                    context = context_gen.generate_context(text, chunk, i)
                    if context:
                        contextualized.append(f"{context}\n\n{chunk}")
                    else:
                        contextualized.append(chunk)
                
                st.success(f"✅ Generated contexts for {len(chunks)} chunks")
                
                # Store in Chroma
                st.write("💾 Storing in local Chroma DB...")
                store_progress = st.progress(0)
                
                db = VectorDBManager(config.CHROMA_PERSIST_DIR)
                db.create_collection(config.COLLECTION_NAME)
                
                metadatas = [
                    processor.create_metadata(filename, i, len(chunks))
                    for i in range(len(chunks))
                ]
                ids = [f"{filename}_chunk_{i}" for i in range(len(chunks))]
                
                db.add_documents(contextualized, metadatas, ids)
                store_progress.progress(1.0)
                
                # Initialize retriever
                all_docs = db.get_all_documents()
                retriever = DualRetriever(db, all_docs)
                
                st.session_state.db = db
                st.session_state.retriever = retriever
                
                st.success("✅ Document ready for querying!")
                
                # Cleanup
                os.remove("temp_file")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

### TAB 2: QUERY ###
with tab2:
    st.header("Ask Questions")
    
    if st.session_state.retriever:
        query = st.text_input("What would you like to know?")
        
        if query and st.button("Search & Generate", type="primary"):
            with st.spinner("Retrieving..."):
                try:
                    # Retrieve
                    retriever = st.session_state.retriever
                    chunks = retriever.retrieve(query, top_k=top_k)
                    
                    st.subheader(f"📚 Retrieved {len(chunks)} chunks")
                    
                    with st.expander("View Retrieved Chunks"):
                        for i, chunk in enumerate(chunks, 1):
                            st.write(f"**Chunk {i}**")
                            st.write(chunk['document'][:400] + "...")
                            st.caption(chunk['metadata'].get('source_file', 'Unknown'))
                    
                    # Generate response (LOCAL - NO COST!)
                    st.write("✨ Generating response...")
                    
                    llm_interface = LocalLLMInterface(
                        config.OLLAMA_HOST,
                        config.RESPONSE_MODEL
                    )
                    response = llm_interface.generate_response(query, chunks)
                    
                    st.subheader("Answer")
                    st.write(response)
                    
                    # Stats
                    st.divider()
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Chunks Retrieved", len(chunks))
                    col2.metric("Generation Model", "Mistral 7B")
                    col3.metric("Cost", "$0.00 ✅")
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    else:
        st.warning("⚠️ Upload and process a document first!")

### TAB 3: INFO ###
with tab3:
    st.header("ℹ️ About This System")
    
    st.write("""
    ### 🎉 100% Local & Free Edition
    
    This system uses:
    - **Ollama** - Local LLM server (free, open-source)
    - **Mistral 7B** - Fast, capable language model (~4GB, free)
    - **Sentence-Transformers** - Local embeddings (~100MB, free)
    - **Chroma DB** - Local vector database (free, open-source)
    - **Streamlit** - UI framework (free, open-source)
    
    ### 💰 Cost Breakdown
    - **API costs:** $0.00
    - **Infrastructure:** $0.00
    - **Per query:** $0.00
    - **Monthly:** $0.00 ✅
    
    Everything runs on your machine!
    
    ### ⚙️ Requirements
    - 4GB RAM minimum
    - 2GB disk space (model downloads)
    - ~30 seconds per query (CPU)
    - ~3 seconds per query (with GPU)
    
    ### 🚀 How It Works
    1. Upload a PDF or text document
    2. System chunks it into pieces
    3. Local Mistral LLM adds context to each chunk
    4. Chunks are embedded and stored locally
    5. Your questions are answered using local search + local LLM
    
    **Everything stays on your computer. 100% private. 100% free.**
    """)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("✅ Advantages")
        st.write("""
        - Zero cost forever
        - Complete privacy
        - Works offline
        - No rate limits
        - No API keys needed
        - Customizable
        """)
    
    with col2:
        st.subheader("⚠️ Trade-offs")
        st.write("""
        - Slower than cloud LLMs
        - Needs 4GB+ RAM
        - Lower quality than GPT-4
        - Single user only
        - Local hardware limits
        """)
    
    st.divider()
    
    st.subheader("🔧 Configuration")
    st.write(f"""
    - **LLM Host:** {config.OLLAMA_HOST}
    - **Model:** {config.CONTEXT_MODEL}
    - **Chunk Size:** {config.CHUNK_SIZE} tokens
    - **DB Location:** {config.CHROMA_PERSIST_DIR}
    """)
