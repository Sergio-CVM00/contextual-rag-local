# Configuration settings for Contextual Retrieval RAG

# LLM Settings (Local Ollama)
OLLAMA_HOST = "http://localhost:11434"
CONTEXT_MODEL = "mistral"        # ~4GB, runs on CPU
RESPONSE_MODEL = "mistral"       # Same model for both

# Alternative models (smaller/larger):
# "neural-chat"    # ~4GB, optimized for chat
# "llama2"         # ~4GB, alternative
# "dolphin-mixtral"# ~26GB if GPU available (very capable)
# "phi"            # ~2.7GB, fast alternative

# Chunking
CHUNK_SIZE = 800
CHUNK_OVERLAP = 0.2

# Retrieval
TOP_K_SEMANTIC = 20
TOP_K_BM25 = 20
TOP_K_FINAL = 10

# Vector DB
CHROMA_PERSIST_DIR = "./data/chroma_db"
COLLECTION_NAME = "contextual_rag"

# Embedding (Local, free)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # ~100MB, runs locally

# Hardware optimization for limited RAM (4GB)
# Uncomment if you have only 4GB RAM:
# CHUNK_SIZE = 400
# TOP_K_FINAL = 5
# CONTEXT_MODEL = "phi"

# Hardware optimization for good CPU (8+ cores)
# Uncomment if you have powerful CPU:
# CHUNK_SIZE = 1200
# TOP_K_FINAL = 15
