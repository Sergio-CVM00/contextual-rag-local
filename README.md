# Contextual Retrieval RAG - Local & Zero Cost

> A fully local, completely free RAG system with contextual retrieval. No API costs, no cloud dependencies, complete privacy.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-green.svg)](https://ollama.ai)
[![Cost: $0](https://img.shields.io/badge/Cost-%240-brightgreen.svg)](https://github.com)

## Features

- **Zero Cost** - No API fees, completely free forever
- **100% Private** - Everything runs locally, your data never leaves your machine
- **Works Offline** - No internet required after initial setup
- **49% Better Retrieval** - Contextual retrieval improves accuracy significantly
- **Fast Setup** - Get running in 15 minutes
- **Modest Hardware** - Runs on 4GB RAM minimum

## What Is This?

This is a Retrieval-Augmented Generation (RAG) system that implements **Contextual Retrieval** using completely local, open-source tools. Upload documents, ask questions, and get AI-powered answers - all without sending your data to any cloud service or paying for API calls.

### How It Works

1. **Upload** - Add your PDF or text documents
2. **Process** - Documents are chunked and enriched with context using local LLM
3. **Store** - Chunks are embedded and stored in local vector database
4. **Query** - Ask questions and get AI-generated answers from your documents
5. **Privacy** - Everything stays on your computer

## Quick Start

### Prerequisites

- Python 3.10 or higher
- 4GB RAM minimum (8GB recommended)
- 5GB free disk space

### 1. Install Ollama

**macOS:**
```bash
brew install ollama
```

**Linux:**
```bash
curl https://ollama.ai/install.sh | sh
```

**Windows:**
Download from [https://ollama.ai/download](https://ollama.ai/download)

### 2. Download LLM Model

```bash
# Start Ollama service (keep running)
ollama serve

# In another terminal, download Mistral 7B (~4GB)
ollama pull mistral
```

### 3. Set Up Project

```bash
# Clone repository
git clone <your-repo-url>
cd contextual-rag-local

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 4. Run the Application

```bash
# Make sure Ollama is running in another terminal
streamlit run app.py
```

Open your browser to `http://localhost:8501` and start uploading documents!

## Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| **LLM** | Ollama + Mistral 7B | Fast, capable, runs on CPU |
| **Embeddings** | Sentence-Transformers | Local, no API needed |
| **Vector DB** | Chroma DB | Open-source, embedded |
| **Search** | BM25 + Semantic | Hybrid retrieval |
| **UI** | Streamlit | Simple, powerful |
| **Cost** | $0 | Everything is free |

## Project Structure

```
contextual-rag-local/
├── app.py                      # Streamlit UI
├── config.py                   # Configuration
├── requirements.txt            # Dependencies
│
├── src/
│   ├── document_processor.py   # Document chunking
│   ├── context_generator.py    # Context generation
│   ├── vector_db.py            # Vector database
│   ├── retrieval.py            # Hybrid retrieval
│   └── llm_interface.py        # Ollama integration
│
├── data/
│   ├── sample_documents/       # Your documents
│   └── chroma_db/              # Vector storage
│
└── README.md
```

## Configuration

Edit `config.py` to customize:

```python
# LLM Settings
OLLAMA_HOST = "http://localhost:11434"
CONTEXT_MODEL = "mistral"       # Model for context generation
RESPONSE_MODEL = "mistral"      # Model for answering

# Chunking
CHUNK_SIZE = 800                # Characters per chunk
CHUNK_OVERLAP = 0.2             # 20% overlap

# Retrieval
TOP_K_SEMANTIC = 20             # Semantic search results
TOP_K_BM25 = 20                 # Keyword search results
TOP_K_FINAL = 10                # Final results to use

# Embeddings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Local embedding model
```

## Hardware Optimization

### Limited RAM (4GB)
```python
CHUNK_SIZE = 400
TOP_K_FINAL = 5
CONTEXT_MODEL = "phi"  # Smaller 2.7GB model
```

### Good CPU (8+ cores)
```python
CHUNK_SIZE = 1200
TOP_K_FINAL = 15
CONTEXT_MODEL = "mistral"
```

### GPU Available
```bash
# Use larger, more capable model
ollama pull dolphin-mixtral

# Update config
CONTEXT_MODEL = "dolphin-mixtral"
RESPONSE_MODEL = "dolphin-mixtral"
```

## Performance

| Task | CPU Time | GPU Time |
|------|----------|----------|
| Context generation/chunk | 2-5s | 0.5-1s |
| Embedding generation | <1s | <1s |
| Semantic search | 100ms | 100ms |
| Response generation | 5-10s | 1-2s |

## Troubleshooting

### Connection Refused Error
```bash
# Ollama isn't running - start it:
ollama serve
```

### Model Not Found
```bash
# Download the model:
ollama pull mistral

# List installed models:
ollama list
```

### Out of Memory
```python
# In config.py, reduce:
CHUNK_SIZE = 400
CONTEXT_MODEL = "phi"  # Smaller model
```

### Slow Performance
- **Use GPU** for 10x speedup
- **Use smaller model** (phi: 2.7GB)
- **Reduce chunk size**

## Usage Tips

1. **Document Quality** - Clean, well-formatted documents work best
2. **Chunk Size** - Smaller chunks (400-600) for specific info, larger (800-1200) for context
3. **Model Selection** - Mistral for balance, Phi for speed, Dolphin-Mixtral for quality (GPU)
4. **Query Formulation** - Be specific in your questions for better results

## Cost Comparison

| Solution | Setup | Monthly | Annual |
|----------|-------|---------|--------|
| **This Project** | Free | **$0** | **$0** |
| OpenAI API | Free | $100-500 | $1,200-6,000 |
| Enterprise RAG | Free | $500-5,000 | $6,000-60,000 |

**Save $1,200-60,000/year by going local!**

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Ollama](https://ollama.ai) - Local LLM runtime
- [Chroma DB](https://www.trychroma.com/) - Vector database
- [Sentence-Transformers](https://www.sbert.net/) - Embeddings
- [Streamlit](https://streamlit.io/) - UI framework

## Resources

- [Ollama Documentation](https://github.com/ollama/ollama)
- [Available Models](https://ollama.ai/library)
- [Contextual Retrieval Blog](https://www.anthropic.com/news/contextual-retrieval)

## Star History

If you find this project useful, please consider giving it a star!

---

**Made with care for privacy-conscious developers**

*Zero cost. Zero tracking. Zero compromises.*
