# 📰 News Article Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that scrapes news articles from major Bangladeshi news outlets, processes them with advanced text chunking techniques, and provides intelligent responses to user queries about current events using local LLM inference.

## 📑 Index


- [🌟 Key Features](#key-features)
- [🗃️ Architecture](#architecture)
- [📁 Project Structure](#project-structure)
- [🚀 Quick Start](#quick-start)
   - [Option 1: Using Docker (Recommended)](#option-1-using-docker-recommended)
   - [Option 2: Manual Docker Build](#option-2-manual-docker-build)
   - [Option 3: Local Installation](#option-3-local-installation)
- [⚙️ LLM Provider Selection](#llm-provider-selection)
   - [Using Groq API](#using-groq-api)
- [📖 Usage Guide](#usage-guide)
   - [Initial Setup](#initial-setup)
   - [Chat Interface Features](#chat-interface-features)
   - [Advanced Features](#advanced-features)
- [⚙️ Configuration](#configuration)
   - [News Sources](#news-sources)
   - [LLM Settings](#llm-settings)
   - [Vector Database](#vector-database)
- [📈 Performance Optimization](#performance-optimization)
   - [For Large Datasets](#for-large-datasets)
- [🙏 Acknowledgments](#acknowledgments)
- [Contact](#contact)

## Key Features

- **Multi-Source News Scraping**: Automatically scrapes from Prothom Alo, The Daily Star, and BDNews24
- **Advanced RAG Pipeline**: Text processing and semantic chunking
- **Flexible LLM Integration**: Choose between local Ollama (Llama 3.2) or cloud-based Groq API
- **Interactive Streamlit UI**: Beautiful web interface with real-time chat and analytics
- **Vector Search Engine**: ChromaDB with sentence transformers for semantic similarity search
- **Comprehensive Analytics**: Database statistics, source distribution, and content metrics
- **Containerized Deployment**: Full Docker support with automated setup
- **Multi-Strategy Querying**: Handles count queries, summaries, trending news, and general search

## Architecture

```
  ┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
  │   News Sources  │ --> │   Web Scraper    │ --> │ Text Processor  │
  │ • Prothom Alo   │     │ • RSS Feeds      │     │ • Deduplication │
  │ • Daily Star    │     │ • Parallel Fetch │     │ • Metadata      │
  │ • BDNews24, etc.│     │ • Content Extract│     │                 │
  └─────────────────┘     └──────────────────┘     └─────────────────┘
                                                           |
┌────────────────────┐    ┌──────────────────┐             v
│   User Query       │    │   RAG Pipeline   │     ┌─────────────────┐
│ • Intent Detection │--> │ • Multi-Strategy │ <-- │ Vector Database │
│ • Entity Extract   │    │ • Source Filter  │     │ • ChromaDB      │
│ • Keyword Focus    │    │                  │     │ • Embeddings    │
└────────────────────┘    └──────────────────┘     └─────────────────┘
                                |
                                v
                       ┌─────────────────┐
                       │   LLM Provider  │
                       │ • Ollama (Local)│
                       │ • Groq API      │
                       │ • Configurable  │
                       └─────────────────┘
```

## Project Structure

```
news-rag-chatbot/
├── src/
│   ├── config.py              # Configuration settings
│   ├── news_scraper.py        # Multi-threaded web scraping
│   ├── text_preprocessor.py   # Text processing
│   ├── chroma_manager.py      # Vector database operations
│   ├── prompts.py             # Prompt templates
│   └── rag_pipeline.py        # RAG Pipeline
├── app.py                     # Main Streamlit application
├── data/                      # Data storage directory
├── logs/                      # Application logs
├── Dockerfile                 # Container configuration
├── docker-compose.yml         # Multi-service orchestration
├── .env                       # Environment variables (create this)
├── requirements.txt           # Python dependencies
├── user_usage_guide.md        # User guide and advanced tips
└── README.md                  # This documentation
```

## Quick Start

### Option 1: Using Docker (Recommended)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/bd-news-rag-chatbot.git
   cd news-rag-chatbot
   ```

2. **Configure LLM Provider:**
   ```bash
   # For Groq API (cloud-based)
   echo "GROQ_API_KEY=your_groq_api_key_here" > .env
   
   # Edit config.py to set USE_LOCAL = False for Groq
   # or keep USE_LOCAL = True for local Ollama
   ```

3. **Run with Docker Compose:**
   ```bash
   docker-compose up --build
   ```

4. **Access the application:**
   Open your browser and go to `http://localhost:8501`

5. **First-time setup:**
   - Click **`🔄Scrape Recent News`** button to fetch articles
   - Wait for processing and indexing to complete
   - Start asking questions!

### Option 2: Manual Docker Build

```bash
# Build the image
docker build -t bd-news-chatbot .

# Run the container
docker run -p 8501:8501 -p 11435:11434 -v $(pwd)/data:/app/data bd-news-chatbot
```

### Option 3: Local Installation

**Prerequisites:**
- Python 3.9+
- Git
- 8GB+ RAM recommended (for local Ollama)
- Groq API key (if using cloud inference)

**Step-by-step setup:**

1. **Configure LLM Provider (Choose One):**

   **Option A: Local Ollama Setup**
   ```bash
   # Install Ollama
   curl -fsSL https://ollama.com/install.sh | sh
   
   # Start Ollama service
   ollama serve
   
   # Install Llama 3.2 model
   ollama pull llama3.2
   ```

   **Option B: Groq API Setup**
   ```bash
   # Create environment file
   echo "GROQ_API_KEY=your_groq_api_key_here" > .env
   
   # Edit config.py and set USE_LOCAL = False
   ```

2. **Clone and setup the project:**
   ```bash
   git clone https://github.com/your-username/bd-news-rag-chatbot.git
   cd bd-news-rag-chatbot
   
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   streamlit run app.py
   ```

## LLM Provider Selection

This project supports two LLM backends:

- **Ollama (local inference)** – runs on your machine (default if `USE_LOCAL = True`)
- **Groq API (cloud inference)** – uses Groq's hosted LLMs (if `USE_LOCAL = False`)

You can select which provider to use by editing the following line in `config.py`:

```python
USE_LOCAL = False   # True = Ollama local, False = Groq API
```

- Set `USE_LOCAL = True` to use Ollama (local).
- Set `USE_LOCAL = False` to use Groq API.

### Using Groq API

To use Groq, you must set your API key in an `.env` file or as an environment variable:

```bash
GROQ_API_KEY=your_groq_api_key_here
```

You can also configure the Groq model in `config.py` under `LLM_PROVIDER`.


## Usage Guide

### Initial Setup

1. **Scrape News Articles:**
   - Click **`🔄Scrape Recent News`** in the sidebar
   - The system will fetch articles from all configured sources
   - Processing and indexing happen automatically

2. **Monitor Progress:**
   - Check the "📈 Database Stats" section for real-time updates
   - View article counts by source
   - Monitor processing status

### Chat Interface Features

**Sample Queries to Try:**
- `What is trending news today?`
- `How many news articles are there?`
- `Tell me about Bangladesh politics news`
- `Latest news from Prothom Alo`



### Advanced Features

**Multi-Chat Sessions:**
- Use **`📝New Chat`** to start fresh conversations
- Previous chat history is preserved and accessible
- Each session maintains independent context

**Database Management:**
- Reset database to clear all articles
- Confirmation prompts prevent accidental data loss
- Real-time statistics tracking

## Configuration

### News Sources

Edit `src/config.py` to add/modify news sources:

```python
NEWS_SOURCES = {
    "prothom_alo": {
        "base_url": "https://en.prothomalo.com",
        "sections": ["bangladesh", "international", "sports"],
        "rss": ["https://en.prothomalo.com/feed/"]
    },
    # Add more sources...
}
```

### LLM Settings

Configure the language model providers:

```python
# Choose provider
USE_LOCAL = True  # True for Ollama, False for Groq

# Ollama Configuration
LLM_CONFIG = {
    "model_name": "llama3.2",
    "temperature": 0.7,
    "max_tokens": 1000,
    "context_window": 4096
}

# Groq Configuration
LLM_PROVIDER = {
    "groq_api_key": os.getenv("GROQ_API_KEY", ""),
    "groq_model": "llama3-8b-8192"
}
```

### Vector Database

Adjust search parameters:

```python
VECTORDB_CONFIG = {
    "embedding_model": "all-MiniLM-L6-v2",
    "similarity_threshold": 0.15,
    "max_results": 15
}
```


## Performance Optimization

### For Large Datasets

1. **Increase batch processing:**
   ```python
   VECTORDB_CONFIG = {
       "batch_size": 100  # Increase for better throughput
   }
   ```

2. **Adjust chunk sizes:**
   ```python
   TEXT_PROCESSING_CONFIG = {
       "chunk_size": 1200,
       "chunk_overlap": 200
   }
   ```

3. **Optimize search parameters:**
   ```python
   SEARCH_CONFIG = {
       "max_context_length": 3000,
   }
   ```

## Acknowledgments

- **Ollama** for local LLM inference
- **Groq** for fast cloud-based LLM inference
- **ChromaDB** for efficient vector storage
- **LangChain** for RAG pipeline components
- **Streamlit** for the interactive web interface
- **Beautiful Soup** & **Feedparser** for web scraping
- **Bangladeshi news outlets** for providing accessible news content


## Contact  
**👤 Name**: MD. AMINUL ISLAM <br>
**📧 Email**: itsmeaminul@gmail.com <br>
**🔗 GitHub**: https://github.com/itsmeaminul <br>
**🔗 LinkedIn**: [linkedin.com/in/md-aminul88](https://linkedin.com/in/md-aminul88/)

