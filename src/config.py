# Updated config.py
"""
Centralized configuration for the BD News RAG Chatbot
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Data paths
RAW_ARTICLES_PATH = DATA_DIR / "raw_articles.json"
PROCESSED_ARTICLES_PATH = DATA_DIR / "processed_articles.json"
CHROMADB_PATH = DATA_DIR / "chromadb"

# Disable ChromaDB telemetry to avoid errors
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_SERVER_NOFILE"] = "1"

# News sources configuration
NEWS_SOURCES = {
    "prothom_alo": {
        "base_url": "https://en.prothomalo.com",
        "sections": ["bangladesh", "international", "sports", "entertainment", "business", "education", "opinion"],
        "rss": [
            "https://en.prothomalo.com/feed/",
            "https://en.prothomalo.com/bangladesh",
            "https://en.prothomalo.com/sports",
            "https://en.prothomalo.com/entertainment",
            "https://en.prothomalo.com/business"
        ]
    },
    "daily_star": {
        "base_url": "https://www.thedailystar.net",
        "sections": ["news", "bangladesh", "sports", "lifestyle", "business"],
        "rss": [
            "https://www.thedailystar.net/frontpage/rss.xml",
            "https://www.thedailystar.net/news/rss.xml",
            "https://www.thedailystar.net/sports/rss.xml",
            "https://www.thedailystar.net/lifestyle/rss.xml"
        ],
        "article_selectors": {
            "title": ["h1.title", "h1", ".article-header h1"],
            "content": [".article-content", ".node-content", ".story-content"],
            "category": [".breadcrumb li:last-child a", ".category a"],
            "author": [".author", ".node-meta .name", ".byline"]
        }
    },
    "bdnews24": {
        "base_url": "https://bdnews24.com",
        "sections": ["news/bangladesh", "news/world", "sport", "news/politics", "news/education", "news/environment", "news/science", "news/business"],
        "rss": [
            "https://bdnews24.com/?widgetName=rssfeed&widgetId=1150&getXmlFeed=true",
        ],
        "article_selectors": {
            "title": ["h1.article-title", "h1.headline", "h1"],
            "content": [".article-body", ".story-content", ".content", "article"],
            "category": [".category", ".breadcrumb a:last-child"],
            "author": [".author", ".byline"]
        }
    }
}

# Scraping configuration
SCRAPING_CONFIG = {
    "max_articles_per_source": 100,
    "request_timeout": 10,
    "delay_between_requests": 1,
    "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "delay_min": 0.2,
    "delay_max": 1.0,
    "max_workers": 8
}

# Text processing configuration
TEXT_PROCESSING_CONFIG = {
    "chunk_size": 1000,
    "chunk_overlap": 150,
    "min_text_length": 100,
    "max_text_length": 8000,
    "title_chunk_size": 600,
    "create_title_chunks": True,
}

# Vector database configuration
VECTORDB_CONFIG = {
    "collection_name": "bd_news_articles",
    "embedding_model": "all-MiniLM-L6-v2",
    "similarity_threshold": 0.15,  # Lowered for better recall
    "max_results": 15,
    "batch_size": 50,
}

# LLM configuration
USE_LOCAL = False   # True = Ollama local, False = Groq API

LLM_PROVIDER = {
    "groq_api_key": os.getenv("GROQ_API_KEY", ""),
    "groq_model": "llama3-8b-8192"
}

LLM_CONFIG = {
    "model_name": "llama3.2",   # "qwen2:1.5b",
    "base_url": "http://localhost:11434",
    "temperature": 0.7,
    "max_tokens": 1000,  # Increased for better responses
    "context_window": 4096
}

# Streamlit UI configuration
UI_CONFIG = {
    "page_title": "News Article Chatbot",
    "page_icon": "📰",
    "layout": "wide",
    "sidebar_width": 300
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": LOGS_DIR / "app.log"
}

# Searching related configuration
SEARCH_CONFIG = {
    "max_context_length": 4000,
}

# Query processing configuration
QUERY_CONFIG = {
    "stop_words": [
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
        "how", "many", "what", "is", "are", "was", "were", "be", "been", "being",
        "tell", "me", "about", "give", "show", "summary", "summarize", "latest", "recent",
        "trending", "today", "news", "articles", "article", "from", "get", "find"
    ],
    "title_phrase_patterns": [
        r"news about (.+)",
        r"article about (.+)", 
        r"story about (.+)",
        r"headlines about (.+)",
        r"reports on (.+)",
        r"coverage of (.+)"
    ]
}