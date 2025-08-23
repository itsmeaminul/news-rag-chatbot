"""
Centralized configuration for the BD News RAG Chatbot - Improved Version
"""
import os
from pathlib import Path

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
        ]
    },
    "bdnews24": {
        "base_url": "https://bdnews24.com",
        "sections": ["bangladesh", "world", "sport", "cricket", "politics", "education", "environment", "science", "business"],
        "rss": [
            "https://bdnews24.com/?widgetName=rssfeed&widgetId=1150&getXmlFeed=true"
        ]
    }
}

# Scraping configuration
SCRAPING_CONFIG = {
    "max_articles_per_source": 50,
    "request_timeout": 10,
    "delay_between_requests": 1,
    "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# Text processing configuration
TEXT_PROCESSING_CONFIG = {
    "chunk_size": 800,  # Reduced for better retrieval
    "chunk_overlap": 150,  # Reduced overlap
    "min_text_length": 100,  # Increased minimum
    "max_text_length": 8000  # Reduced maximum
}

# Vector database configuration - Improved settings
VECTORDB_CONFIG = {
    "collection_name": "bd_news_articles",
    "embedding_model": "all-MiniLM-L6-v2",
    "similarity_threshold": 0.15,  # Lowered threshold for better recall
    "max_results": 15,  # Increased max results
    "batch_size": 50  # Added batch size for processing
}

# LLM configuration
LLM_CONFIG = {
    "model_name": "llama3.2",
    "base_url": "http://localhost:11434",
    "temperature": 0.7,
    "max_tokens": 800,  # Increased for better responses
    "context_window": 4096
}

# Streamlit UI configuration
UI_CONFIG = {
    "page_title": "BD News RAG Chatbot",
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

# Search configuration for better results
SEARCH_CONFIG = {
    "semantic_weight": 0.7,
    "keyword_weight": 0.3,
    "min_similarity_threshold": 0.25,
    "max_context_length": 3000,
    "fallback_search_terms": ["bangladesh", "news", "politics", "sports", "business", "international"]
}

# Query processing configuration
QUERY_CONFIG = {
    "stop_words": [
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
        "how", "many", "what", "is", "are", "was", "were", "be", "been", "being",
        "tell", "me", "about", "give", "show", "summary", "summarize", "latest", "recent",
        "trending", "today", "news", "articles", "article", "from", "get", "find"
    ],
    "intent_confidence_threshold": 0.6,
    "max_search_terms": 5
}