"""
Centralized configuration for the BD News RAG Chatbot
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
        "sections": ["news/bangladesh", "news/world", "sport", "news/politics", "news/education", "news/environment", "news/science", "news/business"],
        "rss": [
            "https://bdnews24.com/?widgetName=rssfeed&widgetId=1150&getXmlFeed=true",
            "https://bdnews24.com/feeds/news.xml",  # Try this if available
            "https://bdnews24.com/rss.xml"  # Common RSS path
        ],
        # Add specific selectors for bdnews24
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
    "max_articles_per_source": 50,
    "request_timeout": 10,
    "delay_between_requests": 1,
    "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# Enhanced Text processing configuration with title emphasis
TEXT_PROCESSING_CONFIG = {
    "chunk_size": 800,
    "chunk_overlap": 150,
    "min_text_length": 100,
    "max_text_length": 8000,
    # Title-specific settings
    "title_chunk_size": 600,  # Smaller chunks for title-focused content
    "title_emphasis_weight": 0.3,  # How much to boost title relevance
    "create_title_chunks": True,  # Always create dedicated title chunks
    "title_keyword_boost": 0.2,  # Boost for chunks containing title keywords
    "max_title_keywords": 8,  # Maximum title keywords to extract
    "title_relevance_threshold": 0.3  # Minimum relevance for title matching
}

# Enhanced Vector database configuration with title support
VECTORDB_CONFIG = {
    "collection_name": "bd_news_articles",
    "embedding_model": "all-MiniLM-L6-v2",
    "similarity_threshold": 0.15,
    "max_results": 15,
    "batch_size": 50,
    # Title-specific settings
    "title_similarity_threshold": 0.25,  # Higher threshold for title matches
    "title_chunk_weight": 1.5,  # Higher weight for title chunks in scoring
    "enable_title_search": True,  # Enable specialized title search methods
    "exact_title_match_boost": 0.4,  # Boost for exact title phrase matches
    "title_keyword_match_boost": 0.2  # Boost for title keyword matches
}

# LLM configuration
LLM_CONFIG = {
    "model_name": "llama3.2",  # "qwen2:1.5b",
    "base_url": "http://localhost:11434",
    "temperature": 0.7,
    "max_tokens": 800,  # Increased for better responses
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

# Enhanced Search configuration with title priority
SEARCH_CONFIG = {
    "semantic_weight": 0.5,  # Reduced to make room for title weight
    "keyword_weight": 0.2,
    "title_weight": 0.3,  # New: dedicated title weight
    "min_similarity_threshold": 0.1,  # Lowered for better title matching
    "max_context_length": 4000,  # Increased for title-rich context
    "fallback_search_terms": ["bangladesh", "news", "politics", "sports", "business", "international"],
    # Title-specific search settings
    "prioritize_title_chunks": True,
    "title_keyword_expansion": True,  # Expand search with title-related terms
    "title_exact_match_boost": 0.4,  # Boost for exact title phrase matches
    "enable_multi_strategy_search": True,  # Enable multiple search strategies
    "title_chunk_priority_multiplier": 1.3  # Multiply title chunk scores
}

# Enhanced Query processing with title focus
QUERY_CONFIG = {
    "stop_words": [
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
        "how", "many", "what", "is", "are", "was", "were", "be", "been", "being",
        "tell", "me", "about", "give", "show", "summary", "summarize", "latest", "recent",
        "trending", "today", "news", "articles", "article", "from", "get", "find"
    ],
    "intent_confidence_threshold": 0.6,
    "max_search_terms": 10,  # Increased to accommodate title keywords
    # Title-specific query processing
    "title_query_indicators": [
        "news about", "article about", "story about", "what happened with",
        "tell me about", "latest on", "update on", "headlines about", "reports on",
        "coverage of", "story on"
    ],
    "title_keyword_minimum_length": 3,  # Minimum length for title keywords
    "extract_title_entities": True,  # Extract named entities from titles
    "enable_exact_phrase_detection": True,  # Detect exact phrases for title matching
    "title_phrase_patterns": [  # Patterns that indicate title-specific queries
        r"news about (.+)",
        r"article about (.+)", 
        r"story about (.+)",
        r"headlines about (.+)",
        r"reports on (.+)",
        r"coverage of (.+)"
    ]
}

# Title-focused enhancement settings
TITLE_ENHANCEMENT_CONFIG = {
    "enable_title_chunks": True,  # Create dedicated title summary chunks
    "enable_keyword_chunks": True,  # Create keyword-focused chunks
    "enable_exact_phrase_search": True,  # Enable exact phrase matching
    "title_repetition_factor": 2,  # How many times to repeat title in content
    "max_title_variants": 3,  # Maximum title chunk variants to create
    "title_context_size": 800,  # Size of title context chunks
    "keyword_density_threshold": 0.3,  # Minimum keyword density for relevance
    "enable_title_boosting": True,  # Enable similarity boosting for title matches
    "title_boost_multiplier": 1.4  # Multiplier for title-matched results
}