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

# News sources configuration
NEWS_SOURCES = {
    "prothom_alo": {
        "base_url": "https://en.prothomalo.com/",
        "sections": ["bangladesh", "international", "sports", "entertainment"]
    },
    "daily_star": {
        "base_url": "https://www.thedailystar.net",
        "sections": ["news", "sports", "lifestyle"]
    },
    "bdnews24": {
        "base_url": "https://bdnews24.com",
        "sections": ["bangladesh", "world", "sports"]
    }
}