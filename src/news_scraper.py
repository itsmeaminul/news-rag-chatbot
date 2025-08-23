"""
RSS-based News Scraper for Bangladeshi news websites
"""
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import requests
from bs4 import BeautifulSoup
import feedparser
from .config import RAW_ARTICLES_PATH, NEWS_SOURCES, SCRAPING_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NewsArticle:
    """Data class for news articles"""

    def __init__(self, title: str, content: str, url: str, source: str,
                 published_date: str = None, author: str = None, category: str = None):
        self.title = title
        self.content = content
        self.url = url
        self.source = source
        self.published_date = published_date or datetime.now().isoformat()
        self.author = author
        self.category = category
        self.scraped_at = datetime.now().isoformat()

    def to_dict(self) -> Dict:
        return {
            "title": self.title,
            "content": self.content,
            "url": self.url,
            "source": self.source,
            "published_date": self.published_date,
            "author": self.author,
            "category": self.category,
            "scraped_at": self.scraped_at,
        }


class NewsScraper:
    """Fetches articles using RSS feeds and web scraping"""

    def __init__(self):
        self.articles: List[NewsArticle] = []
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': SCRAPING_CONFIG.get("user_agent")
        })

    def fetch_rss_feed(self, source: str, section: Optional[str], rss_url: str) -> List[NewsArticle]:
        """Fetch and parse a single RSS feed"""
        try:
            logger.info(f"Fetching RSS feed from {source}:{section} - {rss_url}")
            response = self.session.get(rss_url, timeout=SCRAPING_CONFIG.get("request_timeout", 10))
            response.raise_for_status()
            feed = feedparser.parse(response.content)

            articles = []
            for entry in feed.entries[:20]:
                try:
                    # Prefer full content if available
                    if hasattr(entry, "content") and entry.content:
                        content = entry.content[0].value if isinstance(entry.content, list) else entry.content.value
                    else:
                        content = entry.get("summary", "")

                    article = NewsArticle(
                        title=entry.get("title", "").strip(),
                        content=content.strip(),
                        url=entry.get("link", "").strip(),
                        source=source,
                        published_date=entry.get("published", None),
                        author=entry.get("author", None),
                        category=section
                    )

                    if article.title and article.content:
                        articles.append(article)

                except Exception as e:
                    logger.warning(f"Error processing RSS entry: {e}")
                    continue

            return articles

        except Exception as e:
            logger.error(f"Error fetching RSS feed: {e}")
            return []

    def extract_article_content(self, url: str) -> str:
        """Extract full article content from webpage"""
        try:
            response = self.session.get(url, timeout=SCRAPING_CONFIG.get("request_timeout", 10))
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            # Remove unwanted tags
            for element in soup.find_all(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                element.decompose()

            # Extract all <p> tags
            paragraphs = [p.get_text().strip() for p in soup.find_all('p') if p.get_text().strip()]
            content = " ".join(paragraphs)

            return content[:5000]  # prevent overly long articles

        except Exception as e:
            logger.warning(f"Error extracting content from {url}: {e}")
            return ""

    def scrape_all_sources(self) -> List[NewsArticle]:
        """Scrape all sources defined in config"""
        all_articles = []
        max_articles = SCRAPING_CONFIG.get("max_articles_per_source", 50)

        for source, info in NEWS_SOURCES.items():
            base_url = info.get("base_url")
            sections = info.get("sections", [])
            rss_feeds = info.get("rss", [])
            source_articles = []

            # If rss feeds are defined explicitly
            if rss_feeds:
                for rss_url in rss_feeds:
                    articles = self.fetch_rss_feed(source, None, rss_url)
                    for art in articles:
                        art.category = self.infer_category_from_url(art.url, sections)
                    source_articles.extend(articles)
                    if len(source_articles) >= max_articles:
                        break
                    time.sleep(SCRAPING_CONFIG.get("delay_between_requests", 1))
            else:
                # Try section-based RSS
                for section in sections:
                    rss_url = f"{base_url.rstrip('/')}/{section}/rss.xml"
                    articles = self.fetch_rss_feed(source, section, rss_url)
                    for art in articles:
                        art.category = section
                    source_articles.extend(articles)
                    if len(source_articles) >= max_articles:
                        break
                    time.sleep(SCRAPING_CONFIG.get("delay_between_requests", 1))

            # Fallback: homepage scrape
            if not source_articles:
                homepage_articles = self.scrape_homepage(source, base_url, sections)
                source_articles.extend(homepage_articles)

            all_articles.extend(source_articles)

        # Deduplicate by URL
        seen = set()
        unique_articles = []
        for art in all_articles:
            if art.url not in seen and art.url:
                seen.add(art.url)
                if len(art.content) < 200:
                    art.content = self.extract_article_content(art.url)
                unique_articles.append(art)

        self.articles = unique_articles
        return unique_articles

    def scrape_homepage(self, source: str, base_url: str, sections: List[str]) -> List[NewsArticle]:
        """Basic homepage scraping if RSS unavailable"""
        try:
            response = self.session.get(base_url, timeout=SCRAPING_CONFIG.get("request_timeout", 10))
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")

            articles = []
            for a in soup.select("a")[:50]:
                title = a.get_text().strip()
                link = a.get("href", "")
                if not title or not link:
                    continue
                if link.startswith("/"):
                    link = base_url.rstrip("/") + link
                if not self.is_valid_article_link(link):
                    continue

                content = self.extract_article_content(link)
                if len(content) > 200:
                    category = self.infer_category_from_url(link, sections)
                    articles.append(NewsArticle(
                        title=title, content=content, url=link,
                        source=source, category=category
                    ))
            return articles
        except Exception as e:
            logger.warning(f"Error scraping homepage for {source}: {e}")
            return []

    def is_valid_article_link(self, url: str) -> bool:
        """Filter out invalid or unwanted links"""
        blacklist = ["auth", "login", "signup", "account", "privacy", "terms", "#"]
        if any(b in url for b in blacklist):
            return False
        return True

    def infer_category_from_url(self, url: str, sections: List[str]) -> Optional[str]:
        """Infer category from URL path by matching config sections"""
        parts = url.split("/")
        for section in sections:
            if section in parts:
                return section
        return None

    def save_articles(self, filepath: str = None):
        if not filepath:
            filepath = RAW_ARTICLES_PATH
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump([a.to_dict() for a in self.articles], f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(self.articles)} articles to {filepath}")


def main():
    scraper = NewsScraper()
    articles = scraper.scrape_all_sources()
    scraper.save_articles()
    print(f"Scraping completed. Total articles: {len(articles)}")

    # summary by source/category
    counts = {}
    for a in articles:
        key = f"{a.source}:{a.category}"
        counts[key] = counts.get(key, 0) + 1
    print("\nArticles by source:category")
    for k, v in counts.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
