"""
RSS-based News Scraper for Bangladeshi news websites
"""
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import requests
import feedparser
from bs4 import BeautifulSoup
from config import RAW_ARTICLES_PATH


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
    """Fetches articles using RSS feeds and web scraping as fallback"""

    # Updated RSS feeds - some sources may not have working RSS
    RSS_FEEDS = {
        "daily_star": [
            "https://www.thedailystar.net/frontpage/rss.xml",
            "https://www.thedailystar.net/news/rss.xml",
            "https://www.thedailystar.net/business/rss.xml"
        ],
        "prothom_alo": [
            "https://en.prothomalo.com/feed/",
            "https://www.prothomalo.com/feed/"
        ],
        "bdnews24": [
            "https://bdnews24.com/rss.xml",
            "https://bdnews24.com/feed"
        ]
    }

    # Fallback: Direct article URLs for scraping
    FALLBACK_URLS = {
        "daily_star": "https://www.thedailystar.net",
        "prothom_alo": "https://en.prothomalo.com",
        "bdnews24": "https://bdnews24.com",
        "dhaka_tribune": "https://www.dhakatribune.com",
        "new_age": "https://www.newagebd.net"
    }

    def __init__(self):
        self.articles: List[NewsArticle] = []
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

    def fetch_rss_feed(self, source: str, url: str) -> List[NewsArticle]:
        """Fetch and parse a single RSS feed"""
        try:
            logger.info(f"Fetching RSS feed from {source}: {url}")
            
            # Try with requests first for better control
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            feed = feedparser.parse(response.content)
            
            if not feed.entries:
                logger.warning(f"No entries found in RSS feed for {source}")
                return []
                
            articles = []
            for entry in feed.entries[:20]:  # Limit to 20 articles per feed
                try:
                    # Get more detailed content if available
                    content = entry.get("summary", "")
                    if hasattr(entry, 'content') and entry.content:
                        content = entry.content[0].value if isinstance(entry.content, list) else entry.content.value
                    
                    article = NewsArticle(
                        title=entry.get("title", "").strip(),
                        content=content.strip(),
                        url=entry.get("link", "").strip(),
                        source=source,
                        published_date=entry.get("published", None),
                        author=entry.get("author", None)
                    )
                    
                    if article.title and article.content:  # Only add if has title and content
                        articles.append(article)
                        
                except Exception as e:
                    logger.warning(f"Error processing RSS entry from {source}: {e}")
                    continue
                    
            logger.info(f"Successfully fetched {len(articles)} articles from {source} RSS")
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching RSS feed for {source} ({url}): {e}")
            return []

    def scrape_website_fallback(self, source: str, base_url: str) -> List[NewsArticle]:
        """Fallback method to scrape articles directly from website"""
        try:
            logger.info(f"Using fallback scraping for {source}: {base_url}")
            
            response = self.session.get(base_url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            articles = []
            
            # Common selectors for news articles
            article_selectors = [
                'article h2 a', 'article h3 a', 'article .title a',
                '.news-item a', '.post-title a', 'h2.title a',
                '.entry-title a', '.headline a', '.story-headline a'
            ]
            
            links_found = set()
            
            for selector in article_selectors:
                elements = soup.select(selector)
                for element in elements[:10]:  # Limit per selector
                    try:
                        title = element.get_text().strip()
                        link = element.get('href', '')
                        
                        if not title or not link:
                            continue
                            
                        # Convert relative URLs to absolute
                        if link.startswith('/'):
                            link = base_url.rstrip('/') + link
                        elif not link.startswith('http'):
                            continue
                            
                        if link in links_found:
                            continue
                            
                        links_found.add(link)
                        
                        # Try to get article content (basic extraction)
                        content = self.extract_article_content(link)
                        
                        if content and len(content.strip()) > 100:  # Minimum content length
                            article = NewsArticle(
                                title=title,
                                content=content,
                                url=link,
                                source=source
                            )
                            articles.append(article)
                            
                            if len(articles) >= 15:  # Limit total articles
                                break
                                
                    except Exception as e:
                        logger.warning(f"Error processing link from {source}: {e}")
                        continue
                        
                if len(articles) >= 15:
                    break
                    
            logger.info(f"Scraped {len(articles)} articles from {source} website")
            return articles
            
        except Exception as e:
            logger.error(f"Error in fallback scraping for {source}: {e}")
            return []

    def extract_article_content(self, url: str) -> str:
        """Extract article content from URL"""
        try:
            response = self.session.get(url, timeout=8)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup.find_all(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                element.decompose()
            
            # Common content selectors
            content_selectors = [
                '.article-content', '.post-content', '.entry-content',
                '.news-content', '.story-content', 'article .content',
                '.article-body', '.post-body', 'main article'
            ]
            
            content = ""
            for selector in content_selectors:
                element = soup.select_one(selector)
                if element:
                    content = element.get_text().strip()
                    if len(content) > 200:  # Good content found
                        break
            
            # Fallback: try to find content in paragraphs
            if not content or len(content) < 200:
                paragraphs = soup.find_all('p')
                content = ' '.join([p.get_text().strip() for p in paragraphs[:10]])
            
            # Clean up content
            content = ' '.join(content.split())  # Remove extra whitespace
            return content[:2000]  # Limit content length
            
        except Exception as e:
            logger.warning(f"Error extracting content from {url}: {e}")
            return ""

    def scrape_all_sources(self) -> List[NewsArticle]:
        """Fetch articles from all defined RSS feeds and fallback sources"""
        all_articles = []
        
        # First try RSS feeds
        for source, urls in self.RSS_FEEDS.items():
            source_articles = []
            
            for url in urls:
                articles = self.fetch_rss_feed(source, url)
                source_articles.extend(articles)
                
                if len(source_articles) >= 20:  # Stop if we have enough articles
                    break
                    
                time.sleep(1)  # Rate limiting
            
            # If RSS failed or insufficient articles, try fallback
            if len(source_articles) < 5 and source in self.FALLBACK_URLS:
                logger.info(f"RSS insufficient for {source}, trying fallback scraping...")
                fallback_articles = self.scrape_website_fallback(source, self.FALLBACK_URLS[source])
                source_articles.extend(fallback_articles)
            
            all_articles.extend(source_articles)
            logger.info(f"Total articles from {source}: {len(source_articles)}")
            time.sleep(2)  # Rate limiting between sources
        
        # Try additional sources if we don't have enough articles
        if len(all_articles) < 30:
            additional_sources = ["dhaka_tribune", "new_age"]
            for source in additional_sources:
                if source in self.FALLBACK_URLS:
                    articles = self.scrape_website_fallback(source, self.FALLBACK_URLS[source])
                    all_articles.extend(articles)
                    time.sleep(2)
        
        # Remove duplicates based on URL
        seen_urls = set()
        unique_articles = []
        for article in all_articles:
            if article.url not in seen_urls and article.url:
                seen_urls.add(article.url)
                unique_articles.append(article)
        
        logger.info(f"Total unique articles scraped: {len(unique_articles)}")
        self.articles = unique_articles
        return unique_articles

    def save_articles(self, filepath: str = None):
        """Save scraped articles to JSON file"""
        if not filepath:
            filepath = RAW_ARTICLES_PATH
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump([a.to_dict() for a in self.articles], f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(self.articles)} articles to {filepath}")

    def load_articles(self, filepath: str = None) -> List[NewsArticle]:
        """Load articles from JSON file"""
        if not filepath:
            filepath = RAW_ARTICLES_PATH
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                articles_data = json.load(f)
            self.articles = [NewsArticle(**data) for data in articles_data]
            logger.info(f"Loaded {len(self.articles)} articles from {filepath}")
            return self.articles
        except FileNotFoundError:
            logger.warning(f"No articles file found at {filepath}")
            return []


def main():
    scraper = NewsScraper()
    articles = scraper.scrape_all_sources()
    scraper.save_articles()

    print(f"Scraping completed. Total articles: {len(articles)}")
    
    # Show summary by source
    source_counts = {}
    for article in articles:
        source_counts[article.source] = source_counts.get(article.source, 0) + 1
    
    print("\nArticles by source:")
    for source, count in source_counts.items():
        print(f"  {source}: {count}")
    
    # Show sample articles
    print(f"\nSample articles:")
    for article in articles[:3]:
        print(f"\nTitle: {article.title}")
        print(f"Source: {article.source}")
        print(f"Content: {article.content[:200]}...")


if __name__ == "__main__":
    main()