"""
RSS-based News Scraper for Bangladeshi news websites
"""
import json
import logging
import time
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Iterable, Tuple

import requests
from bs4 import BeautifulSoup
import feedparser
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config import RAW_ARTICLES_PATH, NEWS_SOURCES, SCRAPING_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Data class for news articles
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

# Scraper with parallel fetching and processing
class NewsScraper:
    """Fetches articles using RSS feeds and parallel web scraping"""

    def __init__(self):
        self.articles: List[NewsArticle] = []
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": SCRAPING_CONFIG.get("user_agent", "Mozilla/5.0")
        })
        self.request_timeout = SCRAPING_CONFIG.get("request_timeout", 10)
        self.delay_min = SCRAPING_CONFIG.get("delay_min", 0.2)
        self.delay_max = SCRAPING_CONFIG.get("delay_max", 1.0)
        self.max_workers = SCRAPING_CONFIG.get("max_workers", 8)
        self.max_articles_per_source = SCRAPING_CONFIG.get("max_articles_per_source", 50)

    # Utility to enforce polite scraping delays
    def _polite_delay(self):
        time.sleep(random.uniform(self.delay_min, self.delay_max))

    def _get(self, url: str) -> Optional[requests.Response]:
        try:
            self._polite_delay()
            r = self.session.get(url, timeout=self.request_timeout)
            r.raise_for_status()
            # Handle odd encodings gracefully
            if not r.encoding or r.encoding.lower() == "iso-8859-1":
                r.encoding = r.apparent_encoding
            return r
        except Exception as e:
            logger.warning(f"GET failed: {url} - {e}")
            return None
    
    # Fetching and parsing RSS feeds
    def fetch_rss_feed(self, source: str, section: Optional[str], rss_url: str) -> List[NewsArticle]:
        """Fetch and parse a single RSS feed"""
        try:
            logger.info(f"[RSS] {source}:{section or '-'} -> {rss_url}")
            resp = self._get(rss_url)
            if not resp:
                return []
            feed = feedparser.parse(resp.content)

            articles = []
            for entry in feed.entries[:50]:
                try:
                    # Prefer full content if available
                    if hasattr(entry, "content") and entry.content:
                        content = entry.content[0].value if isinstance(entry.content, list) else entry.content.value
                    else:
                        content = entry.get("summary", "") or ""

                    article = NewsArticle(
                        title=(entry.get("title") or "").strip(),
                        content=(content or "").strip(),
                        url=(entry.get("link") or "").strip(),
                        source=source,
                        published_date=entry.get("published") or entry.get("updated"),
                        author=entry.get("author", None),
                        category=section,
                    )
                    # Keep only minimally valid entries
                    if article.url and (article.title or article.content):
                        articles.append(article)
                except Exception as e:
                    logger.warning(f"Error processing RSS entry from {rss_url}: {e}")
                    continue

            return articles

        except Exception as e:
            logger.error(f"Error fetching RSS feed ({rss_url}): {e}")
            return []

    # Content Extraction
    def extract_article_content(self, url: str, source: str) -> str:
        """Extract full article content using site-specific selectors"""
        try:
            resp = self._get(url)
            if not resp:
                return ""
            soup = BeautifulSoup(resp.content, "html.parser")

            # Common unwanted tags
            for element in soup.find_all(["script", "style", "nav", "footer", "header", "aside"]):
                element.decompose()

            selectors = NEWS_SOURCES.get(source, {}).get("article_selectors", {})
            content_selectors = selectors.get("content", [])

            paragraphs = []
            # Try site-specific content containers first
            for sel in content_selectors:
                container = soup.select_one(sel)
                if container:
                    paragraphs = [p.get_text().strip() for p in container.find_all("p") if p.get_text().strip()]
                    if paragraphs:
                        break

            # Fallback: all <p>
            if not paragraphs:
                paragraphs = [p.get_text().strip() for p in soup.find_all("p") if p.get_text().strip()]

            return " ".join(paragraphs)[:5000]

        except Exception as e:
            logger.warning(f"Error extracting content from {url}: {e}")
            return ""

    def enrich_article_metadata_from_html(self, url: str, article: NewsArticle, source: str) -> NewsArticle:
        """Fetch HTML and fill missing title/author/category using selectors"""
        resp = self._get(url)
        if not resp:
            return article

        soup = BeautifulSoup(resp.content, "html.parser")
        selectors = NEWS_SOURCES.get(source, {}).get("article_selectors", {})

        def pick_first(selectors_list: List[str]) -> Optional[str]:
            for sel in selectors_list:
                el = soup.select_one(sel)
                if el:
                    text = el.get_text(strip=True)
                    if text:
                        return text
            return None

        if not article.title and "title" in selectors:
            val = pick_first(selectors["title"])
            if val: article.title = val

        if not article.author and "author" in selectors:
            val = pick_first(selectors["author"])
            if val: article.author = val

        if not article.category and "category" in selectors:
            val = pick_first(selectors["category"])
            if val: article.category = val

        return article

    # Homepage scraping if RSS unavailable
    def scrape_homepage(self, source: str, base_url: str, sections: List[str]) -> List[NewsArticle]:
        """Basic homepage scraping if RSS unavailable"""
        resp = self._get(base_url)
        if not resp:
            return []
        soup = BeautifulSoup(resp.content, "html.parser")

        articles = []
        for a in soup.select("a")[:80]:
            title = a.get_text().strip()
            link = a.get("href", "")
            if not title or not link:
                continue
            if link.startswith("/"):
                link = base_url.rstrip("/") + link
            if not self.is_valid_article_link(link):
                continue

            content = self.extract_article_content(link, source)
            if len(content) > 200:
                category = self.infer_category_from_url(link, sections)
                articles.append(NewsArticle(
                    title=title, content=content, url=link,
                    source=source, category=category
                ))
            if len(articles) >= self.max_articles_per_source:
                break
        return articles

    # Parallel fetching and processing
    def _parallel_fetch_rss_for_source(self, source: str, rss_feeds: List[str], sections: List[str]) -> List[NewsArticle]:
        """Fetch multiple RSS feeds for a source in parallel"""
        results: List[NewsArticle] = []

        # Prepare tasks
        tasks: List[Tuple[str, Optional[str], str]] = []
        for rss_url in rss_feeds:
            tasks.append((source, None, rss_url))

        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            future_map = {ex.submit(self.fetch_rss_feed, s, sec, url): (s, sec, url) for (s, sec, url) in tasks}
            for fut in as_completed(future_map):
                s, sec, url = future_map[fut]
                try:
                    arts = fut.result()
                    # Infer category from URL if possible
                    for a in arts:
                        if not a.category:
                            a.category = self.infer_category_from_url(a.url, sections)
                    results.extend(arts)
                except Exception as e:
                    logger.warning(f"RSS task failed for {s}:{url} - {e}")

                if len(results) >= self.max_articles_per_source:
                    break

        # Trim per source cap
        return results[:self.max_articles_per_source]

    def _parallel_enrich_articles(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """Fetch full content + metadata in parallel for articles with short bodies"""
        def work(art: NewsArticle) -> NewsArticle:
            try:
                if len(art.content or "") < 200:
                    art.content = self.extract_article_content(art.url, art.source)
                    art = self.enrich_article_metadata_from_html(art.url, art, art.source)
                return art
            except Exception as e:
                logger.warning(f"Enrichment failed: {art.url} - {e}")
                return art

        enriched: List[NewsArticle] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = [ex.submit(work, a) for a in articles]
            for fut in as_completed(futures):
                try:
                    enriched.append(fut.result())
                except Exception as e:
                    logger.warning(f"Enrichment future failed: {e}")

        return enriched

    # Public Orchestration
    def scrape_all_sources(self) -> List[NewsArticle]:
        """Scrape all sources defined in config (parallel where possible)"""
        all_articles: List[NewsArticle] = []

        for source, info in NEWS_SOURCES.items():
            base_url = info.get("base_url")
            sections = info.get("sections", [])
            rss_feeds = info.get("rss", [])
            source_articles: List[NewsArticle] = []

            if rss_feeds:
                source_articles = self._parallel_fetch_rss_for_source(source, rss_feeds, sections)
            else:
                # Try section-based RSS in parallel
                rss_urls = [f"{base_url.rstrip('/')}/{sec}/rss.xml" for sec in sections]
                source_articles = self._parallel_fetch_rss_for_source(source, rss_urls, sections)

            # Fallback to homepage if nothing came from RSS
            if not source_articles:
                logger.info(f"[Fallback] Homepage scrape for {source}")
                source_articles = self.scrape_homepage(source, base_url, sections)

            all_articles.extend(source_articles)

        # Deduplicate by URL
        # Deuplicate while preserving order
        seen = set()
        unique_articles = [a for a in all_articles if not (a.url in seen or seen.add(a.url)) and a.url]

        # Enrich in parallel (fetch full body + missing metadata)
        unique_articles = self._parallel_enrich_articles(unique_articles)

        self.articles = unique_articles
        return unique_articles

    # Helpers
    # Simple heuristic to filter out non-article links
    def is_valid_article_link(self, url: str) -> bool:
        blacklist = ["auth", "login", "signup", "account", "privacy", "terms", "#"]
        return not any(b in url for b in blacklist)

    def infer_category_from_url(self, url: str, sections: List[str]) -> Optional[str]:
        parts = url.lower().split("/")
        for section in sections:
            sec_parts = section.lower().split("/")
            if all(sp in parts for sp in sec_parts):
                return section.split("/")[-1]  # return last part for clarity
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
    counts: Dict[str, int] = {}
    for a in articles:
        key = f"{a.source}:{a.category}"
        counts[key] = counts.get(key, 0) + 1
    print("\nArticles by source:category")
    for k, v in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
