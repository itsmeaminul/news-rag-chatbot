"""
Text preprocessing module for cleaning and chunking news articles
"""
import json
import logging
import re
from datetime import datetime
from hashlib import md5
from typing import Dict, List, Set

from langchain.text_splitter import RecursiveCharacterTextSplitter

from .config import TEXT_PROCESSING_CONFIG, RAW_ARTICLES_PATH, PROCESSED_ARTICLES_PATH

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextProcessor:
    """Enhanced text preprocessing and chunking class with title emphasis"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=TEXT_PROCESSING_CONFIG['chunk_size'],
            chunk_overlap=TEXT_PROCESSING_CONFIG['chunk_overlap'],
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ";", ":", " ", ""]
        )
        
        # Title-specific splitter for smaller, focused chunks
        self.title_splitter = RecursiveCharacterTextSplitter(
            chunk_size=TEXT_PROCESSING_CONFIG.get('title_chunk_size', 600),
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", " ", ""]
        )
        
        self.processed_chunks = []
        self.duplicate_hashes: Set[str] = set()
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove HTML entities and special characters
        text = re.sub(r'&[a-zA-Z0-9#]+;', '', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove excessive punctuation
        text = re.sub(r'[.]{3,}', '...', text)
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        
        # Remove social media handles and hashtags
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#\w+', '', text)
        
        # Remove common boilerplate text
        boilerplate_patterns = [
            r'subscribe to our newsletter',
            r'follow us on',
            r'share this article',
            r'read more:',
            r'related articles:',
            r'advertisement',
            r'ads by google',
            r'click here to',
            r'terms and conditions',
            r'privacy policy'
        ]
        
        for pattern in boilerplate_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Clean up whitespace again
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_metadata(self, article: Dict) -> Dict:
        """Extract and clean metadata from article"""
        metadata = {
            'title': self.clean_text(article.get('title', '')),
            'source': article.get('source', ''),
            'url': article.get('url', ''),
            'author': self.clean_text(article.get('author', '') or ''),
            'published_date': article.get('published_date', ''),
            'scraped_at': article.get('scraped_at', ''),
            'category': article.get('category', ''),
            'word_count': 0,
            'char_count': 0
        }
        
        return metadata
    
    def is_duplicate(self, text: str) -> bool:
        """Check if text is a duplicate using content hash"""
        text_hash = md5(text.encode('utf-8')).hexdigest()
        
        if text_hash in self.duplicate_hashes:
            return True
        
        self.duplicate_hashes.add(text_hash)
        return False
    
    def is_valid_text(self, text: str) -> bool:
        """Check if text is valid for processing"""
        if not text or len(text.strip()) == 0:
            return False

        if len(text) < 50:  
            return False

        if len(text) > TEXT_PROCESSING_CONFIG['max_text_length']:
            return False

        words = text.split()
        if len(words) < 10:  
            return False

        return True
    
    def extract_title_keywords(self, title: str) -> List[str]:
        """Extract keywords from title for enhanced searchability"""
        if not title:
            return []
        
        # Remove common title words
        title_stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'has', 'have', 'had', 'will', 'would', 'could', 'should', 'may', 'might',
            'says', 'said', 'tells', 'told', 'asks', 'asked'
        }
        
        # Extract words and filter
        words = re.findall(r'\b[a-zA-Z]{3,}\b', title.lower())
        keywords = [word for word in words if word not in title_stopwords]
        
        return keywords[:6]  # Limit to top 6 keywords
    
    def calculate_title_relevance(self, chunk_text: str, title: str) -> float:
        """Calculate how relevant a chunk is to the article title"""
        if not title:
            return 0.0
        
        title_words = set(word.lower() for word in title.split() if len(word) > 2)
        chunk_words = set(word.lower() for word in chunk_text.split() if len(word) > 2)
        
        if not title_words:
            return 0.0
        
        # Calculate overlap
        overlap = len(title_words.intersection(chunk_words))
        relevance = overlap / len(title_words)
        
        # Boost if title appears verbatim in chunk
        if title.lower() in chunk_text.lower():
            relevance += 0.5
        
        # Boost if multiple title words appear close together
        title_phrase_patterns = [
            ' '.join(title_words) for title_words in [title.split()[i:i+3] for i in range(len(title.split())-2)]
        ]
        for pattern in title_phrase_patterns:
            if len(pattern) > 10 and pattern.lower() in chunk_text.lower():
                relevance += 0.3
                break
        
        return min(relevance, 1.0)
    
    def create_chunks(self, article: Dict) -> List[Dict]:
        """Create text chunks from article with enhanced title emphasis"""
        content = article.get('content', '')
        title = article.get('title', '')
        
        # Clean the content and title
        cleaned_content = self.clean_text(content)
        cleaned_title = self.clean_text(title)
        
        # Validate content
        if not self.is_valid_text(cleaned_content):
            logger.warning(f"Invalid content for article: {cleaned_title[:50]}...")
            return []
        
        # Check for duplicates
        if self.is_duplicate(cleaned_content):
            logger.info(f"Duplicate content detected: {cleaned_title[:50]}...")
            return []
        
        # Extract metadata and title keywords
        metadata = self.extract_metadata(article)
        metadata['word_count'] = len(cleaned_content.split())
        metadata['char_count'] = len(cleaned_content)
        
        title_keywords = self.extract_title_keywords(cleaned_title)
        
        processed_chunks = []
        chunk_counter = 0
        
        # Strategy 1: Create dedicated title summary chunk
        if TEXT_PROCESSING_CONFIG.get('create_title_chunks', True):
            title_summary = f"HEADLINE: {cleaned_title}\n\n"
            title_summary += f"ARTICLE TITLE: {cleaned_title}\n\n"
            title_summary += f"KEY TOPICS: {', '.join(title_keywords)}\n\n"
            title_summary += f"SUMMARY: {cleaned_content[:800]}"
            
            if len(title_summary) > 100:
                title_chunk_data = {
                    'chunk_id': f"{metadata['source']}_{md5(article['url'].encode()).hexdigest()[:8]}_title",
                    'text': title_summary,
                    'metadata': {
                        **metadata,
                        'chunk_index': chunk_counter,
                        'chunk_type': 'title_summary',
                        'is_title_chunk': True,
                        'title_keywords': title_keywords,
                        'title_relevance_score': 1.0,
                        'processed_at': datetime.now().isoformat()
                    }
                }
                processed_chunks.append(title_chunk_data)
                chunk_counter += 1
        
        # Strategy 2: Create title-emphasized content chunks
        title_emphasized_text = f"{cleaned_title}\n\n{cleaned_title}\n\n{cleaned_content}"
        
        # Use regular splitter for main content
        text_chunks = self.text_splitter.split_text(title_emphasized_text)
        
        for i, chunk in enumerate(text_chunks):
            if self.is_valid_text(chunk):
                # Calculate title relevance score
                title_relevance = self.calculate_title_relevance(chunk, cleaned_title)
                
                # Determine chunk type
                chunk_type = 'content'
                if title_relevance > 0.7:
                    chunk_type = 'title_relevant_content'
                elif cleaned_title.lower() in chunk.lower()[:200]:
                    chunk_type = 'title_opening_content'
                
                chunk_data = {
                    'chunk_id': f"{metadata['source']}_{md5(article['url'].encode()).hexdigest()[:8]}_{chunk_counter}",
                    'text': chunk,
                    'metadata': {
                        **metadata,
                        'chunk_index': chunk_counter,
                        'chunk_type': chunk_type,
                        'title_keywords': title_keywords,
                        'title_relevance_score': title_relevance,
                        'contains_title_keywords': title_relevance > 0.3,
                        'chunk_size': len(chunk),
                        'processed_at': datetime.now().isoformat()
                    }
                }
                processed_chunks.append(chunk_data)
                chunk_counter += 1
        
        # Strategy 3: Create keyword-focused chunks if we have strong title keywords
        if len(title_keywords) >= 2:
            keyword_text = f"Key Topics: {', '.join(title_keywords)}\n"
            keyword_text += f"Article: {cleaned_title}\n\n"
            
            # Find content sections that contain multiple title keywords
            content_sentences = cleaned_content.split('. ')
            keyword_rich_content = []
            
            for sentence in content_sentences:
                keyword_count = sum(1 for keyword in title_keywords if keyword.lower() in sentence.lower())
                if keyword_count >= 2:
                    keyword_rich_content.append(sentence)
            
            if keyword_rich_content:
                keyword_text += '. '.join(keyword_rich_content[:5])
                
                if len(keyword_text) > 200:
                    keyword_chunk_data = {
                        'chunk_id': f"{metadata['source']}_{md5(article['url'].encode()).hexdigest()[:8]}_keywords",
                        'text': keyword_text,
                        'metadata': {
                            **metadata,
                            'chunk_index': chunk_counter,
                            'chunk_type': 'keyword_focused',
                            'title_keywords': title_keywords,
                            'title_relevance_score': 0.9,
                            'contains_title_keywords': True,
                            'chunk_size': len(keyword_text),
                            'processed_at': datetime.now().isoformat()
                        }
                    }
                    processed_chunks.append(keyword_chunk_data)
                    chunk_counter += 1
        
        # Update total chunks count for all chunks
        for chunk in processed_chunks:
            chunk['metadata']['total_chunks'] = len(processed_chunks)
        
        logger.info(f"Created {len(processed_chunks)} chunks (with title emphasis) for: {cleaned_title[:50]}...")
        return processed_chunks
    
    def get_stop_words(self) -> set:
        """Return set of stop words for title keyword extraction"""
        return {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'has', 'have', 'had', 'will', 'would', 'could', 'should', 'may', 'might'
        }
    
    def process_articles(self, articles: List[Dict]) -> List[Dict]:
        """Process all articles into chunks with title emphasis"""
        logger.info(f"Processing {len(articles)} articles with title emphasis...")
        
        all_chunks = []
        processed_count = 0
        skipped_count = 0
        title_chunks_created = 0
        
        for article in articles:
            try:
                chunks = self.create_chunks(article)
                if chunks:
                    all_chunks.extend(chunks)
                    processed_count += 1
                    
                    # Count title chunks
                    title_chunks_created += sum(1 for chunk in chunks 
                                              if chunk['metadata'].get('is_title_chunk', False))
                else:
                    skipped_count += 1
                    
            except Exception as e:
                logger.error(f"Error processing article {article.get('title', 'Unknown')}: {e}")
                skipped_count += 1
                continue
        
        logger.info(f"Processing complete. Processed: {processed_count}, Skipped: {skipped_count}")
        logger.info(f"Total chunks: {len(all_chunks)}, Title chunks: {title_chunks_created}")
        
        self.processed_chunks = all_chunks
        return all_chunks
    
    def save_processed_chunks(self, filepath: str = None) -> None:
        """Save processed chunks to JSON file"""
        if not filepath:
            filepath = PROCESSED_ARTICLES_PATH
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.processed_chunks, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved {len(self.processed_chunks)} processed chunks to {filepath}")
    
    def load_processed_chunks(self, filepath: str = None) -> List[Dict]:
        """Load processed chunks from JSON file"""
        if not filepath:
            filepath = PROCESSED_ARTICLES_PATH
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            
            self.processed_chunks = chunks
            logger.info(f"Loaded {len(chunks)} processed chunks from {filepath}")
            return chunks
            
        except FileNotFoundError:
            logger.warning(f"No processed chunks file found at {filepath}")
            return []
        except Exception as e:
            logger.error(f"Error loading processed chunks from {filepath}: {e}")
            return []
    
    def get_processing_stats(self) -> Dict:
        """Get enhanced statistics about processed chunks"""
        if not self.processed_chunks:
            return {}
        
        total_chunks = len(self.processed_chunks)
        sources = set(chunk['metadata']['source'] for chunk in self.processed_chunks)
        total_words = sum(chunk['metadata']['word_count'] for chunk in self.processed_chunks)
        total_chars = sum(chunk['metadata']['char_count'] for chunk in self.processed_chunks)
        
        # Title-specific stats
        title_chunks = sum(1 for chunk in self.processed_chunks 
                          if chunk['metadata'].get('is_title_chunk', False))
        high_title_relevance = sum(1 for chunk in self.processed_chunks 
                                 if chunk['metadata'].get('title_relevance_score', 0) > 0.5)
        
        avg_chunk_size = sum(len(chunk['text']) for chunk in self.processed_chunks) / total_chunks
        
        return {
            'total_chunks': total_chunks,
            'unique_sources': len(sources),
            'sources': list(sources),
            'total_words': total_words,
            'total_characters': total_chars,
            'average_chunk_size': round(avg_chunk_size, 2),
            'duplicate_count': len(self.duplicate_hashes),
            'title_chunks_created': title_chunks,
            'high_title_relevance_chunks': high_title_relevance,
            'title_enhancement_ratio': round(title_chunks / total_chunks, 3)
        }


def main():
    """Main function to run enhanced text processing"""
    # Load raw articles
    try:
        with open(RAW_ARTICLES_PATH, 'r', encoding='utf-8', errors='ignore') as f:
            articles = json.load(f)
    except FileNotFoundError:
        logger.error(f"Raw articles file not found at {RAW_ARTICLES_PATH}")
        return
    
    # Process articles with title emphasis
    processor = TextProcessor()
    chunks = processor.process_articles(articles)
    processor.save_processed_chunks()
    
    # Print enhanced statistics
    stats = processor.get_processing_stats()
    print("\nEnhanced Processing Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()