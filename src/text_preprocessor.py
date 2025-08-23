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
    """Text preprocessing and chunking class"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=TEXT_PROCESSING_CONFIG['chunk_size'],
            chunk_overlap=TEXT_PROCESSING_CONFIG['chunk_overlap'],
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ";", ":", " ", ""]
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

        # Accept shorter texts
        if len(text) < 50:  
            return False

        # Don't reject long news articles
        if len(text) > TEXT_PROCESSING_CONFIG['max_text_length']:
            return False

        # Word count check
        words = text.split()
        if len(words) < 10:  
            return False

        return True
    
    def create_chunks(self, article: Dict) -> List[Dict]:
        """Create text chunks from article with metadata"""
        content = article.get('content', '')
        title = article.get('title', '')
        
        # Clean the content
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
        
        # Extract metadata
        metadata = self.extract_metadata(article)
        metadata['word_count'] = len(cleaned_content.split())
        metadata['char_count'] = len(cleaned_content)
        
        # Create full text for chunking (title + content)
        full_text = f"{cleaned_title}\n\n{cleaned_content}"
        
        # Split into chunks
        text_chunks = self.text_splitter.split_text(full_text)
        
        processed_chunks = []
        for i, chunk in enumerate(text_chunks):
            if self.is_valid_text(chunk):
                chunk_data = {
                    'chunk_id': f"{metadata['source']}_{md5(article['url'].encode()).hexdigest()[:8]}_{i}",
                    'text': chunk,
                    'metadata': {
                        **metadata,
                        'chunk_index': i,
                        'total_chunks': len(text_chunks),
                        'chunk_size': len(chunk),
                        'processed_at': datetime.now().isoformat()
                    }
                }
                processed_chunks.append(chunk_data)
        
        logger.info(f"Created {len(processed_chunks)} chunks for: {cleaned_title[:50]}...")
        return processed_chunks
    
    def process_articles(self, articles: List[Dict]) -> List[Dict]:
        """Process all articles into chunks"""
        logger.info(f"Processing {len(articles)} articles...")
        
        all_chunks = []
        processed_count = 0
        skipped_count = 0
        
        for article in articles:
            try:
                chunks = self.create_chunks(article)
                if chunks:
                    all_chunks.extend(chunks)
                    processed_count += 1
                else:
                    skipped_count += 1
                    
            except Exception as e:
                logger.error(f"Error processing article {article.get('title', 'Unknown')}: {e}")
                skipped_count += 1
                continue
        
        logger.info(f"Processing complete. Processed: {processed_count}, Skipped: {skipped_count}, Total chunks: {len(all_chunks)}")
        
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
        """Get statistics about processed chunks"""
        if not self.processed_chunks:
            return {}
        
        total_chunks = len(self.processed_chunks)
        sources = set(chunk['metadata']['source'] for chunk in self.processed_chunks)
        total_words = sum(chunk['metadata']['word_count'] for chunk in self.processed_chunks)
        total_chars = sum(chunk['metadata']['char_count'] for chunk in self.processed_chunks)
        
        avg_chunk_size = sum(len(chunk['text']) for chunk in self.processed_chunks) / total_chunks
        
        return {
            'total_chunks': total_chunks,
            'unique_sources': len(sources),
            'sources': list(sources),
            'total_words': total_words,
            'total_characters': total_chars,
            'average_chunk_size': round(avg_chunk_size, 2),
            'duplicate_count': len(self.duplicate_hashes)
        }


def main():
    """Main function to run text processing"""
    # Load raw articles
    try:
        with open(RAW_ARTICLES_PATH, 'r', encoding='utf-8', errors='ignore') as f:
            articles = json.load(f)
    except FileNotFoundError:
        logger.error(f"Raw articles file not found at {RAW_ARTICLES_PATH}")
        return
    
    # Process articles
    processor = TextProcessor()
    chunks = processor.process_articles(articles)
    processor.save_processed_chunks()
    
    # Print statistics
    stats = processor.get_processing_stats()
    print("\nProcessing Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()