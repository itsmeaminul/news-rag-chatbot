"""
ChromaDB vector database manager for storing and retrieving news article chunks
"""
import json
import logging
from typing import Dict, List, Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import torch

from .config import VECTORDB_CONFIG, CHROMADB_PATH, PROCESSED_ARTICLES_PATH

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChromaManager:
    """ChromaDB manager for vector operations"""

    def __init__(self):
        # Initialize ChromaDB client with telemetry disabled
        self.client = chromadb.PersistentClient(
            path=str(CHROMADB_PATH),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Initialize embedding model with proper device handling
        self.embedding_model = self._initialize_embedding_model()

        # Get or create collection
        self.collection = self._get_or_create_collection()

        logger.info("ChromaDB manager initialized successfully")
    
    def _initialize_embedding_model(self):
        """Initialize embedding model with proper device handling"""
        try:
            # Set device and check CUDA availability
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"Using device: {device}")
            
            # Load model with explicit device mapping
            model = SentenceTransformer(
                VECTORDB_CONFIG['embedding_model'],
                device=device
            )
            
            # Ensure model is properly loaded and has weights
            if hasattr(model, 'eval'):
                model.eval()  # Set to evaluation mode
                
            logger.info(f"Embedding model loaded successfully on {device}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            # Fallback: try without device specification
            try:
                model = SentenceTransformer(VECTORDB_CONFIG['embedding_model'])
                logger.info("Embedding model loaded without device specification")
                return model
            except Exception as fallback_error:
                logger.error(f"Fallback model loading also failed: {fallback_error}")
                raise

    def _get_or_create_collection(self):
        """Get existing collection or create new one"""
        try:
            collection = self.client.get_collection(VECTORDB_CONFIG['collection_name'])
            logger.info(f"Retrieved existing collection: {VECTORDB_CONFIG['collection_name']}")
        except Exception:
            collection = self.client.create_collection(
                name=VECTORDB_CONFIG['collection_name'],
                metadata={"description": "BD News articles collection with title emphasis"}
            )
            logger.info(f"Created new collection: {VECTORDB_CONFIG['collection_name']}")

        return collection

    def _prepare_metadata(self, metadata: Dict) -> Dict:
        """Prepare metadata for ChromaDB (only string/int/float values)"""
        clean_metadata = {}

        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                # Convert bool to str for safety
                if isinstance(value, bool):
                    clean_metadata[key] = str(value)
                else:
                    clean_metadata[key] = value
            elif isinstance(value, list):
                clean_metadata[key] = ", ".join(map(str, value))
            elif value is None:
                continue
            else:
                clean_metadata[key] = str(value)

        return clean_metadata

    def add_chunks(self, chunks: List[Dict]) -> bool:
        """Add text chunks to the vector database with title emphasis"""
        if not chunks:
            logger.warning("No chunks provided to add")
            return False

        try:
            texts = []
            metadatas = []
            ids = []

            logger.info(f"Preparing {len(chunks)} chunks (with title focus) for embedding...")

            for chunk in chunks:
                texts.append(chunk['text'])
                metadatas.append(self._prepare_metadata(chunk['metadata']))
                ids.append(chunk['chunk_id'])

            # Generate embeddings with proper error handling
            logger.info("Generating embeddings with title awareness...")
            try:
                embeddings = self.embedding_model.encode(
                    texts,
                    show_progress_bar=True,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    device=self.embedding_model.device  # Explicit device
                ).tolist()
            except Exception as embed_error:
                logger.error(f"Embedding generation failed: {embed_error}")
                # Try without device specification as fallback
                embeddings = self.embedding_model.encode(
                    texts,
                    show_progress_bar=True,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                ).tolist()

            # Add to collection in batches to avoid memory issues
            batch_size = 50  # Reduced batch size for stability
            for i in range(0, len(texts), batch_size):
                end_idx = min(i + batch_size, len(texts))

                self.collection.add(
                    documents=texts[i:end_idx],
                    embeddings=embeddings[i:end_idx],
                    metadatas=metadatas[i:end_idx],
                    ids=ids[i:end_idx]
                )

                logger.info(f"Added batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")

            logger.info(f"Successfully added {len(chunks)} chunks to vector database")
            return True

        except Exception as e:
            logger.error(f"Error adding chunks to vector database: {e}")
            return False

    def search_similar(self, query: str, n_results: int = None,
                       where: Optional[Dict] = None) -> List[Dict]:
        """Search for similar chunks using vector similarity"""
        if n_results is None:
            n_results = VECTORDB_CONFIG['max_results']

        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query]).tolist()[0]

            # Prepare search parameters
            search_params = {
                'query_embeddings': [query_embedding],
                'n_results': n_results,
                'include': ['documents', 'metadatas', 'distances']
            }
            
            # Add where clause if provided and valid
            if where and self._is_valid_where_clause(where):
                search_params['where'] = where

            # Search in collection
            results = self.collection.query(**search_params)

            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    result = {
                        'text': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i],
                        'similarity': 1 - results['distances'][0][i]  # Convert distance to similarity
                    }
                    formatted_results.append(result)

            logger.info(f"Found {len(formatted_results)} similar chunks for query")
            return formatted_results

        except Exception as e:
            logger.error(f"Error searching vector database: {e}")
            return []

    def search_with_title_emphasis(self, query: str, n_results: int = None) -> List[Dict]:
        """Enhanced search that gives higher weight to title matches"""
        if n_results is None:
            n_results = VECTORDB_CONFIG['max_results']
        
        try:
            # Step 1: Search for title chunks specifically
            title_chunk_results = []
            try:
                title_results = self.collection.get(
                    where={"is_title_chunk": {"$eq": "True"}},
                    limit=n_results * 2,
                    include=['documents', 'metadatas']
                )
                
                # Filter title chunks by query relevance
                if title_results['documents']:
                    for i in range(len(title_results['documents'])):
                        doc_text = title_results['documents'][i]
                        metadata = title_results['metadatas'][i]
                        
                        # Check if query terms appear in the title chunk
                        query_words = set(query.lower().split())
                        doc_words = set(doc_text.lower().split())
                        overlap = len(query_words.intersection(doc_words))
                        
                        if overlap > 0:
                            similarity = min(0.95, overlap / len(query_words) + 0.3)
                            result = {
                                'text': doc_text,
                                'metadata': metadata,
                                'similarity': similarity,
                                'match_type': 'title_chunk'
                            }
                            title_chunk_results.append(result)
            except Exception as e:
                logger.warning(f"Error searching title chunks: {e}")
            
            # Step 2: Regular semantic search
            semantic_results = self.search_similar(query, n_results * 2)
            
            # Step 3: Boost results that have high title relevance
            for result in semantic_results:
                title_relevance = float(result.get('metadata', {}).get('title_relevance_score', 0))
                is_title_chunk = result.get('metadata', {}).get('is_title_chunk') == 'True'
                
                if is_title_chunk:
                    result['similarity'] = min(0.98, result.get('similarity', 0) + 0.25)
                    result['match_type'] = 'title_chunk_semantic'
                elif title_relevance > 0.5:
                    result['similarity'] = min(0.95, result.get('similarity', 0) + 0.15)
                    result['match_type'] = 'high_title_relevance'
                elif title_relevance > 0.3:
                    result['similarity'] = min(0.9, result.get('similarity', 0) + 0.1)
                    result['match_type'] = 'medium_title_relevance'
                else:
                    result['match_type'] = 'content_match'
            
            # Step 4: Combine and deduplicate results
            all_results = title_chunk_results + semantic_results
            
            # Deduplicate by chunk_id
            seen_chunks = set()
            final_results = []
            
            for result in all_results:
                chunk_id = result.get('metadata', {}).get('chunk_id')
                if chunk_id and chunk_id not in seen_chunks:
                    seen_chunks.add(chunk_id)
                    final_results.append(result)
                elif not chunk_id:  # Fallback for results without chunk_id
                    final_results.append(result)
            
            # Sort by similarity (title matches first)
            final_results.sort(key=lambda x: x.get('similarity', 0), reverse=True)
            
            logger.info(f"Title-emphasized search found {len(final_results)} results")
            return final_results[:n_results]
            
        except Exception as e:
            logger.error(f"Error in title-emphasized search: {e}")
            # Fallback to regular search
            return self.search_similar(query, n_results)

    def search_by_title_keywords(self, query: str, n_results: int = 10) -> List[Dict]:
        """Search specifically for articles whose titles contain query keywords"""
        try:
            query_words = [word.strip().lower() for word in query.split() if len(word.strip()) > 2]
            
            if not query_words:
                return []
            
            # Get all articles and check title matches
            all_results = self.collection.get(
                include=['documents', 'metadatas']
            )
            
            matching_results = []
            if all_results['documents']:
                for i in range(len(all_results['documents'])):
                    metadata = all_results['metadatas'][i]
                    title = metadata.get('title', '').lower()
                    doc_text = all_results['documents'][i]
                    
                    # Score based on how many query words appear in title
                    matches = sum(1 for word in query_words if word in title)
                    
                    # Also check title keywords in metadata
                    title_keywords = metadata.get('title_keywords', '')
                    if isinstance(title_keywords, str):
                        title_keywords_lower = title_keywords.lower()
                        keyword_matches = sum(1 for word in query_words if word in title_keywords_lower)
                        matches += keyword_matches * 0.5  # Weight keyword matches slightly less
                    
                    if matches > 0:
                        similarity = min(0.95, matches / len(query_words) + 0.2)
                        result = {
                            'text': doc_text,
                            'metadata': metadata,
                            'similarity': similarity,
                            'title_keyword_matches': matches,
                            'match_type': 'title_keyword'
                        }
                        matching_results.append(result)
            
            # Sort by similarity
            matching_results.sort(key=lambda x: x['similarity'], reverse=True)
            
            logger.info(f"Title keyword search found {len(matching_results)} results")
            return matching_results[:n_results]
            
        except Exception as e:
            logger.error(f"Error in title keyword search: {e}")
            return []

    def search_by_exact_title_phrase(self, phrase: str, n_results: int = 5) -> List[Dict]:
        """Search for articles with titles containing the exact phrase"""
        try:
            phrase_lower = phrase.lower().strip()
            if len(phrase_lower) < 3:
                return []
            
            # Get all articles
            all_results = self.collection.get(
                include=['documents', 'metadatas']
            )
            
            exact_matches = []
            if all_results['documents']:
                for i in range(len(all_results['documents'])):
                    metadata = all_results['metadatas'][i]
                    title = metadata.get('title', '').lower()
                    
                    if phrase_lower in title:
                        similarity = 0.98  # Very high similarity for exact matches
                        result = {
                            'text': all_results['documents'][i],
                            'metadata': metadata,
                            'similarity': similarity,
                            'match_type': 'exact_title_phrase'
                        }
                        exact_matches.append(result)
            
            logger.info(f"Exact title phrase search found {len(exact_matches)} results")
            return exact_matches[:n_results]
            
        except Exception as e:
            logger.error(f"Error in exact title phrase search: {e}")
            return []

    def _is_valid_where_clause(self, where: Dict) -> bool:
        """Validate where clause structure for ChromaDB"""
        try:
            # ChromaDB expects either a single condition or $and/$or operators
            if not isinstance(where, dict):
                return False
            
            # Check if it has multiple top-level keys (which causes the error)
            if len(where) > 1:
                # Convert to $and format
                return False
            
            return True
        except Exception:
            return False

    def search_by_metadata(self, where: Dict, n_results: int = None) -> List[Dict]:
        """Search chunks by metadata filters"""
        if n_results is None:
            n_results = VECTORDB_CONFIG['max_results']

        try:
            # Validate where clause
            if not self._is_valid_where_clause(where):
                logger.warning(f"Invalid where clause: {where}")
                return []

            results = self.collection.get(
                where=where,
                limit=n_results,
                include=['documents', 'metadatas']
            )

            formatted_results = []
            if results['documents']:
                for i in range(len(results['documents'])):
                    result = {
                        'text': results['documents'][i],
                        'metadata': results['metadatas'][i],
                        'similarity': 0.8  # Default similarity for metadata matches
                    }
                    formatted_results.append(result)

            logger.info(f"Found {len(formatted_results)} chunks matching metadata filters")
            return formatted_results

        except Exception as e:
            logger.error(f"Error searching by metadata: {e}")
            return []

    def search_by_category(self, category: str, n_results: int = 10) -> List[Dict]:
        """Search by category only"""
        return self.search_by_metadata({"category": {"$eq": category}}, n_results=n_results)

    def search_by_source(self, source: str, n_results: int = 10) -> List[Dict]:
        """Search by source only"""
        return self.search_by_metadata({"source": {"$eq": source}}, n_results=n_results)

    def combined_search(self, query: str, source: str = None, category: str = None, n_results: int = 10) -> List[Dict]:
        """Enhanced combined semantic and metadata search with title prioritization"""
        try:
            results = []
            
            # First, try title-emphasized semantic search
            title_results = self.search_with_title_emphasis(query, n_results=n_results * 2)
            
            # Then filter results manually if needed
            for result in title_results:
                metadata = result['metadata']
                match = True
                
                if source and metadata.get('source', '').lower() != source.lower():
                    match = False
                
                if category and metadata.get('category', '').lower() != category.lower():
                    match = False
                    
                if match:
                    results.append(result)
                    
                if len(results) >= n_results:
                    break
            
            # If we don't have enough results, try broader search
            if len(results) < n_results // 2:
                broader_results = self.search_similar(query, n_results=n_results)
                existing_chunk_ids = {r.get('metadata', {}).get('chunk_id') for r in results}
                
                for result in broader_results:
                    if len(results) >= n_results:
                        break
                        
                    metadata = result['metadata']
                    chunk_id = metadata.get('chunk_id')
                    
                    if chunk_id in existing_chunk_ids:
                        continue
                        
                    match = True
                    if source and metadata.get('source', '').lower() != source.lower():
                        match = False
                    if category and metadata.get('category', '').lower() != category.lower():
                        match = False
                        
                    if match:
                        results.append(result)
            
            logger.info(f"Combined search found {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error in combined search: {e}")
            return []

    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection including title-related metrics"""
        try:
            count = self.collection.count()

            # Get sample of metadata to understand structure
            sample = self.collection.get(limit=min(500, count), include=['metadatas'])

            sources = set()
            categories = set()
            title_chunks = 0
            high_title_relevance = 0

            if sample['metadatas']:
                for metadata in sample['metadatas']:
                    if 'source' in metadata:
                        sources.add(metadata['source'])
                    if 'category' in metadata:
                        categories.add(metadata['category'])
                    if metadata.get('is_title_chunk') == 'True':
                        title_chunks += 1
                    if float(metadata.get('title_relevance_score', 0)) > 0.5:
                        high_title_relevance += 1

            return {
                'total_chunks': count,
                'unique_sources': len(sources),
                'sources': list(sources),
                'unique_categories': len(categories),
                'categories': list(categories),
                'collection_name': self.collection.name,
                'title_chunks': title_chunks,
                'high_title_relevance_chunks': high_title_relevance,
                'title_enhancement_enabled': True
            }

        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}

    def delete_collection(self) -> bool:
        """Delete the entire collection"""
        try:
            self.client.delete_collection(VECTORDB_CONFIG['collection_name'])
            logger.info(f"Deleted collection: {VECTORDB_CONFIG['collection_name']}")
            return True
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            return False

    def reset_collection(self) -> bool:
        """Reset collection by deleting and recreating"""
        try:
            self.delete_collection()
            self.collection = self._get_or_create_collection()
            logger.info("Collection reset successfully")
            return True
        except Exception as e:
            logger.error(f"Error resetting collection: {e}")
            return False

    def load_and_index_articles(self, filepath: str = None) -> bool:
        """Load processed articles and index them in vector database"""
        if not filepath:
            filepath = PROCESSED_ARTICLES_PATH

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                chunks = json.load(f)

            logger.info(f"Loaded {len(chunks)} chunks from {filepath}")

            # Check if collection already has data
            current_count = self.collection.count()
            if current_count > 0:
                logger.warning(f"Collection already contains {current_count} chunks. Use reset_collection() first if needed.")
                return False

            # Add chunks to vector database
            success = self.add_chunks(chunks)

            if success:
                stats = self.get_collection_stats()
                logger.info(f"Indexing complete. Collection stats: {stats}")

            return success

        except FileNotFoundError:
            logger.error(f"Processed articles file not found at {filepath}")
            return False
        except Exception as e:
            logger.error(f"Error loading and indexing articles: {e}")
            return False


def main():
    """Main function to test enhanced ChromaDB operations"""
    # Initialize ChromaDB manager
    db_manager = ChromaManager()

    # Load and index articles
    success = db_manager.load_and_index_articles()

    if success:
        # Test title-focused searches
        test_queries = [
            "Bangladesh cricket team",
            "Prime Minister Sheikh Hasina",
            "Dhaka traffic",
            "coronavirus vaccine",
            "education policy"
        ]

        for query in test_queries:
            print(f"\n{'='*60}")
            print(f"Testing query: '{query}'")
            print(f"{'='*60}")
            
            # Test different search methods
            title_results = db_manager.search_by_title_keywords(query, n_results=3)
            print(f"\nTitle keyword search found {len(title_results)} results")
            
            enhanced_results = db_manager.search_with_title_emphasis(query, n_results=3)
            print(f"Title-emphasized search found {len(enhanced_results)} results")
            
            if enhanced_results:
                print(f"\nTop result:")
                result = enhanced_results[0]
                print(f"Similarity: {result['similarity']:.3f}")
                print(f"Match type: {result.get('match_type', 'unknown')}")
                print(f"Title: {result['metadata'].get('title', 'No title')}")
                print(f"Source: {result['metadata'].get('source', 'Unknown')}")
                print(f"Text preview: {result['text'][:200]}...")

        # Print collection stats
        stats = db_manager.get_collection_stats()
        print(f"\n{'='*60}")
        print("\nCollection Statistics:")
        print(f"{'='*60}")
        for key, value in stats.items():
            print(f"{key}: {value}")
    else:
        print("Failed to load and index articles")


if __name__ == "__main__":
    main()