"""
ChromaDB vector database manager for storing and retrieving news article chunks using semantic retrieval (RAG)
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
    """
    ChromaManager handles all operations related to the ChromaDB vector database for semantic retrieval (RAG).
    
    Responsibilities:
    - Initialize and manage the ChromaDB persistent client and collection.
    - Load and use a SentenceTransformer embedding model for text embeddings.
    - Add article chunks (with metadata) to the vector database.
    - Perform semantic search and metadata-based filtering.
    - Provide combined search (semantic + metadata).
    - Retrieve collection statistics.
    - Reset or delete the collection.
    - Load and index articles from a JSON file.
    """

    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=str(CHROMADB_PATH),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        self.embedding_model = self._initialize_embedding_model()
        self.collection = self._get_or_create_collection()
        logger.info("ChromaDB manager initialized successfully")

    # Initialize the embedding model
    def _initialize_embedding_model(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {device}")
        model = SentenceTransformer(
            VECTORDB_CONFIG['embedding_model'],
            device=device
        )
        if hasattr(model, 'eval'):
            model.eval()
        logger.info(f"Embedding model loaded successfully on {device}")
        return model

    # Get or create the ChromaDB collection
    def _get_or_create_collection(self):
        try:
            collection = self.client.get_collection(VECTORDB_CONFIG['collection_name'])
            logger.info(f"Retrieved existing collection: {VECTORDB_CONFIG['collection_name']}")
        except Exception:
            collection = self.client.create_collection(
                name=VECTORDB_CONFIG['collection_name'],
                metadata={"description": "BD News articles collection"}
            )
            logger.info(f"Created new collection: {VECTORDB_CONFIG['collection_name']}")
        return collection

    # Prepare metadata by ensuring all values are strings or simple types
    def _prepare_metadata(self, metadata: Dict) -> Dict:
        clean_metadata = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                clean_metadata[key] = str(value) if isinstance(value, bool) else value
            elif isinstance(value, list):
                clean_metadata[key] = ", ".join(map(str, value))
            elif value is None:
                continue
            else:
                clean_metadata[key] = str(value)
        return clean_metadata

    # Add chunks to the vector database
    def add_chunks(self, chunks: List[Dict]) -> bool:
        if not chunks:
            logger.warning("No chunks provided to add")
            return False
        try:
            texts = [chunk['text'] for chunk in chunks]
            metadatas = [self._prepare_metadata(chunk['metadata']) for chunk in chunks]
            ids = [chunk['chunk_id'] for chunk in chunks]
            logger.info(f"Preparing {len(chunks)} chunks for embedding...")

            embeddings = self.embedding_model.encode(
                texts,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True
            ).tolist()

            batch_size = 50
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
    
    # Semantic search with optional metadata filtering
    def search_similar(self, query: str, n_results: int = None, where: Optional[Dict] = None) -> List[Dict]:
        """Semantic search using embedding model, with optional metadata filter"""
        if n_results is None:
            n_results = VECTORDB_CONFIG['max_results']
        try:
            query_embedding = self.embedding_model.encode([query]).tolist()[0]
            search_params = {
                'query_embeddings': [query_embedding],
                'n_results': n_results,
                'include': ['documents', 'metadatas', 'distances']
            }
            if where and self._is_valid_where_clause(where):
                search_params['where'] = where
            results = self.collection.query(**search_params)
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    result = {
                        'text': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i],
                        'similarity': 1 - results['distances'][0][i]
                    }
                    formatted_results.append(result)
            logger.info(f"Found {len(formatted_results)} similar chunks for query")
            return formatted_results
        except Exception as e:
            logger.error(f"Error searching vector database: {e}")
            return []
    # Validate the structure of the where clause
    def _is_valid_where_clause(self, where: Dict) -> bool:
        try:
            if not isinstance(where, dict):
                return False
            if len(where) > 1:
                return False
            return True
        except Exception:
            return False

    # Search by metadata only
    def search_by_metadata(self, where: Dict, n_results: int = None) -> List[Dict]:
        if n_results is None:
            n_results = VECTORDB_CONFIG['max_results']
        try:
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
                        'similarity': 0.8
                    }
                    formatted_results.append(result)
            logger.info(f"Found {len(formatted_results)} chunks matching metadata filters")
            return formatted_results
        except Exception as e:
            logger.error(f"Error searching by metadata: {e}")
            return []

    # Convenience methods for common metadata searches
    def search_by_category(self, category: str, n_results: int = 10) -> List[Dict]:
        return self.search_by_metadata({"category": {"$eq": category}}, n_results=n_results)

    # Convenience methods for common metadata searches
    def search_by_source(self, source: str, n_results: int = 10) -> List[Dict]:
        return self.search_by_metadata({"source": {"$eq": source}}, n_results=n_results)

    # Combined semantic and metadata search
    def combined_search(self, query: str, source: str = None, category: str = None, n_results: int = 10) -> List[Dict]:
        """Semantic search with optional metadata filtering"""
        try:
            where = {}
            if source:
                where["source"] = {"$eq": source}
            if category:
                where["category"] = {"$eq": category}
            if where:
                if not self._is_valid_where_clause(where):
                    logger.warning("Combined search: invalid where clause, falling back to semantic only.")
                    where = None
            results = self.search_similar(query, n_results=n_results, where=where if where else None)
            logger.info(f"Combined search found {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"Error in combined search: {e}")
            return []

    # Get collection statistics
    def get_collection_stats(self) -> Dict:
        try:
            count = self.collection.count()
            sample = self.collection.get(limit=min(500, count), include=['metadatas'])
            sources = set()
            categories = set()
            if sample['metadatas']:
                for metadata in sample['metadatas']:
                    if 'source' in metadata:
                        sources.add(metadata['source'])
                    if 'category' in metadata:
                        categories.add(metadata['category'])
            return {
                'total_chunks': count,
                'unique_sources': len(sources),
                'sources': list(sources),
                'unique_categories': len(categories),
                'categories': list(categories),
                'collection_name': self.collection.name,
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}

    # Delete the entire collection
    def delete_collection(self) -> bool:
        try:
            self.client.delete_collection(VECTORDB_CONFIG['collection_name'])
            logger.info(f"Deleted collection: {VECTORDB_CONFIG['collection_name']}")
            return True
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            return False

    # Reset the collection by deleting and recreating it
    def reset_collection(self) -> bool:
        try:
            self.delete_collection()
            self.collection = self._get_or_create_collection()
            logger.info("Collection reset successfully")
            return True
        except Exception as e:
            logger.error(f"Error resetting collection: {e}")
            return False
    
    # Load and index articles from a JSON file
    def load_and_index_articles(self, filepath: str = None) -> bool:
        if not filepath:
            filepath = PROCESSED_ARTICLES_PATH
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            logger.info(f"Loaded {len(chunks)} chunks from {filepath}")
            current_count = self.collection.count()
            if current_count > 0:
                logger.warning(f"Collection already contains {current_count} chunks. Use reset_collection() first if needed.")
                return False
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
    """Main function to test ChromaDB semantic retrieval"""
    db_manager = ChromaManager()
    success = db_manager.load_and_index_articles()
    if success:
        test_queries = [
            "Bangladesh cricket team",
            "Dhaka traffic",
            "coronavirus vaccine",
            "education policy"
        ]
        for query in test_queries:
            print(f"\n{'='*60}")
            print(f"Testing query: '{query}'")
            print(f"{'='*60}")
            results = db_manager.search_similar(query, n_results=3)
            print(f"\nSemantic search found {len(results)} results")
            if results:
                print(f"\nTop result:")
                result = results[0]
                print(f"Similarity: {result['similarity']:.3f}")
                print(f"Title: {result['metadata'].get('title', 'No title')}")
                print(f"Source: {result['metadata'].get('source', 'Unknown')}")
                print(f"Text preview: {result['text'][:200]}...")
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
