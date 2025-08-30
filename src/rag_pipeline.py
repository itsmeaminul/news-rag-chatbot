"""
RAG (Retrieval-Augmented Generation) pipeline for news chatbot
"""
import logging
import os
from typing import Dict, List

import requests
from sentence_transformers import SentenceTransformer, CrossEncoder

from .config import LLM_CONFIG, USE_LOCAL, LLM_PROVIDER, VECTORDB_CONFIG, SEARCH_CONFIG
from .chroma_manager import ChromaManager
from .prompts import PromptTemplates, ResponseMessages

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["CHROMA_TELEMETRY"] = "false"

# Document Retriever
class RAGRetriever:
    """
    Document retriever with embedding-based search.

    This class is responsible for retrieving relevant documents from the vector store
    using semantic similarity based on sentence embeddings.
    """
    
    def __init__(self, vector_store: ChromaManager):
        self.vector_store = vector_store
        self.embedding_model = SentenceTransformer(VECTORDB_CONFIG['embedding_model'])
    
    def retrieve_documents(self, query: str, top_k: int = 20) -> List[Dict]:
        """
        Retrieve candidate documents using semantic similarity.

        Args:
            query (str): The user query.
            top_k (int): Number of top documents to retrieve.

        Returns:
            List[Dict]: List of retrieved documents with scores and metadata.
        """
        try:
            results = self.vector_store.search_similar(query=query, n_results=top_k)
            
            # Add retrieval score and metadata
            for i, result in enumerate(results):
                result['retrieval_rank'] = i + 1
                result['retrieval_score'] = result.get('similarity', 0)
            
            logger.info(f"Retrieved {len(results)} candidate documents")
            return results
            
        except Exception as e:
            logger.error(f"Error in document retrieval: {e}")
            return []

# Re-ranker
class ResultReranker:
    """
    Re-rank retrieved documents using cross-encoder and fallback scoring.

    This class uses a cross-encoder model to re-rank the retrieved documents for better relevance.
    If the cross-encoder is unavailable, it falls back to using the retrieval score.
    """
    
    def __init__(self):
        try:
            self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            logger.info("Cross-encoder loaded for re-ranking")
        except Exception as e:
            self.cross_encoder = None
            logger.warning(f"Could not load cross-encoder: {e}. Using fallback re-ranking.")
    
    def rerank_results(self, candidates: List[Dict], query: str) -> List[Dict]:
        """
        Re-rank candidates using cross-encoder or fallback.

        Args:
            candidates (List[Dict]): List of candidate documents.
            query (str): The user query.

        Returns:
            List[Dict]: Re-ranked list of documents.
        """
        if not candidates:
            return candidates
        
        try:
            # Strategy 1: Cross-encoder re-ranking
            if self.cross_encoder and len(candidates) > 1:
                pairs = [[query, doc['text']] for doc in candidates]
                cross_scores = self.cross_encoder.predict(pairs)
                
                for i, doc in enumerate(candidates):
                    doc['cross_encoder_score'] = float(cross_scores[i])
            else:
                # Fallback: use retrieval score
                for doc in candidates:
                    doc['cross_encoder_score'] = doc.get('similarity', 0.5)
            
            # Final score (just cross-encoder for simplicity)
            for doc in candidates:
                doc['final_score'] = doc['cross_encoder_score']
            
            # Sort by final score
            reranked = sorted(candidates, key=lambda x: x['final_score'], reverse=True)
            
            logger.info(f"Re-ranked {len(reranked)} documents")
            return reranked
            
        except Exception as e:
            logger.error(f"Error in re-ranking: {e}")
            return candidates

# Context Optimizer
class ContextOptimizer:
    """
    Optimize context for LLM input.

    This class selects and formats the most relevant document snippets to fit within the LLM's context window.
    """
    
    def __init__(self):
        self.max_context_length = SEARCH_CONFIG.get('max_context_length', 4000)
    
    def optimize_context(self, documents: List[Dict]) -> str:
        """
        Create optimized context from selected documents.

        Args:
            documents (List[Dict]): List of documents to include in the context.

        Returns:
            str: Optimized context string.
        """
        if not documents:
            return ""
        
        context_parts = []
        current_length = 0
        
        sorted_docs = sorted(documents, key=lambda x: x.get('final_score', 0), reverse=True)
        
        for i, doc in enumerate(sorted_docs):
            snippet = self._format_document_snippet(doc, i + 1)
            
            if current_length + len(snippet) > self.max_context_length:
                remaining_space = self.max_context_length - current_length - 200
                if remaining_space > 500:
                    truncated_text = doc['text'][:remaining_space] + "..."
                    snippet = self._format_document_snippet({**doc, 'text': truncated_text}, i + 1)
                    context_parts.append(snippet)
                break
            
            context_parts.append(snippet)
            current_length += len(snippet)
        
        optimized_context = "\n\n".join(context_parts)
        logger.info(f"Optimized context: {len(optimized_context)} characters from {len(context_parts)} documents")
        return optimized_context
    
    def _format_document_snippet(self, doc: Dict, doc_num: int) -> str:
        """
        Format a document snippet for context.

        Args:
            doc (Dict): Document dictionary.
            doc_num (int): Document number in the context.

        Returns:
            str: Formatted snippet string.
        """
        metadata = doc['metadata']
        
        snippet = f"Document {doc_num}:\n"
        snippet += f"Title: {metadata.get('title', 'Untitled')}\n"
        snippet += f"Source: {metadata.get('source', 'Unknown')}\n"
        snippet += f"Date: {metadata.get('published_date', 'Unknown')}\n"
        snippet += f"Content: {doc['text']}\n"
        
        return snippet


class OllamaLLM:
    """
    Ollama LLM client for local inference.

    This class handles communication with a locally hosted Ollama LLM server for generating responses.
    """

    def __init__(self):
        self.base_url = LLM_CONFIG['base_url']
        self.model_name = LLM_CONFIG['model_name']
        self.temperature = LLM_CONFIG['temperature']
        self.max_tokens = LLM_CONFIG['max_tokens']
        self._test_connection()

    def _test_connection(self) -> bool:
        """
        Test connection to Ollama server.

        Returns:
            bool: True if connection is successful, False otherwise.
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            logger.info("Successfully connected to Ollama server")
            return True
        except Exception as e:
            logger.warning(f"Could not connect to Ollama server: {e}")
            return False

    def generate(self, prompt: str, system_prompt: str = None) -> str:
        """
        Generate response from Ollama.

        Args:
            prompt (str): The user prompt.
            system_prompt (str, optional): System prompt for the LLM.

        Returns:
            str: Generated response.
        """
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                    "top_k": 40,
                    "top_p": 0.9
                }
            }

            if system_prompt:
                payload["system"] = system_prompt

            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=600
            )
            response.raise_for_status()

            result = response.json()
            return result.get('response', '').strip()

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return ResponseMessages.INITIALIZATION_ERROR

class GroqLLM:
    """
    Groq API client.

    This class handles communication with the Groq API for generating responses using hosted LLMs.
    """

    def __init__(self, model_name=None):
        self.api_key = LLM_PROVIDER["groq_api_key"]
        self.model = model_name or LLM_PROVIDER["groq_model"]
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"

        if not self.api_key:
            raise EnvironmentError("GROQ_API_KEY missing in .env")

    def generate(self, prompt: str, system_prompt: str = None) -> str:
        """
        Generate response from Groq API.

        Args:
            prompt (str): The user prompt.
            system_prompt (str, optional): System prompt for the LLM.

        Returns:
            str: Generated response.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1500,
        }

        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return f"Error generating response from Groq API: {e}"

class RAGPipeline:
    """
    RAG pipeline.

    This class orchestrates the entire Retrieval-Augmented Generation process,
    including retrieval, re-ranking, context optimization, and LLM response generation.
    """
    
    def __init__(self):
        self.db_manager = ChromaManager()
        self.retriever = RAGRetriever(self.db_manager)
        self.reranker = ResultReranker()
        self.optimizer = ContextOptimizer()
        self.llm = OllamaLLM() if USE_LOCAL else GroqLLM()
        
        logger.info("RAG pipeline initialized")
    
    def process_query(self, user_query: str) -> str:
        """
        Query processing with better intent detection.

        Args:
            user_query (str): The user's input query.

        Returns:
            str: The generated response.
        """
        try:
            # Check if database has articles
            stats = self.db_manager.get_collection_stats()
            if stats.get('total_chunks', 0) == 0:
                return ResponseMessages.NO_ARTICLES_LOADED

            # Detect query intent
            query_lower = user_query.lower()
            
            # Handle count queries for articles
            if any(phrase in query_lower for phrase in ["how many articles", "how many news", "count of articles"]):
                return self._handle_count_query(user_query, stats)
            
            # Handle yesterday's news specifically
            if "yesterday" in query_lower and any(word in query_lower for word in ["news", "articles", "published"]):
                return self._handle_date_query("yesterday")
            
            # Handle topic-specific counts
            if "how many" in query_lower and "about" in query_lower:
                topic = self._extract_topic(user_query)
                if topic:
                    return self._handle_topic_count(topic)
            
            # Handle summary requests
            if any(word in query_lower for word in ["summary", "summarize"]) and any(word in query_lower for word in ["article", "news", "topic"]):
                topic = self._extract_topic(user_query)
                if topic:
                    return self._generate_summary(topic)
            
            # Handle trending news
            if any(phrase in query_lower for phrase in ["trending news", "trending today", "what is trending"]):
                return self._handle_trending_news()
            
            # Default: semantic search
            return self._handle_general_query(user_query)

        except Exception as e:
            logger.error(f"Error processing query '{user_query}': {e}")
            return ResponseMessages.DATABASE_ERROR

    def _handle_count_query(self, query: str, stats: Dict) -> str:
            """Handle queries asking for article counts"""
            total_chunks = stats.get('total_chunks', 0)
            unique_sources = stats.get('unique_sources', 0)
            sources = stats.get('sources', [])
            
            response = f"📊 **Database Statistics**\n\n"
            response += f"I currently have **{total_chunks}** article chunks from **{unique_sources}** news sources.\n\n"
            
            if sources:
                response += "**Sources include:**\n"
                for source in sources:
                    response += f"• {source}\n"
            
            return response

    def _handle_date_query(self, date_ref: str) -> str:
        """Handle date-specific article queries"""
        from datetime import datetime, timedelta
        
        if date_ref == "yesterday":
            target_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        else:
            target_date = date_ref
        
        # Search for articles from that date
        results = self.db_manager.search_by_metadata({"published_date": {"$contains": target_date}}, n_results=100)
        
        count = len(results)
        if count == 0:
            return f"No articles found from {date_ref}. Note: The articles in the database may not include yesterday's news if they haven't been scraped recently."
        
        # Group by source
        sources = {}
        for result in results:
            source = result['metadata'].get('source', 'Unknown')
            sources[source] = sources.get(source, 0) + 1
        
        response = f"📅 **Articles from {date_ref}**\n\n"
        response += f"Found **{count}** articles:\n\n"
        for source, cnt in sources.items():
            response += f"• {source}: {cnt} articles\n"
        
        return response

    def _handle_topic_count(self, topic: str) -> str:
        """Handle count queries for specific topics"""
        results = self.db_manager.search_similar(topic, n_results=50)
        
        # Filter by relevance threshold
        relevant_results = [r for r in results if r.get('similarity', 0) > 0.3]
        count = len(relevant_results)
        
        if count == 0:
            return f"No articles found about '{topic}'."
        
        # Group by source
        sources = {}
        for result in relevant_results:
            source = result['metadata'].get('source', 'Unknown')
            sources[source] = sources.get(source, 0) + 1
        
        response = f"📰 **Articles about '{topic}'**\n\n"
        response += f"Found **{count}** relevant articles:\n\n"
        for source, cnt in sorted(sources.items(), key=lambda x: x[1], reverse=True):
            response += f"• {source}: {cnt} articles\n"
        
        # Add sample titles
        response += "\n**Sample articles:**\n"
        for i, result in enumerate(relevant_results[:3], 1):
            title = result['metadata'].get('title', 'Untitled')
            response += f"{i}. {title}\n"
        
        return response

    def _generate_summary(self, topic: str) -> str:
        """Generate summary for a specific topic"""
        # Retrieve relevant articles
        results = self.db_manager.search_similar(topic, n_results=10)
        
        if not results:
            return f"No articles found about '{topic}' to summarize."
        
        # Filter highly relevant results
        relevant_results = [r for r in results if r.get('similarity', 0) > 0.25]
        
        if not relevant_results:
            return f"Found some articles but they don't seem closely related to '{topic}'."
        
        # Optimize context for summary
        context = self.optimizer.optimize_context(relevant_results[:5])
        
        # Generate summary using LLM
        prompt = PromptTemplates.get_summary_prompt(topic, context)
        summary = self.llm.generate(prompt, PromptTemplates.SYSTEM_PROMPT)
        
        # Add article references
        response = f"📝 **Summary: {topic}**\n\n{summary}\n\n"
        response += "**Based on articles:**\n"
        for i, result in enumerate(relevant_results[:3], 1):
            title = result['metadata'].get('title', 'Untitled')
            source = result['metadata'].get('source', 'Unknown')
            response += f"{i}. {title} - {source}\n"
        
        return response

    def _handle_trending_news(self) -> str:
        """Handle trending news queries"""
        # Get recent articles (you might want to sort by date)
        results = self.db_manager.search_similar("bangladesh news today trending", n_results=15)
        
        if not results:
            return "No trending news articles available. Please scrape recent news first."
        
        # Group by category/topic
        categories = {}
        for result in results:
            category = result['metadata'].get('category', 'General')
            if category not in categories:
                categories[category] = []
            categories[category].append(result)
        
        # Generate response
        context = self.optimizer.optimize_context(results[:10])
        prompt = PromptTemplates.get_trending_prompt(context)
        trending_summary = self.llm.generate(prompt, PromptTemplates.SYSTEM_PROMPT)
        
        response = f"📈 **Trending News Today**\n\n{trending_summary}\n\n"
        
        return response

    def _extract_topic(self, query: str) -> str:
        """Extract topic from query"""
        # Remove common query words
        stop_phrases = ["how many", "articles", "news", "about", "tell me", "summary of", 
                    "summarize", "what is the", "were published", "show me"]
        
        topic = query.lower()
        for phrase in stop_phrases:
            topic = topic.replace(phrase, "")
        
        # Clean up
        topic = topic.strip().strip("?.,!").strip()
        
        # If topic is too short, return None
        if len(topic) < 3:
            return None
        
        return topic

    def _handle_general_query(self, user_query: str) -> str:
        """Handle general queries with standard RAG pipeline"""
        # Your existing RAG pipeline code
        print("📚 Retrieving relevant documents...")
        candidates = self.retriever.retrieve_documents(user_query, top_k=20)
        if not candidates:
            return ResponseMessages.get_no_relevant_articles(user_query)

        print("📊 Re-ranking results...")
        ranked_docs = self.reranker.rerank_results(candidates, user_query)

        print("🎯 Selecting top documents...")
        top_docs = ranked_docs[:10]

        print("📝 Optimizing context...")
        context = self.optimizer.optimize_context(top_docs)

        print("🤖 Generating response...")
        prompt = PromptTemplates.get_general_search_prompt(user_query, context)
        response = self.llm.generate(prompt, PromptTemplates.SYSTEM_PROMPT)
        
        sources_section = self._format_sources_section(top_docs)
        
        return response + sources_section

    def _format_sources_section(self, documents: List[Dict]) -> str:
        if not documents:
            return ""
        
        sources_str = "\n\n**Sources:**\n"
        for i, doc in enumerate(documents[:5], 1):
            metadata = doc['metadata']
            sources_str += f"{i}. **{metadata.get('title', 'Untitled')}** - {metadata.get('source', 'Unknown')}\n"
            sources_str += f"   Date: {metadata.get('published_date', 'Unknown')}\n"
            sources_str += f"   Link: {metadata.get('url', 'N/A')}\n"
            sources_str += f"   Snippet: {doc['text'][:200]}...\n\n"
        
        return sources_str

def main():
    print("🚀 Starting Simplified RAG Pipeline...")
    
    try:
        pipeline = RAGPipeline()
        test_queries = [
            "What is trending news today?",
            "Tell me about Bangladesh politics",
            "Recent news from Daily Star about sports",
            "Summary of coronavirus vaccine news",
            "Bangladesh cricket team performance"
        ]

        for query in test_queries:
            print(f"\n📝 Query: {query}")
            print("=" * 60)
            response = pipeline.process_query(query)
            print(f"🤖 Response: {response}")
            print("=" * 80)
            
    except Exception as e:
        print(f"Error running pipeline: {e}")


if __name__ == "__main__":
    main()
