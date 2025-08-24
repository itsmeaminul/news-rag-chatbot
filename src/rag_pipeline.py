"""
Improved RAG (Retrieval-Augmented Generation) pipeline for news chatbot
"""
import logging
import re
import os
from typing import Dict, List, Optional

import requests

from .config import LLM_CONFIG, VECTORDB_CONFIG
from .chroma_manager import ChromaManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["CHROMA_TELEMETRY"] = "false"

class OllamaLLM:
    """Ollama LLM client for local inference"""

    def __init__(self):
        self.base_url = LLM_CONFIG['base_url']
        self.model_name = LLM_CONFIG['model_name']
        self.temperature = LLM_CONFIG['temperature']
        self.max_tokens = LLM_CONFIG['max_tokens']

        # Test connection
        self._test_connection()

    def _test_connection(self) -> bool:
        """Test connection to Ollama server"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            logger.info("Successfully connected to Ollama server")
            return True
        except Exception as e:
            logger.warning(f"Could not connect to Ollama server: {e}")
            return False

    def generate(self, prompt: str, system_prompt: str = None) -> str:
        """Generate response using Ollama"""
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
            return "I apologize, but I'm having trouble generating a response right now. Please try again later."


class ImprovedQueryProcessor:
    """Improved query processor with better intent extraction"""

    def __init__(self):
        # News sources mapping
        self.source_mapping = {
            'prothom alo': 'prothom_alo',
            'prothomalo': 'prothom_alo',
            'daily star': 'daily_star',
            'thedailystar': 'daily_star',
            'bdnews24': 'bdnews24',
            'bdnews': 'bdnews24'
        }
        
        # Category mapping
        self.category_mapping = {
            'sports': ['sports', 'sport', 'cricket', 'football', 'soccer'],
            'bangladesh': ['bangladesh', 'politics', 'govt', 'government', 'political'],
            'international': ['international', 'world', 'global'],
            'entertainment': ['entertainment', 'culture', 'movie', 'cinema'],
            'business': ['business', 'economy', 'economic', 'trade', 'finance'],
            'lifestyle': ['lifestyle', 'health', 'fashion']
        }

    def extract_intent_and_entities(self, query: str) -> Dict:
        """Extract intent and entities from user query with improved accuracy"""
        query_lower = query.lower().strip()
        
        intent_data = {
            'intent': 'general_search',
            'topic': None,
            'source': None,
            'category': None,
            'search_terms': [],
            'original_query': query
        }

        # Intent detection patterns
        if re.search(r'\b(how many|count|number of|total)\b.*\barticles?\b', query_lower):
            if re.search(r'\babout\b', query_lower):
                intent_data['intent'] = 'count_topic'
                # Extract topic after "about"
                topic_match = re.search(r'\babout\s+([^?]+)', query_lower)
                if topic_match:
                    intent_data['topic'] = topic_match.group(1).strip()
            else:
                intent_data['intent'] = 'count_articles'
        
        elif re.search(r'\b(summary|summarize|tell me about|what.*about)\b', query_lower):
            intent_data['intent'] = 'summary'
            # Extract topic for summary
            for pattern in [r'summary.*?of\s+([^?]+)', r'summarize\s+([^?]+)', r'tell me about\s+([^?]+)', r'what.*?about\s+([^?]+)']:
                match = re.search(pattern, query_lower)
                if match:
                    intent_data['topic'] = match.group(1).strip()
                    break
        
        elif re.search(r'\b(trending|latest|recent|today.*news|current.*news|top.*news)\b', query_lower):
            intent_data['intent'] = 'trending'

        # Extract source (be more careful with source extraction)
        source_found = None
        for source_key, source_variants in [
            ('prothom_alo', ['prothom alo', 'prothomalo']),
            ('daily_star', ['daily star', 'thedailystar']),
            ('bdnews24', ['bdnews24', 'bdnews'])
        ]:
            for variant in source_variants:
                if variant in query_lower:
                    source_found = source_key
                    break
            if source_found:
                break
        
        if source_found:
            intent_data['source'] = source_found

        # Extract category (improved category detection)
        category_found = None
        for category, keywords in self.category_mapping.items():
            for keyword in keywords:
                if re.search(r'\b' + re.escape(keyword) + r'\b', query_lower):
                    category_found = category
                    break
            if category_found:
                break
        
        if category_found:
            intent_data['category'] = category_found

        # Extract search terms (clean up the query for better searching)
        search_terms = self._extract_search_terms(query_lower, intent_data)
        intent_data['search_terms'] = search_terms

        logger.info(f"Extracted intent: {intent_data}")
        return intent_data

    def _extract_search_terms(self, query: str, intent_data: Dict) -> List[str]:
        """Extract meaningful search terms from query"""
        # Remove common stop words and intent-specific words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'how', 'many', 'what', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'tell', 'me', 'about', 'give', 'show', 'summary', 'summarize', 'latest', 'recent',
            'trending', 'today', 'news', 'articles', 'article'
        }
        
        # Remove source and category terms that we already extracted
        if intent_data.get('source'):
            stop_words.update(['prothom', 'alo', 'daily', 'star', 'bdnews24', 'bdnews'])
        
        # Split and clean
        words = re.findall(r'\b\w+\b', query)
        search_terms = []
        
        for word in words:
            word = word.lower()
            if len(word) > 2 and word not in stop_words:
                search_terms.append(word)
        
        return search_terms[:5]  # Limit to top 5 terms


class ImprovedRAGPipeline:
    """Improved RAG pipeline with better search and error handling"""

    def __init__(self):
        self.db_manager = ChromaManager()
        self.llm = OllamaLLM()
        self.query_processor = ImprovedQueryProcessor()

        self.system_prompt = """You are a helpful AI assistant that answers questions about Bangladeshi news articles. 

        Your responses should be:
        - Accurate and based only on the provided context
        - Clear and concise
        - Informative and helpful
        - Written in a conversational tone
        - Always mention the news sources when available and relevant
        - Include specific sources for important facts or claims

        If you cannot find relevant information in the provided context, politely say so and suggest alternative ways to help.
        Always attribute information to the specific news sources when possible.
        """

    def _smart_search(self, query_data: Dict) -> List[Dict]:
        """Improved search strategy that tries multiple approaches"""
        results = []
        
        try:
            # Strategy 1: Combined search with source and category filters
            if query_data.get('source') and query_data.get('category'):
                search_query = ' '.join(query_data.get('search_terms', [query_data['original_query']]))
                results = self.db_manager.combined_search(
                    query=search_query,
                    source=query_data['source'],
                    category=query_data['category'],
                    n_results=10
                )
                logger.info(f"Combined search found {len(results)} results")
            
            # Strategy 2: Search with source filter only
            elif query_data.get('source') and not results:
                search_query = ' '.join(query_data.get('search_terms', [query_data['original_query']]))
                results = self.db_manager.combined_search(
                    query=search_query,
                    source=query_data['source'],
                    n_results=10
                )
                logger.info(f"Source-filtered search found {len(results)} results")
            
            # Strategy 3: Search with category filter only
            elif query_data.get('category') and not results:
                search_query = ' '.join(query_data.get('search_terms', [query_data['original_query']]))
                results = self.db_manager.combined_search(
                    query=search_query,
                    category=query_data['category'],
                    n_results=10
                )
                logger.info(f"Category-filtered search found {len(results)} results")
            
            # Strategy 4: Pure semantic search
            if not results:
                search_query = query_data.get('topic') or ' '.join(query_data.get('search_terms', [])) or query_data['original_query']
                results = self.db_manager.search_similar(search_query, n_results=10)
                logger.info(f"Semantic search found {len(results)} results")
            
            # Strategy 5: Broader search if still no results
            if not results:
                broader_terms = ['bangladesh', 'news', 'politics', 'sports', 'business']
                for term in broader_terms:
                    if term in query_data['original_query'].lower():
                        results = self.db_manager.search_similar(term, n_results=5)
                        logger.info(f"Broader search with '{term}' found {len(results)} results")
                        break
            
        except Exception as e:
            logger.error(f"Error in smart search: {e}")
        
        return results

    def _format_context(self, retrieved_docs: List[Dict]) -> str:
        """Format retrieved documents into context string with proper source attribution"""
        if not retrieved_docs:
            return "No relevant articles found."

        context_parts = []
        for i, doc in enumerate(retrieved_docs[:5], 1):  # Limit to top 5 for context
            metadata = doc.get('metadata', {})
            text = doc.get('text', '')

            source = metadata.get('source', 'Unknown')
            category = metadata.get('category', 'Unknown')
            title = metadata.get('title', 'No title')
            url = metadata.get('url', '')
            
            # Format source nicely
            source_display = self._format_source_name(source)

            context_part = f"Article {i} (Source: {source_display}, Category: {category}):\n"
            context_part += f"Title: {title}\n"
            if url:
                context_part += f"URL: {url}\n"
            context_part += f"Content: {text[:1200]}...\n\n"
            context_parts.append(context_part)

        return "".join(context_parts)

    # Helper method to format source names
    def _format_source_name(self, source: str) -> str:
        """Format source names for better display"""
        source_mapping = {
            'prothom_alo': 'Prothom Alo',
            'daily_star': 'The Daily Star',
            'bdnews24': 'BDNews24',
            'bdnews': 'BDNews24'
        }
        return source_mapping.get(source, source.title().replace('_', ' '))

    def _handle_count_articles(self, query_data: Dict) -> str:
        """Handle article count queries"""
        try:
            stats = self.db_manager.get_collection_stats()
            total_chunks = stats.get('total_chunks', 0)
            
            if total_chunks == 0:
                return "No articles are currently indexed in the database. Please scrape some news first."
            
            unique_sources = stats.get('unique_sources', 0)
            sources = stats.get('sources', [])
            
            response = f"I have {total_chunks} article chunks from {unique_sources} news sources."
            
            if sources:
                response += f"\n\nSources include: {', '.join(sources)}"
                
            return response
            
        except Exception as e:
            logger.error(f"Error handling count articles: {e}")
            return "I encountered an error while counting articles."

    def _handle_count_topic(self, query_data: Dict) -> str:
        """Handle topic-specific count queries"""
        try:
            results = self._smart_search(query_data)
            relevant_results = [r for r in results if r.get('similarity', 0) > 0.3]
            
            topic = query_data.get('topic', 'the topic')
            
            if not relevant_results:
                return f"I couldn't find any articles about '{topic}'."
                
            return f"I found {len(relevant_results)} article chunks related to '{topic}'."
            
        except Exception as e:
            logger.error(f"Error handling count topic: {e}")
            return f"I encountered an error while counting articles about '{query_data.get('topic', 'the topic')}'."

    def _handle_summary(self, query_data: Dict) -> str:
        """Handle summary requests"""
        try:
            results = self._smart_search(query_data)

            if not results:
                topic = query_data.get('topic', 'that topic')
                return f"I couldn't find any articles about '{topic}'. Please try a different topic or check if articles are loaded."

            relevant_results = [r for r in results if r.get('similarity', 0) > 0.3]

            if not relevant_results:
                topic = query_data.get('topic', 'that topic')
                return f"I found some articles but they don't seem closely related to '{topic}'. Could you be more specific?"

            context = self._format_context(relevant_results)
            topic = query_data.get('topic', query_data['original_query'])
            
            prompt = f"""Based on the following news articles, provide a clear and informative summary about '{topic}': 

            {context}

            Please provide a comprehensive summary that:
            1. Covers the key points from these articles
            2. Attributes information to specific news sources when relevant
            3. Highlights any consensus or differences between sources
            4. Is well-structured and easy to understand"""

            return self.llm.generate(prompt, self.system_prompt)

        except Exception as e:
            logger.error(f"Error handling summary: {e}")
            return f"I couldn't generate a summary due to an error: {str(e)}"

    def _handle_trending(self, query_data: Dict) -> str:
        """Handle trending/latest news queries"""
        try:
            # For trending, try to get a variety of recent articles
            results = self._smart_search(query_data)
            
            if not results:
                # Fallback: get any recent articles
                all_results = self.db_manager.collection.get(limit=10, include=['documents', 'metadatas'])
                if all_results['documents']:
                    results = []
                    for i in range(len(all_results['documents'])):
                        results.append({
                            'text': all_results['documents'][i],
                            'metadata': all_results['metadatas'][i],
                            'similarity': 0.5
                        })
            
            if not results:
                return "I don't have any news articles to show you. Please scrape some news first using the 'Scrape News' button."

            context = self._format_context(results)
            
            prompt = f"""Based on these news articles, provide a summary of recent/trending news:

            {context}

            Please highlight the most important current events and news stories from Bangladesh."""

            return self.llm.generate(prompt, self.system_prompt)

        except Exception as e:
            logger.error(f"Error handling trending: {e}")
            return "I encountered an error while fetching trending news."

    def _extract_sources_from_results(self, results: List[Dict]) -> List[str]:
        """Extract unique sources from search results"""
        sources = set()
        for result in results:
            source = result.get('metadata', {}).get('source')
            if source:
                sources.add(self._format_source_name(source))
        return list(sources)

    def _handle_general_search(self, query_data: Dict) -> str:
        """Handle general search queries with source attribution"""
        try:
            results = self._smart_search(query_data)
            
            if not results:
                return f"I couldn't find any articles related to '{query_data['original_query']}'. Please try different keywords or check if articles are loaded."
            
            # Filter by similarity threshold
            relevant_results = [r for r in results if r.get('similarity', 0) > 0.15]
            if not relevant_results:
                relevant_results = results  # fallback, always pass something
            
            if not relevant_results:
                return f"I found some articles but they don't seem closely related to '{query_data['original_query']}'. Could you try rephrasing your question?"
            
            context = self._format_context(relevant_results)
            sources = self._extract_sources_from_results(relevant_results)
            
            prompt = f"""Based on the following news articles, please answer this question: "{query_data['original_query']}"

            {context}

            Please provide a helpful and informative response that:
            1. Directly addresses the question
            2. Cites specific news sources for important information (sources available: {', '.join(sources)})
            3. Is based only on the provided articles
            4. Is clear and concise"""

            return self.llm.generate(prompt, self.system_prompt)
            
        except Exception as e:
            logger.error(f"Error handling general search: {e}")
            return f"I encountered an error while searching for information about '{query_data['original_query']}'."

    def process_query(self, user_query: str) -> str:
        """Main method to process user queries with improved error handling"""
        try:
            # Check if database has articles
            stats = self.db_manager.get_collection_stats()
            if stats.get('total_chunks', 0) == 0:
                return "No news articles are currently loaded. Please use the 'Scrape News' button in the sidebar to load some articles first."

            # Extract intent and entities from query
            query_data = self.query_processor.extract_intent_and_entities(user_query)
            intent = query_data['intent']

            # Route to appropriate handler
            if intent == 'count_articles':
                return self._handle_count_articles(query_data)
            elif intent == 'count_topic':
                return self._handle_count_topic(query_data)
            elif intent == 'summary':
                return self._handle_summary(query_data)
            elif intent == 'trending':
                return self._handle_trending(query_data)
            else:
                return self._handle_general_search(query_data)

        except Exception as e:
            logger.error(f"Error processing query '{user_query}': {e}")
            return "I encountered an error while processing your question. Please try again or rephrase your query."


# Create an alias for backward compatibility
RAGPipeline = ImprovedRAGPipeline


def main():
    """Test the improved RAG pipeline"""
    rag = ImprovedRAGPipeline()
    test_queries = [
        "How many articles do you have?",
        "Bangladesh politics news",
        "Today's sports news",
        "What is trending news today?",
        "Give me summary of economy news",
        "How many articles about cricket?"
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        response = rag.process_query(query)
        print(f"Response: {response}")
        print("=" * 80)


if __name__ == "__main__":
    main()