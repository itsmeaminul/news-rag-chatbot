"""
RAG (Retrieval-Augmented Generation) pipeline for news chatbot
"""
import logging
import re
import os
from typing import Dict, List, Optional, Set

import requests

from .config import LLM_CONFIG, USE_LOCAL, LLM_PROVIDER
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

class GroqLLM:
    """Groq API client"""

    def __init__(self, model_name=None):
        from .config import LLM_PROVIDER
        self.api_key = LLM_PROVIDER["groq_api_key"]
        self.model = model_name or LLM_PROVIDER["groq_model"]
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"

        if not self.api_key:
            raise EnvironmentError("❌ GROQ_API_KEY missing in .env")

    def generate(self, prompt: str, system_prompt: str = None) -> str:
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
            "max_tokens": 500,
        }

        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return f"Error generating response from Groq API: {e}"


class TitleFocusedQueryProcessor:
    """Query processor with strong title emphasis"""

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
        
        self.category_mapping = {
            'sports': ['sports', 'sport', 'cricket', 'football', 'soccer', 'match', 'tournament', 'team', 'player'],
            'bangladesh': ['bangladesh', 'politics', 'govt', 'government', 'political', 'minister', 'parliament', 'prime minister'],
            'international': ['international', 'world', 'global', 'foreign', 'diplomatic', 'embassy'],
            'entertainment': ['entertainment', 'culture', 'movie', 'cinema', 'film', 'actor', 'celebrity', 'music'],
            'business': ['business', 'economy', 'economic', 'trade', 'finance', 'market', 'investment', 'bank'],
            'lifestyle': ['lifestyle', 'health', 'fashion', 'food', 'travel', 'hospital', 'medical']
        }

    def extract_intent_and_entities(self, query: str) -> Dict:
        """Extract intent and entities from user query with enhanced title-matching focus"""
        query_lower = query.lower().strip()
        
        intent_data = {
            'intent': 'general_search',
            'topic': None,
            'source': None,
            'category': None,
            'search_terms': [],
            'title_keywords': [],
            'is_title_query': False,
            'exact_phrase': None,
            'original_query': query
        }

        # Detect if user is asking about specific news by title-like phrases
        title_indicators = [
            r'news about (.+)',
            r'article about (.+)',
            r'story about (.+)',
            r'what happened with (.+)',
            r'tell me about (.+?) news',
            r'(.+?) news',
            r'latest on (.+)',
            r'update on (.+)',
            r'headlines about (.+)',
            r'reports on (.+)',
            r'what is the matter\?*$',
            r'what.*happened.*\?*$'
        ]
        
        for pattern in title_indicators:
            match = re.search(pattern, query_lower)
            if match:
                intent_data['is_title_query'] = True
                if match.groups():
                    intent_data['topic'] = match.group(1).strip()
                    intent_data['exact_phrase'] = match.group(1).strip()
                break

        # Extract title-specific keywords (more comprehensive)
        title_keywords = self._extract_title_keywords(query)
        intent_data['title_keywords'] = title_keywords
        
        # Standard intent detection with title awareness
        if re.search(r'\b(how many|count|number of|total)\b.*\barticles?\b', query_lower):
            if re.search(r'\babout\b', query_lower):
                intent_data['intent'] = 'count_topic'
                topic_match = re.search(r'\babout\s+([^?]+)', query_lower)
                if topic_match:
                    intent_data['topic'] = topic_match.group(1).strip()
                    intent_data['title_keywords'].extend(topic_match.group(1).strip().split())
            else:
                intent_data['intent'] = 'count_articles'
        
        elif re.search(r'\b(summary|summarize|tell me about|what.*about)\b', query_lower):
            intent_data['intent'] = 'summary'
            # Extract topic for summary
            for pattern in [r'summary.*?of\s+([^?]+)', r'summarize\s+([^?]+)', r'tell me about\s+([^?]+)', r'what.*?about\s+([^?]+)']:
                match = re.search(pattern, query_lower)
                if match:
                    intent_data['topic'] = match.group(1).strip()
                    intent_data['is_title_query'] = True
                    break
        
        elif re.search(r'\b(trending|latest|recent|today.*news|current.*news|top.*news|breaking)\b', query_lower):
            intent_data['intent'] = 'trending'

        # Enhanced source extraction
        source_found = self._extract_source(query_lower)
        if source_found:
            intent_data['source'] = source_found

        # Enhanced category extraction  
        category_found = self._extract_category(query_lower)
        if category_found:
            intent_data['category'] = category_found

        # Extract comprehensive search terms with title priority
        search_terms = self._extract_search_terms(query_lower, intent_data)
        intent_data['search_terms'] = search_terms

        logger.info(f"Enhanced intent extraction: {intent_data}")
        return intent_data

    def _extract_title_keywords(self, query: str) -> List[str]:
        """Extract keywords that are likely to appear in news titles"""
        # Remove common query phrases first
        cleaned_query = query.lower()
        
        # Remove question patterns
        patterns_to_remove = [
            r'tell me about\s+',
            r'what is\s+',
            r'what are\s+',
            r'how is\s+',
            r'news about\s+',
            r'article about\s+',
            r'latest\s+',
            r'recent\s+',
            r'today\s*\'*s*\s*',
            r'trending\s+',
            r'update on\s+',
            r'headlines about\s+',
            r'what is the matter\?*',
            r'what.*happened.*\?*'
        ]
        
        for pattern in patterns_to_remove:
            cleaned_query = re.sub(pattern, '', cleaned_query)
        
        # Extract meaningful words
        words = re.findall(r'\b[a-zA-Z]{3,}\b', cleaned_query)
        
        # Enhanced important keywords for Bangladesh news
        important_keywords = {
            'bangladesh', 'bangladeshi', 'dhaka', 'chittagong', 'sylhet', 'rajshahi', 'khulna', 'barishal',
            'government', 'minister', 'parliament', 'election', 'awami', 'league', 'bnp',
            'cricket', 'football', 'match', 'tournament', 'team', 'player', 'tigers',
            'economy', 'business', 'market', 'trade', 'investment', 'taka', 'bank',
            'covid', 'coronavirus', 'vaccine', 'hospital', 'health', 'medical',
            'university', 'student', 'education', 'school', 'college', 'exam',
            'police', 'officer', 'court', 'justice', 'law', 'crime', 'arrest', 'investigation',
            'rohingya', 'refugee', 'myanmar', 'border', 'camp', 'west', 'bengal', 'india',
            'garments', 'textile', 'export', 'import', 'factory', 'workers',
            'flood', 'cyclone', 'weather', 'climate', 'disaster', 'relief'
        }
        
        stop_words = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use'
        }
        
        title_keywords = []
        for word in words:
            word = word.lower()
            if word in important_keywords or (len(word) > 3 and word not in stop_words):
                title_keywords.append(word)
        
        return title_keywords[:8]  # Increased limit for better matching

    def _extract_source(self, query_lower: str) -> Optional[str]:
        """Enhanced source extraction"""
        for source_key, source_variants in [
            ('prothom_alo', ['prothom alo', 'prothomalo', 'prothom-alo']),
            ('daily_star', ['daily star', 'thedailystar', 'daily-star']),
            ('bdnews24', ['bdnews24', 'bdnews', 'bd news'])
        ]:
            for variant in source_variants:
                if variant in query_lower:
                    return source_key
        return None

    def _extract_category(self, query_lower: str) -> Optional[str]:
        """Enhanced category extraction with better matching"""
        for category, keywords in self.category_mapping.items():
            for keyword in keywords:
                if re.search(r'\b' + re.escape(keyword) + r'\b', query_lower):
                    return category
        return None

    def _extract_search_terms(self, query: str, intent_data: Dict) -> List[str]:
        """Extract search terms with priority on title-relevant words"""
        # Start with title keywords if available
        search_terms = intent_data.get('title_keywords', []).copy()
        
        # Add exact phrase if specified
        if intent_data.get('exact_phrase'):
            phrase_words = [word for word in intent_data['exact_phrase'].split() if len(word) > 2]
            search_terms.extend(phrase_words)
        
        # Add topic if specified
        if intent_data.get('topic'):
            topic_words = [word for word in intent_data['topic'].split() if len(word) > 2]
            search_terms.extend(topic_words)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_terms = []
        for term in search_terms:
            term_lower = term.lower()
            if term_lower not in seen:
                seen.add(term_lower)
                unique_terms.append(term)
        
        return unique_terms[:10]  # Increased limit


class EnhancedRAGPipeline:
    """Enhanced RAG pipeline with clean, structured output"""

    def __init__(self):
        self.db_manager = ChromaManager()
        if USE_LOCAL:
            self.llm = OllamaLLM()
            logger.info("✅ Using local Ollama LLM")
        else:
            self.llm = GroqLLM()
            logger.info("✅ Using Groq API")
            
        self.query_processor = TitleFocusedQueryProcessor()

        self.system_prompt = """You are a helpful AI assistant that answers questions about Bangladeshi news articles with accuracy and clarity.

        Your responses should be:
        - Accurate and based only on the provided context
        - Clear, concise, and well-structured
        - Organized with key points when appropriate
        - Written in a conversational but informative tone
        - Focused on the most important and relevant information

        Response Format Guidelines:
        1. Start with a brief summary or direct answer
        2. Follow with key points or details when relevant (use bullet points or numbered lists for clarity)
        3. Include important facts, dates, names, and locations
        4. DO NOT mention source labels like [TITLE MATCH] or [TITLE SUMMARY] in your response
        5. Keep the response focused and avoid unnecessary repetition

        If you cannot find relevant information in the provided context, politely say so and suggest alternative ways to help.
        """

    def _title_first_search(self, query_data: Dict) -> List[Dict]:
        """Search strategy that tries multiple approaches"""
        results = []
        
        try:
            # Strategy 1: Exact title phrase search (highest priority)
            if query_data.get('exact_phrase'):
                exact_results = self.db_manager.search_by_exact_title_phrase(
                    query_data['exact_phrase'], n_results=5
                )
                if exact_results:
                    logger.info(f"Exact title phrase search found {len(exact_results)} results")
                    results.extend(exact_results)
            
            # Strategy 2: Title keyword search
            if query_data.get('title_keywords') or query_data.get('is_title_query'):
                search_terms = query_data.get('title_keywords', [])
                if query_data.get('topic'):
                    search_terms.extend(query_data['topic'].split())
                
                if search_terms:
                    title_query = ' '.join(search_terms[:5])  # Focus on key terms
                    title_results = self.db_manager.search_by_title_keywords(title_query, n_results=8)
                    
                    if title_results:
                        logger.info(f"Title keyword search found {len(title_results)} results")
                        # Avoid duplicates
                        existing_ids = {r.get('metadata', {}).get('chunk_id') for r in results}
                        for result in title_results:
                            chunk_id = result.get('metadata', {}).get('chunk_id')
                            if chunk_id not in existing_ids:
                                results.append(result)
            
            # Strategy 3: Title-emphasized semantic search
            if query_data.get('title_keywords') or query_data.get('topic'):
                search_query = query_data.get('topic') or ' '.join(query_data.get('title_keywords', []))
                if search_query:
                    title_semantic_results = self.db_manager.search_with_title_emphasis(
                        search_query, n_results=10
                    )
                    
                    # Filter out duplicates and add new results
                    existing_chunk_ids = {r.get('metadata', {}).get('chunk_id') for r in results}
                    for result in title_semantic_results:
                        chunk_id = result.get('metadata', {}).get('chunk_id')
                        if chunk_id not in existing_chunk_ids:
                            results.append(result)
            
            # Strategy 4: Combined search with filters
            if query_data.get('source') or query_data.get('category'):
                search_query = query_data.get('topic') or ' '.join(query_data.get('search_terms', []))
                if search_query:
                    filtered_results = self.db_manager.combined_search(
                        query=search_query,
                        source=query_data.get('source'),
                        category=query_data.get('category'),
                        n_results=8
                    )
                    
                    existing_chunk_ids = {r.get('metadata', {}).get('chunk_id') for r in results}
                    for result in filtered_results:
                        chunk_id = result.get('metadata', {}).get('chunk_id')
                        if chunk_id not in existing_chunk_ids:
                            results.append(result)
            
            # Strategy 5: Regular semantic search as fallback
            if len(results) < 5:
                original_query = query_data['original_query']
                semantic_results = self.db_manager.search_similar(original_query, n_results=12)
                
                existing_chunk_ids = {r.get('metadata', {}).get('chunk_id') for r in results}
                for result in semantic_results:
                    chunk_id = result.get('metadata', {}).get('chunk_id')
                    if chunk_id not in existing_chunk_ids:
                        results.append(result)
            
            # Re-rank results with enhanced title bias
            results = self._rerank_with_title_bias(results, query_data)
            
            logger.info(f"Title-first search found {len(results)} total results")
            return results[:15]  # Return top 15 results
            
        except Exception as e:
            logger.error(f"Error in title-first search: {e}")
            # Ultimate fallback
            return self.db_manager.search_similar(query_data['original_query'], n_results=10)

    def _rerank_with_title_bias(self, results: List[Dict], query_data: Dict) -> List[Dict]:
        """Re-rank results with strong bias toward title matches"""
        title_keywords = set(word.lower() for word in query_data.get('title_keywords', []))
        query_topic = query_data.get('topic', '').lower()
        exact_phrase = query_data.get('exact_phrase', '').lower()
        
        for result in results:
            metadata = result.get('metadata', {})
            title = metadata.get('title', '').lower()
            chunk_type = metadata.get('chunk_type', '')
            is_title_chunk = metadata.get('is_title_chunk') == 'True'
            
            # Base similarity
            base_similarity = result.get('similarity', 0)
            
            # Title match bonuses
            title_bonus = 0
            
            # Bonus for exact phrase matches in title
            if exact_phrase and exact_phrase in title:
                title_bonus += 0.4
            
            # Bonus for title chunks
            if is_title_chunk:
                title_bonus += 0.2
            
            # Bonus for title keyword matches
            if title_keywords:
                title_words = set(title.split())
                keyword_matches = len(title_keywords.intersection(title_words))
                if keyword_matches > 0:
                    title_bonus += (keyword_matches / len(title_keywords)) * 0.25
            
            # Bonus for topic in title
            if query_topic and query_topic in title:
                title_bonus += 0.3
            
            # Bonus for high title relevance score
            title_relevance = float(metadata.get('title_relevance_score', 0))
            if title_relevance > 0.7:
                title_bonus += 0.15
            elif title_relevance > 0.5:
                title_bonus += 0.1
            
            # Bonus for chunk type
            if 'title' in chunk_type:
                title_bonus += 0.1
            
            # Apply bonus but cap at 0.99
            result['similarity'] = min(0.99, base_similarity + title_bonus)
            result['title_bonus'] = title_bonus
        
        # Sort by enhanced similarity
        results.sort(key=lambda x: x.get('similarity', 0), reverse=True)
        return results

    def _format_context_for_llm(self, retrieved_docs: List[Dict]) -> str:
        """Format context for LLM without exposing internal labels"""
        if not retrieved_docs:
            return "No relevant articles found."

        context_parts = []
        for i, doc in enumerate(retrieved_docs[:6], 1):  # Limit to top 6 results
            metadata = doc.get('metadata', {})
            text = doc.get('text', '')
            
            title = metadata.get('title', 'No title')
            source = self._format_source_name(metadata.get('source', 'Unknown'))
            category = metadata.get('category', 'Unknown')
            
            context_part = f"Article {i}:\n"
            context_part += f"Title: {title}\n"
            context_part += f"Source: {source}\n"
            context_part += f"Category: {category}\n"
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

    def _extract_source_urls(self, results: List[Dict]) -> List[str]:
        """Extract unique source URLs from search results for citation"""
        seen_urls = set()
        source_urls = []
        
        for result in results:
            metadata = result.get('metadata', {})
            url = metadata.get('url', '').strip()
            title = metadata.get('title', 'No title')
            source = self._format_source_name(metadata.get('source', 'Unknown'))
            
            if url and url not in seen_urls:
                seen_urls.add(url)
                source_urls.append({
                    'title': title,
                    'source': source,
                    'url': url
                })
        
        return source_urls[:5]  # Limit to top 5 sources

    def _format_sources_section(self, source_urls: List[Dict]) -> str:
        """Format the sources section with links"""
        if not source_urls:
            return ""
        
        sources_text = "\n\n**Sources:**\n"
        for i, source_info in enumerate(source_urls, 1):
            sources_text += f"{i}. **{source_info['title']}** - {source_info['source']}\n"
            sources_text += f"   Link: {source_info['url']}\n"
        
        return sources_text

    def _handle_count_articles(self, query_data: Dict) -> str:
        """Handle article count queries"""
        try:
            stats = self.db_manager.get_collection_stats()
            total_chunks = stats.get('total_chunks', 0)
            
            if total_chunks == 0:
                return "No articles are currently indexed in the database. Please scrape some news first."
            
            unique_sources = stats.get('unique_sources', 0)
            sources = stats.get('sources', [])
            title_chunks = stats.get('title_chunks', 0)
            
            response = f"I have **{total_chunks}** article chunks from **{unique_sources}** news sources in the database."
            
            if sources:
                formatted_sources = [self._format_source_name(s) for s in sources]
                response += f"\n\n**Available Sources:**\n"
                for source in formatted_sources:
                    response += f"• {source}\n"
                
            return response
            
        except Exception as e:
            logger.error(f"Error handling count articles: {e}")
            return "I encountered an error while counting articles."

    def _handle_count_topic(self, query_data: Dict) -> str:
        """Handle topic-specific count queries"""
        try:
            results = self._title_first_search(query_data)
            relevant_results = [r for r in results if r.get('similarity', 0) > 0.25]
            
            topic = query_data.get('topic', query_data.get('exact_phrase', 'the topic'))
            
            if not relevant_results:
                return f"I couldn't find any articles about **'{topic}'**. Try rephrasing your query or check if articles are loaded."
            
            response = f"I found **{len(relevant_results)}** article chunks related to **'{topic}'**."
            
            # Group by sources for better insight
            sources = {}
            for result in relevant_results:
                source = result.get('metadata', {}).get('source', 'Unknown')
                formatted_source = self._format_source_name(source)
                sources[formatted_source] = sources.get(formatted_source, 0) + 1
            
            if sources:
                response += f"\n\n**Articles by Source:**\n"
                for source, count in sources.items():
                    response += f"• {source}: {count} articles\n"
                
            return response
            
        except Exception as e:
            logger.error(f"Error handling count topic: {e}")
            return f"I encountered an error while counting articles about '{topic}'."

    def _handle_summary(self, query_data: Dict) -> str:
        """Handle summary requests"""
        try:
            results = self._title_first_search(query_data)

            if not results:
                topic = query_data.get('topic', query_data.get('exact_phrase', 'that topic'))
                return f"I couldn't find any articles about '{topic}'. Please try a different topic or check if articles are loaded."

            relevant_results = [r for r in results if r.get('similarity', 0) > 0.2]

            if not relevant_results:
                topic = query_data.get('topic', query_data.get('exact_phrase', 'that topic'))
                return f"I found some articles but they don't seem closely related to '{topic}'. Could you be more specific?"

            context = self._format_context_for_llm(relevant_results)
            topic = query_data.get('topic', query_data.get('exact_phrase', query_data['original_query']))
            source_urls = self._extract_source_urls(relevant_results)
            
            prompt = f"""Based on the following news articles, provide a comprehensive summary about '{topic}': 

            {context}

            Please provide a well-structured summary that:
            1. Starts with a brief overview
            2. Includes key points or important details (use bullet points when appropriate)
            3. Mentions specific facts, dates, names, and locations when relevant
            4. Organizes information clearly and logically
            5. Focuses on the most important information without repetition

            Format your response in a clear, readable manner with proper structure."""

            llm_response = self.llm.generate(prompt, self.system_prompt)
            
            # Add sources section
            sources_section = self._format_sources_section(source_urls)
            
            return llm_response + sources_section

        except Exception as e:
            logger.error(f"Error handling summary: {e}")
            return f"I couldn't generate a summary due to an error: {str(e)}"

    def _handle_trending(self, query_data: Dict) -> str:
        """Handle trending/latest news queries"""
        try:
            # For trending, we want a variety of recent articles
            results = self._title_first_search(query_data)
            
            if not results:
                # Fallback: get any recent articles
                all_results = self.db_manager.collection.get(limit=15, include=['documents', 'metadatas'])
                if all_results['documents']:
                    results = []
                    for i in range(len(all_results['documents'])):
                        results.append({
                            'text': all_results['documents'][i],
                            'metadata': all_results['metadatas'][i],
                            'similarity': 0.6
                        })
            
            if not results:
                return "I don't have any news articles to show you. Please scrape some news first using the 'Scrape News' button."

            context = self._format_context_for_llm(results)
            source_urls = self._extract_source_urls(results)
            
            # prompt = f"""Based on these news articles, provide a summary of recent/trending news from Bangladesh:

            # {context}

            # Please provide a well-organized summary that:
            # 1. Starts with a brief overview of current events
            # 2. Lists the most important news stories
            # 3. Organizes information by topic or significance
            # 4. Uses clear, structured format with bullet points for key points
            # 5. Focuses on the most newsworthy and current information

            # Format your response clearly and avoid repetition."""

            prompt = f"""Based on these news articles, provide a summary of recent/trending news from Bangladesh:
            
            {context} 

            Please follow these rules:
            1. If relevant information is found, provide a well-organized summary that:
            - Starts with a brief overview of current events
            - Lists the most important news stories
            - Organizes information by topic or significance
            - Uses clear, structured format with bullet points for key points
            - Focuses on the most newsworthy and current information

            2. If no relevant information is found in the provided articles:
            - Clearly state that no direct updates are available on the requested topic
            - Avoid generating irrelevant or unrelated content
            - Suggest reliable sources (e.g., BDNews24, The Daily Star, official websites, or social media) where the user can find up-to-date information
            - Keep the response concise, polite, and professional

            Format the response clearly and avoid repetition.
            """

            llm_response = self.llm.generate(prompt, self.system_prompt)
            
            # Add sources section
            sources_section = self._format_sources_section(source_urls)
            
            return llm_response + sources_section

        except Exception as e:
            logger.error(f"Error handling trending: {e}")
            return "I encountered an error while fetching trending news."

    def _handle_general_search(self, query_data: Dict) -> str:
        """Handle general search queries"""
        try:
            results = self._title_first_search(query_data)
            
            if not results:
                return f"I couldn't find any articles related to '{query_data['original_query']}'. Please try different keywords or check if articles are loaded."
            
            # Filter by similarity threshold, but be more lenient for title matches
            relevant_results = []
            for result in results:
                similarity = result.get('similarity', 0)
                title_bonus = result.get('title_bonus', 0)
                
                if title_bonus > 0.1 or similarity > 0.15:
                    relevant_results.append(result)
            
            if not relevant_results:
                relevant_results = results[:5]  # Always show something
            
            context = self._format_context_for_llm(relevant_results)
            source_urls = self._extract_source_urls(relevant_results)
            
            prompt = f"""Based on the following news articles, please answer this question: "{query_data['original_query']}"

            {context}

            Please provide a helpful and informative response that:
            1. Directly answers the user's question with a clear summary
            2. Includes key points or important details (use bullet points when helpful)
            3. Mentions specific facts, dates, names, and locations when relevant
            4. Is accurate and based only on the provided articles
            5. Is well-structured and easy to understand

            Format your response clearly and focus on the most relevant information."""

            llm_response = self.llm.generate(prompt, self.system_prompt)
            
            # Add sources section
            sources_section = self._format_sources_section(source_urls)
            
            return llm_response + sources_section
            
        except Exception as e:
            logger.error(f"Error handling general search: {e}")
            return f"I encountered an error while searching for information about '{query_data['original_query']}'."

    def process_query(self, user_query: str) -> str:
        """Main method to process user queries with clean output"""
        try:
            # Check if database has articles
            stats = self.db_manager.get_collection_stats()
            if stats.get('total_chunks', 0) == 0:
                return "No news articles are currently loaded. Please use the 'Scrape Recent News' button in the sidebar to load some articles first."

            # Extract intent and entities from query
            query_data = self.query_processor.extract_intent_and_entities(user_query)
            intent = query_data['intent']

            # Route to appropriate handlera
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


# Create aliases for backward compatibility
RAGPipeline = EnhancedRAGPipeline
TitleFocusedQueryProcessor = TitleFocusedQueryProcessor
ImprovedQueryProcessor = TitleFocusedQueryProcessor
ImprovedRAGPipeline = EnhancedRAGPipeline


def main():
    """Test the enhanced RAG pipeline with clean output"""
    rag = EnhancedRAGPipeline()
    test_queries = [
        "How many articles do you have?",
        "What is trending news today?",
        "Summarize economy news from Daily Star",
        "How many articles about coronavirus vaccine?",
        "Latest news from Prothom Alo"
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        print("=" * 60)
        response = rag.process_query(query)
        print(f"Response: {response}")
        print("=" * 80)


if __name__ == "__main__":
    main()