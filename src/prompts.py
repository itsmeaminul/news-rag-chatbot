"""
Prompt templates for the RAG pipeline
"""
from typing import List

class PromptTemplates:
    """Collection of prompt templates for the news chatbot"""
    
    # System prompt for the LLM
    SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions about Bangladeshi news articles with accuracy and clarity.

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
6. Nony provide the context related news articles

If you cannot find relevant information in the provided context, politely say so and suggest alternative ways to help.
"""

    # Template for summary queries
    SUMMARY_TEMPLATE = """Based on the following news articles, provide a comprehensive summary about '{topic}': 

{context}

Please provide a well-structured summary that:
1. Starts with a brief overview
2. Includes important details (use bullet points when appropriate)
3. Mentions specific facts, dates, names, and locations which are relevant
4. Organizes information clearly and logically
5. Focuses on the most important information without repetition
6. Nony provide the context related news articles

Format your response in a clear, readable manner with proper structure."""

    # Template for trending/latest news queries
    TRENDING_TEMPLATE = """Based on these news articles, provide a summary of recent/trending news from Bangladesh:
            
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

Format the response clearly and avoid repetition."""

    # Template for general search queries
    GENERAL_SEARCH_TEMPLATE = """Based on the following news articles, please answer this question: "{query}"

{context}

Please provide a helpful and informative response that:
1. Directly answers the user's question with a clear summary
2. Includes key points or important details (use bullet points when helpful)
3. Mentions specific facts, dates, names, and locations when relevant
4. Is accurate and based only on the provided articles
5. Is well-structured and easy to understand
6. Nony provide the context related news articles

Format your response clearly and focus on the most relevant information."""

    @classmethod
    def get_summary_prompt(cls, topic: str, context: str) -> str:
        """Get formatted summary prompt"""
        return cls.SUMMARY_TEMPLATE.format(topic=topic, context=context)
    
    @classmethod
    def get_trending_prompt(cls, context: str) -> str:
        """Get formatted trending news prompt"""
        return cls.TRENDING_TEMPLATE.format(context=context)
    
    @classmethod
    def get_general_search_prompt(cls, query: str, context: str) -> str:
        """Get formatted general search prompt"""
        return cls.GENERAL_SEARCH_TEMPLATE.format(query=query, context=context)


class ResponseMessages:
    """Collection of standard response messages"""
    
    # Error messages
    NO_ARTICLES_LOADED = "No news articles are currently loaded. Please use the 'Scrape Recent News' button in the sidebar to load some articles first."
    
    DATABASE_ERROR = "I encountered an error while processing your question. Please try again or rephrase your query."
    
    INITIALIZATION_ERROR = "I apologize, but I'm having trouble generating a response right now. Please try again later."
    
    # Not found messages
    NO_RELEVANT_ARTICLES = "I couldn't find any articles related to '{query}'. Please try different keywords or check if articles are loaded."
    
    NO_CLOSE_MATCHES = "I found some articles but they don't seem closely related to '{topic}'. Could you be more specific?"
    
    NO_TITLE_MATCHES = "I couldn't find any articles about '{topic}'. Try rephrasing your query or check if articles are loaded."
    
    # Success messages for counts
    COUNT_ARTICLES_RESPONSE = """I have **{total_chunks}** article chunks from **{unique_sources}** news sources in the database."""
    
    COUNT_ARTICLES_WITH_SOURCES = """I have **{total_chunks}** article chunks from **{unique_sources}** news sources in the database.

**Available Sources:**
{sources_list}"""
    
    COUNT_TOPIC_RESPONSE = """I found **{count}** article chunks related to **'{topic}'**."""
    
    COUNT_TOPIC_WITH_SOURCES = """I found **{count}** article chunks related to **'{topic}'**.

**Articles by Source:**
{sources_breakdown}"""
    
    # Trending news fallback
    NO_TRENDING_NEWS = "I don't have any news articles to show you. Please scrape some news first using the 'Scrape News' button."
    
    @classmethod
    def get_no_relevant_articles(cls, query: str) -> str:
        """Get formatted no relevant articles message"""
        return cls.NO_RELEVANT_ARTICLES.format(query=query)
    
    @classmethod
    def get_no_close_matches(cls, topic: str) -> str:
        """Get formatted no close matches message"""
        return cls.NO_CLOSE_MATCHES.format(topic=topic)
    
    @classmethod
    def get_no_title_matches(cls, topic: str) -> str:
        """Get formatted no title matches message"""
        return cls.NO_TITLE_MATCHES.format(topic=topic)
    
    @classmethod
    def get_count_articles_response(cls, total_chunks: int, unique_sources: int, sources: List[str] = None) -> str:
        """Get formatted count articles response"""
        if sources:
            sources_list = '\n'.join(f"• {source}" for source in sources)
            return cls.COUNT_ARTICLES_WITH_SOURCES.format(
                total_chunks=total_chunks,
                unique_sources=unique_sources,
                sources_list=sources_list
            )
        else:
            return cls.COUNT_ARTICLES_RESPONSE.format(
                total_chunks=total_chunks,
                unique_sources=unique_sources
            )
    
    @classmethod
    def get_count_topic_response(cls, count: int, topic: str, sources_breakdown: str = None) -> str:
        """Get formatted count topic response"""
        if sources_breakdown:
            return cls.COUNT_TOPIC_WITH_SOURCES.format(
                count=count,
                topic=topic,
                sources_breakdown=sources_breakdown
            )
        else:
            return cls.COUNT_TOPIC_RESPONSE.format(count=count, topic=topic)


class PromptBuilder:
    """Builder class for constructing complex prompts"""
    
    def __init__(self):
        self.context = ""
        self.query = ""
        self.topic = ""
        self.additional_instructions = []
    
    def with_context(self, context: str) -> 'PromptBuilder':
        """Add context to the prompt"""
        self.context = context
        return self
    
    def with_query(self, query: str) -> 'PromptBuilder':
        """Add query to the prompt"""
        self.query = query
        return self
    
    def with_topic(self, topic: str) -> 'PromptBuilder':
        """Add topic to the prompt"""
        self.topic = topic
        return self
    
    def add_instruction(self, instruction: str) -> 'PromptBuilder':
        """Add additional instruction"""
        self.additional_instructions.append(instruction)
        return self
    
    def build_summary_prompt(self) -> str:
        """Build summary prompt with custom instructions"""
        prompt = PromptTemplates.get_summary_prompt(self.topic, self.context)
        
        if self.additional_instructions:
            prompt += "\n\nAdditional instructions:\n"
            for instruction in self.additional_instructions:
                prompt += f"- {instruction}\n"
        
        return prompt
    
    def build_general_prompt(self) -> str:
        """Build general search prompt with custom instructions"""
        prompt = PromptTemplates.get_general_search_prompt(self.query, self.context)
        
        if self.additional_instructions:
            prompt += "\n\nAdditional instructions:\n"
            for instruction in self.additional_instructions:
                prompt += f"- {instruction}\n"
        
        return prompt
    
    def reset(self) -> 'PromptBuilder':
        """Reset builder state"""
        self.context = ""
        self.query = ""
        self.topic = ""
        self.additional_instructions = []
        return self