"""
Prompt templates for the RAG pipeline
"""
from typing import List


class PromptTemplates:
    """
    Collection of prompt templates and utility methods for generating prompts used by the news chatbot.
    This class provides structured prompt templates for various chatbot tasks, including:
    - System persona and behavior definition
    - General search and question answering
    - Executive summaries
    - Trending news identification
    - Handling cases with no relevant sources
    Each template enforces strict grounding to the provided news article context, mandates source attribution, and ensures objective, well-structured responses. The class also includes class methods to format and retrieve these prompts with dynamic content.
    Methods:
        get_summary_prompt(topic: str, context: str) -> str:
            Returns a formatted executive summary prompt for a given topic and context.
        get_trending_prompt(context: str) -> str:
            Returns a formatted prompt for identifying trending news stories from the given context.
        get_general_search_prompt(query: str, context: str) -> str:
            Returns a formatted prompt for answering a user's question based on the provided news articles context.
        get_no_sources_response(query: str, suggestions: List[str] = None) -> str:
            Returns a formatted response when no relevant sources are found, including alternative search suggestions.
    """
    """Collection of prompt templates for the news chatbot"""
    
    # SYSTEM_PROMPT: Refined persona and added stricter rules for synthesis and grounding.
    SYSTEM_PROMPT = """You are an expert AI News Analyst specializing in Bangladeshi current events. Your primary role is to provide accurate, objective, and well-structured answers based exclusively on the provided news article excerpts.

**Core Directives:**
1.  **Absolute Grounding:** Base your entire response on the provided context. Do not use any external knowledge.
2.  **Synthesize, Don't Just List:** When multiple documents are provided, synthesize the information into a single, coherent response. Identify common themes and highlight key differences or developments mentioned across sources.
3.  **Cite Sources Inline:** Attribute information to its source as you present it (e.g., "According to The Daily Star..."). Use the `Source` and `Date` metadata provided with each document.
4.  **Handle Contradictions:** If sources present conflicting information, point this out directly (e.g., "While Prothom Alo reports X, BDNews24 states Y.").
5.  **Structured Formatting:** Use Markdown for clarity. Employ headings, bold text for key terms, and bullet points for lists.
6.  **Objective Tone:** Maintain a neutral, factual tone. Avoid speculation, opinion, or any information not explicitly stated in the text.

**CRITICAL RULE:** If the provided context does not contain the information needed to answer the question, you must state: "Based on the provided articles, I cannot answer this question." Do not attempt to answer from outside knowledge."""

    # GENERAL_SEARCH_TEMPLATE: More direct, with a clear output structure.
    GENERAL_SEARCH_TEMPLATE = """Based on the following news articles, provide a detailed answer to the user's question.

**User Question:** "{query}"

**News Articles Context:**
---
{context}
---

**Instructions for Your Response:**

1.  **Direct Answer:** Begin with a concise, direct answer to the question.
2.  **Detailed Explanation:** Elaborate on the direct answer with key facts, figures, names, and dates from the articles. Organize this section with bullet points for readability.
3.  **Synthesis Across Sources:** If multiple articles discuss the topic, synthesize their information. Note any agreements or discrepancies.
4.  **Source Attribution:** Cite sources inline (e.g., "The Daily Star reported...").
5.  **Conclusion:** End with a brief concluding sentence that summarizes the main findings.

**Example Response Structure:**
**Answer:** [Direct answer to the query]

**Key Details:**
* [Detail 1 from the articles, with source attribution]
* [Detail 2 from the articles, with source attribution]
* [And so on...]

**Conclusion:** [A brief summary of the information]

**CRITICAL:** Answer *only* from the provided "News Articles Context". If the context does not contain the answer, state that clearly and do not provide any other information."""

    # SUMMARY_TEMPLATE: Focuses on creating a high-level executive summary.
    SUMMARY_TEMPLATE = """You are tasked with creating a high-level executive summary about '{topic}' using the provided news articles.

**News Articles Context:**
---
{context}
---

**Instructions for Your Summary:**

1.  **Headline:** Create a short, descriptive headline for the summary.
2.  **Executive Overview:** Write a 2-3 sentence paragraph that captures the most critical information and the overall situation regarding the topic.
3.  **Key Themes/Developments:** Identify 3-5 main themes or developments from the articles. Present each as a bullet point with a brief explanation.
4.  **Source Synthesis:** Mention the primary sources that contribute to this summary. For example: "This summary is based on reporting from The Daily Star, Prothom Alo, and BDNews24."

**CRITICAL:** The summary must be a synthesis of the information provided in the context. Do not add any outside information. If the articles are insufficient to create a summary, state that."""

    # TRENDING_TEMPLATE: Guides the model to act like an editor identifying top stories.
    TRENDING_TEMPLATE = """You are a news editor. Your task is to identify the top trending stories from today's news cycle based on the provided articles. Pay close attention to the 'Date' metadata to determine recency.

**News Articles Context:**
---
{context}
---

**Instructions for Your Trending News Report:**

1.  **Overall Headline:** Start with a headline summarizing the day's biggest news.
2.  **Top 3 Trending Stories:**
    * Identify the three most significant or most frequently covered topics in the provided articles.
    * For each story, provide:
        * A **bolded headline**.
        * A 2-3 sentence summary of the key events.
        * Mention the sources covering the story.
3.  **Other Notable News:** Briefly list any other important topics in a bulleted list.

**CRITICAL:** Your report must be based *exclusively* on the provided articles. Prioritize the most recent articles based on their 'Date' metadata."""

    # NO_SOURCES_TEMPLATE: This template is well-designed and remains unchanged.
    NO_SOURCES_TEMPLATE = """I couldn't find relevant information about "{query}" in the currently available news articles.

To get the most up-to-date information on this topic, I recommend checking:

**Primary Bangladeshi News Sources:**
• The Daily Star (thedailystar.net) - Leading English daily
• BDNews24 (bdnews24.com) - Comprehensive news coverage  
• Dhaka Tribune (dhakatribune.com) - International perspective
• New Age (newagebd.net) - Independent journalism
• United News of Bangladesh (UNB) - National news agency

**For Specific Topics:**
• Government updates: Press Information Department (PID)
• Economic news: The Financial Express, Bangladesh Bank
• Sports: Cricfrenzy, ESPN Cricinfo (for cricket)

**Alternative Search Suggestions:**
Try searching for related terms like: {suggestions}

Would you like me to help you with any other topics from the available articles?"""

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
    
    @classmethod
    def get_no_sources_response(cls, query: str, suggestions: List[str] = None) -> str:
        """Get formatted no sources response with suggestions"""
        if suggestions:
            suggestions_text = ", ".join(suggestions)
        else:
            suggestions_text = "[related keywords based on your query]"
        
        return cls.NO_SOURCES_TEMPLATE.format(query=query, suggestions=suggestions_text)


class ResponseMessages:
    """
    Collection of standard response message templates for the news RAG chatbot.
    This class provides user-facing messages for various scenarios encountered during
    news article search and retrieval, such as errors, no results found, successful
    queries, and outdated data warnings. Messages are designed to guide users with
    actionable suggestions and clear status updates.
    Attributes:
        NO_ARTICLES_LOADED (str): Message shown when no articles are loaded in the database.
        DATABASE_ERROR (str): Message for technical/database errors during search.
        INITIALIZATION_ERROR (str): Message for database initialization or connectivity issues.
        NO_RELEVANT_ARTICLES (str): Message when no articles match the user's query.
        NO_CLOSE_MATCHES (str): Message when only loosely related articles are found.
        NO_TITLE_MATCHES (str): Message when no articles match the topic in their titles.
        COUNT_ARTICLES_RESPONSE (str): Message summarizing the number of loaded articles and sources.
        COUNT_ARTICLES_WITH_SOURCES (str): Message listing available news sources.
        COUNT_TOPIC_RESPONSE (str): Message summarizing the number of articles found for a topic.
        COUNT_TOPIC_WITH_SOURCES (str): Message with a breakdown of sources for a topic.
        NO_TRENDING_NEWS (str): Message when trending news cannot be shown.
        OUTDATED_ARTICLES_WARNING (str): Warning message for outdated article data.
    Methods:
        get_no_relevant_articles(query: str) -> str:
            Returns a formatted message when no relevant articles are found for the query.
        get_no_close_matches(topic: str) -> str:
            Returns a formatted message when only loosely related articles are found.
        get_no_title_matches(topic: str) -> str:
            Returns a formatted message when no articles match the topic in their titles.
        get_count_articles_response(total_chunks: int, unique_sources: int, sources: List[str] = None) -> str:
            Returns a formatted message summarizing the number of loaded articles and sources.
        get_count_topic_response(count: int, topic: str, sources_breakdown: str = None) -> str:
            Returns a formatted message summarizing the number of articles found for a topic,
            optionally including a breakdown by source.
        get_outdated_warning(topic: str, date_range: str, available_info: str) -> str:
            Returns a formatted warning message when articles are outdated.
    """
    """Collection of standard response messages with improved source handling"""
    
    # Error messages
    NO_ARTICLES_LOADED = """No news articles are currently loaded in the database. 

To get started:
1. Use the 'Scrape Recent News' button in the sidebar to load fresh articles
2. Wait for the scraping process to complete
3. Then ask your question again

I'll be ready to help once articles are available!"""
    
    DATABASE_ERROR = """I encountered a technical error while searching the news database. 

Please try:
1. Rephrasing your question with different keywords
2. Checking if articles are properly loaded
3. Trying again in a moment

If the problem persists, consider re-scraping the news articles."""
    
    INITIALIZATION_ERROR = """I'm currently having trouble accessing the news database.

This might be due to:
- Database initialization issues
- No articles loaded yet
- Technical connectivity problems

Please try loading news articles first, then ask your question again."""
    
    # Enhanced not found messages with better guidance
    NO_RELEVANT_ARTICLES = """I couldn't find any articles related to '{query}' in the current database.

**Suggestions:**
• Try broader keywords (e.g., if searching for "Dhaka Metro Rail", try "metro" or "transportation")
• Check spelling and try alternative terms
• Ensure articles are recently scraped and loaded
• Consider that the topic might not be covered in current articles

**Alternative:** I can help you search for other topics that might be available in the loaded articles."""
    
    NO_CLOSE_MATCHES = """I found some articles but none closely match '{topic}'.

**Try these approaches:**
• Use more general terms (e.g., "economy" instead of "GDP growth rate")
• Try Bengali/English alternatives if applicable
• Search for related topics or broader categories
• Check if the topic is typically covered by Bangladeshi news sources

**Available:** I can show you what topics are currently available in the database."""
    
    NO_TITLE_MATCHES = """I couldn't find articles specifically about '{topic}' in the loaded database.

**Recommendations:**
• Try different phrasing or keywords
• Use broader search terms
• Ensure the topic is relevant to Bangladeshi news
• Check that recent articles have been scraped

**Help:** Would you like me to search for related topics or show what's currently available?"""
    
    # Enhanced success messages with source details
    COUNT_ARTICLES_RESPONSE = """I have **{total_chunks}** article chunks from **{unique_sources}** news sources loaded and ready for search.

**Database Status:** ✅ Ready to answer questions
**Coverage:** Recent Bangladeshi news from multiple sources
**Last Updated:** [Timestamp when articles were scraped]"""
    
    COUNT_ARTICLES_WITH_SOURCES = """I have **{total_chunks}** article chunks from **{unique_sources}** news sources loaded.

**Available News Sources:**
{sources_list}

**Coverage Areas:** Politics, Economy, Sports, International, Technology, Health, Education, and more.
**Status:** ✅ Ready to answer questions about recent Bangladeshi news."""
    
    COUNT_TOPIC_RESPONSE = """I found **{count}** relevant article chunks about **'{topic}'**.

**Coverage:** Multiple perspectives and recent developments
**Ready to provide:** Detailed summary, key facts, timeline, and source-attributed information."""
    
    COUNT_TOPIC_WITH_SOURCES = """I found **{count}** article chunks related to **'{topic}'**.

**Coverage by Source:**
{sources_breakdown}

**Available Information:** Recent developments, key facts, analysis, and source-verified details."""
    
    # Enhanced trending news handling
    NO_TRENDING_NEWS = """No news articles are currently available to show trending topics.

**Next Steps:**
1. Click the 'Scrape Recent News' button in the sidebar
2. Wait for the scraping process to complete (usually 30-60 seconds)
3. Ask for trending news again

**What you'll get:** Latest stories from major Bangladeshi news sources with full source attribution and publication dates."""
    
    # New: Template for when articles are old/outdated
    OUTDATED_ARTICLES_WARNING = """⚠️ **Note:** The available articles appear to be from {date_range}. For the most current news on '{topic}', consider:

**Current Sources:**
• Visit news websites directly for breaking news
• Check social media accounts of major news outlets
• Look for government press releases on official websites

**Existing Information:** Based on available articles, here's what I found:
{available_info}"""
    
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
    
    @classmethod
    def get_outdated_warning(cls, topic: str, date_range: str, available_info: str) -> str:
        """Get formatted outdated articles warning"""
        return cls.OUTDATED_ARTICLES_WARNING.format(
            topic=topic, 
            date_range=date_range, 
            available_info=available_info
        )


class PromptBuilder:
    """Prompt builder class for constructing complex prompts with better source handling"""
    
    def __init__(self):
        self.context = ""
        self.query = ""
        self.topic = ""
        self.additional_instructions = []
        self.source_metadata = {}
        self.require_recent = False
    
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
    
    def with_source_metadata(self, metadata: dict) -> 'PromptBuilder':
        """Add source metadata (publication dates, source names, etc.)"""
        self.source_metadata = metadata
        return self
    
    def require_recent_sources(self, require: bool = True) -> 'PromptBuilder':
        """Flag that response should prioritize recent sources"""
        self.require_recent = require
        return self
    
    def add_instruction(self, instruction: str) -> 'PromptBuilder':
        """Add additional instruction"""
        self.additional_instructions.append(instruction)
        return self
    
    def _build_source_instruction(self) -> str:
        """Build source-specific instructions based on metadata"""
        if not self.source_metadata:
            return ""
        
        instructions = []
        
        if 'date_range' in self.source_metadata:
            instructions.append(f"Articles span from {self.source_metadata['date_range']}")
        
        if 'total_sources' in self.source_metadata:
            instructions.append(f"Information from {self.source_metadata['total_sources']} different publications")
        
        if self.require_recent:
            instructions.append("Prioritize the most recent information and emphasize publication dates")
        
        if 'source_reliability' in self.source_metadata:
            instructions.append("Note the credibility and reliability of sources when presenting information")
        
        return "\n".join(f"- {instruction}" for instruction in instructions)
    
    def build_summary_prompt(self) -> str:
        """Build summary prompt with enhanced source handling"""
        prompt = PromptTemplates.get_summary_prompt(self.topic, self.context)
        
        # Add source-specific instructions
        source_instructions = self._build_source_instruction()
        if source_instructions:
            prompt += f"\n\nSource-Specific Instructions:\n{source_instructions}\n"
        
        # Add custom instructions
        if self.additional_instructions:
            prompt += "\n\nAdditional Instructions:\n"
            for instruction in self.additional_instructions:
                prompt += f"- {instruction}\n"
        
        return prompt
    
    def build_general_prompt(self) -> str:
        """Build general search prompt with enhanced source handling"""
        prompt = PromptTemplates.get_general_search_prompt(self.query, self.context)
        
        # Add source-specific instructions
        source_instructions = self._build_source_instruction()
        if source_instructions:
            prompt += f"\n\nSource-Specific Instructions:\n{source_instructions}\n"
        
        # Add custom instructions
        if self.additional_instructions:
            prompt += "\n\nAdditional Instructions:\n"
            for instruction in self.additional_instructions:
                prompt += f"- {instruction}\n"
        
        return prompt
    
    def build_trending_prompt(self) -> str:
        """Build trending prompt with enhanced source and recency handling"""
        prompt = PromptTemplates.get_trending_prompt(self.context)
        
        # Add recency emphasis for trending queries
        prompt += "\n\nTrending News Priority Instructions:\n"
        prompt += "- Focus on the most recent articles first\n"
        prompt += "- Highlight developing or ongoing stories\n"
        prompt += "- Emphasize publication dates and timeline\n"
        prompt += "- Group stories by topic when multiple sources cover the same event\n"
        prompt += "- Distinguish between breaking news and ongoing coverage\n"
        
        # Add source-specific instructions
        source_instructions = self._build_source_instruction()
        if source_instructions:
            prompt += f"\nSource Metadata:\n{source_instructions}\n"
        
        # Add custom instructions
        if self.additional_instructions:
            prompt += "\nAdditional Instructions:\n"
            for instruction in self.additional_instructions:
                prompt += f"- {instruction}\n"
        
        return prompt
    
    def reset(self) -> 'PromptBuilder':
        """Reset builder state"""
        self.context = ""
        self.query = ""
        self.topic = ""
        self.additional_instructions = []
        self.source_metadata = {}
        self.require_recent = False
        return self


class SourceAttributionHelper:
    """Helper class for managing source attribution and metadata"""
    
    @staticmethod
    def extract_source_info(article_chunk: str) -> dict:
        """Extract source information from article chunk"""
        # This would parse metadata from your article chunks
        # Assuming format includes source name, date, etc.
        source_info = {
            'publication': 'Unknown',
            'date': 'Unknown',
            'url': None,
            'reliability_score': None
        }
        
        # Add parsing logic based on your chunk format
        # Example: if chunks contain metadata headers
        if '[SOURCE:' in article_chunk:
            # Parse source metadata
            pass
        
        return source_info
    
    @staticmethod
    def format_citation(publication: str, date: str = None, url: str = None) -> str:
        """Format a proper citation for the response"""
        if date and date != 'Unknown':
            return f"According to {publication} (published {date})"
        else:
            return f"According to {publication}"
    
    @staticmethod
    def group_sources_by_topic(articles: List[dict]) -> dict:
        """Group articles by topic/theme for better organization"""
        # Implementation would depend on your article structure
        grouped = {}
        
        for article in articles:
            topic = article.get('topic', 'General')
            if topic not in grouped:
                grouped[topic] = []
            grouped[topic].append(article)
        
        return grouped


class QueryClassifier:
    """Classify queries to determine the best prompt template"""
    
    TRENDING_KEYWORDS = [
        'trending', 'latest', 'recent', 'current', 'today', 'this week', 
        'breaking', 'news today', 'what\'s happening', 'updates', 'now'
    ]
    
    SUMMARY_KEYWORDS = [
        'summary', 'about', 'explain', 'overview', 'tell me about', 
        'what is', 'background', 'details about'
    ]
    
    @classmethod
    def classify_query(cls, query: str) -> str:
        """Classify query type to choose appropriate template"""
        query_lower = query.lower()
        
        # Check for trending/latest news
        if any(keyword in query_lower for keyword in cls.TRENDING_KEYWORDS):
            return 'trending'
        
        # Check for summary requests
        if any(keyword in query_lower for keyword in cls.SUMMARY_KEYWORDS):
            return 'summary'
        
        # Default to general search
        return 'general'
    
    @classmethod
    def suggest_alternatives(cls, query: str) -> List[str]:
        """Suggest alternative search terms if no results found"""
        # Simple keyword extraction and suggestion logic
        words = query.lower().split()
        suggestions = []
        
        # Remove common stop words and suggest variations
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        meaningful_words = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Take first 3-5 meaningful words as suggestions
        suggestions.extend(meaningful_words[:5])
        
        return suggestions