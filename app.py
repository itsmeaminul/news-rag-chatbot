""""
Streamlit web application for BD News RAG Chatbot
"""
import json
import logging
import re
from datetime import datetime
from typing import List

import streamlit as st

from src.config import UI_CONFIG, RAW_ARTICLES_PATH, PROCESSED_ARTICLES_PATH
from src.rag_pipeline import RAGPipeline
from src.chroma_manager import ChromaManager
from src.news_scraper import NewsScraper
from src.text_preprocessor import TextProcessor


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatbotUI:
    """Streamlit UI for the news chatbot"""

    def __init__(self):
        self.setup_page_config()
        self.initialize_session_state()
        self.rag_pipeline = None
        self.db_manager = None

    def setup_page_config(self):
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title=UI_CONFIG['page_title'],
            page_icon=UI_CONFIG['page_icon'],
            layout=UI_CONFIG['layout'],
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS for better formatting
        st.markdown("""
        <style>
        .source-section {
            background-color: #f0f2f6;
            padding: 10px;
            border-radius: 5px;
            margin-top: 10px;
        }
        .source-link {
            color: #0066cc;
            text-decoration: none;
            font-size: 0.9em;
        }
        .source-link:hover {
            text-decoration: underline;
        }
        .key-points {
            background-color: #fff3cd;
            padding: 10px;
            border-radius: 5px;
            border-left: 4px solid #ffc107;
        }
        </style>
        """, unsafe_allow_html=True)

    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'messages' not in st.session_state:
            st.session_state.messages = []

        if 'rag_initialized' not in st.session_state:
            st.session_state.rag_initialized = False

        if 'db_stats' not in st.session_state:
            st.session_state.db_stats = {}

        if 'articles_scraped' not in st.session_state:
            st.session_state.articles_scraped = 0

        # Add chat session ID for tracking
        if 'chat_session_id' not in st.session_state:
            st.session_state.chat_session_id = datetime.now().isoformat()
        
        # Add state for reset confirmation
        if 'confirming_reset' not in st.session_state:
            st.session_state.confirming_reset = False

    def load_rag_pipeline(self):
        """Load and initialize RAG pipeline"""
        if not st.session_state.rag_initialized:
            try:
                with st.spinner("Initializing RAG pipeline..."):
                    self.rag_pipeline = RAGPipeline()
                    self.db_manager = ChromaManager()
                    st.session_state.rag_initialized = True
                st.success("RAG pipeline initialized successfully!")
            except Exception as e:
                st.error(f"Error initializing RAG pipeline: {e}")
                logger.error(f"RAG initialization error: {e}")
                return False
        else:
            self.rag_pipeline = RAGPipeline()
            self.db_manager = ChromaManager()

        return True

    def render_sidebar(self):
        """Render sidebar with controls and information"""
        with st.sidebar:
            # New Chat Button
            if st.button("📝 New Chat", key="new_chat_btn", use_container_width=True, type="primary"):
                self.start_new_chat()

            # --- Previous Chat Sessions ---
            if st.session_state.get('chat_history_sessions'):
                st.subheader("📜 Chat History")
                for idx, session in enumerate(reversed(st.session_state.chat_history_sessions)):
                    label = session.get('label', 'No question')

                    # Highlight if active
                    is_active = st.session_state.get('active_chat_idx') == len(st.session_state.chat_history_sessions) - 1 - idx
                    button_label = f"👉 {label[:20]}..." if is_active else f"{label[:30]}..."

                    if st.button(button_label, key=f"prev_session_{idx}"):
                        st.session_state.messages = session['messages']
                        st.session_state.active_chat_idx = len(st.session_state.chat_history_sessions) - 1 - idx
                        st.rerun()
            
            st.divider()

            st.subheader("🗄️ Database Management")
            # Action Buttons
            if st.button("🔄 Scrape Recent News", key="sidebar_scrape_btn", use_container_width=True):
                self.scrape_news()
            
            # The "Reset Database" button now sets a flag to show the confirmation UI.
            if st.button("🗑️ Reset Database", key="sidebar_reset_btn", use_container_width=True):
                st.session_state.confirming_reset = True
            
            # If the confirmation flag is set, display the confirmation options.
            if st.session_state.get('confirming_reset', False):
                st.warning("Are you sure? This will delete all the scrapped news from the database.")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✅ Confirm", key="confirm_reset_action", use_container_width=True, type="primary"):
                        self.reset_database()
                        st.session_state.confirming_reset = False
                        st.rerun()
                with col2:
                    if st.button("❌ Cancel", key="cancel_reset_action", use_container_width=True):
                        st.session_state.confirming_reset = False
                        st.rerun()

            # Database Statistics
            with st.expander("📈 Database Stats", expanded=False):
                self.display_database_stats()

            st.divider()

            # Sample Queries
            st.subheader("💡 Sample Queries")
            sample_queries = [
                "What is the trending news today?",
                "How many articles are there?",
                "Tell me about Bangladesh politics",
                "Summary of recent sports news",
                "Latest news from Prothom Alo",
            ]

            for query in sample_queries:
                if st.button(f"💬 {query}", key=f"sample_{hash(query)}", use_container_width=True):
                    st.session_state.sample_query = query

    def start_new_chat(self):
        """Start a new chat session while preserving previous chat history"""
        if st.session_state.messages:
            # Use the first user message as the session label
            first_question = next(
                (msg["content"] for msg in st.session_state.messages if msg["role"] == "user"),
                "No question"
            )

            if 'chat_history_sessions' not in st.session_state:
                st.session_state.chat_history_sessions = []

            # Prevent duplicates (store unique based on messages or label)
            if not any(s['messages'] == st.session_state.messages for s in st.session_state.chat_history_sessions):
                st.session_state.chat_history_sessions.append({
                    'label': first_question,
                    'messages': st.session_state.messages.copy()
                })

        # Start a fresh chat
        st.session_state.messages = []
        st.session_state.chat_session_id = datetime.now().isoformat()
        st.session_state.active_chat_idx = None  # Reset active
        st.success("Started new chat session! Previous chat history is preserved.")
        st.rerun()

    def scrape_news(self):
        """Scrape news, process them, and index in DB"""
        try:
            with st.spinner("Scraping news articles... This may take a few minutes."):
                scraper = NewsScraper()
                articles = scraper.scrape_all_sources()
                scraper.save_articles()

                st.session_state.articles_scraped = len(articles)
                st.success(f"✅ Scraped {len(articles)} articles!")

            # Auto process after scraping
            self.process_articles(auto_trigger=True)

            # Auto index after processing
            self.index_articles(auto_trigger=True)

        except Exception as e:
            st.error(f"Error scraping news: {e}")
            logger.error(f"Scraping error: {e}")

    def process_articles(self, auto_trigger=False):
        """Process scraped articles"""
        try:
            with open(RAW_ARTICLES_PATH, 'r', encoding='utf-8', errors='ignore') as f:
                articles = json.load(f)

            with st.spinner("Processing articles..."):
                processor = TextProcessor()
                chunks = processor.process_articles(articles)
                processor.save_processed_chunks()
                stats = processor.get_processing_stats()

            if not auto_trigger:
                st.success(f"✅ Processed {len(chunks)} chunks!")

        except FileNotFoundError:
            st.error("No scraped articles found. Please scrape articles first.")
        except Exception as e:
            st.error(f"Error processing articles: {e}")
            logger.error(f"Processing error: {e}")

    def index_articles(self, auto_trigger=False):
        """Index processed articles in DB"""
        try:
            if not self.load_rag_pipeline():
                return

            with open(PROCESSED_ARTICLES_PATH, 'r') as f:
                chunks = json.load(f)

            with st.spinner("Indexing articles..."):
                self.db_manager.reset_collection()
                success = self.db_manager.add_chunks(chunks)

            if success:
                st.session_state.db_stats = self.db_manager.get_collection_stats()
                if not auto_trigger:
                    st.success(f"✅ Indexed {len(chunks)} chunks!")

        except FileNotFoundError:
            st.error("No processed articles found. Please process articles first.")
        except Exception as e:
            st.error(f"Error indexing articles: {e}")
            logger.error(f"Indexing error: {e}")

    # This function is now simplified to only perform the reset action.
    # The confirmation UI is handled in the render_sidebar method.
    def reset_database(self):
        """Reset vector database after confirmation."""
        try:
            if not self.load_rag_pipeline():
                return

            with st.spinner("Resetting database..."):
                success = self.db_manager.reset_collection()

                if success:
                    st.success("✅ Successfully reset database! Database is empty now.")
                    st.session_state.db_stats = {}
                else:
                    st.error("Failed to reset database.")
        except Exception as e:
            st.error(f"Error resetting database: {e}")
            logger.error(f"Database reset error: {e}")

    def display_database_stats(self):
        """Display database statistics in sidebar"""
        if not st.session_state.db_stats and hasattr(self, 'db_manager') and self.db_manager:
            try:
                st.session_state.db_stats = self.db_manager.get_collection_stats()
            except Exception as e:
                logger.error(f"Error getting database stats: {e}")

        stats = st.session_state.db_stats

        if stats:
            st.write(f"**Total Articles:** {st.session_state.articles_scraped}")
            st.write(f"**Unique Sources:** {stats.get('unique_sources', 0)}")
            sources = stats.get('sources', [])
            if sources:
                st.write("**Sources:**")
                for source in sources:
                    st.write(f"• {source}")
        else:
            st.info("No database statistics available. Scrape news to see stats.")

    def extract_sources_from_response(self, response: str) -> List[dict]:
        """Extract source information from response"""
        sources = []
        
        # Look for sources section in response
        if "**Sources:**" in response:
            sources_section = response.split("**Sources:**")[1]
            source_lines = sources_section.split('\n')
            
            current_source = {}
            for line in source_lines:
                line = line.strip()
                if line and line[0].isdigit():
                    # New source entry
                    if current_source:
                        sources.append(current_source)
                    
                    # Extract title and source name
                    match = re.search(r'\d+\.\s*\*\*(.*?)\*\*\s*-\s*(.*?)$', line)
                    if match:
                        current_source = {
                            'title': match.group(1).strip(),
                            'source': match.group(2).strip(),
                            'url': ''
                        }
                elif line.startswith('Link:') and current_source:
                    current_source['url'] = line.replace('Link:', '').strip()
            
            # Add the last source if exists
            if current_source:
                sources.append(current_source)
        
        return sources

    def format_response_with_sources(self, response: str) -> tuple:
        """Format response by separating main content from sources"""
        if "**Sources:**" in response:
            parts = response.split("**Sources:**")
            main_response = parts[0].strip()
            sources = self.extract_sources_from_response(response)
            return main_response, sources
        
        return response, []

    def render_sources_section(self, sources: List[dict]):
        """Render sources section in Streamlit"""
        if not sources:
            return
        
        st.markdown("---")
        st.markdown("### 📰 Sources")
        
        for i, source in enumerate(sources, 1):
            with st.expander(f"{i}. {source['title']} - {source['source']}", expanded=False):
                if source['url']:
                    st.markdown(f"[Read full article]({source['url']})")
                else:
                    st.write("URL not available")

    def render_main_chat(self):
        """Render main chat interface"""
        st.title(f"{UI_CONFIG['page_icon']} {UI_CONFIG['page_title']}")
        st.markdown("Welcome to the Bangladesh News RAG Chatbot! Ask me anything about recent news.")

        # Display current chat session info
        st.caption(f"Chat session: {st.session_state.chat_session_id[:19]}")

        # Initialize RAG pipeline
        if not self.load_rag_pipeline():
            st.error("Failed to initialize chatbot. Please check the logs.")
            return

        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Display sources if available
                if message["role"] == "assistant" and "sources" in message:
                    self.render_sources_section(message["sources"])

        # Handle sample query from sidebar
        if hasattr(st.session_state, 'sample_query'):
            query = st.session_state.sample_query
            del st.session_state.sample_query
            self.process_user_query(query)

        # Chat input
        prompt = st.chat_input("Ask me about Bangladesh news...")
        if prompt:
            self.process_user_query(prompt)

    def process_user_query(self, query: str):
        """Process user query and display clean formatted response"""
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = self.rag_pipeline.process_query(query)
                    
                    # Separate main response from sources
                    main_response, sources = self.format_response_with_sources(response)
                    
                    # Display main response
                    st.markdown(main_response)
                    
                    # Display sources section
                    if sources:
                        self.render_sources_section(sources)
                    
                    # Store message with sources for chat history
                    message_data = {"role": "assistant", "content": main_response}
                    if sources:
                        message_data["sources"] = sources
                    
                    st.session_state.messages.append(message_data)
                    
                except Exception as e:
                    error_msg = f"I encountered an error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    logger.error(f"Query processing error: {e}")

    def run(self):
        """Main application runner"""
        self.render_sidebar()
        self.render_main_chat()



def main():
    try:
        app = ChatbotUI()
        app.run()
    except Exception as e:
        st.error(f"Application error: {e}")
        logger.error(f"Application error: {e}")


if __name__ == "__main__":
    main()