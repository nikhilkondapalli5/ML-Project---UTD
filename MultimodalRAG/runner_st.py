import os
import logging
from PIL import Image
from rag_generator import RAGGenerator
from typing import Dict, List, Tuple
import streamlit as st

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StreamlitRAGInterface:
    def __init__(self):
        # Store the generator instance in session state to persist its internal history
        if 'generator' not in st.session_state:
            st.session_state.generator = RAGGenerator()
            logger.info("Initialized RAGGenerator in session state.")
        
        # Access the generator from session state
        self.generator = st.session_state.generator
        self.image_dir = "pic_data"
        
        # No longer need a separate conversation history in session state
        # if 'conversation_history' not in st.session_state:
        #     st.session_state.conversation_history = []

    def display_conversation_history(self):
        """Display past conversations in sidebar using generator's history"""
        with st.sidebar:
            st.markdown("### Conversation History")
            if st.button("Clear History"):
                # Reset the generator's internal history
                self.generator.reset_conversation()
                logger.info("Cleared generator conversation history via Streamlit button.")
                # Rerun to reflect the cleared history
                st.rerun()
            
            # Display previous conversations from the generator's history
            # Access the generator's history (list of {'user': ..., 'assistant': ...} dicts)
            history = self.generator.conversation_history
            if not history:
                st.info("No conversation history yet.")
                
            # Display up to the second to last exchange in the sidebar expander
            # The last exchange is displayed in the main area
            for i, exchange in enumerate(history[:-1], 1):
                st.markdown(f"**Q{i}: {exchange['user'][:30]}...**")
                with st.expander("View Details"):
                    st.markdown("**Question:**")
                    st.write(exchange['user'])
                    st.markdown("**Answer:**")
                    st.write(exchange['assistant'])

    def format_sources_for_display(self, sources: List[Dict]) -> None:
        """Format and display sources in Streamlit"""
        if not sources:
            st.warning("No sources used.")
            return
            
        st.markdown("#### Sources:")
        # Dynamically adjust columns based on number of sources
        num_sources = len(sources)
        num_cols = min(num_sources, 3) # Max 3 columns
        cols = st.columns(num_cols) if num_sources > 0 else []
        
        for i, source in enumerate(sources, 1):
            col_index = (i - 1) % num_cols
            with cols[col_index]:
                with st.expander(f"Source {i}"):
                    if source['type'] == 'text':
                        st.markdown(f"**Text from page {source['page']}**")
                        st.markdown(f"Preview: {source['content_preview']}")
                    else:
                        st.markdown(f"**Image from page {source['page']}**")
                        # Use the corrected path handling logic
                        image_path = source['path']
                        image_path = image_path.replace('\\', '/') # Normalize separators
                        
                        # Construct absolute path relative to the script location if needed
                        if not os.path.isabs(image_path):
                            script_dir = os.path.dirname(__file__)
                            abs_image_path = os.path.abspath(os.path.join(script_dir, image_path))
                        else:
                            abs_image_path = image_path

                        logger.info(f"Attempting to display image: {abs_image_path}")
                        
                        if os.path.exists(abs_image_path):
                            try:
                                st.image(abs_image_path, caption=f"Source Image {i}")
                            except Exception as e:
                                st.warning(f"Error displaying image: {str(e)}")
                                st.markdown(f"Image path: {abs_image_path}")
                        else:
                            st.warning(f"Image not found: {abs_image_path}")
                            st.markdown("**Debug Info:**")
                            st.markdown(f"- Original path: {source['path']}")
                            st.markdown(f"- Current working directory: {os.getcwd()}")
                            st.markdown(f"- Script directory: {os.path.dirname(__file__)}")
                            st.markdown(f"- Attempted absolute path: {abs_image_path}")

    def run_interface(self):
        """Run the Streamlit interface"""
        st.title("Multimodal RAG System - Corporate Document Analysis")
        
        # Help section in sidebar
        with st.sidebar:
            st.markdown("### Help")
            st.markdown("""
            **Available Features:**
            - Enter your question in the text input
            - View answer and sources below
            - Sources include both text and images
            - Past conversations stored and used for context
            - View past conversations in the sidebar
            - Clear history using the button in the sidebar
            """)
            st.markdown("---")
        
        # Display conversation history in sidebar
        self.display_conversation_history()
        
        st.markdown("---")
        
        # Main query interface
        query = st.text_input("Enter your question:", 
                            placeholder="e.g., What is the relationship between LTIMindtree and IBM?")
        
        # Display the latest conversation exchange if it exists
        if self.generator.conversation_history:
            last_exchange = self.generator.conversation_history[-1]
            st.markdown("### Last Interaction")
            st.markdown("**Your Question:**")
            st.write(last_exchange['user'])
            st.markdown("**Answer:**")
            st.write(last_exchange['assistant'])
            # Note: Sources for the last interaction are not stored directly in the generator's history.
            # We might need to store them separately in session state if needed for redisplay without re-querying.
            # For now, sources are displayed only immediately after a query.
            st.markdown("---")


        if query:
            # Check if this query is the same as the last one to avoid reprocessing on simple reruns
            is_new_query = not self.generator.conversation_history or self.generator.conversation_history[-1]['user'] != query
            
            if is_new_query:
                try:
                    with st.spinner("Processing your query..."):
                        # Process query using the generator (which uses its internal history)
                        result = self.generator.generate_answer(
                            query=query,
                            n_text_results=5,
                            n_image_results=3,
                            temperature=0.1
                        )
                        
                        # The generator already updated its internal history.
                        # No need to manually append to st.session_state.conversation_history
                        # st.session_state.conversation_history.append({
                        #     'query': query,
                        #     'answer': result['answer'],
                        #     'sources': result['sources']
                        # })
                        
                        # Store the sources of the *latest* query in session state for potential redisplay
                        st.session_state.latest_sources = result['sources']

                        # Rerun to display the new results immediately
                        st.rerun()

                except Exception as e:
                    logger.error(f"Error processing query: {e}")
                    st.error(f"An error occurred: {str(e)}")
            else:
                 # If it's the same query, display the sources from the last run
                 if 'latest_sources' in st.session_state:
                     st.markdown("### Sources for Last Answer")
                     self.format_sources_for_display(st.session_state.latest_sources)


def main():
    interface = StreamlitRAGInterface()
    interface.run_interface()

if __name__ == "__main__":
    main()