import os
import logging
from PIL import Image
from rag_generator import RAGGenerator
from typing import Dict, List, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TerminalRAGInterface:
    """
    A terminal interface for the multimodal RAG system.
    """
    def __init__(self):
        self.generator = RAGGenerator()
        self.image_dir = "pic_data"
        
    def format_sources_for_display(self, sources: List[Dict]) -> str:
        """Format sources for display in the terminal"""
        if not sources:
            return "No sources used."
            
        source_text = "\nSources:\n"
        for i, source in enumerate(sources, 1):
            if source['type'] == 'text':
                source_text += f"\nSource {i}: Text from page {source['page']}\n"
                source_text += f"Preview: {source['content_preview']}\n"
            else:
                source_text += f"\nSource {i}: Image from page {source['page']}\n"
                source_text += f"Path: {source['path']}\n"
                
        return source_text
        
    def run_interface(self):
        """Run the terminal interface"""
        print("\nMultimodal RAG System - Corporate Document Analysis")
        print("================================================")
        print("Type 'exit' to quit, 'help' for commands\n")
        
        while True:
            try:
                # Get user input
                query = input("\nEnter your question: ").strip()
                
                if query.lower() == 'exit':
                    print("\nGoodbye!")
                    break
                    
                if query.lower() == 'help':
                    print("\nAvailable commands:")
                    print("- exit: Exit the program")
                    print("- help: Show this help message")
                    print("Just type your question to query the system\n")
                    continue
                
                if not query:
                    print("Please enter a question.")
                    continue
                
                # Process query
                print("\nProcessing your query...")
                result = self.generator.generate_answer(
                    query=query,
                    n_text_results=5,
                    n_image_results=3,
                    temperature=0.1
                )
                
                # Display results
                print("\nAnswer:")
                print("-------")
                print(result['answer'])
                
                # Display sources
                print(self.format_sources_for_display(result['sources']))
                
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                logger.error(f"Error: {e}")
                print(f"\nAn error occurred: {str(e)}")

if __name__ == "__main__":
    interface = TerminalRAGInterface()
    interface.run_interface()