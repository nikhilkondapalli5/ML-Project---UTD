import json
import logging
import time
from typing import Dict, List, Tuple
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from rag_generator import RAGGenerator
from multimodal_retriever import MultimodalRetriever

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGEvaluator:
    """
    Evaluates the performance of the multimodal RAG system
    """
    def __init__(self):
        self.generator = RAGGenerator()
        self.retriever = MultimodalRetriever()
        self.test_queries = [
            "What were the financial highlights from the last fiscal year?",
            "Explain the company's sustainability initiatives.",
            "Who are the members of the board of directors?",
            "What are the key risk factors mentioned in the report?",
            "How has the company performed in terms of revenue growth?",
            "What is the company's approach to digital transformation?",
            "What are the strategic priorities for the coming year?",
            "Describe the company's corporate governance structure.",
            "What investments has the company made in R&D?",
            "How does the company address environmental concerns?"
        ]
        
    def measure_retrieval_performance(self) -> Dict:
        """
        Measure retrieval performance metrics
        
        Returns:
            Dictionary with metrics
        """
        results = {
            'query': [],
            'text_results_count': [],
            'image_results_count': [],
            'retrieval_time': []
        }
        
        logger.info("Measuring retrieval performance...")
        for query in tqdm(self.test_queries):
            try:
                # Measure retrieval time
                start_time = time.time()
                retrieval_results = self.retriever.hybrid_query(
                    query=query,
                    n_text_results=5,
                    n_image_results=3
                )
                retrieval_time = time.time() - start_time
                
                # Record results
                results['query'].append(query)
                results['text_results_count'].append(len(retrieval_results['text_results']))
                results['image_results_count'].append(len(retrieval_results['image_results']))
                results['retrieval_time'].append(retrieval_time)
                
            except Exception as e:
                logger.error(f"Error evaluating query '{query}': {e}")
                # Record failure
                results['query'].append(query)
                results['text_results_count'].append(0)
                results['image_results_count'].append(0)
                results['retrieval_time'].append(0)
        
        return results
    
    def measure_generation_performance(self) -> Dict:
        """
        Measure generation performance metrics
        
        Returns:
            Dictionary with metrics
        """
        results = {
            'query': [],
            'answer_length': [],
            'sources_count': [],
            'generation_time': [],
            'has_image_sources': []
        }
        
        logger.info("Measuring generation performance...")
        for query in tqdm(self.test_queries):
            try:
                # Measure generation time
                start_time = time.time()
                answer_results = self.generator.generate_answer(
                    query=query,
                    n_text_results=5,
                    n_image_results=3
                )
                generation_time = time.time() - start_time
                
                # Count image sources
                image_sources = [s for s in answer_results['sources'] if s['type'] == 'image']
                
                # Record results
                results['query'].append(query)
                results['answer_length'].append(len(answer_results['answer']))
                results['sources_count'].append(len(answer_results['sources']))
                results['generation_time'].append(generation_time)
                results['has_image_sources'].append(len(image_sources) > 0)
                
            except Exception as e:
                logger.error(f"Error generating answer for '{query}': {e}")
                # Record failure
                results['query'].append(query)
                results['answer_length'].append(0)
                results['sources_count'].append(0)
                results['generation_time'].append(0)
                results['has_image_sources'].append(False)
        
        return results
    
    def run_evaluation(self, save_results: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run a full evaluation on the RAG system
        
        Args:
            save_results: Whether to save results to disk
            
        Returns:
            Tuple of DataFrames with retrieval and generation performance
        """
        # Measure retrieval performance
        retrieval_metrics = self.measure_retrieval_performance()
        retrieval_df = pd.DataFrame(retrieval_metrics)
        
        # Measure generation performance
        generation_metrics = self.measure_generation_performance()
        generation_df = pd.DataFrame(generation_metrics)
        
        # Save results if requested
        if save_results:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            retrieval_df.to_csv(f"evaluation_retrieval_{timestamp}.csv", index=False)
            generation_df.to_csv(f"evaluation_generation_{timestamp}.csv", index=False)
            
            # Save combined metrics as JSON
            with open(f"evaluation_metrics_{timestamp}.json", 'w') as f:
                json.dump({
                    'retrieval': {
                        'avg_time': retrieval_df['retrieval_time'].mean(),
                        'avg_text_results': retrieval_df['text_results_count'].mean(),
                        'avg_image_results': retrieval_df['image_results_count'].mean()
                    },
                    'generation': {
                        'avg_time': generation_df['generation_time'].mean(),
                        'avg_answer_length': generation_df['answer_length'].mean(),
                        'avg_sources_count': generation_df['sources_count'].mean(),
                        'pct_with_images': generation_df['has_image_sources'].mean() * 100
                    }
                }, f, indent=2)
        
        return retrieval_df, generation_df
    
    def plot_performance_metrics(self, retrieval_df: pd.DataFrame, generation_df: pd.DataFrame, save_plot: bool = True):
        """
        Create visualizations of performance metrics
        
        Args:
            retrieval_df: DataFrame with retrieval metrics
            generation_df: DataFrame with generation metrics
            save_plot: Whether to save the plots to disk
        """
        # Create figure with 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot retrieval time
        axes[0, 0].bar(range(len(retrieval_df)), retrieval_df['retrieval_time'])
        axes[0, 0].set_title('Retrieval Time by Query')
        axes[0, 0].set_ylabel('Time (seconds)')
        axes[0, 0].set_xticks(range(len(retrieval_df)))
        axes[0, 0].set_xticklabels(range(1, len(retrieval_df) + 1))
        axes[0, 0].set_xlabel('Query Number')
        
        # Plot retrieval counts
        width = 0.35
        x = range(len(retrieval_df))
        axes[0, 1].bar([i - width/2 for i in x], retrieval_df['text_results_count'], width, label='Text')
        axes[0, 1].bar([i + width/2 for i in x], retrieval_df['image_results_count'], width, label='Images')
        axes[0, 1].set_title('Retrieved Results by Query')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(range(1, len(retrieval_df) + 1))
        axes[0, 1].set_xlabel('Query Number')
        axes[0, 1].legend()
        
        # Plot generation time
        axes[1, 0].bar(range(len(generation_df)), generation_df['generation_time'])
        axes[1, 0].set_title('Generation Time by Query')
        axes[1, 0].set_ylabel('Time (seconds)')
        axes[1, 0].set_xticks(range(len(generation_df)))
        axes[1, 0].set_xticklabels(range(1, len(generation_df) + 1))
        axes[1, 0].set_xlabel('Query Number')
        
        # Plot answer length and sources
        ax1 = axes[1, 1]
        ax1.set_title('Answer Length and Sources by Query')
        ax1.set_xlabel('Query Number')
        ax1.set_ylabel('Answer Length (chars)', color='tab:blue')
        ax1.bar(range(len(generation_df)), generation_df['answer_length'], color='tab:blue', alpha=0.7)
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.set_xticks(range(len(generation_df)))
        ax1.set_xticklabels(range(1, len(generation_df) + 1))
        
        ax2 = ax1.twinx()
        ax2.set_ylabel('Sources Count', color='tab:red')
        ax2.plot(range(len(generation_df)), generation_df['sources_count'], 'o-', color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')
        
        plt.tight_layout()
        
        if save_plot:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            plt.savefig(f"evaluation_plots_{timestamp}.png", dpi=300)
            
        plt.show()


# Run evaluation if executed directly
if __name__ == "__main__":
    evaluator = RAGEvaluator()
    logger.info("Starting RAG system evaluation...")
    
    # Run evaluation
    retrieval_df, generation_df = evaluator.run_evaluation()
    
    # Plot results
    evaluator.plot_performance_metrics(retrieval_df, generation_df)
    
    # Print summary
    print("\nEvaluation Summary:")
    print(f"Average Retrieval Time: {retrieval_df['retrieval_time'].mean():.3f} seconds")
    print(f"Average Generation Time: {generation_df['generation_time'].mean():.3f} seconds")
    print(f"Average Answer Length: {generation_df['answer_length'].mean():.1f} characters")
    print(f"Average Number of Sources: {generation_df['sources_count'].mean():.1f}")
    print(f"Percentage of Answers with Images: {generation_df['has_image_sources'].mean()*100:.1f}%")