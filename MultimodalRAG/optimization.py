import logging
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, List, Tuple, Any
from multimodal_retriever import MultimodalRetriever
from rag_generator import RAGGenerator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGOptimizer:
    """
    Optimizes and fine-tunes the parameters of the multimodal RAG system.
    """
    def __init__(self):
        self.retriever = MultimodalRetriever()
        self.generator = RAGGenerator()
        
        # Test queries for optimization
        self.test_queries = [
            "What were the financial highlights from the last fiscal year?",
            "Explain the company's sustainability initiatives.",
            "Who are the members of the board of directors?",
            "What are the key risk factors mentioned in the report?"
        ]
        
        # Cache to avoid redundant API calls
        self.cache = {}
    
    def _get_cache_key(self, query: str, n_text: int, n_image: int, temp: float) -> str:
        """Generate a cache key for a specific parameter combination"""
        return f"{query}_{n_text}_{n_image}_{temp}"
    
    def evaluate_params(self, query: str, n_text_results: int, n_image_results: int, temperature: float) -> Dict:
        """
        Evaluate system with specific parameter settings
        
        Args:
            query: Text query
            n_text_results: Number of text chunks to retrieve
            n_image_results: Number of images to retrieve
            temperature: LLM temperature
            
        Returns:
            Dictionary with performance metrics
        """
        # Check cache first
        cache_key = self._get_cache_key(query, n_text_results, n_image_results, temperature)
        if cache_key in self.cache:
            return self.cache[cache_key]
            
        try:
            # Measure retrieval time
            retrieval_start = time.time()
            retrieval_results = self.retriever.hybrid_query(
                query=query, 
                n_text_results=n_text_results,
                n_image_results=n_image_results
            )
            retrieval_time = time.time() - retrieval_start
            
            # Measure generation time
            gen_start = time.time()
            answer_results = self.generator.generate_answer(
                query=query,
                n_text_results=n_text_results,
                n_image_results=n_image_results,
                temperature=temperature
            )
            generation_time = time.time() - gen_start
            
            # Calculate metrics
            metrics = {
                'retrieval_time': retrieval_time,
                'generation_time': generation_time,
                'total_time': retrieval_time + generation_time,
                'answer_length': len(answer_results['answer']),
                'text_results': len(retrieval_results['text_results']),
                'image_results': len(retrieval_results['image_results']),
                'sources': len(answer_results['sources'])
            }
            
            # Store in cache
            self.cache[cache_key] = metrics
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating parameters: {e}")
            return {
                'retrieval_time': 0,
                'generation_time': 0,
                'total_time': 0,
                'answer_length': 0,
                'text_results': 0,
                'image_results': 0,
                'sources': 0
            }
    
    def optimize_text_chunk_count(self, max_chunks: int = 10) -> Dict:
        """
        Find optimal number of text chunks to retrieve
        
        Args:
            max_chunks: Maximum number of chunks to try
            
        Returns:
            Dictionary with optimization results
        """
        logger.info("Optimizing text chunk count...")
        results = {
            'chunk_counts': list(range(1, max_chunks + 1)),
            'retrieval_times': [],
            'generation_times': [],
            'total_times': [],
            'answer_lengths': []
        }
        
        for n_chunks in tqdm(range(1, max_chunks + 1)):
            # Run tests with different queries and average results
            chunk_metrics = []
            for query in self.test_queries:
                metrics = self.evaluate_params(
                    query=query,
                    n_text_results=n_chunks,
                    n_image_results=3,  # Fixed for this test
                    temperature=0.1     # Fixed for this test
                )
                chunk_metrics.append(metrics)
            
            # Average metrics across queries
            avg_metrics = {
                k: np.mean([m[k] for m in chunk_metrics])
                for k in chunk_metrics[0].keys()
            }
            
            # Record results
            results['retrieval_times'].append(avg_metrics['retrieval_time'])
            results['generation_times'].append(avg_metrics['generation_time'])
            results['total_times'].append(avg_metrics['total_time'])
            results['answer_lengths'].append(avg_metrics['answer_length'])
        
        # Find optimal chunk count (balancing time and answer quality)
        optimal_idx = np.argmin(results['total_times'])
        optimal_chunks = results['chunk_counts'][optimal_idx]
        
        # Create a combined metric: (answer_length / total_time)
        efficiency = [
            a / (t if t > 0 else 1) 
            for a, t in zip(results['answer_lengths'], results['total_times'])
        ]
        efficiency_idx = np.argmax(efficiency)
        efficient_chunks = results['chunk_counts'][efficiency_idx]
        
        return {
            'results': results,
            'optimal_chunks_time': optimal_chunks,
            'optimal_chunks_efficiency': efficient_chunks
        }
    
    def optimize_image_count(self, max_images: int = 5) -> Dict:
        """
        Find optimal number of images to retrieve
        
        Args:
            max_images: Maximum number of images to try
            
        Returns:
            Dictionary with optimization results
        """
        logger.info("Optimizing image count...")
        results = {
            'image_counts': list(range(0, max_images + 1)),
            'retrieval_times': [],
            'generation_times': [],
            'total_times': []
        }
        
        for n_images in tqdm(range(0, max_images + 1)):
            # Run tests with different queries and average results
            image_metrics = []
            for query in self.test_queries:
                metrics = self.evaluate_params(
                    query=query,
                    n_text_results=5,  # Fixed based on previous optimization
                    n_image_results=n_images,
                    temperature=0.1     # Fixed for this test
                )
                image_metrics.append(metrics)
            
            # Average metrics across queries
            avg_metrics = {
                k: np.mean([m[k] for m in image_metrics])
                for k in image_metrics[0].keys()
            }
            
            # Record results
            results['retrieval_times'].append(avg_metrics['retrieval_time'])
            results['generation_times'].append(avg_metrics['generation_time'])
            results['total_times'].append(avg_metrics['total_time'])
        
        # Find optimal image count
        optimal_idx = np.argmin(results['total_times'])
        optimal_images = results['image_counts'][optimal_idx]
        
        return {
            'results': results,
            'optimal_images': optimal_images
        }
    
    def optimize_temperature(self) -> Dict:
        """
        Find optimal temperature setting for the LLM
        
        Returns:
            Dictionary with optimization results
        """
        logger.info("Optimizing LLM temperature...")
        temperatures = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
        results = {
            'temperatures': temperatures,
            'generation_times': [],
            'answer_lengths': [],
            'uniqueness_scores': []  # Higher is better - more diverse responses
        }
        
        # Storage for answer texts to compute uniqueness
        all_answers = []
        
        for temp in tqdm(temperatures):
            # Run tests with different queries and average results
            temp_metrics = []
            temp_answers = []
            
            for query in self.test_queries:
                metrics = self.evaluate_params(
                    query=query,
                    n_text_results=5,  # Fixed based on previous optimization
                    n_image_results=3,  # Fixed based on previous optimization
                    temperature=temp
                )
                temp_metrics.append(metrics)
                
                # Get the actual answer for uniqueness calculation
                answer = self.generator.generate_answer(
                    query=query,
                    n_text_results=5,
                    n_image_results=3,
                    temperature=temp
                )['answer']
                temp_answers.append(answer)
            
            # Store answers for uniqueness calculation
            all_answers.append(temp_answers)
            
            # Average metrics across queries
            avg_metrics = {
                k: np.mean([m[k] for m in temp_metrics])
                for k in temp_metrics[0].keys()
            }
            
            # Record results
            results['generation_times'].append(avg_metrics['generation_time'])
            results['answer_lengths'].append(avg_metrics['answer_length'])
        
        # Calculate uniqueness scores (comparing answers at different temperatures)
        for i, temp_answers in enumerate(all_answers):
            # Compare with answers from first temperature
            if i == 0:
                results['uniqueness_scores'].append(0)  # First set is baseline
            else:
                # Simple uniqueness score: average difference in answer length
                uniqueness = np.mean([
                    abs(len(a1) - len(a2)) / (max(len(a1), len(a2)) or 1)
                    for a1, a2 in zip(temp_answers, all_answers[0])
                ])
                results['uniqueness_scores'].append(uniqueness)
        
        # Balance between generation time and uniqueness
        # Higher score is better: uniqueness / generation_time
        balanced_scores = [
            u / (t if t > 0 else 1) 
            for u, t in zip(results['uniqueness_scores'], results['generation_times'])
        ]
        optimal_idx = np.argmax(balanced_scores)
        optimal_temp = results['temperatures'][optimal_idx]
        
        return {
            'results': results,
            'optimal_temperature': optimal_temp
        }
    
    def run_optimization(self, save_results: bool = True) -> Dict:
        """
        Run full parameter optimization
        
        Args:
            save_results: Whether to save results to disk
            
        Returns:
            Dictionary with all optimization results
        """
        # Optimize text chunk count
        text_opt = self.optimize_text_chunk_count()
        optimal_chunks = text_opt['optimal_chunks_efficiency']
        
        # Optimize image count
        image_opt = self.optimize_image_count()
        optimal_images = image_opt['optimal_images']
        
        # Optimize temperature
        temp_opt = self.optimize_temperature()
        optimal_temp = temp_opt['optimal_temperature']
        
        # Combine results
        optimization_results = {
            'text_chunk_optimization': text_opt,
            'image_count_optimization': image_opt,
            'temperature_optimization': temp_opt,
            'recommended_settings': {
                'n_text_results': optimal_chunks,
                'n_image_results': optimal_images,
                'temperature': optimal_temp
            }
        }
        
        # Save results if requested
        if save_results:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            with open(f"optimization_results_{timestamp}.json", 'w') as f:
                # Convert numpy values to Python native types for JSON serialization
                def convert_np(obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    return obj
                
                # Filter and convert results
                serializable_results = json.loads(
                    json.dumps(optimization_results, default=convert_np)
                )
                json.dump(serializable_results, f, indent=2)
            
            # Create optimization plots
            self.plot_optimization_results(text_opt, image_opt, temp_opt, timestamp)
        
        return optimization_results
    
    def plot_optimization_results(self, text_opt: Dict, image_opt: Dict, temp_opt: Dict, timestamp: str):
        """
        Create plots for optimization results
        
        Args:
            text_opt: Text chunk optimization results
            image_opt: Image count optimization results
            temp_opt: Temperature optimization results
            timestamp: Timestamp string for filename
        """
        # Create figure with 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Text chunk count vs. performance
        ax1 = axes[0, 0]
        x = text_opt['results']['chunk_counts']
        
        # Main axis: retrieval and generation times
        ax1.plot(x, text_opt['results']['retrieval_times'], 'o-', label='Retrieval Time')
        ax1.plot(x, text_opt['results']['generation_times'], 'o-', label='Generation Time')
        ax1.plot(x, text_opt['results']['total_times'], 'o-', label='Total Time')
        ax1.set_xlabel('Number of Text Chunks')
        ax1.set_ylabel('Time (seconds)')
        ax1.set_title('Text Chunk Count vs. Processing Time')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Secondary axis: answer length
        ax1b = ax1.twinx()
        ax1b.plot(x, text_opt['results']['answer_lengths'], 'r--', label='Answer Length')
        ax1b.set_ylabel('Answer Length (chars)', color='r')
        ax1b.tick_params(axis='y', labelcolor='r')
        
        # Mark optimal points
        opt_time_idx = x.index(text_opt['optimal_chunks_time'])
        opt_eff_idx = x.index(text_opt['optimal_chunks_efficiency'])
        ax1.axvline(x=x[opt_time_idx], color='g', linestyle='--', alpha=0.5)
        ax1.axvline(x=x[opt_eff_idx], color='purple', linestyle='--', alpha=0.5)
        ax1.text(x[opt_time_idx], ax1.get_ylim()[1] * 0.9, f"Fastest: {x[opt_time_idx]}", 
                 color='g', ha='center', bbox=dict(facecolor='white', alpha=0.7))
        ax1.text(x[opt_eff_idx], ax1.get_ylim()[1] * 0.8, f"Most Efficient: {x[opt_eff_idx]}", 
                 color='purple', ha='center', bbox=dict(facecolor='white', alpha=0.7))
        
        # Plot 2: Image count vs. performance
        ax2 = axes[0, 1]
        x = image_opt['results']['image_counts']
        
        ax2.plot(x, image_opt['results']['retrieval_times'], 'o-', label='Retrieval Time')
        ax2.plot(x, image_opt['results']['generation_times'], 'o-', label='Generation Time')
        ax2.plot(x, image_opt['results']['total_times'], 'o-', label='Total Time')
        ax2.set_xlabel('Number of Images')
        ax2.set_ylabel('Time (seconds)')
        ax2.set_title('Image Count vs. Processing Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Mark optimal points
        opt_img_idx = x.index(image_opt['optimal_images'])
        ax2.axvline(x=x[opt_img_idx], color='g', linestyle='--', alpha=0.5)
        ax2.text(x[opt_img_idx], ax2.get_ylim()[1] * 0.9, f"Optimal: {x[opt_img_idx]}", 
                 color='g', ha='center', bbox=dict(facecolor='white', alpha=0.7))
        
        # Plot 3: Temperature vs. performance
        ax3 = axes[1, 0]
        x = temp_opt['results']['temperatures']
        
        # Main axis: generation time
        ax3.plot(x, temp_opt['results']['generation_times'], 'o-', label='Generation Time')
        ax3.set_xlabel('Temperature')
        ax3.set_ylabel('Time (seconds)')
        ax3.set_title('Temperature vs. Performance')
        ax3.legend(loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        # Secondary axis: uniqueness score
        ax3b = ax3.twinx()
        ax3b.plot(x, temp_opt['results']['uniqueness_scores'], 'r--', label='Uniqueness Score')
        ax3b.set_ylabel('Uniqueness Score', color='r')
        ax3b.tick_params(axis='y', labelcolor='r')
        
        # Mark optimal temperature
        opt_temp_idx = x.index(temp_opt['optimal_temperature'])
        ax3.axvline(x=x[opt_temp_idx], color='g', linestyle='--', alpha=0.5)
        ax3.text(x[opt_temp_idx], ax3.get_ylim()[1] * 0.9, f"Optimal: {x[opt_temp_idx]}", 
                 color='g', ha='center', bbox=dict(facecolor='white', alpha=0.7))
        
        # Plot 4: Summary of recommended settings
        ax4 = axes[1, 1]
        ax4.axis('off')  # Turn off axis
        
        recommended = {
            'Text Chunks': text_opt['optimal_chunks_efficiency'],
            'Images': image_opt['optimal_images'],
            'Temperature': temp_opt['optimal_temperature']
        }
        
        # Create a table with recommended settings
        table_data = [['Parameter', 'Recommended Value']]
        for param, value in recommended.items():
            table_data.append([param, str(value)])
        
        table = ax4.table(
            cellText=table_data,
            loc='center',
            cellLoc='center',
            colWidths=[0.5, 0.5]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2)
        
        # Add a title to the table
        ax4.set_title('Recommended RAG System Settings', pad=20)
        
        plt.tight_layout()
        plt.savefig(f"optimization_plots_{timestamp}.png", dpi=300)
        plt.close()


# Run optimization if executed directly
if __name__ == "__main__":
    optimizer = RAGOptimizer()
    logger.info("Starting RAG system parameter optimization...")
    
    # Run optimization
    results = optimizer.run_optimization()
    
    # Print recommended settings
    print("\nRecommended Settings:")
    print(f"Text Chunks: {results['recommended_settings']['n_text_results']}")
    print(f"Images: {results['recommended_settings']['n_image_results']}")
    print(f"Temperature: {results['recommended_settings']['temperature']}")