import os
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional
import chromadb
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer
import numpy as np
from dotenv import load_dotenv
from datetime import datetime
import easyocr

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SentenceTransformerEmbeddingFunction:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        self.model.to(self.device)
        
    def __call__(self, input: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts using SentenceTransformer model
        Args:
            input: List of texts to generate embeddings for
        Returns:
            List of embeddings as float arrays
        """
        try:
            # Generate embeddings for all texts at once
            embeddings = self.model.encode(input, convert_to_tensor=True)
            # Convert to numpy, normalize, and convert to list
            embeddings_np = embeddings.cpu().numpy()
            normalized_embeddings = embeddings_np / np.linalg.norm(embeddings_np, axis=1, keepdims=True)
            return normalized_embeddings.tolist()
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

class ContentLabeler:
    """Helper class to classify and label content using ML models"""
    
    # Content type categories
    CONTENT_TYPES = {
        'financial': ['revenue', 'profit', 'margin', 'balance', 'income', 'cash flow'],
        'operational': ['operations', 'process', 'workflow', 'efficiency'],
        'strategic': ['strategy', 'vision', 'mission', 'goals'],
        'technological': ['technology', 'digital', 'innovation', 'IT'],
        'sustainability': ['ESG', 'environmental', 'social', 'governance'],
    }
    
    @staticmethod
    def detect_content_type(text: str) -> List[str]:
        """Detect content types based on keyword presence"""
        text_lower = text.lower()
        detected_types = []
        
        for content_type, keywords in ContentLabeler.CONTENT_TYPES.items():
            if any(keyword in text_lower for keyword in keywords):
                detected_types.append(content_type)
                
        return detected_types if detected_types else ['general']
    
    @staticmethod
    def get_metadata(text: str, filename: str) -> Dict:
        """Generate metadata for a chunk of text"""
        # Parse page and chunk numbers from filename
        match = re.match(r'text_(\d+)_(\d+)\.txt', filename)
        if not match:
            raise ValueError(f"Invalid filename format: {filename}")
        
        page_num, chunk_num = map(int, match.groups())
        
        return {
            'page_number': page_num,
            'chunk_number': chunk_num,
            'content_types': ','.join(ContentLabeler.detect_content_type(text)),
            'timestamp': datetime.now().isoformat(),
            'word_count': len(text.split()),
            'char_count': len(text)
        }

class ImageAnalyzer:
    """Handles image analysis using CLIP model"""
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model.to(self.device)
        self.reader = easyocr.Reader(['en'])  # Initialize EasyOCR

    def extract_text(self, image_path: str) -> str:
        """Extract text from image using EasyOCR"""
        try:
            results = self.reader.readtext(image_path)
            # Combine all detected text with spaces
            extracted_text = ' '.join([text[1] for text in results])
            return extracted_text
        except Exception as e:
            logger.error(f"Error extracting text from image {image_path}: {e}")
            return ""

    def analyze_image(self, image_path: str, max_retries: int = 3) -> Dict:
        """Analyze an image and return its description, type, and extracted text"""
        categories = [
            "a table or spreadsheet",
            "a graph or chart",
            "a diagram or flowchart",
            "a photograph",
            "an illustration",
            "a logo or brand image",
            "a map",
            "text or document"
        ]
        
        for attempt in range(max_retries):
            try:
                # Load and convert image
                image = Image.open(image_path).convert('RGB')
                
                # Extract text from image
                extracted_text = self.extract_text(image_path)
                
                # Process image and text
                inputs = self.processor(
                    images=image,
                    text=categories,
                    return_tensors="pt",
                    padding=True
                )
                
                # Move inputs to the same device as model
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    
                    if not hasattr(outputs, 'logits_per_image') or outputs.logits_per_image is None:
                        raise ValueError("Model did not produce valid logits")
                        
                    logits = outputs.logits_per_image
                    
                    if logits.shape[0] == 0:
                        raise ValueError("Empty logits tensor")
                        
                    # Get probabilities
                    probs = logits.softmax(dim=-1)
                    probs_np = probs.cpu().numpy()
                    
                    if probs_np.shape[-1] != len(categories):
                        raise ValueError(f"Unexpected probability shape: {probs_np.shape}")
                    
                    # Get the most likely category
                    category_idx = int(probs_np[0].argmax())
                    category = categories[category_idx]
                    confidence = float(probs_np[0][category_idx])
                
                # Generate meaningful descriptions based on category and extracted text
                description = ""
                if category == "a table or spreadsheet":
                    description = f"Table containing data. Extracted content: {extracted_text[:200]}..."
                elif category == "a graph or chart":
                    description = f"Graph/Chart visualization. Labels and values: {extracted_text[:200]}..."
                elif category == "a diagram or flowchart":
                    description = f"Diagram/Flowchart showing: {extracted_text[:200]}..."
                else:
                    description = f"This image appears to be {category}"
                
                return {
                    "type": category,
                    "description": description,
                    "confidence": confidence,
                    "extracted_text": extracted_text
                }
                
            except Exception as e:
                if attempt == max_retries - 1:  # Last attempt
                    logger.error(f"Error analyzing image {image_path} after {max_retries} attempts: {e}")
                    return {
                        "type": "unknown",
                        "description": "",
                        "confidence": 0.0,
                        "extracted_text": ""
                    }
                else:
                    logger.warning(f"Attempt {attempt + 1} failed for {image_path}: {e}. Retrying...")
                    continue

class EmbeddingsProcessor:
    def __init__(self):
        load_dotenv()
        
        # Initialize CLIP-based image analyzer
        self.image_analyzer = ImageAnalyzer()
        
        # Initialize ChromaDB
        self.db_path = "chroma_db"
        self.client = chromadb.PersistentClient(path=self.db_path)
        
        # Initialize embedding function using SentenceTransformer model
        self.embedding_function = SentenceTransformerEmbeddingFunction()
        
        self._setup_collections(force_recreate=True)
    
    def _setup_collections(self, force_recreate: bool = False):
        """Set up ChromaDB collections for text and images"""
        # Delete existing collections if force_recreate
        if force_recreate:
            try:
                self.client.delete_collection("text_embeddings")
                self.client.delete_collection("image_embeddings")
            except:
                pass
        
        # Text collection
        self.text_collection = self.client.create_collection(
            name="text_embeddings",
            metadata={"description": "Text embeddings from documents"},
            embedding_function=self.embedding_function
        )
        
        # Image collection  
        self.image_collection = self.client.create_collection(
            name="image_embeddings",
            metadata={"description": "Image embeddings and metadata"}
        )
    
    def get_text_embedding(self, text: str) -> List[float]:
        """Generate embeddings for text using SentenceTransformer model"""
        return self.embedding_function([text])[0]
    
    def process_data_folder(self, data_folder: str = "data", image_folder: str = "pic_data") -> None:
        """Process all text files and images in the data folders"""
        # Process text files
        data_path = Path(data_folder)
        if not data_path.exists():
            raise ValueError(f"Data folder not found: {data_folder}")
        
        # Get all text files
        text_files = sorted(data_path.glob("text_*_*.txt"))
        
        logger.info("Processing text files...")
        for file_path in text_files:
            try:
                # Read the text content
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                
                if not content:
                    logger.warning(f"Empty content in {file_path}")
                    continue
                
                # Generate metadata
                metadata = ContentLabeler.get_metadata(content, file_path.name)
                
                # Generate embedding
                embedding = self.get_text_embedding(content)
                
                # Add to ChromaDB
                self.text_collection.add(
                    documents=[content],
                    embeddings=[embedding],
                    metadatas=[metadata],
                    ids=[file_path.stem]
                )
                
                logger.info(f"Processed text file: {file_path.name}")
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                continue

        # Process images
        image_path = Path(image_folder)
        if not image_path.exists():
            logger.warning(f"Image folder not found: {image_folder}")
            return
            
        # Get all image files
        image_files = sorted(image_path.glob("page_*_img_*.png"))
        
        logger.info("Processing images...")
        for image_file in image_files:
            try:
                # Extract page number from filename
                page_match = re.match(r'page_(\d+)_img_(\d+)', image_file.stem)
                if not page_match:
                    logger.warning(f"Invalid image filename format: {image_file}")
                    continue
                    
                page_num = int(page_match.group(1))
                img_num = int(page_match.group(2))
                
                # Analyze image
                analysis = self.image_analyzer.analyze_image(str(image_file))
                
                # Create metadata
                metadata = {
                    "page_number": page_num,
                    "image_number": img_num,
                    "image_path": str(image_file),
                    "type": analysis["type"],
                    "description": analysis["description"],
                    "confidence": analysis["confidence"],
                    "extracted_text": analysis["extracted_text"],
                    "timestamp": datetime.now().isoformat()
                }
                
                # Add to ChromaDB
                self.image_collection.add(
                    documents=[str(image_file)],
                    metadatas=[metadata],
                    ids=[image_file.stem]
                )
                
                logger.info(f"Processed image: {image_file.name} - Type: {analysis['type']}" + 
                          (f", Description: {analysis['description']}" if analysis['description'] else "") +
                          f", Confidence: {analysis['confidence']:.2f}" +
                          (f", Extracted Text: {analysis['extracted_text']}" if analysis['extracted_text'] else ""))
                
            except Exception as e:
                logger.error(f"Error processing image {image_file}: {e}")
                continue
    
    def process_image(self, image_path: str, page_number: int) -> Dict:
        """Process a single image and store its analysis"""
        # Analyze image
        analysis = self.image_analyzer.analyze_image(image_path)
        
        # Create metadata
        metadata = {
            "page_number": page_number,
            "image_path": str(image_path),
            "type": analysis["type"],
            "description": analysis["description"],
            "confidence": analysis["confidence"],
            "extracted_text": analysis["extracted_text"],
            "timestamp": datetime.now().isoformat()
        }
        
        # Add to ChromaDB
        self.image_collection.add(
            documents=[str(image_path)],
            embeddings=[],  # No embeddings for now
            metadatas=[metadata],
            ids=[Path(image_path).stem]
        )
        
        logger.info(f"Processed image {image_path}")
        return metadata
    
    def query_similar_content(self, query: str, n_results: int = 5) -> List[Dict]:
        """Query the database for similar content"""
        try:
            # Use the same embedding function as the collection
            query_embedding = self.embedding_function([query])[0]
            
            results = self.text_collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
            
            return [
                {
                    'text': doc,
                    'metadata': meta,
                    'distance': dist
                }
                for doc, meta, dist in zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )
            ]
            
        except Exception as e:
            logger.error(f"Error querying database: {e}")
            raise

def main():
    try:
        processor = EmbeddingsProcessor()
        processor.process_data_folder()
        
        # Example query
        query_results = processor.query_similar_content(
            "What are the key financial highlights?",
            n_results=3
        )
        
        print("\nExample Query Results:")
        for i, result in enumerate(query_results, 1):
            print(f"\nResult {i}:")
            print(f"Distance: {result['distance']:.4f}")
            print(f"Page: {result['metadata']['page_number']}")
            print(f"Content Types: {result['metadata']['content_types']}")
            print(f"Text Preview: {result['text'][:200]}...")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    main()