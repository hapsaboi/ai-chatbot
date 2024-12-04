import torch
from typing import Dict, List, Union, Tuple, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from functools import lru_cache
from datetime import datetime
from pathlib import Path
from config import ChatbotConfig
from data_processor import DataProcessor
from model_manager import ModelManager
from utils import setup_logging

class UniversityChatbot:
    def __init__(self, model_name: str = None, force_reload: bool = False):
        """Initialize the University Chatbot
        
        Args:
            model_name: Name of the model to use. If None, uses default Mistral API
            force_reload: Whether to force reload the knowledge base
        """
        self.logger = setup_logging(__name__)
        self.data_processor = DataProcessor()
        self.model_manager = ModelManager(model_name=model_name)  # Pass model_name here
        self.init_time = datetime.now()
        
        # Initialize knowledge base and embeddings
        self._load_knowledge_base(force_reload)
        
        # Initialize cache
        self.response_cache = {}
        
    def _load_knowledge_base(self, force_reload: bool = False) -> None:
        """Load knowledge base and compute embeddings"""
        try:
            self.logger.info("Loading knowledge base...")
            datasets = self.data_processor.load_datasets()
            self.knowledge_base = self.data_processor.create_knowledge_base(datasets)
            
            embeddings_path = ChatbotConfig.MODEL_DIR / f"embeddings_v{ChatbotConfig.MODEL_VERSION}.pt"
            
            if not force_reload and embeddings_path.exists():
                self.logger.info("Loading pre-computed embeddings...")
                self.knowledge_embeddings = torch.load(embeddings_path)
            else:
                self.logger.info("Computing new embeddings...")
                self.knowledge_embeddings = self.model_manager.get_embeddings(
                    self.knowledge_base,
                    batch_size=ChatbotConfig.BATCH_SIZE
                )
                embeddings_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(self.knowledge_embeddings, embeddings_path)
                
            self.logger.info("Knowledge base loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading knowledge base: {str(e)}")
            raise

    def _get_relevant_knowledge(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.3
    ) -> Tuple[List[str], float]:
        """Get relevant knowledge entries for the query"""
        try:
            query_embedding = self.model_manager.get_embeddings([query])
            
            similarities = cosine_similarity(
                query_embedding.cpu().numpy(),
                self.knowledge_embeddings.cpu().numpy()
            )[0]
            
            valid_indices = np.where(similarities >= threshold)[0]
            top_indices = valid_indices[np.argsort(similarities[valid_indices])[-top_k:]][::-1]
            
            if len(top_indices) == 0:
                return [], 0.0
                
            relevant_knowledge = [self.knowledge_base[i] for i in top_indices]
            confidence = float(similarities[top_indices[0]])
            
            return relevant_knowledge, confidence
            
        except Exception as e:
            self.logger.error(f"Error getting relevant knowledge: {str(e)}")
            return [], 0.0

    def _identify_query_type(self, query: str) -> str:
        """Identify the type of query"""
        query_lower = query.lower()
        
        type_keywords = {
            'admission': ['admission', 'requirements', 'apply', 'entry', 'enroll'],
            'program': ['program', 'course', 'degree', 'curriculum', 'study'],
            'faculty': ['faculty', 'professor', 'teacher', 'lecturer', 'dr'],
            'facility': ['facility', 'facilities', 'campus', 'amenities'],
            'location': ['location', 'address', 'where', 'located', 'place'],
            'research': ['research', 'publication', 'paper', 'journal']
        }
        
        for query_type, keywords in type_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                return query_type
                
        return 'general'

    def _create_context(self, query: str, relevant_knowledge: List[str], query_type: str) -> str:
        """Create context for the model"""
        # Group information by category
        categorized_info = {}
        for info in relevant_knowledge:
            if ': ' in info:
                category, detail = info.split(': ', 1)
                if category not in categorized_info:
                    categorized_info[category] = []
                categorized_info[category].append(detail)
            else:
                if 'General' not in categorized_info:
                    categorized_info['General'] = []
                categorized_info['General'].append(info)

        # Build context
        context = "Information:\n"
        for category, details in categorized_info.items():
            if details:  # Only add categories that have information
                context += f"\n{category}:\n"
                for detail in details:
                    context += f"- {detail}\n"
        
        context += f"\nQuestion: {query}"
        
        return context

    @lru_cache(maxsize=ChatbotConfig.CACHE_SIZE)
    def get_response(
        self,
        query: str,
        confidence_threshold: float = 0.4
    ) -> Dict[str, Union[str, List[str], float]]:
        """Generate response using the model"""
        try:
            # Get relevant knowledge
            relevant_knowledge, confidence = self._get_relevant_knowledge(query)
            query_type = self._identify_query_type(query)
            
            # self.logger.info(f"Query type identified as: {query_type}")
            # self.logger.info(f"Confidence score: {confidence}")
            
            # If we have specific knowledge, use it
            if relevant_knowledge and confidence >= confidence_threshold:
                context = self._create_factual_context(query, relevant_knowledge)
                response = self.model_manager.generate_response(context)
                return {
                    'status': 'success',
                    'response': response,
                    'confidence': confidence,
                    'relevant_knowledge': relevant_knowledge
                }
            
            # For general guidance, keep focus on our university
            context = (
                "You are an information assistant for Emirates Aviation University. "
                "While answering questions where specific information isn't available in our database, "
                "keep these guidelines in mind:\n"
                "1. Focus on our university's context as an aviation-focused institution\n"
                "2. Don't make specific claims about our programs or facilities\n"
                "3. Provide general guidance that aligns with our aviation and engineering focus\n"
                "4. Don't mention or compare with other universities\n"
                "5. If the question is unrelated to university education, politely explain that you "
                "assist with university and education-related questions\n\n"
                f"Question: {query}\n\n"
                "Remember: Provide helpful guidance without making specific claims about our "
                "programs, facilities, or policies that aren't in our database."
            )
            
            response = self.model_manager.generate_response(context)
            return {
                'status': 'general_guidance',
                'response': response,
                'confidence': 0.3,
                'relevant_knowledge': []
            }
                
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            return {
                'status': 'error',
                'response': "I encountered an error while processing your question.",
                'confidence': 0.0,
                'relevant_knowledge': []
            }
            
    def _create_factual_context(self, query: str, relevant_knowledge: List[str]) -> str:
        """Create context for factual responses"""
        return (
            "You are Emirates Aviation University's information assistant. Answer using ONLY "
            "the following verified information. Do not add or infer any additional information.\n\n"
            "Verified Information:\n" + 
            "\n".join(f"- {info}" for info in relevant_knowledge) +
            f"\n\nQuestion: {query}"
        )
    def refresh_knowledge(self) -> bool:
        """Refresh the knowledge base and embeddings"""
        try:
            self._load_knowledge_base(force_reload=True)
            self.response_cache.clear()
            self.get_response.cache_clear()
            return True
        except Exception as e:
            self.logger.error(f"Error refreshing knowledge: {str(e)}")
            return False

    def get_stats(self) -> Dict[str, Union[int, float, str]]:
        """Get chatbot statistics"""
        return {
            'knowledge_base_size': len(self.knowledge_base),
            'cache_size': len(self.response_cache),
            'cache_hits': self.get_response.cache_info().hits,
            'cache_misses': self.get_response.cache_info().misses,
            'uptime': str(datetime.now() - self.init_time),
            'model_version': ChatbotConfig.MODEL_VERSION,
            'last_refresh': self.init_time.isoformat(),
            'api_mode': self.model_manager.use_api
        }