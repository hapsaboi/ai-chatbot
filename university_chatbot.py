# import torch
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, GPT2LMHeadModel, GPT2Tokenizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
from typing import List, Tuple, Dict, Union
import re
import logging
from datetime import datetime
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chatbot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class UniversityChatbot:
    def __init__(self, data_path: str = "dataset/"):
        """Initialize the University Chatbot"""
        logger.info("Initializing University Chatbot...")
        
        # Initialize device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize models
        self._initialize_models()
        
        # Load data
        self.data_path = data_path
        self._load_data()
        
        # Create knowledge base and embeddings
        self.knowledge_base = self._create_knowledge_base()
        self._initialize_embeddings()
        
        # Initialize category patterns
        self.category_patterns = {
            'location': r'where|location|address|campus',
            'programs': r'program|course|study|degree|major',
            'facilities': r'facility|facilities|amenities|campus|building',
            'faculty': r'professor|teacher|instructor|faculty|staff|teach',
            'admissions': r'admission|apply|enroll|requirement|criteria',
            'research': r'research|publication|paper|study|project',
            'events': r'event|activity|workshop|seminar|conference',
            'services': r'service|support|help|assistance|aid'
        }
        
        logger.info("Chatbot initialization complete!")

    def _initialize_models(self):
        """Initialize transformer models"""
        try:
            logger.info("Loading embedding model...")
            self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            
            logger.info("Loading language model...")
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = GPT2LMHeadModel.from_pretrained('gpt2-medium').to(self.device)
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            raise

    def _load_data(self):
        """Load all required datasets"""
        try:
            logger.info("Loading datasets...")
            self.basic_info = pd.read_csv(f'{self.data_path}university_basic_info.csv', encoding='utf-8', on_bad_lines='skip')
            self.programs = pd.read_csv(f'{self.data_path}programs_and_courses.csv', encoding='utf-8', on_bad_lines='skip')
            self.facilities = pd.read_csv(f'{self.data_path}facilities.csv', encoding='utf-8', on_bad_lines='skip')
            self.accreditations = pd.read_csv(f'{self.data_path}accreditations_partnerships.csv', encoding='utf-8', on_bad_lines='skip')
            self.faculties = pd.read_csv(f'{self.data_path}faculties_programs.csv', encoding='utf-8', on_bad_lines='skip')
            self.events = pd.read_csv(f'{self.data_path}events_and_news.csv', encoding='utf-8', on_bad_lines='skip')
            self.faculty_info = pd.read_csv(f'{self.data_path}faculty_info.csv', encoding='utf-8', on_bad_lines='skip')
            self.features = pd.read_csv(f'{self.data_path}key_features.csv', encoding='utf-8', on_bad_lines='skip')
            self.research = pd.read_csv(f'{self.data_path}research_and_publications.csv', encoding='utf-8', on_bad_lines='skip')
            self.services = pd.read_csv(f'{self.data_path}student_services.csv', encoding='utf-8', on_bad_lines='skip')
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def _create_knowledge_base(self) -> List[str]:
        """Create knowledge base from loaded data"""
        logger.info("Creating knowledge base...")
        knowledge_texts = []
        
        try:
            # Process basic information
            university_name = self.basic_info.loc[self.basic_info['name'] == 'university_name', 'value'].iloc[0]
            location = self.basic_info.loc[self.basic_info['name'] == 'location', 'value'].iloc[0]
            chancellor = self.basic_info.loc[self.basic_info['name'] == 'chancellor', 'value'].iloc[0]
            vice_chancellor = self.basic_info.loc[self.basic_info['name'] == 'vice_chancellor', 'value'].iloc[0]
            founding_year = self.basic_info.loc[self.basic_info['name'] == 'founding_year', 'value'].iloc[0]
            
            knowledge_texts.append(f"{university_name} is located in {location}")
            knowledge_texts.append(f"The university was founded in {founding_year}")
            knowledge_texts.append(f"The Chancellor is {chancellor}")
            knowledge_texts.append(f"The Vice Chancellor is {vice_chancellor}")
            
            # Process faculty programs
            for faculty in self.faculties['faculty'].unique():
                faculty_programs = self.faculties[self.faculties['faculty'] == faculty]['program_type'].tolist()
                program_text = f"The {faculty} offers the following programs: {', '.join(faculty_programs)}"
                knowledge_texts.append(program_text)
            
            # Process programs
            for _, row in self.programs.iterrows():
                program_text = (f"The {row['program_name']} is a {row['level']} program in {row['department']}. "
                              f"Courses include: {row['courses']}")
                knowledge_texts.append(program_text)
            
            # Process facilities
            for category in self.facilities['category'].unique():
                category_facilities = self.facilities[self.facilities['category'] == category]['facility_name'].tolist()
                facility_text = f"The {category} facilities include: {', '.join(category_facilities)}"
                knowledge_texts.append(facility_text)
            
            # Process accreditations and partnerships
            accred_list = self.accreditations[self.accreditations['type'] == 'accreditation']['organization'].tolist()
            partner_list = self.accreditations[self.accreditations['type'] == 'partnership']['organization'].tolist()
            knowledge_texts.append(f"The university is accredited by: {', '.join(accred_list)}")
            knowledge_texts.append(f"The university has partnerships with: {', '.join(partner_list)}")
            
            # Process faculty information
            for _, row in self.faculty_info.iterrows():
                faculty_text = f"Dr. {row['name']} is a {row['title']} in the {row['department']}"
                if pd.notna(row['specialization']):
                    faculty_text += f", specializing in {row['specialization']}"
                if pd.notna(row['research_interests']):
                    faculty_text += f". Research interests include: {row['research_interests']}"
                knowledge_texts.append(faculty_text)
            
            # Process research
            for _, row in self.research.iterrows():
                if pd.notna(row['publication_title']) and pd.notna(row['publication_venue']):
                    research_text = f"Research: {row['publication_title']} published in {row['publication_venue']}"
                    knowledge_texts.append(research_text)
            
            # Process services
            services_list = self.services['service_name'].tolist()
            knowledge_texts.append(f"Student services available: {', '.join(services_list)}")
            
            logger.info(f"Created knowledge base with {len(knowledge_texts)} entries")
            return knowledge_texts
            
        except Exception as e:
            logger.error(f"Error creating knowledge base: {str(e)}")
            raise

    def _initialize_embeddings(self):
        """Initialize embeddings for the knowledge base"""
        try:
            logger.info("Creating knowledge embeddings...")
            self.knowledge_embeddings = self.embedding_model.encode(
                self.knowledge_base,
                batch_size=32,
                show_progress_bar=True,
                convert_to_tensor=True
            )
        except Exception as e:
            logger.error(f"Error creating embeddings: {str(e)}")
            raise

    def preprocess_query(self, query: str) -> str:
        """Clean and standardize the query"""
        query = query.lower().strip()
        query = re.sub(r'[^\w\s]', ' ', query)
        query = re.sub(r'\s+', ' ', query)
        return query

    def get_query_category(self, query: str) -> str:
        """Determine the category of the query"""
        query = self.preprocess_query(query)
        for category, pattern in self.category_patterns.items():
            if re.search(pattern, query):
                return category
        return 'general'

    def get_relevant_knowledge(self, query: str, top_k: int = 3) -> Tuple[List[str], np.ndarray]:
        """Get relevant knowledge based on query category and semantic similarity"""
        query = self.preprocess_query(query)
        category = self.get_query_category(query)
        
        # Get query embedding
        query_embedding = self.embedding_model.encode([query], convert_to_tensor=True)
        
        # Calculate similarities
        with torch.no_grad():
            similarities = cosine_similarity(
                query_embedding.cpu().numpy(),
                self.knowledge_embeddings.cpu().numpy()
            )[0]
        
        # Boost similarity scores based on category matching
        boosted_similarities = similarities.copy()
        for i, knowledge in enumerate(self.knowledge_base):
            if category != 'general' and category in knowledge.lower():
                boosted_similarities[i] *= 1.2  # Category match boost
                
            # Term matching boost
            query_terms = set(query.lower().split())
            knowledge_terms = set(knowledge.lower().split())
            matching_terms = query_terms.intersection(knowledge_terms)
            if matching_terms:
                term_boost = 1 + (len(matching_terms) / len(query_terms)) * 0.3
                boosted_similarities[i] *= term_boost
        
        # Get top-k relevant knowledge
        top_indices = np.argsort(boosted_similarities)[-top_k:][::-1]
        relevant_knowledge = [self.knowledge_base[i] for i in top_indices]
        relevant_similarities = similarities[top_indices]
        
        return relevant_knowledge, relevant_similarities

    def format_response(self, response: str) -> str:
        """Clean and format the response"""
        response = re.sub(r'You are.*?so:', '', response, flags=re.DOTALL)
        response = re.sub(r'Question:.*?Answer:', '', response, flags=re.DOTALL)
        response = re.sub(r'http\S+', '', response)
        response = re.sub(r'Q:.*', '', response)
        response = re.sub(r'A:.*', '', response)
        response = re.sub(r'\s+', ' ', response).strip()
        
        if response and not response.endswith(('.', '!', '?')):
            response += '.'
                
        return response

    def get_response(self, query: str) -> Dict[str, Union[str, List[str], float]]:
        """Generate a response to the given query"""
        try:
            # Get relevant knowledge
            relevant_knowledge, similarities = self.get_relevant_knowledge(query)
            
            # Check confidence
            confidence_threshold = 0.3
            if not relevant_knowledge or np.max(similarities) < confidence_threshold:
                return {
                    'status': 'success',
                    'response': "I apologize, but I don't have enough information to answer that question accurately.",
                    'confidence': float(np.max(similarities)) if similarities.size > 0 else 0.0,
                    'relevant_knowledge': []
                }
            # Prepare context
            context = (
                "You are helping a student with questions about Emirates Aviation University. "
                "You should not add information that is not relevant to the current question"
                "You dont have to use all the relevant knowledge, some of it might not be correct, dont fabricate new information"
                "Use ONLY the following facts to answer - if you're not sure, say so:\n\n"
                f"{' '.join(relevant_knowledge)}\n\n"
                f"Question: {query}\n"
                "Answer: "
            )
            
            # Generate response
            inputs = self.tokenizer(context, return_tensors='pt', truncation=True, 
                                  max_length=1024).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    num_beams=5,
                    no_repeat_ngram_size=3,
                    top_k=50,
                    top_p=0.85,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    early_stopping=True
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = self.format_response(response)
            
            confidence = float(np.max(similarities))
            
            return {
                'status': 'success',
                'response': response,
                'confidence': confidence,
                'relevant_knowledge': relevant_knowledge
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'response': "I apologize, but I encountered an error while processing your question.",
                'confidence': 0.0,
                'relevant_knowledge': []
            }

    def get_response_confidence(self, response: str, query: str) -> float:
        """Calculate confidence score for a response"""
        try:
            with torch.no_grad():
                response_embedding = self.embedding_model.encode([response])
                query_embedding = self.embedding_model.encode([query])
                confidence = cosine_similarity(response_embedding, query_embedding)[0][0]
            return float(confidence)
        except Exception as e:
            logger.error(f"Error calculating confidence: {str(e)}")
            return 0.0

if __name__ == "__main__":
    # Example usage
    chatbot = UniversityChatbot()
    result = chatbot.get_response("What engineering programs do you offer?")
    print(json.dumps(result, indent=2))