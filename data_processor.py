# data_processor.py
import pandas as pd
import numpy as np
import torch
import re
import json
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from transformers import GPT2Tokenizer
from functools import lru_cache
from datetime import datetime
from torch.utils.data import Dataset


class ChatbotDataset(Dataset):
    """Dataset for chatbot training"""
    def __init__(self, encodings):
        self.encodings = encodings
        
    def __len__(self):
        return len(self.encodings['input_ids'])
        
    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': self.encodings['labels'][idx]
        }
        
class DataProcessor:
    def __init__(
        self, 
        data_dir: str = "./dataset",  # Changed default to match your structure
        cache_dir: Optional[str] = None
    ):
        # Convert string paths to Path objects
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else self.data_dir / 'cache'
        self.cache_dir.mkdir(exist_ok=True)
        
        # Setup logging
        import logging
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        # Initialize tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize cache
        self._knowledge_base_cache = None
        self._last_cache_update = None
        
        # Define expected columns for each dataset
        self.expected_columns = {
            'basic_info': ['name', 'value'],
            'programs': ['program_name', 'level', 'department', 'courses'],
            'facilities': ['facility_name', 'category'],
            'accreditations': ['type', 'organization'],
            'faculties': ['faculty', 'program_type'],
            'events': ['date', 'type', 'title', 'description'],
            'faculty_info': ['name', 'title', 'department', 'email', 'phone', 
                           'education', 'specialization', 'research_interests'],
            'research': ['faculty_name', 'research_interest', 'publication_title', 'publication_venue'],
            'services': ['service_name'],
            'admissions': ['program_level', 'requirement_type', 'requirement']
        }
        
    @lru_cache(maxsize=32)
    def load_datasets(self) -> Dict[str, pd.DataFrame]:
        """Load all required datasets"""
        try:
            self.logger.info("Loading datasets...")
            datasets = {}
            data_files = {
                'basic_info': 'university_basic_info.csv',
                'programs': 'programs_and_courses.csv',
                'facilities': 'facilities.csv',
                'accreditations': 'accreditations_partnerships.csv',
                'faculties': 'faculties_programs.csv',
                'events': 'events_and_news.csv',
                'faculty_info': 'faculty_info.csv',
                'research': 'research_and_publications.csv',
                'services': 'student_services.csv',
                'admissions': 'admissions_info.csv'
            }
            
            for key, filename in data_files.items():
                file_path = self.data_dir / filename
                cache_path = self.cache_dir / f"{key}_processed.parquet"
                
                if cache_path.exists():
                    datasets[key] = pd.read_parquet(cache_path)
                    self.logger.info(f"Loaded {filename} from cache")
                elif file_path.exists():
                    df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
                    # Preprocess the dataframe
                    df = self._preprocess_dataframe(df, key)
                    # Save to cache
                    df.to_parquet(cache_path)
                    datasets[key] = df
                    self.logger.info(f"Loaded and processed {filename}")
                else:
                    self.logger.warning(f"Missing dataset: {filename}")
                    
            return datasets
            
        except Exception as e:
            self.logger.error(f"Error loading datasets: {str(e)}")
            raise
            
    def _preprocess_dataframe(self, df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
        """Preprocess dataframe based on its type"""
        try:
            # Remove empty rows and columns
            df = df.dropna(how='all').dropna(axis=1, how='all')
            
            # Convert text columns to string and clean them
            text_columns = df.select_dtypes(include=['object']).columns
            for col in text_columns:
                df[col] = df[col].astype(str).apply(self.preprocess_text)
            
            # Dataset-specific preprocessing
            if dataset_type == 'faculty_info':
                # Ensure proper formatting of contact information
                if 'email' in df.columns:
                    df['email'] = df['email'].str.strip().str.lower()
                if 'phone' in df.columns:
                    df['phone'] = df['phone'].str.strip()
                    
            elif dataset_type == 'events':
                # Convert dates if present
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'], format='%d %B %Y', errors='coerce')
                    
            elif dataset_type == 'programs':
                # Clean up course listings
                if 'courses' in df.columns:
                    df['courses'] = df['courses'].str.strip()
                    
            elif dataset_type == 'basic_info':
                # Ensure proper formatting of university info
                df['name'] = df['name'].str.strip().str.lower()
                df['value'] = df['value'].str.strip()
                
            return df
            
        except Exception as e:
            self.logger.error(f"Error preprocessing {dataset_type}: {str(e)}")
            return df
            
    def create_knowledge_base(self, datasets: Dict[str, pd.DataFrame], force_refresh: bool = False) -> List[str]:
        """Create knowledge base from datasets"""
        try:
            cache_file = self.cache_dir / 'knowledge_base.json'
            
            # Check cache
            if not force_refresh and cache_file.exists():
                cache_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
                if (datetime.now() - cache_time).days < 1:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        self._knowledge_base_cache = json.load(f)
                        self.logger.info("Loaded knowledge base from cache")
                        return self._knowledge_base_cache
            
            self.logger.info("Creating knowledge base...")
            knowledge_texts = []
            
            # Process basic university information
            if 'basic_info' in datasets:
                for _, row in datasets['basic_info'].iterrows():
                    knowledge_texts.append(f"{row['name'].title()}: {row['value']}")
            
            # Process accreditations
            if 'accreditations' in datasets:
                for _, row in datasets['accreditations'].iterrows():
                    knowledge_texts.append(
                        f"The university is accredited by {row['organization']} "
                        f"as {row['type']}"
                    )
            
            # Process admission requirements
            if 'admissions' in datasets:
                admissions_df = datasets['admissions']
                for level in admissions_df['program_level'].unique():
                    level_reqs = admissions_df[admissions_df['program_level'] == level]
                    reqs_text = f"Admission requirements for {level} programs:\n"
                    for _, row in level_reqs.iterrows():
                        reqs_text += f"- For {row['requirement_type']}: {row['requirement']}\n"
                    knowledge_texts.append(reqs_text)
            
            # Process programs and courses
            if 'programs' in datasets:
                for _, row in datasets['programs'].iterrows():
                    program_text = (
                        f"The {row['program_name']} is a {row['level']} program in the "
                        f"{row['department']} department. The courses include: {row['courses']}"
                    )
                    knowledge_texts.append(program_text)
            
            # Process facilities
            if 'facilities' in datasets:
                facilities_df = datasets['facilities']
                for category in facilities_df['category'].unique():
                    facilities = facilities_df[facilities_df['category'] == category]
                    facilities_text = f"The {category} facilities include: "
                    facilities_text += ', '.join(facilities['facility_name'].tolist())
                    knowledge_texts.append(facilities_text)
            
            # Process faculty information
            if 'faculty_info' in datasets:
                for _, row in datasets['faculty_info'].iterrows():
                    faculty_text = (
                        f"{row['name']} is a {row['title']} in the {row['department']}. "
                        f"Contact: {row['email']}, {row['phone']}. "
                    )
                    if pd.notna(row['specialization']):
                        faculty_text += f"Specialization: {row['specialization']}. "
                    if pd.notna(row['research_interests']):
                        faculty_text += f"Research interests: {row['research_interests']}"
                    knowledge_texts.append(faculty_text)
            
            # Process research and publications
            if 'research' in datasets:
                for _, row in datasets['research'].iterrows():
                    research_text = (
                        f"Research by {row['faculty_name']} on {row['research_interest']}: "
                        f"'{row['publication_title']}' published in {row['publication_venue']}"
                    )
                    knowledge_texts.append(research_text)
            
            # Process events and news
            if 'events' in datasets:
                for _, row in datasets['events'].iterrows():
                    event_text = f"{row['type']}: {row['title']} on {row['date']}"
                    if pd.notna(row['description']):
                        event_text += f". {row['description']}"
                    knowledge_texts.append(event_text)
            
            # Process student services
            if 'services' in datasets:
                services_list = datasets['services']['service_name'].tolist()
                knowledge_texts.append(
                    f"Available student services: {', '.join(services_list)}"
                )
            
            # Cache the knowledge base
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(knowledge_texts, f, ensure_ascii=False, indent=2)
            
            self._knowledge_base_cache = knowledge_texts
            self._last_cache_update = datetime.now()
            
            self.logger.info(f"Created knowledge base with {len(knowledge_texts)} entries")
            return knowledge_texts
            
        except Exception as e:
            self.logger.error(f"Error creating knowledge base: {str(e)}")
            raise
            
    def _generate_questions(self, knowledge: str) -> List[str]:
        """Generate relevant questions from knowledge text"""
        questions = []
        knowledge_lower = knowledge.lower()
        
        try:
            # Program-related questions
            if 'program' in knowledge_lower or 'course' in knowledge_lower:
                try:
                    # More robust program name extraction
                    if 'the ' in knowledge:
                        program_name = knowledge.split('the ')[1].split(' is')[0]
                    else:
                        program_name = knowledge.split(' is')[0]
                except:
                    program_name = "program"  # Fallback
                    
                questions.extend([
                    f"What can you tell me about the {program_name}?",
                    f"What are the courses offered in {program_name}?",
                    "What programs do you offer?",
                    f"Tell me about the {program_name} curriculum.",
                    f"What are the admission requirements for {program_name}?",
                    f"What career opportunities are available after {program_name}?",
                    f"How long does it take to complete the {program_name}?",
                    f"What are the prerequisites for {program_name}?"
                ])
                
            # Facility-related questions
            if 'facility' in knowledge_lower or 'facilities' in knowledge_lower:
                try:
                    if 'the ' in knowledge and ' facilities include' in knowledge:
                        facility_type = knowledge.split('the ')[1].split(' facilities')[0]
                    else:
                        facility_type = "campus"  # Fallback
                except:
                    facility_type = "campus"  # Fallback
                    
                questions.extend([
                    "What facilities are available?",
                    f"Tell me about the {facility_type} facilities",
                    "What campus facilities do you have?",
                    f"What can you tell me about {facility_type} facilities?",
                    "Where are the study areas located?",
                    "What recreational facilities do you have?",
                    "Are there any research facilities?",
                    "What equipment is available in the facilities?"
                ])
                
            # Faculty-related questions
            if any(title in knowledge_lower for title in ['professor', 'dr.', 'lecturer', 'faculty']):
                try:
                    name = knowledge.split(' is ')[0] if ' is ' in knowledge else "faculty member"
                except:
                    name = "faculty member"  # Fallback
                    
                questions.extend([
                    f"Who is {name}?",
                    f"What does {name} teach?",
                    f"What is {name}'s specialization?",
                    f"How can I contact {name}?",
                    f"What research does {name} do?",
                    "Tell me about the faculty members",
                    "Who are the professors?",
                    "What are the faculty qualifications?"
                ])
                
            # Research-related questions
            if 'research' in knowledge_lower or 'publication' in knowledge_lower:
                questions.extend([
                    "What research is being conducted?",
                    "What are the recent publications?",
                    "Tell me about the research activities.",
                    "What are the research areas?",
                    "Who are the researchers?",
                    "What are the key research achievements?",
                    "Are there research opportunities for students?",
                    "What research facilities are available?"
                ])
                
            # Events-related questions
            if 'event' in knowledge_lower or 'news' in knowledge_lower:
                questions.extend([
                    "What events are happening?",
                    "Tell me about upcoming events.",
                    "What's new at the university?",
                    "Are there any events planned?",
                    "What activities are happening?",
                    "When is the next event?",
                    "What type of events do you organize?",
                    "Where are events usually held?"
                ])
                
            # Services-related questions
            if 'service' in knowledge_lower:
                questions.extend([
                    "What student services are available?",
                    "What support services do you offer?",
                    "Tell me about student services.",
                    "What services can students access?",
                    "How can students get support?",
                    "What facilities and services are provided?",
                    "Is there counseling available?",
                    "What medical services are offered?"
                ])
                
            # Admission-related questions
            if 'admission' in knowledge_lower or 'requirement' in knowledge_lower:
                questions.extend([
                    "What are the admission requirements?",
                    "How can I apply?",
                    "What documents are needed for admission?",
                    "Tell me about the application process.",
                    "What are the entry requirements?",
                    "When can I apply?",
                    "Is there an application fee?",
                    "What are the admission criteria?"
                ])
                
            # Basic information questions
            if ':' in knowledge and not questions:
                try:
                    topic = knowledge.split(':')[0].strip()
                    questions.extend([
                        f"What is the {topic}?",
                        f"Tell me about the {topic}.",
                        f"Can you provide information about {topic}?",
                        f"What do you know about {topic}?",
                        f"Give me details about {topic}."
                    ])
                except:
                    # Fallback general questions
                    questions.extend([
                        "What can you tell me about this?",
                        "Can you provide more information?",
                        "Tell me more about this topic.",
                        "What are the details?",
                        "Can you explain this further?"
                    ])
                    
            # If no specific questions generated, add general questions
            if not questions:
                questions.extend([
                    "Can you tell me more about this?",
                    "What additional information is available?",
                    "Could you explain this in detail?",
                    "What should I know about this?",
                    "Can you provide more details?"
                ])
                
            return questions
            
        except Exception as e:
            self.logger.error(f"Error generating questions: {str(e)}")
            # Return default questions if there's an error
            return [
                "What can you tell me about this?",
                "Can you provide more information?",
                "Tell me more about this.",
                "What are the important details?",
                "Could you explain this further?"
            ]
    def validate_data(self) -> bool:
        """Validate all datasets against expected structure"""
        try:
            self.logger.info("Validating datasets...")
            datasets = self.load_datasets()
            required_columns={
                'basic_info': ['name', 'value'],
                'programs': ['program_name', 'level', 'department', 'courses'],
                'facilities': ['facility_name', 'category'],
                'accreditations': ['type', 'organization'],
                'faculty_info': ['name', 'title', 'department', 'email', 'phone'],
                'events': ['date', 'type', 'title', 'description'],
                'research': ['faculty_name', 'research_interest', 'publication_title', 'publication_venue'],
                'services': ['service_name'],
                'admissions': ['program_level', 'requirement_type', 'requirement']
            }
            
            # Check each dataset
            for dataset_name, req_cols in required_columns.items():
                if dataset_name not in datasets:
                    self.logger.warning(f"Optional dataset missing: {dataset_name}")
                    continue
                    
                df = datasets[dataset_name]
                missing_cols = [col for col in req_cols if col not in df.columns]
                
                if missing_cols:
                    self.logger.error(f"Missing columns in {dataset_name}: {missing_cols}")
                    return False
                    
                # Check for empty values in required columns
                empty_counts = df[req_cols].isna().sum()
                if empty_counts.any():
                    self.logger.warning(
                        f"Found empty values in {dataset_name}:\n{empty_counts[empty_counts > 0]}"
                    )
            
            # Validate relationships between datasets
            if 'programs' in datasets and 'faculty_info' in datasets:
                program_depts = set(datasets['programs']['department'])
                faculty_depts = set(datasets['faculty_info']['department'])
                
                orphaned_depts = program_depts - faculty_depts
                if orphaned_depts:
                    self.logger.warning(f"Programs without faculty in departments: {orphaned_depts}")
            
            self.logger.info("Data validation successful!")
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating data: {str(e)}")
            return False

    @staticmethod
    def preprocess_text(text: str) -> str:
        """Clean and standardize text"""
        if not isinstance(text, str):
            return ""
            
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Standardize formatting
        text = text.replace(' :', ':')
        
        # Handle common abbreviations
        abbreviations = {
            'dept.': 'Department',
            'prof.': 'Professor',
            'dr.': 'Dr.',
            'phd': 'PhD',
            'msc': 'MSc',
            'bsc': 'BSc',
            'ba': 'BA',
            'ma': 'MA'
        }
        
        for abbr, full in abbreviations.items():
            text = re.sub(rf'\b{abbr}\b', full, text, flags=re.IGNORECASE)
        
        return text

    def prepare_training_data(
        self, 
        datasets: Dict[str, pd.DataFrame],
        validation_split: float = 0.2,
        max_examples: Optional[int] = None
    ) -> Tuple[Dataset, Dataset]:
        """Prepare training data with validation split"""
        try:
            self.logger.info("Preparing training data...")
            knowledge_base = self.create_knowledge_base(datasets)
            training_texts = []
            
            for knowledge in knowledge_base:
                questions = self._generate_questions(knowledge)
                
                for question in questions:
                    training_example = (
                        f"Question: {question}\n"
                        f"Answer: {knowledge}\n"
                        f"{self.tokenizer.eos_token}"
                    )
                    training_texts.append(training_example)
            
            # Limit number of examples if specified
            if max_examples and len(training_texts) > max_examples:
                training_texts = training_texts[:max_examples]
            
            # Tokenize all texts
            encodings = self.tokenizer(
                training_texts,
                truncation=True,
                max_length=512,
                padding='max_length',
                return_tensors='pt'
            )
            
            # Create labels
            encodings['labels'] = encodings['input_ids'].clone()
            
            # Create full dataset
            full_dataset = ChatbotDataset(encodings)
            
            # Split into train and validation
            dataset_size = len(full_dataset)
            val_size = int(dataset_size * validation_split)
            train_size = dataset_size - val_size
            
            # Use PyTorch's random_split with generators for reproducibility
            generator = torch.Generator().manual_seed(42)
            train_dataset, val_dataset = torch.utils.data.random_split(
                full_dataset, 
                [train_size, val_size],
                generator=generator
            )
            
            self.logger.info(
                f"Prepared {train_size} training examples and {val_size} validation examples"
            )
            return train_dataset, val_dataset
            
        except Exception as e:
            self.logger.error(f"Error preparing training data: {str(e)}")
            raise      
    def get_dataset_stats(self) -> Dict[str, Dict[str, int]]:
        """Get statistics about the datasets"""
        try:
            datasets = self.load_datasets()
            stats = {}
            
            for name, df in datasets.items():
                stats[name] = {
                    'rows': len(df),
                    'columns': len(df.columns),
                    'missing_values': df.isna().sum().sum(),
                    'duplicate_rows': df.duplicated().sum()
                }
                
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting dataset stats: {str(e)}")
            raise

    def refresh_cache(self) -> None:
        """Refresh all cached data"""
        try:
            self.logger.info("Refreshing cache...")
            
            # Clear function cache
            self.load_datasets.cache_clear()
            
            # Remove cached files
            for cache_file in self.cache_dir.glob('*'):
                cache_file.unlink()
                
            # Reset instance cache
            self._knowledge_base_cache = None
            self._last_cache_update = None
            
            self.logger.info("Cache refresh complete")
            
        except Exception as e:
            self.logger.error(f"Error refreshing cache: {str(e)}")
            raise