# config.py
import os
from pathlib import Path
from dotenv import load_dotenv
import torch

load_dotenv()

class ChatbotConfig:
    # Paths
    BASE_DIR = Path(__file__).parent
    MODEL_DIR = BASE_DIR / os.getenv('MODEL_DIR', 'models')
    DATA_DIR = BASE_DIR / os.getenv('DATA_DIR', 'dataset')
    LOG_DIR = BASE_DIR / os.getenv('LOG_DIR', 'logs')
    
    # Ensure directories exist
    MODEL_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)
    
    # Model settings
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BASE_MODEL = os.getenv('BASE_MODEL', 'gpt2')
    CHAT_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
    HF_TOKEN = os.getenv('HF_TOKEN')
    USE_API = True  # Set to False to use local model
    EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
    MODEL_VERSION = "1.0.0"
    
    # Training settings
    LEARNING_RATE = 2e-5
    EPOCHS = 3
    TRAIN_BATCH_SIZE = 8
    EVAL_BATCH_SIZE = 16
    MAX_LENGTH = 512
    WARMUP_STEPS = 100
    GRADIENT_ACCUMULATION_STEPS = 4
    
    # Inference settings
    MAX_NEW_TOKENS = 50
    NUM_BEAMS = 3
    TOP_K = 30
    TOP_P = 0.7
    TEMPERATURE = 0.6
    CONFIDENCE_THRESHOLD = 0.4
    BATCH_SIZE = 32
    
    # Cache settings
    CACHE_SIZE = 1000
    RESPONSE_CACHE_TTL = 3600  # 1 hour
    
    # Training optimization
    FP16 = torch.cuda.is_available()  # Use mixed precision if GPU available
    GRADIENT_CHECKPOINTING = True
    NUM_WORKERS = 4  # For data loading
    
    # Logging
    WANDB_PROJECT = os.getenv('WANDB_PROJECT', 'university-chatbot')
    WANDB_ENTITY = os.getenv('WANDB_ENTITY', None)