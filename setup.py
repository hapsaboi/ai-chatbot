# scripts/setup.py
"""Initial setup script"""
import subprocess
from pathlib import Path
from config import ChatbotConfig

def setup_environment():
    """Set up the project environment"""
    # Create required directories
    for dir_path in [ChatbotConfig.MODEL_DIR, ChatbotConfig.DATA_DIR, ChatbotConfig.LOG_DIR]:
        dir_path.mkdir(exist_ok=True)
        print(f"Created directory: {dir_path}")
    
    # Install requirements
    subprocess.run(["pip", "install", "-r", "requirements.txt"])
    print("\nInstalled required packages")
    
    # Create empty .env file if it doesn't exist
    env_file = Path(".env")
    if not env_file.exists():
        env_file.write_text("""
MODEL_DIR=models
DATA_DIR=dataset
LOG_DIR=logs
BASE_MODEL=distilgpt2
EMBEDDING_MODEL=all-MiniLM-L6-v2
WANDB_PROJECT=university-chatbot
WANDB_ENTITY=your-username
        """.strip())
        print("\nCreated .env file")
    
    print("\nSetup complete!")

if __name__ == "__main__":
    setup_environment()