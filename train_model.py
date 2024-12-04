# training_flow.py
from trainer import OptimizedChatbotTrainer
from chatbot import UniversityChatbot
import logging
from pathlib import Path

def train_and_verify():
    """Train the model and verify it's being used"""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Step 1: Train the model
    logger.info("Step 1: Training the model...")
    trainer = OptimizedChatbotTrainer()
    trainer.train()
    
    # Step 2: Initialize chatbot (will automatically use trained model)
    logger.info("\nStep 2: Initializing chatbot with trained model...")
    chatbot = UniversityChatbot()
    
    # Step 3: Verify it's using the trained model
    logger.info("\nStep 3: Testing responses...")
    test_queries = [
        "What programs do you offer?",
        "Tell me about the faculty of engineering",
        "What are the admission requirements?"
    ]
    
    for query in test_queries:
        logger.info(f"\nQuery: {query}")
        response = chatbot.get_response(query)
        logger.info(f"Response: {response['response']}")
        logger.info(f"Confidence: {response['confidence']:.2f}")

def check_model_version():
    """Check which model version is being used"""
    from config import ChatbotConfig
    
    model_path = ChatbotConfig.MODEL_DIR / f"model_v{ChatbotConfig.MODEL_VERSION}"
    if model_path.exists():
        print(f"Using trained model at: {model_path}")
        # List files in model directory
        print("\nModel files:")
        for file in model_path.iterdir():
            print(f"- {file.name}")
    else:
        print("No trained model found, will use base model")

if __name__ == "__main__":
    # Check model status
    print("Checking model status:")
    check_model_version()
    
    # Ask if user wants to train
    response = input("\nDo you want to train the model? (y/n): ")
    if response.lower() == 'y':
        train_and_verify()
    else:
        print("Using existing model configuration")