# check_model.py
import logging
from chatbot import UniversityChatbot
from config import ChatbotConfig
from pathlib import Path

def check_current_model():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Check for trained model
        model_path = ChatbotConfig.MODEL_DIR / f"model_v{ChatbotConfig.MODEL_VERSION}"
        logger.info(f"Checking for trained model at: {model_path}")
        
        if model_path.exists():
            logger.info("✓ Trained model found!")
            # Initialize chatbot (will use trained model)
            chatbot = UniversityChatbot()
            
            # Test a query
            query = "What programs do you offer?"
            logger.info(f"\nTesting query: {query}")
            response = chatbot.get_response(query)
            
            logger.info("\nResponse details:")
            logger.info(f"Status: {response['status']}")
            logger.info(f"Confidence: {response['confidence']:.2f}")
            logger.info(f"Response: {response['response']}")
            
        else:
            logger.warning("✗ No trained model found - will use base model")
            logger.info(f"Base model: {ChatbotConfig.BASE_MODEL}")
            
    except Exception as e:
        logger.error(f"Error checking model: {str(e)}")

if __name__ == "__main__":
    check_current_model()