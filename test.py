# test_trained_model.py
import logging
from chatbot import UniversityChatbot
from pathlib import Path

def test_model_responses():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Initialize chatbot
    chatbot = UniversityChatbot()
    
    # Test queries
    test_queries = [
        "What is the admission requirement for engineering?",
        "Tell me about the faculty of engineering",
        "What facilities do you have?",
        "What research areas are available?"
    ]
    
    logger.info("\nTesting responses...")
    for query in test_queries:
        logger.info(f"\nQuery: {query}")
        response = chatbot.get_response(query)
        
        logger.info(f"Status: {response['status']}")
        logger.info(f"Confidence: {response['confidence']:.2f}")
        logger.info(f"Response: {response['response']}")
        
        if 'alternative_responses' in response:
            logger.info("\nAlternative responses:")
            for i, alt in enumerate(response['alternative_responses'], 1):
                logger.info(f"{i}. {alt}")
                
        logger.info("-" * 80)

if __name__ == "__main__":
    test_model_responses()