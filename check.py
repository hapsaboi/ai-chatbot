# test_chatbot.py
import logging
from pathlib import Path
from chatbot import UniversityChatbot

def test_chatbot():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize chatbot
        logger.info("Initializing chatbot...")
        chatbot = UniversityChatbot()
        
        # Test basic information queries
        test_queries = [
            "What programs do you offer?",
            "Tell me about the faculty of engineering",
            "What are the admission requirements?",
            "What facilities are available?",
            "Tell me about research activities",
            "What student services do you provide?",
            "What are the accreditations of the university?",
            "Tell me about upcoming events",
        ]
        
        logger.info("\nTesting various queries...")
        for query in test_queries:
            logger.info(f"\nQuery: {query}")
            response = chatbot.get_response(query)
            logger.info(f"Status: {response['status']}")
            logger.info(f"Confidence: {response['confidence']:.2f}")
            logger.info(f"Response: {response['response']}")
            logger.info("Relevant Knowledge:")
            for i, knowledge in enumerate(response['relevant_knowledge'], 1):
                logger.info(f"{i}. {knowledge[:100]}...")
            logger.info("-" * 80)
        
        # Test statistics
        logger.info("\nTesting chatbot statistics...")
        stats = chatbot.get_stats()
        logger.info("Chatbot Stats:")
        for key, value in stats.items():
            logger.info(f"{key}: {value}")
        
        # Test cache
        logger.info("\nTesting cache functionality...")
        # Ask the same question twice
        query = "What programs do you offer?"
        logger.info(f"First attempt - Query: {query}")
        response1 = chatbot.get_response(query)
        logger.info(f"Second attempt - Query: {query}")
        response2 = chatbot.get_response(query)
        logger.info("Cache working: " + str(response1['response'] == response2['response']))
        
        # Test knowledge refresh
        logger.info("\nTesting knowledge refresh...")
        success = chatbot.refresh_knowledge()
        logger.info(f"Knowledge refresh {'successful' if success else 'failed'}")
        
        # Test cache clearing
        logger.info("\nTesting cache clearing...")
        chatbot.clear_cache()
        stats_after = chatbot.get_stats()
        logger.info(f"Cache size after clearing: {stats_after['cache_size']}")
        
        # Test error handling
        logger.info("\nTesting error handling...")
        response = chatbot.get_response("")
        logger.info(f"Empty query response status: {response['status']}")
        
        # Test low confidence response
        logger.info("\nTesting low confidence handling...")
        response = chatbot.get_response("xyz123 completely irrelevant query")
        logger.info(f"Low confidence response status: {response['status']}")
        logger.info(f"Low confidence response: {response['response']}")
        
        logger.info("\nAll tests completed!")
        return chatbot
        
    except Exception as e:
        logger.error(f"Error during testing: {str(e)}")
        raise

if __name__ == "__main__":
    chatbot = test_chatbot()