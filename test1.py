# test_chatbot.py
import logging
from pathlib import Path
from data_processor import DataProcessor

def test_chatbot():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize processor
        processor = DataProcessor(data_dir="./dataset")
        
        # Load and validate data
        logger.info("Loading and validating data...")
        datasets = processor.load_datasets()
        is_valid = processor.validate_data()
        
        if not is_valid:
            logger.error("Data validation failed!")
            return
            
        # Create knowledge base
        logger.info("\nCreating knowledge base...")
        knowledge_base = processor.create_knowledge_base(datasets)
        
        # Print some examples
        logger.info("\nSample knowledge entries:")
        for entry in knowledge_base[:3]:
            logger.info(f"- {entry}")
            
        # Generate some questions
        logger.info("\nSample questions for first knowledge entry:")
        questions = processor._generate_questions(knowledge_base[0])
        for q in questions[:3]:
            logger.info(f"Q: {q}")
            
        # Get dataset statistics
        logger.info("\nDataset statistics:")
        stats = processor.get_dataset_stats()
        for dataset, stat in stats.items():
            logger.info(f"{dataset}: {stat}")
            
        return processor, datasets, knowledge_base
        
    except Exception as e:
        logger.error(f"Error during testing: {str(e)}")
        raise

if __name__ == "__main__":
    processor, datasets, knowledge_base = test_chatbot()