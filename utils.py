# utils.py
import logging
from typing import Any, Dict
import json
from pathlib import Path
import torch
import numpy as np
from config import ChatbotConfig
from datetime import datetime

def setup_logging(name: str) -> logging.Logger:
    """Set up logging configuration"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(
        ChatbotConfig.LOG_DIR / f"{name}_{datetime.now():%Y%m%d}.log"
    )
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    c_handler.setFormatter(formatter)
    f_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)
    
    return logger

def save_json(data: Dict[str, Any], filepath: Path):
    """Save data to JSON file"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_json(filepath: Path) -> Dict[str, Any]:
    """Load data from JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)
